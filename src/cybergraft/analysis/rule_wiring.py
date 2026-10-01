"""Direction B: does the specific wiring constrain which learning rules are viable?

Runs a 4-rule sweep (offline_ls / online_ls / rstdp / rstdp_temporal) on the real
connectome and on null models (weight_shuffle, cell_type_shuffle), then regresses
the efficiency ratio ``rho = rstdp_gain / oracle_gain`` on wiring concentration.

Hypotheses (see newIdea.md §5):
  H1: ``rho`` decreases with concentration ``conc`` (wiring modulates the rule gap).
  H2: the per-step-reward rule ``rstdp_temporal`` flattens the slope (mechanism is
      temporal-credit collapse, not saturation/alignment).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.experiment.bypass_helpers import (
    _build_stimulus,
    _run_fixed_drive,
    generate_informative_source,
)
from cybergraft.experiment.bypass_selection import select_bypass_candidates
from cybergraft.experiment.bypass_training import (
    output_rate,
    rate_similarity,
    run_rstdp_training,
)
from cybergraft.experiment.lesion import Lesion
from cybergraft.experiment.mapping_interface import MappingInterface
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard

from .null_models import shuffle_cell_types, weight_shuffle

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024)"

RULES = ["offline_ls", "online_ls", "rstdp", "rstdp_temporal"]


def _resolve_ids(graph, readout_type, lesion_type, input_type):
    def ids(ct):
        return [n.id for n in graph.nodes if n.cell_type == ct]
    return ids(readout_type), ids(lesion_type), ids(input_type)


def _graph_features(graph, output_ids, lesion_ids, candidate_ids):
    W = graph.weights
    drive = np.asarray(W[output_ids, :].sum(axis=0)).ravel()
    total = float(W[output_ids, :].sum())
    lesioned = float(W[np.ix_(output_ids, lesion_ids)].sum())
    conc = lesioned / total if total > 0 else 0.0
    cand = np.asarray(W[np.ix_(output_ids, candidate_ids)].sum(axis=0)).ravel() if candidate_ids else np.zeros(0, dtype=np.float64)
    s = float(cand.sum())
    hhi = float(np.sum((cand / s) ** 2)) if s > 0 else 0.0
    pos = np.sort(drive[drive > 0])[::-1]
    hub_frac = float(pos[:5].sum() / total) if total > 0 else 0.0
    coo = W.tocoo()
    edges = set(zip(coo.row.tolist(), coo.col.tolist()))
    recip = float(sum(1 for (i, j) in edges if (j, i) in edges) / len(edges)) if edges else 0.0
    n = graph.node_count
    sparsity = float(W.nnz) / (n * (n - 1)) if n > 1 else 0.0
    return {"conc": conc, "hhi": hhi, "hub_frac": hub_frac, "reciprocity": recip, "sparsity": sparsity}


def _sweep_rules_on_graph(
    graph,
    *,
    readout_type,
    lesion_type,
    input_type,
    lesion_fraction,
    max_candidates,
    n_source,
    parameters,
    duration_ms,
    stimulus_cfg,
    map_cfg,
    n_seeds,
    base_seed,
):
    output_ids, lesion_ids, input_ids = _resolve_ids(graph, readout_type, lesion_type, input_type)
    graph.groups = {"output": output_ids, "lesion": lesion_ids, "input": input_ids}

    dt_ms = parameters.dt_ms
    n_steps = int(round(duration_ms / dt_ms))

    drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
    ranked = sorted(lesion_ids, key=lambda n: (-drive_onto[n], n))
    n_lesion = int(np.ceil(lesion_fraction * len(lesion_ids)))
    lesion_target = ranked[:n_lesion]
    lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

    sel = select_bypass_candidates(
        lesioned_graph,
        output_group="output",
        lesion_group="lesion",
        max_candidates=max_candidates,
        seed=base_seed,
    )
    candidate_ids = sel["candidate_ids"]
    if not candidate_ids:
        return None
    n_target = len(candidate_ids)

    feats = _graph_features(graph, output_ids, lesion_ids, candidate_ids)

    source_onset_ms = float(stimulus_cfg["start_ms"])
    source_duration_ms = float(stimulus_cfg["duration_ms"])

    mapping_kwargs = {
        "weight_min": float(map_cfg["weight_min"]),
        "tau_pre_ms": float(map_cfg["tau_pre_ms"]),
        "tau_post_ms": float(map_cfg["tau_post_ms"]),
        "init_scale": float(map_cfg["init_scale"]),
    }
    mapping_gain = float(map_cfg["gain"])
    rstdp_lr = float(map_cfg.get("rstdp_lr", 0.05))
    temporal_lr_scale = float(map_cfg.get("temporal_lr_scale", 1.0))
    n_episodes = int(map_cfg.get("n_episodes", 20))
    tau_elig_ms = float(map_cfg.get("tau_elig_ms", 20.0))
    weight_max = float(map_cfg["weight_max"])
    delta_lr = float(map_cfg.get("delta_lr", 0.01))
    delta_epochs = int(map_cfg.get("delta_epochs", 300))

    per_seed_gain = {r: [] for r in RULES}
    per_seed_recovery = {r: [] for r in RULES}
    per_seed_lesioned = []

    for i in range(n_seeds):
        seed = base_seed + i
        stimulus = _build_stimulus(stimulus_cfg, graph.groups, seed, dt_ms)
        act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
        baseline_rate = output_rate(act0.spikes, output_ids)
        act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
        lesioned_rate = output_rate(act_lesion.spikes, output_ids)
        reward_baseline = rate_similarity(lesioned_rate, baseline_rate)
        per_seed_lesioned.append(float(reward_baseline))
        missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)

        baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
        command_rate = (baseline_rate / baseline_max).astype(np.float64) if baseline_max > 0 else np.zeros_like(baseline_rate, dtype=np.float64)

        source = generate_informative_source(
            n_steps, n_source, 0.0, command_rate, dt_ms,
            source_onset_ms, source_duration_ms, seed,
        )
        source_f = source.astype(np.float64)

        w_offline, _, _, _ = np.linalg.lstsq(source_f, missing, rcond=None)
        offline_spikes = _run_fixed_drive(
            lesioned_graph, parameters, candidate_ids, source @ w_offline, mapping_gain, duration_ms, stimulus
        )
        r_off = rate_similarity(output_rate(offline_spikes, output_ids), baseline_rate)
        per_seed_recovery["offline_ls"].append(float(r_off))
        per_seed_gain["offline_ls"].append(float(r_off - reward_baseline))

        w = np.zeros(n_source, dtype=np.float64)
        online_curve = []
        for _ in range(delta_epochs):
            error = missing - source_f @ w
            w = w + delta_lr * (source_f.T @ error / n_steps)
            np.clip(w, 0.0, weight_max, out=w)
            online_spikes = _run_fixed_drive(
                lesioned_graph, parameters, candidate_ids, source @ w, mapping_gain, duration_ms, stimulus
            )
            online_curve.append(float(rate_similarity(output_rate(online_spikes, output_ids), baseline_rate)))
        r_on = online_curve[-1]
        per_seed_recovery["online_ls"].append(float(r_on))
        per_seed_gain["online_ls"].append(float(r_on - reward_baseline))

        mapping = MappingInterface(n_source=n_source, n_target=n_target, weight_max=weight_max, seed=seed, **mapping_kwargs)
        res, _ = run_rstdp_training(
            graph=lesioned_graph, parameters=parameters, mapping=mapping,
            target_node_ids=candidate_ids, source_signal=source, mapping_gain=mapping_gain,
            rstdp_lr=rstdp_lr, duration_ms=duration_ms, baseline_rate=baseline_rate,
            reward_baseline=reward_baseline, n_episodes=n_episodes, stimulus=stimulus,
            tau_elig_ms=tau_elig_ms,
        )
        r_rstdp = rate_similarity(output_rate(res, output_ids), baseline_rate)
        per_seed_recovery["rstdp"].append(float(r_rstdp))
        per_seed_gain["rstdp"].append(float(r_rstdp - reward_baseline))

        mapping_t = MappingInterface(n_source=n_source, n_target=n_target, weight_max=weight_max, seed=seed, **mapping_kwargs)
        res_t, _ = run_rstdp_training(
            graph=lesioned_graph, parameters=parameters, mapping=mapping_t,
            target_node_ids=candidate_ids, source_signal=source, mapping_gain=mapping_gain,
            rstdp_lr=rstdp_lr * temporal_lr_scale, duration_ms=duration_ms, baseline_rate=baseline_rate,
            reward_baseline=reward_baseline, n_episodes=n_episodes, stimulus=stimulus,
            tau_elig_ms=tau_elig_ms, temporal_reward=True,
        )
        r_tmp = rate_similarity(output_rate(res_t, output_ids), baseline_rate)
        per_seed_recovery["rstdp_temporal"].append(float(r_tmp))
        per_seed_gain["rstdp_temporal"].append(float(r_tmp - reward_baseline))

    out: Dict[str, Any] = {"features": feats}
    out["lesioned_recovery"] = float(np.mean(per_seed_lesioned))
    out["headroom"] = 1.0 - out["lesioned_recovery"]
    for r in RULES:
        gains = np.asarray(per_seed_gain[r], dtype=np.float64)
        out[f"{r}_gain"] = float(gains.mean())
        out[f"{r}_gain_std"] = float(gains.std())
        out[f"{r}_recovery"] = float(np.asarray(per_seed_recovery[r]).mean())
    oracle_g = out["offline_ls_gain"]
    out["oracle_gain"] = oracle_g
    if oracle_g >= 0.02:
        out["rho"] = out["rstdp_gain"] / oracle_g
        out["rho_temporal"] = out["rstdp_temporal_gain"] / oracle_g
        out["rho_online"] = out["online_ls_gain"] / oracle_g
        out["delta"] = (oracle_g - out["rstdp_gain"]) / out["headroom"] if out["headroom"] > 0 else float("nan")
    else:
        out["rho"] = out["rho_temporal"] = out["rho_online"] = float("nan")
        out["delta"] = float("nan")
    return out


def _regress(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if xs.size < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "n": int(xs.size), "p_perm": float("nan")}
    slope, intercept = np.polyfit(xs, ys, 1)
    obs = abs(slope)
    rng = np.random.default_rng(0)
    count = 0
    n_perm = 1000
    for _ in range(n_perm):
        p = np.polyfit(rng.permutation(xs), ys, 1)[0]
        if abs(p) >= obs:
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    return {"slope": float(slope), "intercept": float(intercept), "n": int(xs.size), "p_perm": float(p_perm)}


def run_rule_wiring_sweep(
    config: Dict[str, Any],
    project_root: Path,
    n_nulls: int = 30,
    n_seeds: int = 3,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
    nulls: Optional[list] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = f"p6-rule-wiring-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
    result_dir = project_root / config["experiment"]["output_dir"] / experiment_id
    result_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_logging(result_dir / "run.log", verbose=verbose)
    guard = MemoryGuard(
        warning_gb=float(config["runtime"]["memory_warning_gb"]),
        abort_gb=float(config["runtime"]["memory_abort_gb"]),
    )
    mlx_smoke = run_mlx_smoke_test()

    data = config["data"]
    mat = Path(mat_path) if mat_path else (project_root / data["mat_file"])
    n_nulls = int(data.get("n_nulls", n_nulls))
    n_seeds = int(config["reproducibility"].get("n_seeds", n_seeds))
    base_seed = int(config["experiment"]["seed"])

    try:
        adapter = ZebrafishAdapter(
            mat_path=mat,
            max_nodes=int(data["max_nodes"]),
            synaptic_delay_ms=float(config["simulation"]["synaptic_delay_ms"]),
        )
        graph, provenance = adapter.build_full_subgraph(
            readout_type=data["readout_type"],
            lesion_type=data["lesion_type"],
            input_type=data["input_type"],
            max_nodes=int(data["max_nodes"]),
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        duration_ms = float(config["simulation"]["duration_ms"])

        kwargs = dict(
            readout_type=str(data["readout_type"]),
            lesion_type=str(data["lesion_type"]),
            input_type=str(data["input_type"]),
            lesion_fraction=float(data["lesion_fraction"]),
            max_candidates=int(data["max_bypass_candidates"]),
            n_source=int(data["n_source"]),
            parameters=parameters,
            duration_ms=duration_ms,
            stimulus_cfg=config["stimulus"],
            map_cfg=config["mapping"],
            n_seeds=n_seeds,
            base_seed=base_seed,
        )

        rows: list = []
        real = _sweep_rules_on_graph(graph, **kwargs)
        if real is not None:
            rows.append({"label": "real", **real, **{f"f_{k}": v for k, v in real["features"].items()}})

        null_pairs = {
            "weight_shuffle": weight_shuffle,
            "cell_type_shuffle": shuffle_cell_types,
        }
        null_names = list(nulls) if nulls is not None else list(null_pairs.keys())
        for name in null_names:
            generator = null_pairs[name]
            for i in range(n_nulls):
                null_graph = generator(graph, seed=base_seed + i)
                row = _sweep_rules_on_graph(null_graph, **kwargs)
                if row is None:
                    continue
                rows.append({"label": name, **row, **{f"f_{k}": v for k, v in row["features"].items()}})
                guard.check(f"after {name} {i}", logger)

        concs = [r["features"]["conc"] for r in rows]
        rho = [r["rho"] for r in rows]
        rho_t = [r["rho_temporal"] for r in rows]

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M6",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_nulls": n_nulls,
            "n_seeds": n_seeds,
            "n_graphs": len(rows),
            "real": {k: v for k, v in rows[0].items() if k != "features"} if rows else None,
            "regression_rho_on_conc": _regress(concs, rho),
            "regression_rho_temporal_on_conc": _regress(concs, rho_t),
            "rows": [
                {
                    "label": r["label"],
                    "conc": r["features"]["conc"],
                    "rho": r["rho"],
                    "rho_temporal": r["rho_temporal"],
                    "rho_online": r["rho_online"],
                    "delta": r["delta"],
                    "oracle_gain": r["oracle_gain"],
                    "headroom": r["headroom"],
                    **r["features"],
                }
                for r in rows
            ],
        }
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed",
            milestone="M6",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "Direction B: 4-rule sweep (offline_ls/online_ls/rstdp/rstdp_temporal) on real + null graphs; regress rho on concentration.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished rule-wiring sweep: n_graphs=%d, rho~conc slope=%.4f (p=%.4f), rho_temporal~conc slope=%.4f (p=%.4f)",
            len(rows),
            metrics["regression_rho_on_conc"]["slope"],
            metrics["regression_rho_on_conc"]["p_perm"],
            metrics["regression_rho_temporal_on_conc"]["slope"],
            metrics["regression_rho_temporal_on_conc"]["p_perm"],
        )
        return result_dir, metrics

    except Exception as exc:
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=0,
            edge_count=0,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="error",
            milestone="M6",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("rule-wiring sweep failed")
        raise


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
