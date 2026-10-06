"""Milestone V2.2 — de-circularization of the direct-weight -> substitutability rule.

The reviewer's core objection is that the graft source is built from the INTACT
output command (``command_rate = baseline_rate / baseline_max``), so "direct weight
predicts substitutability" could be near-tautological: the source already encodes the
output, and a high direct-weight site can trivially re-inject it. This experiment
shows the rule survives when the recovery task is HELD OUT (unseen at fit time).

Two sub-experiments, both on READOUT substitutability (oracle-only, zebrafish,
``weight_ranked`` lesion at 50%):

  **(A) Held-out condition — cross-condition prediction (PRIMARY).** Run the
  per-neuron scan at 3 stimulus frequencies (5, 6, 8 Hz). For every ordered pair
  (A,B), A≠B, fit ``substitutability ~ direct_weight`` by OLS on A and use the
  (slope, intercept) to predict substitutability on B; report held-out R² and the
  Spearman rank correlation between A's and B's substitutability (ranking transfer).

  **(B) Held-out source — temporal split of the mapping (SECONDARY).** At the
  primary frequency (5 Hz), fit the source->target mapping ``W`` on the first 70%
  of the active window and evaluate the graft on the remaining 30% (unseen time).

Reuses the graft pipeline from :mod:`cybergraft.analysis.grafting` and the
``_select_lesion`` / ``_permutation_p`` helpers from
:mod:`cybergraft.analysis.lesion_robustness`; does not modify either module.
"""

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.experiment.bypass_helpers import (
    _build_stimulus,
    _run_fixed_drive,
    generate_informative_source,
)
from cybergraft.experiment.bypass_training import output_rate, rate_similarity
from cybergraft.experiment.lesion import Lesion
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard

from cybergraft.analysis.grafting import _resolve_ids
from cybergraft.analysis.lesion_robustness import _fit, _permutation_p, _select_lesion

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024)"


def _run_seed_context(
    graph,
    lesioned_graph,
    parameters,
    stimulus_cfg: Dict[str, Any],
    groups: Dict[str, list],
    output_ids: List[int],
    seed: int,
    dt_ms: float,
    duration_ms: float,
    n_steps: int,
    n_source: int,
    onset_ms: float,
    duration_ms_src: float,
) -> Dict[str, Any]:
    """Run one seed's intact + lesioned sims and build the informative source.

    Returns the stimulus, intact/lesioned output rates, reward baseline, missing
    drive, and the (full-window) source + least-squares drive.
    """
    stimulus = _build_stimulus(stimulus_cfg, groups, seed, dt_ms)
    act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
    baseline_rate = output_rate(act0.spikes, output_ids)
    act_les = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
    lesioned_rate = output_rate(act_les.spikes, output_ids)
    reward_baseline = rate_similarity(lesioned_rate, baseline_rate)
    missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)

    baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
    command_rate = (baseline_rate / baseline_max).astype(np.float64) if baseline_max > 0 else np.zeros_like(baseline_rate, dtype=np.float64)
    source = generate_informative_source(n_steps, n_source, 0.0, command_rate, dt_ms, onset_ms, duration_ms_src, seed)
    w_offline, _, _, _ = np.linalg.lstsq(source.astype(np.float64), missing, rcond=None)
    drive = source @ w_offline
    return {
        "stimulus": stimulus,
        "baseline_rate": baseline_rate,
        "lesioned_rate": lesioned_rate,
        "reward_baseline": reward_baseline,
        "missing": missing,
        "source": source,
        "drive": drive,
    }


def run_decircularization(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 3,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Held-out condition + held-out source de-circularization of the rule."""
    experiment_id = f"v2-decircularization-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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
    n_seeds = int(config["reproducibility"].get("n_seeds", n_seeds))
    base_seed = int(config["experiment"]["seed"])

    dec = config.get("decircularization", {})
    freqs = [float(f) for f in dec.get("condition_frequencies", [5.0, 6.0, 8.0])]
    split_frac = float(dec.get("split_frac", 0.7))
    n_perm = int(dec.get("n_permutations", 500))
    freq_keys = [f"{f:g}" for f in freqs]
    primary_freq = freqs[0]

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
        dt_ms = parameters.dt_ms
        n_steps = int(round(duration_ms / dt_ms))

        output_ids, lesion_ids = _resolve_ids(graph, data["readout_type"], data["lesion_type"])
        graph.groups = {"output": output_ids, "lesion": lesion_ids, "input": [n.id for n in graph.nodes if n.cell_type == data["input_type"]]}

        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        mapping_gain = float(config["mapping"]["gain"])

        # Lesion is frequency-independent: weight_ranked @ 50%.
        rng = np.random.default_rng(base_seed)
        lesion_target = _select_lesion(graph, lesion_ids, drive_onto, float(data["lesion_fraction"]), "weight_ranked", rng)
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)
        lesion_set = set(lesion_target)
        output_set = set(output_ids)

        neurons = [
            n.id for n in graph.nodes
            if n.id not in output_set and n.id not in lesion_set and drive_onto[n.id] > 0
        ]
        X = np.asarray([drive_onto[j] for j in neurons], dtype=np.float64)

        perm_rng = np.random.default_rng(base_seed + 1_000_000)

        # ---- Part A: per-frequency per-neuron scan (full-window graftability) ----
        Y: Dict[str, np.ndarray] = {}
        in_sample: Dict[str, Dict[str, Any]] = {}
        for f in freqs:
            key = f"{f:g}"
            cfg_f = deepcopy(config)
            cfg_f["stimulus"]["frequency_hz"] = f

            per_neuron_gain: Dict[int, list] = {j: [] for j in neurons}
            per_seed_headroom: List[float] = []
            for i in range(n_seeds):
                ctx = _run_seed_context(
                    graph, lesioned_graph, parameters, cfg_f["stimulus"], graph.groups,
                    output_ids, base_seed + i, dt_ms, duration_ms, n_steps, n_source,
                    source_onset_ms, source_duration_ms,
                )
                per_seed_headroom.append(1.0 - float(ctx["reward_baseline"]))
                for j in neurons:
                    spikes = _run_fixed_drive(lesioned_graph, parameters, [j], ctx["drive"], mapping_gain, duration_ms, ctx["stimulus"])
                    rec = rate_similarity(output_rate(spikes, output_ids), ctx["baseline_rate"])
                    per_neuron_gain[j].append(float(rec - ctx["reward_baseline"]))
                guard.check(f"after seed {i} @ {key}Hz", logger)

            headroom = float(np.mean(per_seed_headroom)) if per_seed_headroom else 0.0
            ys = np.asarray([
                (float(np.mean(per_neuron_gain[j])) / headroom) if headroom > 0 else float("nan")
                for j in neurons
            ], dtype=np.float64)
            Y[key] = ys

            finite = np.isfinite(ys)
            xs_f = X[finite]
            ys_f = ys[finite]
            slope, intercept, r, r2 = _fit(xs_f, ys_f)
            p_perm = float("nan")
            if np.isfinite(slope) and xs_f.size >= 2 and np.std(xs_f) > 0.0:
                p_perm, _ = _permutation_p(xs_f, ys_f, slope, n_perm, perm_rng)
            in_sample[key] = {
                "frequency_hz": f,
                "slope": slope,
                "intercept": intercept,
                "r": r,
                "r_squared": r2,
                "p_perm": p_perm,
                "n_neurons": int(xs_f.size),
                "headroom": headroom,
            }
            logger.info(
                "condition %s Hz: slope=%+.5f r=%+.4f R2=%.4f p_perm=%.4f n=%d headroom=%.4f",
                key, slope, r, r2, p_perm, int(xs_f.size), headroom,
            )

        # ---- cross-condition: held-out R² + Spearman (3x3 matrices) ----
        held_out_r2: Dict[str, Dict[str, Any]] = {}
        spearman: Dict[str, Dict[str, Any]] = {}
        offdiag_r2: List[float] = []
        offdiag_rho: List[float] = []
        offdiag_rho_p: List[float] = []
        for ka in freq_keys:
            held_out_r2[ka] = {}
            spearman[ka] = {}
            for kb in freq_keys:
                if ka == kb:
                    held_out_r2[ka][kb] = None
                    spearman[ka][kb] = 1.0
                    continue
                mask = np.isfinite(Y[ka]) & np.isfinite(Y[kb])
                xa = X[mask]
                ya = Y[ka][mask]
                yb = Y[kb][mask]
                if xa.size < 2 or np.std(xa) == 0.0:
                    held_out_r2[ka][kb] = float("nan")
                    spearman[ka][kb] = float("nan")
                    continue
                slope_ab, intercept_ab = np.polyfit(xa, ya, 1)
                pred_b = slope_ab * xa + intercept_ab
                ss_res = float(np.sum((yb - pred_b) ** 2))
                ss_tot = float(np.sum((yb - np.mean(yb)) ** 2))
                r2_ho = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
                rho_ab, p_ab = spearmanr(ya, yb)
                held_out_r2[ka][kb] = r2_ho
                spearman[ka][kb] = float(rho_ab)
                offdiag_r2.append(r2_ho)
                offdiag_rho.append(float(rho_ab))
                offdiag_rho_p.append(float(p_ab))

        mean_held_out_r2 = float(np.mean(offdiag_r2)) if offdiag_r2 else float("nan")
        mean_spearman = float(np.mean(offdiag_rho)) if offdiag_rho else float("nan")
        mean_spearman_p = float(np.max(offdiag_rho_p)) if offdiag_rho_p else float("nan")

        # ---- Part B: held-out source (temporal split) at the primary frequency ----
        onset_step = int(round(source_onset_ms / dt_ms))
        end_step = min(n_steps, onset_step + int(round(source_duration_ms / dt_ms)))
        active_steps = max(1, end_step - onset_step)
        split = onset_step + int(round(split_frac * active_steps))

        per_neuron_test_gain: Dict[int, list] = {j: [] for j in neurons}
        per_seed_test_headroom: List[float] = []
        for i in range(n_seeds):
            ctx = _run_seed_context(
                graph, lesioned_graph, parameters, config["stimulus"], graph.groups,
                output_ids, base_seed + i, dt_ms, duration_ms, n_steps, n_source,
                source_onset_ms, source_duration_ms,
            )
            src = ctx["source"].astype(np.float64)
            missing = ctx["missing"]
            w_split, _, _, _ = np.linalg.lstsq(src[:split], missing[:split], rcond=None)
            drive_test = src[split:] @ w_split
            drive_full = np.zeros(n_steps, dtype=np.float64)
            drive_full[split:] = drive_test

            reward_baseline_test = rate_similarity(ctx["lesioned_rate"][split:], ctx["baseline_rate"][split:])
            per_seed_test_headroom.append(1.0 - float(reward_baseline_test))
            for j in neurons:
                spikes = _run_fixed_drive(lesioned_graph, parameters, [j], drive_full, mapping_gain, duration_ms, ctx["stimulus"])
                rec_test = rate_similarity(output_rate(spikes, output_ids)[split:], ctx["baseline_rate"][split:])
                per_neuron_test_gain[j].append(float(rec_test - reward_baseline_test))
            guard.check(f"after held-out source seed {i}", logger)

        headroom_test = float(np.mean(per_seed_test_headroom)) if per_seed_test_headroom else 0.0
        ys_test = np.asarray([
            (float(np.mean(per_neuron_test_gain[j])) / headroom_test) if headroom_test > 0 else float("nan")
            for j in neurons
        ], dtype=np.float64)
        finite_t = np.isfinite(ys_test)
        xs_t = X[finite_t]
        ys_t = ys_test[finite_t]
        slope_t, intercept_t, r_t, r2_t = _fit(xs_t, ys_t)
        p_perm_t = float("nan")
        if np.isfinite(slope_t) and xs_t.size >= 2 and np.std(xs_t) > 0.0:
            p_perm_t, _ = _permutation_p(xs_t, ys_t, slope_t, n_perm, perm_rng)
        held_out_source = {
            "frequency_hz": primary_freq,
            "split_frac": split_frac,
            "split_step": split,
            "slope": slope_t,
            "intercept": intercept_t,
            "r": r_t,
            "r_squared": r2_t,
            "p_perm": p_perm_t,
            "n_neurons": int(xs_t.size),
            "headroom": headroom_test,
        }
        logger.info(
            "held-out source (%g Hz, split=%.2f): slope=%+.5f r=%+.4f R2=%.4f p_perm=%.4f n=%d headroom=%.4f",
            primary_freq, split_frac, slope_t, r_t, r2_t, p_perm_t, int(xs_t.size), headroom_test,
        )

        # ---- verdict ----
        held_out_r2_pass = bool(mean_held_out_r2 > 0)
        all_held_out_r2_positive = bool(all(v > 0 for v in offdiag_r2))
        spearman_pass = bool(mean_spearman > 0 and mean_spearman_p < 0.05)
        held_out_source_pass = bool(np.isfinite(slope_t) and slope_t > 0 and p_perm_t < 0.05)
        part_a_pass = bool(held_out_r2_pass and spearman_pass)
        overall_pass = bool(part_a_pass and held_out_source_pass)
        verdict = {
            "part_a_held_out_r2_pass": held_out_r2_pass,
            "all_held_out_r2_positive": all_held_out_r2_positive,
            "part_a_spearman_pass": spearman_pass,
            "part_b_held_out_source_pass": held_out_source_pass,
            "overall_pass": overall_pass,
        }

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "V2.2",
            "backend": "oracle",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_seeds": n_seeds,
            "conditions": freqs,
            "in_sample": in_sample,
            "held_out_r2": held_out_r2,
            "spearman": spearman,
            "mean_held_out_r2": mean_held_out_r2,
            "mean_spearman": mean_spearman,
            "mean_spearman_p": mean_spearman_p,
            "held_out_source": held_out_source,
            "verdict": verdict,
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
            milestone="V2.2",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": (
                    "Milestone V2.2 de-circularization: does the direct-weight -> substitutability "
                    "rule survive when the recovery task is held out? (A) cross-condition prediction "
                    "at 5/6/8 Hz (fit on A, predict B, held-out R2 + Spearman ranking transfer); "
                    "(B) held-out source: temporal split (split_frac=0.7) of the offline-LS source->"
                    "target mapping, fit on the first 70% of the active window and evaluated on the last 30%."
                ),
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)

        logger.info("mean held-out R2=%.4f, mean Spearman rho=%.4f (p<=%.4g)", mean_held_out_r2, mean_spearman, mean_spearman_p)
        logger.info("verdict=%s", verdict)
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
            milestone="V2.2",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("de-circularization experiment failed")
        raise


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
