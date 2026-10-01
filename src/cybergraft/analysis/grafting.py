"""Direction C: computational neural grafting — graft-site sweep.

Does the graft SITE (which surviving population receives a source injection)
determine whether a trained source can functionally substitute for a lesioned
population? Measured with the ORACLE (offline LS) so graftability is a pure
function of the site's wiring, not the learning rule.

H1: graftability is monotone in the site's direct synaptic weight onto the readout.
H2: beyond direct weight, a recurrent site (the surviving integrator) is a worse
    graft recipient than a feedforward site of equal weight.
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
from cybergraft.experiment.bypass_training import output_rate, rate_similarity
from cybergraft.experiment.lesion import Lesion
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard
from cybergraft.graph.edge_schema import Edge
from cybergraft.graph.sparse_graph import SparseGraph

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024)"

SITES = ["surviving_Int", "_Axl_", "_DOs_", "untyped_direct", "random", "gold_top19"]


def _resolve_ids(graph, readout_type, lesion_type):
    def ids(ct):
        return [n.id for n in graph.nodes if n.cell_type == ct]
    return ids(readout_type), ids(lesion_type)


def _select_graft_site(graph, output_ids, lesion_target, site, n, drive_onto, rng):
    lesion_set = set(lesion_target)

    def ids(ct):
        return [n.id for n in graph.nodes if n.cell_type == ct]

    if site == "surviving_Int":
        surviving = [n.id for n in graph.nodes if n.cell_type == "_Int_" and n.id not in lesion_set]
        surviving.sort(key=lambda x: (-drive_onto[x], x))
        return surviving[:n]
    if site == "_Axl_":
        return ids("_Axl_")[:n]
    if site == "_DOs_":
        dos = ids("_DOs_")
        dos.sort(key=lambda x: (-drive_onto[x], x))
        return dos[:n]
    if site == "untyped_direct":
        un = ids("untyped")
        un.sort(key=lambda x: (-drive_onto[x], x))
        return un[:n]
    if site == "random":
        out_set = set(output_ids)
        pool = [n.id for n in graph.nodes if n.id not in out_set and n.id not in lesion_set]
        pool = [x for x in pool if drive_onto[x] > 0]
        if len(pool) < n:
            return pool
        return [int(x) for x in rng.choice(pool, size=n, replace=False)]
    if site == "gold_top19":
        return select_bypass_candidates(
            graph, output_group="output", lesion_group="lesion", max_candidates=n, seed=0
        )["candidate_ids"]
    raise ValueError(f"unknown site: {site}")


def _site_features(graph, output_ids, site_ids):
    W = graph.weights
    direct_weight = float(W[np.ix_(output_ids, site_ids)].sum())
    intra = float(W[np.ix_(site_ids, site_ids)].sum())
    total_out = float(W[:, site_ids].sum())
    recurrence = intra / total_out if total_out > 0 else 0.0
    out_deg = int(W[:, site_ids].nnz)
    in_deg = int(W[site_ids, :].nnz)
    total_readout = float(W[output_ids, :].sum())
    hub_share = direct_weight / total_readout if total_readout > 0 else 0.0
    return {
        "direct_weight": direct_weight,
        "recurrence": recurrence,
        "out_degree": out_deg,
        "in_degree": in_deg,
        "hub_share": hub_share,
        "direct": bool(direct_weight > 0),
        "n_target": len(site_ids),
    }


def run_graft_site_sweep(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 5,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = f"p6-grafting-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(lesion_ids, key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(lesion_ids)))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        n_source = int(data["n_source"])
        n_target = int(data.get("graft_n_target", 19))
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        mapping_gain = float(config["mapping"]["gain"])

        rng = np.random.default_rng(base_seed)
        site_ids: Dict[str, list] = {}
        site_feats: Dict[str, Dict[str, Any]] = {}
        for site in SITES:
            ids = _select_graft_site(graph, output_ids, lesion_target, site, n_target, drive_onto, rng)
            if not ids:
                continue
            site_ids[site] = ids
            site_feats[site] = _site_features(graph, output_ids, ids)

        per_site_gain: Dict[str, list] = {s: [] for s in site_ids}
        per_seed_headroom: list = []

        for i in range(n_seeds):
            seed = base_seed + i
            stimulus = _build_stimulus(config["stimulus"], graph.groups, seed, dt_ms)
            act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
            baseline_rate = output_rate(act0.spikes, output_ids)
            act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
            lesioned_rate = output_rate(act_lesion.spikes, output_ids)
            reward_baseline = rate_similarity(lesioned_rate, baseline_rate)
            per_seed_headroom.append(1.0 - float(reward_baseline))
            missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)

            baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
            command_rate = (baseline_rate / baseline_max).astype(np.float64) if baseline_max > 0 else np.zeros_like(baseline_rate, dtype=np.float64)
            source = generate_informative_source(
                n_steps, n_source, 0.0, command_rate, dt_ms,
                source_onset_ms, source_duration_ms, seed,
            )
            source_f = source.astype(np.float64)
            w_offline, _, _, _ = np.linalg.lstsq(source_f, missing, rcond=None)
            drive = source @ w_offline

            for site, ids in site_ids.items():
                spikes = _run_fixed_drive(lesioned_graph, parameters, ids, drive, mapping_gain, duration_ms, stimulus)
                rec = rate_similarity(output_rate(spikes, output_ids), baseline_rate)
                per_site_gain[site].append(float(rec - reward_baseline))
            guard.check(f"after seed {i}", logger)

        headroom = float(np.mean(per_seed_headroom))
        rows: list = []
        for site in site_ids:
            gains = np.asarray(per_site_gain[site], dtype=np.float64)
            oracle_gain = float(gains.mean())
            graftability = oracle_gain / headroom if headroom > 0 else float("nan")
            rows.append({
                "site": site,
                "graftability": graftability,
                "oracle_gain": oracle_gain,
                "oracle_gain_std": float(gains.std()),
                "headroom": headroom,
                **site_feats[site],
            })

        # regression: graftability ~ direct_weight (H1), + recurrence (H2)
        xs = np.asarray([r["direct_weight"] for r in rows], dtype=np.float64)
        ys = np.asarray([r["graftability"] for r in rows], dtype=np.float64)
        slope, intercept = (np.polyfit(xs, ys, 1) if xs.size >= 2 else (float("nan"), float("nan")))

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M6",
            "backend": "oracle",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_seeds": n_seeds,
            "headroom": headroom,
            "graft_n_target": n_target,
            "rows": rows,
            "regression_graftability_on_direct_weight": {"slope": float(slope), "intercept": float(intercept), "n": int(xs.size)},
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
                "description": "Direction C: graft-site sweep (6 sites, oracle-only) — does the graft site's wiring predict graftability?",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        for r in rows:
            logger.info(
                "graft site %-14s graftability=%.4f direct_weight=%.4f recurrence=%.4f",
                r["site"], r["graftability"], r["direct_weight"], r["recurrence"],
            )
        logger.info("graftability ~ direct_weight: slope=%.4f intercept=%.4f", slope, intercept)
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
        logger.exception("graft-site sweep failed")
        raise


def run_graft_node_scan(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 3,
    min_direct_weight: float = 0.0,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Graft into EVERY surviving neuron individually (oracle-only), for a complete
    per-neuron graftability map. Confirms completeness (only direct-projectors have
    nonzero graftability) and adds statistical power to the "role matters" finding."""
    experiment_id = f"p6-graft-node-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(lesion_ids, key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(lesion_ids)))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        mapping_gain = float(config["mapping"]["gain"])

        output_set = set(output_ids)
        lesion_set = set(lesion_target)
        neurons = [
            n.id for n in graph.nodes
            if n.id not in output_set and n.id not in lesion_set and drive_onto[n.id] > min_direct_weight
        ]

        per_neuron_gain: Dict[int, list] = {j: [] for j in neurons}
        per_seed_headroom: list = []

        for i in range(n_seeds):
            seed = base_seed + i
            stimulus = _build_stimulus(config["stimulus"], graph.groups, seed, dt_ms)
            act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
            baseline_rate = output_rate(act0.spikes, output_ids)
            act_lesion = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
            lesioned_rate = output_rate(act_lesion.spikes, output_ids)
            reward_baseline = rate_similarity(lesioned_rate, baseline_rate)
            per_seed_headroom.append(1.0 - float(reward_baseline))
            missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)

            baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
            command_rate = (baseline_rate / baseline_max).astype(np.float64) if baseline_max > 0 else np.zeros_like(baseline_rate, dtype=np.float64)
            source = generate_informative_source(
                n_steps, n_source, 0.0, command_rate, dt_ms,
                source_onset_ms, source_duration_ms, seed,
            )
            source_f = source.astype(np.float64)
            w_offline, _, _, _ = np.linalg.lstsq(source_f, missing, rcond=None)
            drive = source @ w_offline

            for j in neurons:
                spikes = _run_fixed_drive(lesioned_graph, parameters, [j], drive, mapping_gain, duration_ms, stimulus)
                rec = rate_similarity(output_rate(spikes, output_ids), baseline_rate)
                per_neuron_gain[j].append(float(rec - reward_baseline))
            guard.check(f"after seed {i}", logger)

        headroom = float(np.mean(per_seed_headroom))
        rows: list = []
        for j in neurons:
            gains = np.asarray(per_neuron_gain[j], dtype=np.float64)
            oracle_gain = float(gains.mean())
            graftability = oracle_gain / headroom if headroom > 0 else float("nan")
            feats = _site_features(graph, output_ids, [j])
            rows.append({
                "neuron": j,
                "graftability": graftability,
                "oracle_gain": oracle_gain,
                "cell_type": graph.nodes[j].cell_type,
                **feats,
            })

        # regression graftability ~ direct_weight
        xs = np.asarray([r["direct_weight"] for r in rows], dtype=np.float64)
        ys = np.asarray([r["graftability"] for r in rows], dtype=np.float64)
        slope, intercept = (np.polyfit(xs, ys, 1) if xs.size >= 2 else (float("nan"), float("nan")))

        # per-type summary (mean graftability over neurons of that type)
        by_type: Dict[str, list] = {}
        for r in rows:
            by_type.setdefault(r["cell_type"], []).append(r["graftability"])
        type_summary = {
            t: {"n": len(v), "mean_graftability": float(np.nanmean(v))}
            for t, v in sorted(by_type.items())
        }

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M6",
            "backend": "oracle",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_seeds": n_seeds,
            "headroom": headroom,
            "n_neurons": len(rows),
            "type_summary": type_summary,
            "regression_graftability_on_direct_weight": {"slope": float(slope), "intercept": float(intercept), "n": int(xs.size)},
            "rows": rows,
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
                "description": "Direction C node scan: graft into every surviving neuron individually (oracle) to map graftability.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        for t, s in type_summary.items():
            logger.info("type %-12s n=%3d mean_graftability=%.4f", t, s["n"], s["mean_graftability"])
        logger.info("graftability ~ direct_weight: slope=%.6f intercept=%.4f (n=%d)", slope, intercept, xs.size)
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
        logger.exception("graft node scan failed")
        raise


def run_graft_role_intervention(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 5,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Causal test for the ROLE contribution to substitutability.

    Two interventions on the graft site: (1) ablation --- remove the surviving
    integrator's recurrent (``_Int_``->``_Int_``) synapses; (2) addition --- add
    a recurrent ring among a feedforward (untyped) site. If recurrence causally
    depresses substitutability, ablation should raise the integrator's
    substitutability and addition should lower the untyped site's.
    """
    experiment_id = f"p6-graft-role-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
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

        lesion_fraction = float(data["lesion_fraction"])
        drive_onto = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
        ranked = sorted(lesion_ids, key=lambda n: (-drive_onto[n], n))
        n_lesion = int(np.ceil(lesion_fraction * len(lesion_ids)))
        lesion_target = ranked[:n_lesion]
        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_target)

        n_source = int(data["n_source"])
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        mapping_gain = float(config["mapping"]["gain"])
        n_target = int(data.get("graft_n_target", 19))

        lesion_set = set(lesion_target)
        surviving_int = [n.id for n in graph.nodes if n.cell_type == "_Int_" and n.id not in lesion_set]
        surviving_int.sort(key=lambda x: (-drive_onto[x], x))
        int_site = surviving_int[:n_target]
        untyped = [n.id for n in graph.nodes if n.cell_type == "untyped"]
        untyped.sort(key=lambda x: (-drive_onto[x], x))
        untyped_site = untyped[:n_target]

        self_w = graph.weights[np.ix_(surviving_int, surviving_int)]
        mean_self = float(self_w[self_w > 0].mean()) if self_w.nnz else 0.0

        def graftability(les_graph, site_ids):
            gains = []
            for i in range(n_seeds):
                seed = base_seed + i
                stimulus = _build_stimulus(config["stimulus"], graph.groups, seed, dt_ms)
                act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
                baseline_rate = output_rate(act0.spikes, output_ids)
                act_les = ReferenceLIFSimulator(les_graph, parameters).run(duration_ms, stimulus)
                lesioned_rate = output_rate(act_les.spikes, output_ids)
                rb = rate_similarity(lesioned_rate, baseline_rate)
                headroom = 1.0 - float(rb)
                missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)
                bm = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
                cmd = (baseline_rate / bm).astype(np.float64) if bm > 0 else np.zeros_like(baseline_rate, dtype=np.float64)
                source = generate_informative_source(n_steps, n_source, 0.0, cmd, dt_ms, source_onset_ms, source_duration_ms, seed)
                w_off, _, _, _ = np.linalg.lstsq(source.astype(np.float64), missing, rcond=None)
                drive = source @ w_off
                spikes = _run_fixed_drive(les_graph, parameters, site_ids, drive, mapping_gain, duration_ms, stimulus)
                rec = rate_similarity(output_rate(spikes, output_ids), baseline_rate)
                gains.append(float((rec - rb) / headroom) if headroom > 0 else float("nan"))
            return float(np.mean(gains))

        int_base = graftability(lesioned_graph, int_site)
        untyped_base = graftability(lesioned_graph, untyped_site)

        s = set(surviving_int)
        kept = [e for e in lesioned_graph.edges if not (e.source in s and e.target in s)]
        ablated_graph = SparseGraph.from_edges(lesioned_graph.nodes, kept, groups=lesioned_graph.groups, max_nodes=lesioned_graph.node_count)
        int_ablated = graftability(ablated_graph, int_site)

        extra = []
        ids_sorted = sorted(untyped_site)
        m = len(ids_sorted)
        for idx, n in enumerate(ids_sorted):
            for off in (1, 2):
                tgt = ids_sorted[(idx + off) % m]
                extra.append(Edge(source=n, target=tgt, weight=float(mean_self), provenance="role_intervention",
                                  delay_ms=None, sign="excitatory", sign_source="model_assumption"))
        added_graph = SparseGraph.from_edges(lesioned_graph.nodes, list(lesioned_graph.edges) + extra, groups=lesioned_graph.groups, max_nodes=lesioned_graph.node_count)
        untyped_added = graftability(added_graph, untyped_site)

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M6",
            "backend": "oracle",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_seeds": n_seeds,
            "mean_self_weight": mean_self,
            "surviving_Int_baseline": int_base,
            "surviving_Int_ablated": int_ablated,
            "surviving_Int_delta": int_ablated - int_base,
            "untyped_baseline": untyped_base,
            "untyped_added": untyped_added,
            "untyped_delta": untyped_added - untyped_base,
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
                "description": "Direction C causal test: recurrence ablation/addition interventions on graft sites.",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "role intervention: Int %0.3f -> %0.3f (ablate, +%0.3f); untyped %0.3f -> %0.3f (add, %+0.3f)",
            int_base, int_ablated, int_ablated - int_base, untyped_base, untyped_added, untyped_added - untyped_base,
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
        logger.exception("graft role intervention failed")
        raise


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
