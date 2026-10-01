"""Fly EPG head-direction compass — substitutability generality test.

Mirrors the zebrafish substitutability finding (grafting.py) on the hemibrain
compass circuit: lesion the EPG+PEG ring integrator, graft a heading source into
each surviving neuron, and ask whether substitutability is predicted by the
site's direct synaptic weight onto the readout (PFL3) and by its structural role
(recurrent ring / upstream velocity / feedforward).

Readout substitutability only (source = intact PFL3 output command), matching the
zebrafish experiment. Excitatory-only.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cybergraft.data_ingest.hemibrain_compass import build_compass_graph
from cybergraft.experiment.bypass_helpers import _run_fixed_drive, generate_informative_source
from cybergraft.experiment.bypass_training import output_rate, rate_similarity
from cybergraft.experiment.lesion import Lesion
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.stimulus import RhythmicStimulus
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard

SOURCE_URLS = {"hemibrain": "https://neuprint.janelia.org"}
LICENSE = "CC BY 4.0 (hemibrain / Scheffer 2020)"

# Structural-role classification by cell type.
_ROLE_MAP = {
    "EPG": "recurrent", "EPGt": "recurrent", "PEG": "recurrent",
    "PEN_a(PEN1)": "upstream", "PEN_b(PEN2)": "upstream",
    "PFL1": "feedforward", "PFL2": "feedforward",
    "DNa02": "downstream",
}


def _role(cell_type: str) -> str:
    if not isinstance(cell_type, str):
        return "other"
    if cell_type in _ROLE_MAP:
        return _ROLE_MAP[cell_type]
    if cell_type.startswith("PFN"):
        return "feedforward"  # goal-direction input, projects onto the readout
    return "other"


def _site_features(graph, output_ids, site_ids):
    W = graph.weights
    direct_weight = float(W[np.ix_(output_ids, site_ids)].sum())
    intra = float(W[np.ix_(site_ids, site_ids)].sum())
    total_out = float(W[:, site_ids].sum())
    recurrence = intra / total_out if total_out > 0 else 0.0
    return {
        "direct_weight": direct_weight,
        "recurrence": recurrence,
        "out_degree": int(W[:, site_ids].nnz),
        "in_degree": int(W[site_ids, :].nnz),
        "direct": bool(direct_weight > 0),
    }


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_compass_node_scan(
    config: Dict[str, Any],
    project_root: Path,
    n_seeds: int = 3,
    verbose: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    experiment_id = f"p7-compass-scan-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{config_hash(config)[:8]}"
    result_dir = project_root / config["experiment"]["output_dir"] / experiment_id
    result_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_logging(result_dir / "run.log", verbose=verbose)
    guard = MemoryGuard(
        warning_gb=float(config["runtime"]["memory_warning_gb"]),
        abort_gb=float(config["runtime"]["memory_abort_gb"]),
    )
    mlx_smoke = run_mlx_smoke_test()

    data = config["data"]
    n_seeds = int(config["reproducibility"].get("n_seeds", n_seeds))
    base_seed = int(config["experiment"]["seed"])

    try:
        graph, provenance = build_compass_graph(config, project_root)
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        duration_ms = float(config["simulation"]["duration_ms"])
        dt_ms = parameters.dt_ms
        n_steps = int(round(duration_ms / dt_ms))

        output_ids = graph.groups["output"]
        lesion_ids = graph.groups["lesion"]
        groups = {k: v for k, v in graph.groups.items()}

        lesioned_graph, _ = Lesion(method="node_silence").apply(graph, lesion_ids)

        n_source = int(data.get("n_source", 32))
        source_onset_ms = float(config["stimulus"]["start_ms"])
        source_duration_ms = float(config["stimulus"]["duration_ms"])
        mapping_gain = float(config["mapping"]["gain"]) if "mapping" in config else 30.0

        output_set = set(output_ids)
        lesion_set = set(lesion_ids)
        neurons = [n.id for n in graph.nodes if n.id not in output_set and n.id not in lesion_set]

        per_neuron_gain: Dict[int, list] = {j: [] for j in neurons}
        per_seed_headroom: list = []

        for i in range(n_seeds):
            seed = base_seed + i
            stimulus = RhythmicStimulus.from_config(config["stimulus"], groups)
            act0 = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
            baseline_rate = output_rate(act0.spikes, output_ids)
            act_les = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
            lesioned_rate = output_rate(act_les.spikes, output_ids)
            reward_baseline = rate_similarity(lesioned_rate, baseline_rate)
            per_seed_headroom.append(1.0 - float(reward_baseline))
            missing = np.clip(baseline_rate - lesioned_rate, 0, None).astype(np.float64)

            baseline_max = float(np.max(baseline_rate)) if baseline_rate.size else 0.0
            command_rate = (baseline_rate / baseline_max).astype(np.float64) if baseline_max > 0 else np.zeros_like(baseline_rate, dtype=np.float64)
            source = generate_informative_source(
                n_steps, n_source, 0.0, command_rate, dt_ms,
                source_onset_ms, source_duration_ms, seed,
            )
            w_offline, _, _, _ = np.linalg.lstsq(source.astype(np.float64), missing, rcond=None)
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
            cell_type = graph.nodes[j].cell_type
            feats = _site_features(graph, output_ids, [j])
            rows.append({
                "neuron": j,
                "cell_type": cell_type,
                "role": _role(cell_type),
                "graftability": graftability,
                "oracle_gain": oracle_gain,
                **feats,
            })

        xs = np.asarray([r["direct_weight"] for r in rows], dtype=np.float64)
        ys = np.asarray([r["graftability"] for r in rows], dtype=np.float64)
        slope, intercept = (np.polyfit(xs, ys, 1) if xs.size >= 2 else (float("nan"), float("nan")))

        by_type: Dict[str, list] = {}
        for r in rows:
            by_type.setdefault(r["cell_type"], []).append(r["graftability"])
        type_summary = {
            t: {"n": len(v), "mean_graftability": float(np.nanmean(v))}
            for t, v in sorted(by_type.items())
        }

        by_role: Dict[str, list] = {}
        for r in rows:
            by_role.setdefault(r["role"], []).append(r["graftability"])
        role_summary = {
            t: {"n": len(v), "mean_graftability": float(np.nanmean(v))}
            for t, v in sorted(by_role.items())
        }

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M7",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "n_seeds": n_seeds,
            "headroom": headroom,
            "n_neurons": len(rows),
            "type_summary": type_summary,
            "role_summary": role_summary,
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
            milestone="M7",
            data_provenance={
                "dataset_versions": {"hemibrain": "v1.2.1"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": "Fly EPG compass: per-neuron substitutability scan (readout substitutability, oracle).",
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)

        for t, s in type_summary.items():
            logger.info("type %-14s n=%3d mean_graftability=%.4f", t, s["n"], s["mean_graftability"])
        for t, s in role_summary.items():
            logger.info("role %-12s n=%3d mean_graftability=%.4f", t, s["n"], s["mean_graftability"])
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
            milestone="M7",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("compass node scan failed")
        raise
