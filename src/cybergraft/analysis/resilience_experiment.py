"""Structural (no-sim) null test for oculomotor readout resilience.

"Direction A" of the redundancy hypothesis: does the real zebrafish connectome's
wiring sustain the ABD_m oculomotor readout after lesioning the most heavily
projecting integrator neurons MORE than null models do?

This is a purely structural analysis. No LIF simulation is run: the metric
``structural_redundancy`` measures the fraction of total presynaptic drive onto
the readout that survives lesioning a ranked subset of the integrator
population, computed directly from the CSR weight matrix.

Two null models randomize one aspect of the graph while leaving the rest fixed:

* ``weight_shuffle``        — permute edge weights among existing edges (topology
  and in/out-degree preserved); asks whether the specific weight arrangement
  matters.
* ``shuffle_cell_types``    — permute cell-type labels across nodes (topology and
  weights preserved); asks whether the specific type identity matters.

Both are deterministic given a seed (``numpy.random.default_rng``). The real
redundancy is compared against the null distribution via mean, z-score, and a
two-sided permutation p-value.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.experiment.bypass_training import output_rate, rate_similarity
from cybergraft.experiment.lesion import Lesion
from cybergraft.graph.sparse_graph import SparseGraph
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.sim.stimulus import RhythmicStimulus
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard

from .null_models import degree_preserving_rewire, shuffle_cell_types, weight_shuffle

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024); cite Vishwanathan 2017 EM / 2024 analysis"


def _experiment_id(config: Dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{config['experiment']['name']}-{timestamp}-{config_hash(config)[:8]}"


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def structural_redundancy(
    graph: SparseGraph,
    *,
    readout_type: str = "ABD_m",
    lesion_type: str = "_Int_",
    lesion_fraction: float = 0.5,
) -> float:
    """Fraction of readout drive that survives lesioning a ranked subset of neurons.

    The lesioned subset is the top ``ceil(lesion_fraction * |lesion_type|)``
    ``lesion_type`` neurons ranked by total presynaptic weight onto the readout
    (matching the lesion selection used by the simulation experiments).

    Returns ``1 - (drive from lesioned subset) / (total drive onto readout)``,
    i.e. a value in [0, 1] where 1.0 means the lesioned neurons contribute no
    drive (readout fully redundant) and 0.0 means they contribute all of it.

    Node sets are resolved by ``cell_type`` (not ``groups``) so the metric is
    well-defined on ``shuffle_cell_types`` null graphs whose labels have moved.
    """
    if not 0.0 < lesion_fraction <= 1.0:
        raise ValueError("lesion_fraction must be in (0, 1]")
    output_ids = [node.id for node in graph.nodes if node.cell_type == readout_type]
    lesion_ids = [node.id for node in graph.nodes if node.cell_type == lesion_type]
    if not output_ids:
        raise ValueError(f"no nodes with cell_type {readout_type!r}")
    if not lesion_ids:
        raise ValueError(f"no nodes with cell_type {lesion_type!r}")

    drive = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
    ranked = sorted(lesion_ids, key=lambda n: (-drive[n], n))
    n_lesion = int(np.ceil(lesion_fraction * len(lesion_ids)))
    lesioned = ranked[:n_lesion]

    total_drive = float(graph.weights[output_ids, :].sum())
    lesioned_drive = float(graph.weights[np.ix_(output_ids, lesioned)].sum())
    if total_drive <= 0.0:
        return 0.0
    return 1.0 - lesioned_drive / total_drive


def _null_stats(values: np.ndarray, real: float, n_nulls: int) -> Dict[str, Any]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    z_score = float((real - mean) / std) if std > 0.0 else 0.0
    frac_gte = float(np.mean(values >= real))
    frac_lte = float(np.mean(values <= real))
    p_one_sided = min(frac_gte, frac_lte)
    p_two_sided = min(1.0, 2.0 * p_one_sided)
    return {
        "null_mean": mean,
        "null_std": std,
        "z_score": z_score,
        "frac_gte": frac_gte,
        "frac_lte": frac_lte,
        "p_one_sided": p_one_sided,
        "p_two_sided": p_two_sided,
        "n_nulls": n_nulls,
    }


def run_resilience_nulls(
    config: Dict[str, Any],
    project_root: Path,
    n_nulls: int = 200,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Run the structural resilience null test and write metrics + manifest.

    Builds the full-connectome subgraph, computes the real ``structural_redundancy``,
    then generates ``n_nulls`` graphs per null model and compares.
    """
    experiment_id = _experiment_id(config)
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

        readout_type = str(data["readout_type"])
        lesion_type = str(data["lesion_type"])
        lesion_fraction = float(data["lesion_fraction"])
        base_seed = int(config["experiment"]["seed"])

        real = structural_redundancy(
            graph,
            readout_type=readout_type,
            lesion_type=lesion_type,
            lesion_fraction=lesion_fraction,
        )

        generators = {
            "weight_shuffle": weight_shuffle,
            "shuffle_cell_types": shuffle_cell_types,
            "degree_preserving_rewire": degree_preserving_rewire,
        }
        nulls: Dict[str, Any] = {}
        for name, generator in generators.items():
            values = np.empty(n_nulls, dtype=np.float64)
            for i in range(n_nulls):
                null_graph = generator(graph, seed=base_seed + i)
                values[i] = structural_redundancy(
                    null_graph,
                    readout_type=readout_type,
                    lesion_type=lesion_type,
                    lesion_fraction=lesion_fraction,
                )
                guard.check(f"after null {name} {i}", logger)
            np.save(result_dir / f"nulls_{name}.npy", values)
            nulls[name] = _null_stats(values, real, n_nulls)

        direction_held = all(nulls[name]["null_mean"] < real for name in generators)

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M5",
            "backend": "structural",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "readout_type": readout_type,
            "lesion_type": lesion_type,
            "lesion_fraction": lesion_fraction,
            "n_nulls": n_nulls,
            "structural_redundancy_real": real,
            "nulls": nulls,
            "hypothesis": "real structural redundancy exceeds both null means",
            "direction_held": direction_held,
        }
        metrics["passed"] = direction_held

        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed" if metrics["passed"] else "failed_acceptance",
            milestone="M5",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": (
                    "Structural (no-sim) null test: does the real zebrafish wiring sustain "
                    "the ABD_m readout after lesioning the top integrator neurons more than "
                    "weight-shuffle and cell-type-shuffle null models?"
                ),
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished structural resilience nulls: real=%.4f, weight_shuffle(null_mean=%.4f, p_two=%.4f), "
            "shuffle_cell_types(null_mean=%.4f, p_two=%.4f), passed=%s",
            real,
            nulls["weight_shuffle"]["null_mean"],
            nulls["weight_shuffle"]["p_two_sided"],
            nulls["shuffle_cell_types"]["null_mean"],
            nulls["shuffle_cell_types"]["p_two_sided"],
            metrics["passed"],
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
            milestone="M5",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M5 structural resilience nulls failed")
        raise


def _resolve_ids(graph: SparseGraph, cell_type: str) -> list[int]:
    """Local ids of every node carrying ``cell_type`` (type-based, not groups-based).

    Type resolution is the same convention as ``structural_redundancy`` so the
    null graphs (whose labels may have moved) are tested consistently.
    """
    return [node.id for node in graph.nodes if node.cell_type == cell_type]


def _build_rhythmic_stimulus(config: Dict[str, Any], input_ids: list[int]) -> RhythmicStimulus:
    """Build the rhythmic stimulus targeting ``input_ids`` (the input population)."""
    return RhythmicStimulus.from_config(config, {"input": list(input_ids)})


def _functional_recovery(
    graph: SparseGraph,
    output_ids: list[int],
    lesion_ids: list[int],
    parameters: LIFParameters,
    stimulus: RhythmicStimulus,
    duration_ms: float,
    lesion_fraction: float,
) -> float:
    """Silence the ranked integrator subset and return lesioned-vs-intact recovery.

    Lesion definition is identical to ``structural_redundancy``: silence the top
    ``ceil(lesion_fraction * |lesion_ids|)`` neurons ranked by total weight onto
    the readout. Recovery is the amplitude-sensitive ``rate_similarity`` between
    the lesioned and intact smoothed output rates (matches the M5 lesioned baseline).
    """
    drive = np.asarray(graph.weights[output_ids, :].sum(axis=0)).ravel()
    ranked = sorted(lesion_ids, key=lambda n: (-drive[n], n))
    n_lesion = int(np.ceil(lesion_fraction * len(lesion_ids)))
    lesioned_graph, _ = Lesion(method="node_silence").apply(graph, ranked[:n_lesion])

    intact = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus).spikes
    lesioned = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus).spikes
    return rate_similarity(output_rate(lesioned, output_ids), output_rate(intact, output_ids))


def run_functional_resilience_nulls(
    config: Dict[str, Any],
    project_root: Path,
    n_nulls: int = 30,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Functional (sim) resilience null test: does the real circuit degrade MORE?

    Builds the full-connectome subgraph, computes the real lesioned-vs-intact
    recovery, then generates ``n_nulls`` graphs per null model and compares. The
    input/lesion/readout populations are all resolved by ``cell_type`` per graph
    (identical to ``structural_redundancy``) so ``shuffle_cell_types`` is a
    non-degenerate null; the rhythmic stimulus spec is held fixed.
    """
    experiment_id = _experiment_id(config)
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
    n_nulls = int(data.get("n_nulls_functional", n_nulls))

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
        readout_type = str(data["readout_type"])
        lesion_type = str(data["lesion_type"])
        input_type = str(data["input_type"])
        lesion_fraction = float(data["lesion_fraction"])
        base_seed = int(config["experiment"]["seed"])

        recovery_real = _functional_recovery(
            graph,
            _resolve_ids(graph, readout_type),
            _resolve_ids(graph, lesion_type),
            parameters,
            _build_rhythmic_stimulus(config["stimulus"], _resolve_ids(graph, input_type)),
            duration_ms,
            lesion_fraction,
        )
        guard.check("after real sim", logger)

        generators = {
            "weight_shuffle": weight_shuffle,
            "shuffle_cell_types": shuffle_cell_types,
            "degree_preserving_rewire": degree_preserving_rewire,
        }
        nulls: Dict[str, Any] = {}
        for name, generator in generators.items():
            values = np.empty(n_nulls, dtype=np.float64)
            for i in range(n_nulls):
                null_graph = generator(graph, seed=base_seed + i)
                values[i] = _functional_recovery(
                    null_graph,
                    _resolve_ids(null_graph, readout_type),
                    _resolve_ids(null_graph, lesion_type),
                    parameters,
                    _build_rhythmic_stimulus(config["stimulus"], _resolve_ids(null_graph, input_type)),
                    duration_ms,
                    lesion_fraction,
                )
                guard.check(f"after null {name} {i}", logger)
            np.save(result_dir / f"nulls_{name}_functional.npy", values)
            nulls[name] = _null_stats(values, recovery_real, n_nulls)

        direction_held = all(nulls[name]["null_mean"] > recovery_real for name in generators)

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M5",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": provenance["cell_type_counts"],
            "readout_type": readout_type,
            "lesion_type": lesion_type,
            "input_type": input_type,
            "lesion_fraction": lesion_fraction,
            "n_nulls": n_nulls,
            "recovery_real": recovery_real,
            "nulls": nulls,
            "hypothesis": "real lesioned recovery is LOWER than both null means (circuit concentrated on integrator)",
            "direction_held": direction_held,
            "stimulus_note": (
                "rhythmic 300/5Hz/360ms; input/lesion/readout resolved by cell_type per graph "
                "(identical to structural_redundancy), stimulus spec held fixed across real + nulls"
            ),
        }
        metrics["passed"] = direction_held

        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed" if metrics["passed"] else "failed_acceptance",
            milestone="M5",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": (
                    "Functional (sim) null test: does the real circuit degrade more (lower "
                    "lesioned recovery) to integrator lesion than weight-shuffle and "
                    "cell-type-shuffle null models?"
                ),
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished functional resilience nulls: real=%.4f, weight_shuffle(null_mean=%.4f, p_two=%.4f), "
            "shuffle_cell_types(null_mean=%.4f, p_two=%.4f), passed=%s",
            recovery_real,
            nulls["weight_shuffle"]["null_mean"],
            nulls["weight_shuffle"]["p_two_sided"],
            nulls["shuffle_cell_types"]["null_mean"],
            nulls["shuffle_cell_types"]["p_two_sided"],
            metrics["passed"],
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
            milestone="M5",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M5 functional resilience nulls failed")
        raise
