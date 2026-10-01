"""M2 zebrafish motor subgraph + lesion experiment orchestration."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.sim.stimulus import DirectCurrentStimulus, PoissonStimulus
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard

from .lesion import Lesion
from .motor_primitive import EyeMovementPrimitive, MotorPrimitive

SOURCE_URLS = {"zebrafish": "https://seunglab.org/zebrafish/"}
LICENSE = "CC BY-NC-ND 4.0 (Vishwanathan 2024); cite Vishwanathan 2017 EM / 2024 analysis"


def _experiment_id(config: Dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{config['experiment']['name']}-{timestamp}-{config_hash(config)[:8]}"


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_stimulus(config: dict, groups: dict, seed: int, dt_ms: float):
    stimulus_type = config.get("type")
    if stimulus_type == "poisson":
        return PoissonStimulus.from_config(config, groups, seed, dt_ms=dt_ms)
    if stimulus_type == "direct_current":
        return DirectCurrentStimulus.from_config(config, groups)
    raise ValueError(f"unsupported stimulus type: {stimulus_type}")


def run_lesion_experiment(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    mat_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
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

    try:
        adapter = ZebrafishAdapter(
            mat_path=mat,
            max_nodes=int(data["max_nodes"]),
            synaptic_delay_ms=float(config["simulation"]["synaptic_delay_ms"]),
        )
        graph, provenance = adapter.build_subgraph(
            include_types=list(data["include_types"]),
            readout_type=data["readout_type"],
            input_type=data["input_type"],
            lesion_type=data["lesion_type"],
        )
        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        primitive = EyeMovementPrimitive(graph.groups["output"])

        n_seeds = int(config["reproducibility"]["n_seeds"])
        base_seed = int(config["experiment"]["seed"])
        duration_ms = float(config["simulation"]["duration_ms"])

        lesion = Lesion(method=config["lesion"]["method"])
        lesioned_graph, lesion_record = lesion.apply(graph, graph.groups["lesion"])

        severities: list[float] = []
        intact_output_spikes: list[int] = []
        lesioned_output_spikes: list[int] = []
        intact_activity_0 = None
        lesioned_activity_0 = None

        for rep in range(n_seeds):
            seed = base_seed + rep
            stimulus = _build_stimulus(config["stimulus"], graph.groups, seed, parameters.dt_ms)
            intact_activity = ReferenceLIFSimulator(graph, parameters).run(duration_ms, stimulus)
            lesioned_activity = ReferenceLIFSimulator(lesioned_graph, parameters).run(duration_ms, stimulus)
            a_intact = primitive.decode(intact_activity)
            a_lesioned = primitive.decode(lesioned_activity)
            severities.append(1.0 - MotorPrimitive.similarity(a_intact, a_lesioned))
            output_nodes = graph.groups["output"]
            intact_output_spikes.append(int(intact_activity.spikes[:, output_nodes].sum()))
            lesioned_output_spikes.append(int(lesioned_activity.spikes[:, output_nodes].sum()))
            if rep == 0:
                intact_activity_0 = intact_activity
                lesioned_activity_0 = lesioned_activity
            guard.check(f"after simulation rep {rep}", logger)

        severity_mean = float(np.mean(severities))
        severity_std = float(np.std(severities))
        intact_output_spikes_mean = float(np.mean(intact_output_spikes))
        lesioned_output_spikes_mean = float(np.mean(lesioned_output_spikes))

        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M2",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "primitive": primitive.name,
            "output_node_count": len(graph.groups["output"]),
            "input_node_count": len(graph.groups["input"]),
            "lesion_node_count": len(graph.groups["lesion"]),
            "lesion_method": lesion_record["method"],
            "lesion_target_count": lesion_record["target_count"],
            "n_seeds": n_seeds,
            "intact_output_spikes": intact_output_spikes_mean,
            "lesioned_output_spikes": lesioned_output_spikes_mean,
            "lesion_severity_mean": severity_mean,
            "lesion_severity_std": severity_std,
            "lesion_severity_per_seed": [float(s) for s in severities],
            "dataset": provenance["dataset"],
            "source_url": provenance["source_url"],
            "cell_type_counts": provenance["cell_type_counts"],
            "acceptance": {
                "min_lesion_severity": float(config["acceptance"]["min_lesion_severity"]),
                "min_intact_output_spikes": int(config["acceptance"]["min_intact_output_spikes"]),
            },
        }
        metrics["passed"] = bool(
            severity_mean >= metrics["acceptance"]["min_lesion_severity"]
            and intact_output_spikes_mean >= metrics["acceptance"]["min_intact_output_spikes"]
        )

        assert intact_activity_0 is not None and lesioned_activity_0 is not None
        intact_activity_0.save(result_dir / "activity_intact.npz")
        lesioned_activity_0.save(result_dir / "activity_lesioned.npz")
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed" if metrics["passed"] else "failed_acceptance",
            milestone="M2",
            data_provenance={
                "dataset_versions": {"zebrafish": "ZConnectome_04292021"},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": {"available": True},
                "license": LICENSE,
                "description": provenance.get("readme", ""),
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished M2 experiment: passed=%s, lesion_severity_mean=%.3f, intact=%.1f, lesioned=%.1f",
            metrics["passed"],
            severity_mean,
            intact_output_spikes_mean,
            lesioned_output_spikes_mean,
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
            milestone="M2",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M2 experiment failed")
        raise
