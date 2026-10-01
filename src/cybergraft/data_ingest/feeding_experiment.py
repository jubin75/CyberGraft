"""M1 feeding pathway experiment orchestration."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.mlx_backend import run_mlx_smoke_test
from cybergraft.sim.stimulus import DirectCurrentStimulus
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard

from .cache import QueryCache
from .fly_pathway import FlyFeedingPathway
from .malecns_queries import MaleCNSQueries
from .neuprint_client import DataUnavailableError, NeuPrintClient
from .subgraph_builder import SubgraphBuilder

SOURCE_URLS = {
    "male_cns": "https://male-cns.janelia.org/",
    "neuprint": "https://neuprint.janelia.org/",
    "calibration_reference": "https://github.com/blendi-remade/fly-brain-minecraft/blob/main/docs/VALIDATION.md",
}


def _experiment_id(config: Dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{config['experiment']['name']}-{timestamp}-{config_hash(config)[:8]}"


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_feeding_experiment(
    config: Dict[str, Any],
    project_root: Path,
    verbose: bool = False,
    client=None,
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

    data_config = config["data"]
    cache_dir = Path(data_config["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir

    if client is None:
        client = NeuPrintClient.from_config(config, project_root)

    data_status = client.status()

    try:
        if not client.available:
            logger.warning("neuPrint unavailable: %s", data_status.get("reason"))
            metrics: Dict[str, Any] = {
                "experiment_id": experiment_id,
                "milestone": "M1",
                "passed": False,
                "data_status": data_status,
            }
            _write_metrics(result_dir / "metrics.json", metrics)
            manifest = build_manifest(
                experiment_id=experiment_id,
                config=config,
                node_count=0,
                edge_count=0,
                project_root=project_root,
                mlx_smoke=mlx_smoke,
                status="unavailable",
                milestone="M1",
                data_provenance={
                    "dataset_versions": {},
                    "source_urls": SOURCE_URLS,
                    "query_hashes": [],
                    "data_status": data_status,
                },
            )
            write_manifest(result_dir / "manifest.json", manifest)
            return result_dir, metrics

        cache = QueryCache(cache_dir)
        queries = MaleCNSQueries(
            client,
            cache,
            dataset=data_config["dataset"],
            source_url="https://neuprint.janelia.org/",
            batch_size=int(data_config.get("batch_size", 200)),
        )
        pathway = FlyFeedingPathway(queries, config).resolve()
        graph, provenance = SubgraphBuilder(
            max_nodes=int(data_config.get("max_nodes", 500)),
            synaptic_delay_ms=float(config["simulation"]["synaptic_delay_ms"]),
        ).build(pathway)

        guard.check("after graph build", logger)

        parameters = LIFParameters.from_config(config["simulation"])
        stimulus = DirectCurrentStimulus.from_config(config["stimulus"], graph.groups)
        activity = ReferenceLIFSimulator(graph, parameters).run(
            config["simulation"]["duration_ms"], stimulus
        )

        guard.check("after simulation", logger)

        output_nodes = graph.groups["output"]
        output_spikes = int(activity.spikes[:, output_nodes].sum())
        total_spikes = activity.total_spikes
        active_nodes = int(np.count_nonzero(activity.spikes.sum(axis=0)))

        metrics = {
            "experiment_id": experiment_id,
            "milestone": "M1",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "duration_ms": float(config["simulation"]["duration_ms"]),
            "dt_ms": parameters.dt_ms,
            "total_spikes": total_spikes,
            "active_node_count": active_nodes,
            "output_node_count": len(output_nodes),
            "output_spikes": output_spikes,
            "mean_firing_rate_hz": float(activity.firing_rate_hz().mean()),
            "data_status": client.status(),
            "input_source": provenance["input_source"],
            "input_types": provenance["input_types"],
            "input_classes": provenance["input_classes"],
            "stage_counts": provenance["stage_counts"],
            "acceptance": {
                "min_total_spikes": int(config["acceptance"]["min_total_spikes"]),
                "min_output_spikes": int(config["acceptance"]["min_output_spikes"]),
            },
        }
        metrics["passed"] = bool(
            metrics["total_spikes"] >= metrics["acceptance"]["min_total_spikes"]
            and metrics["output_spikes"] >= metrics["acceptance"]["min_output_spikes"]
        )

        activity.save(result_dir / "activity.npz")
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="passed" if metrics["passed"] else "failed_acceptance",
            milestone="M1",
            data_provenance={
                "dataset_versions": {"male-cns": "v1.0"},
                "source_urls": SOURCE_URLS,
                "query_hashes": provenance["query_hashes"],
                "data_status": data_status,
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info(
            "Finished M1 experiment: passed=%s, output_spikes=%d",
            metrics["passed"],
            output_spikes,
        )
        return result_dir, metrics

    except DataUnavailableError as exc:
        logger.exception("M1 experiment data unavailable")
        metrics = {
            "experiment_id": experiment_id,
            "milestone": "M1",
            "passed": False,
            "data_status": data_status,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_metrics(result_dir / "metrics.json", metrics)
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=0,
            edge_count=0,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="unavailable",
            milestone="M1",
            data_provenance={
                "dataset_versions": {},
                "source_urls": SOURCE_URLS,
                "query_hashes": [],
                "data_status": data_status,
            },
        )
        write_manifest(result_dir / "manifest.json", manifest)
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
            milestone="M1",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M1 experiment failed")
        raise
