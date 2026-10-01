"""Command-line M0 toy experiment runner."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cybergraft.graph.edge_schema import Edge
from cybergraft.graph.node_schema import Node
from cybergraft.graph.sparse_graph import SparseGraph
from cybergraft.graph.validator import HARD_NODE_LIMIT
from cybergraft.utils.config import load_config
from cybergraft.utils.logging import configure_logging
from cybergraft.utils.manifest import build_manifest, config_hash, write_manifest
from cybergraft.utils.memory_guard import MemoryGuard

from .lif_reference import LIFParameters, ReferenceLIFSimulator
from .mlx_backend import run_mlx_smoke_test
from .stimulus import DirectCurrentStimulus


def build_toy_graph(config: Dict[str, Any], rng: np.random.Generator) -> SparseGraph:
    """Build a deterministic, layered, feed-forward sparse M0 graph."""
    graph_config = config["toy_graph"]
    layers = [int(size) for size in graph_config["layers"]]
    if len(layers) < 2 or any(size <= 0 for size in layers):
        raise ValueError("toy_graph.layers must contain at least two positive layer sizes")
    node_count = sum(layers)
    max_nodes = min(int(config["experiment"]["max_nodes"]), HARD_NODE_LIMIT)
    if node_count > max_nodes:
        raise ValueError(f"Toy graph has {node_count} nodes; configured limit is {max_nodes}")
    offsets = np.cumsum([0] + layers)
    nodes = []
    for layer_index, layer_size in enumerate(layers):
        for local_index in range(layer_size):
            node_id = int(offsets[layer_index] + local_index)
            if layer_index == 0:
                subsystem, input_class, output_class = "toy_input", "stimulus", "interneuron"
            elif layer_index == len(layers) - 1:
                subsystem, input_class, output_class = "toy_output", "interneuron", "readout"
            else:
                subsystem, input_class, output_class = "toy_hidden", "interneuron", "interneuron"
            nodes.append(
                Node(
                    id=node_id,
                    species="synthetic",
                    cell_type=f"toy_layer_{layer_index}",
                    subsystem=subsystem,
                    input_class=input_class,
                    output_class=output_class,
                )
            )
    fanout = int(graph_config["fanout"])
    weight = float(graph_config["weight"])
    weight_jitter = float(graph_config.get("weight_jitter", 0.0))
    if fanout <= 0 or weight <= 0 or weight_jitter < 0:
        raise ValueError("toy_graph fanout and weight must be positive; weight_jitter must be non-negative")
    edges = []
    for layer_index in range(len(layers) - 1):
        source_ids = np.arange(offsets[layer_index], offsets[layer_index + 1], dtype=np.int32)
        target_ids = np.arange(offsets[layer_index + 1], offsets[layer_index + 2], dtype=np.int32)
        degree = min(fanout, len(target_ids))
        for source in source_ids:
            targets = rng.choice(target_ids, size=degree, replace=False)
            for target in targets:
                sampled_weight = max(0.0, weight + rng.uniform(-weight_jitter, weight_jitter))
                edges.append(
                    Edge(
                        source=int(source),
                        target=int(target),
                        weight=sampled_weight,
                        delay_ms=float(config["simulation"]["synaptic_delay_ms"]),
                        sign="excitatory",
                        sign_source="model_assumption",
                        provenance="m0_toy_generator",
                    )
                )
    groups = {
        "input": list(range(int(offsets[0]), int(offsets[1]))),
        "output": list(range(int(offsets[-2]), int(offsets[-1]))),
    }
    return SparseGraph.from_edges(nodes, edges, groups=groups, max_nodes=max_nodes)


def _experiment_id(config: Dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{config['experiment']['name']}-{timestamp}-{config_hash(config)[:8]}"


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(config: Dict[str, Any], project_root: Path, verbose: bool = False) -> Tuple[Path, Dict[str, Any]]:
    """Execute a bounded reference experiment and return its result directory and metrics."""
    if config.get("milestone") == "M1":
        from cybergraft.data_ingest.feeding_experiment import run_feeding_experiment

        return run_feeding_experiment(config, project_root=project_root, verbose=verbose)
    if config.get("milestone") == "M2":
        from cybergraft.experiment.lesion_experiment import run_lesion_experiment

        return run_lesion_experiment(config, project_root=project_root, verbose=verbose)
    if config.get("milestone") == "M3":
        from cybergraft.experiment.bypass_experiment import run_bypass_experiment

        return run_bypass_experiment(config, project_root=project_root, verbose=verbose)
    if config["runtime"]["backend"] != "reference":
        raise ValueError("M0 CLI currently supports runtime.backend=reference only")
    experiment_id = _experiment_id(config)
    result_dir = project_root / config["experiment"]["output_dir"] / experiment_id
    result_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_logging(result_dir / "run.log", verbose=verbose)
    guard = MemoryGuard(
        warning_gb=float(config["runtime"]["memory_warning_gb"]),
        abort_gb=float(config["runtime"]["memory_abort_gb"]),
    )
    graph: Optional[SparseGraph] = None
    mlx_smoke = run_mlx_smoke_test()
    try:
        logger.info("Starting %s using reference backend", experiment_id)
        logger.info("MLX smoke test: %s", mlx_smoke)
        guard.check("before graph build", logger)
        rng = np.random.default_rng(int(config["experiment"]["seed"]))
        graph = build_toy_graph(config, rng)
        if graph.node_count > HARD_NODE_LIMIT:
            raise ValueError(f"node_count {graph.node_count} exceeds hard limit {HARD_NODE_LIMIT}")
        guard.check("after graph build", logger)
        parameters = LIFParameters.from_config(config["simulation"])
        stimulus = DirectCurrentStimulus.from_config(config["stimulus"], graph.groups)
        activity = ReferenceLIFSimulator(graph, parameters).run(config["simulation"]["duration_ms"], stimulus)
        guard.check("after simulation", logger)
        output_nodes = graph.groups["output"]
        output_spikes = int(activity.spikes[:, output_nodes].sum())
        active_nodes = int(np.count_nonzero(activity.spikes.sum(axis=0)))
        metrics: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "milestone": "M0",
            "backend": "reference",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "duration_ms": float(config["simulation"]["duration_ms"]),
            "dt_ms": parameters.dt_ms,
            "total_spikes": activity.total_spikes,
            "active_node_count": active_nodes,
            "output_node_count": len(output_nodes),
            "output_spikes": output_spikes,
            "mean_firing_rate_hz": float(activity.firing_rate_hz().mean()),
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
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.info("Finished M0 experiment: passed=%s, output_spikes=%d", metrics["passed"], output_spikes)
        return result_dir, metrics
    except Exception as exc:
        manifest = build_manifest(
            experiment_id=experiment_id,
            config=config,
            node_count=graph.node_count if graph else 0,
            edge_count=graph.edge_count if graph else 0,
            project_root=project_root,
            mlx_smoke=mlx_smoke,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
        write_manifest(result_dir / "manifest.json", manifest)
        logger.exception("M0 experiment failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CyberGraft M0 toy LIF experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment configuration")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    project_root = config_path.parent.parent
    result_dir, metrics = run_experiment(config, project_root=project_root, verbose=args.verbose)
    print(json.dumps({"result_dir": str(result_dir), "passed": metrics["passed"]}, sort_keys=True))
    if not metrics["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
