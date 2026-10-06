#!/usr/bin/env python
"""V3 null-model validation (PRD §26): recurrent structure vs. shuffled nulls.

Does the zebrafish connectome's *recurrent structure* — the recurrent spectral
radius ``rho_R`` and the mean SCC-based recurrence — exceed what is expected from
degree-, weight-, and type-shuffled nulls?

This runner is **structural-only** (cheap, no LIF simulation). For each null
model it generates ``n_seeds`` randomized graphs and recomputes ``rho_R`` and
mean recurrence, then reports the real value, the null mean ± std, and a
z-score ``(real - null_mean) / null_std`` (guarded when ``null_std == 0``).

Note: the *dynamical* null (shuffling the attractor-support ``A_i`` /
computational-graft ``G_comp`` labels against the real network) is a
supplementary follow-up per PRD §26 ("如果计算量过大，优先保留 real-network
mechanistic result"); it is intentionally out of scope here.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cybergraft.analysis import null_models
from cybergraft.attractor_analysis import structural
from cybergraft.data_ingest.zebrafish_adapter import ZebrafishAdapter
from cybergraft.utils.config import load_config

# (null name, generator, n_seeds)
NULL_SPECS = [
    ("weight_shuffle", null_models.weight_shuffle, 10),
    ("shuffle_cell_types", null_models.shuffle_cell_types, 10),
    ("degree_preserving_rewire", null_models.degree_preserving_rewire, 5),
]


def _graph_metrics(graph) -> dict:
    """Structural metrics for one graph: rho_R, mean recurrence, max SCC size."""
    scores = structural.structural_scores(graph)
    recurrence = np.asarray(scores["recurrence"]["recurrence"], dtype=np.float64)
    return {
        "rho_R": float(scores["spectral_radius"]["rho_R"]),
        "mean_recurrence": float(np.nanmean(recurrence)),
        "max_scc": int(scores["scc"]["max_component_size"]),
    }


def _summarize(real: float, samples: list) -> dict:
    """Real value vs null samples: mean, std, and z-score (guarded if std==0)."""
    arr = np.asarray(samples, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std == 0.0:
        z = 0.0 if mean == real else float("nan")
    else:
        z = float((real - mean) / std)
    return {
        "real": float(real),
        "mean": mean,
        "std": std,
        "z": z,
        "samples": [float(v) for v in arr.tolist()],
    }


def main(config_path: str) -> int:
    config = load_config(config_path)
    project_root = Path.cwd()
    data = config["data"]

    # Build the E/I graph exactly as the V3 runner does.
    adapter = ZebrafishAdapter(
        mat_path=project_root / data["mat_file"],
        max_nodes=int(data["max_nodes"]),
        synaptic_delay_ms=float(config["simulation"]["synaptic_delay_ms"]),
    )
    ei = config.get("ei", {})
    graph, _ = adapter.build_full_subgraph(
        readout_type=data["readout_type"],
        lesion_type=data["lesion_type"],
        input_type=data["input_type"],
        max_nodes=int(data["max_nodes"]),
        inhibitory_fraction=float(ei.get("inhibitory_fraction", 0.0)),
        inhibitory_gain=float(ei.get("inhibitory_gain", 1.0)),
    )
    print(f"[v3-nulls] graph nodes={graph.node_count} edges={graph.edge_count}")

    real = _graph_metrics(graph)
    print(
        f"[v3-nulls] real rho_R={real['rho_R']:.4f} "
        f"mean_recurrence={real['mean_recurrence']:.4f} max_scc={real['max_scc']}"
    )

    base_seed = int(config["experiment"]["seed"])
    nulls: dict = {}
    for name, generator, n_seeds in NULL_SPECS:
        rho_samples: list = []
        rec_samples: list = []
        for i in range(n_seeds):
            null_graph = generator(graph, base_seed + i)
            m = _graph_metrics(null_graph)
            rho_samples.append(m["rho_R"])
            rec_samples.append(m["mean_recurrence"])
        nulls[name] = {
            "n_seeds": n_seeds,
            "rho_R": _summarize(real["rho_R"], rho_samples),
            "mean_recurrence": _summarize(real["mean_recurrence"], rec_samples),
        }

    experiment_id = f"v3-nulls-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    result_dir = project_root / "results" / experiment_id
    result_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment_id": experiment_id,
        "config": config_path,
        "graph": {"node_count": graph.node_count, "edge_count": graph.edge_count},
        "real": real,
        "nulls": nulls,
        "note": "structural-only null validation (PRD §26); dynamical A_i/G_comp null is a supplementary follow-up.",
    }
    out_path = result_dir / "null_validation.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("\n===== V3 NULL-MODEL VALIDATION =====")
    print(f"real rho_R           = {real['rho_R']:.4f}")
    print(f"real mean recurrence = {real['mean_recurrence']:.4f}")
    print(f"real max SCC size    = {real['max_scc']}")
    for name, res in nulls.items():
        for metric in ("rho_R", "mean_recurrence"):
            s = res[metric]
            print(
                f"{name:<24} {metric:<16} real={s['real']:.4f} "
                f"null={s['mean']:.4f}±{s['std']:.4f} z={s['z']:+.2f}"
            )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/v3_zfish.yaml"))
