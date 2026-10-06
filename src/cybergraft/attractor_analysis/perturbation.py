"""Node perturbation assays (PRD §9, §11): transient silencing -> state impact.

V3 estimates each candidate node's contribution to persistent/attractor-supporting
dynamics by *transiently silencing* it (no edge deletion, no weight change — only
the node's activity inside a time window is held at rest) and measuring how the
network state deviates and recovers.

Two complementary scores are produced:

  * ``A_state``  — does silencing the node reduce multi-initial-state convergence?
    (PRD §9.2: ``baseline convergence score - post-silencing convergence score``)
  * ``A_persist``— how large / long / permanent a state deviation silencing the
    node induces (PRD §9.3 / §11.2: max deviation, recovery time, persistent shift).

Both are RAW directional scalars (higher = node contributes more to state
support); the z-scoring and ``A_i`` combination happens in ``scoring.py``.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from cybergraft.graph.sparse_graph import SparseGraph
from cybergraft.sim.lif_reference import LIFParameters

from .convergence import pairwise_convergence
from .dynamics import DynamicsResult, run_dynamics


def silence_window(dt_ms: float, start_ms: float, duration_ms: float) -> Tuple[int, int]:
    """Convert a silencing window (ms) to ``(start_step, end_step)`` (half-open)."""
    start = int(round(start_ms / dt_ms))
    end = int(round((start_ms + duration_ms) / dt_ms))
    return start, end


def trajectory_deviation(baseline: DynamicsResult, perturbed: DynamicsResult) -> np.ndarray:
    """Per-timestep network-state deviation ``||V_perturbed(t) - V_baseline(t)||_2``."""
    if baseline.voltage is None or perturbed.voltage is None:
        raise ValueError("voltage trajectories required; rerun with record_voltage=True")
    diff = perturbed.voltage - baseline.voltage
    return np.linalg.norm(diff, axis=1)


def perturbation_metrics(
    baseline: DynamicsResult,
    perturbed: DynamicsResult,
    *,
    silence_end_step: int,
    deviation_threshold: float = 0.1,
) -> dict:
    """Per-node perturbation outcomes (PRD §11.2).

    Returns ``max_deviation`` (peak network-state deviation, in voltage units),
    ``recovery_time`` (steps after release until deviation falls below
    ``deviation_threshold * max_deviation``; ``None`` if it never does), and
    ``persistent_shift`` (mean deviation over the final 5 steps — a residual,
    long-lived shift).
    """
    dev = trajectory_deviation(baseline, perturbed)
    max_deviation = float(dev.max())
    threshold = max(deviation_threshold * max_deviation, 1e-9)

    tail = dev[silence_end_step:]
    below = np.nonzero(tail < threshold)[0]
    recovery_time = int(below[0]) if below.size else None
    persistent_shift = float(np.mean(dev[-5:])) if dev.size >= 5 else float(dev[-1])
    return {
        "max_deviation": max_deviation,
        "recovery_time": recovery_time,
        "persistent_shift": persistent_shift,
    }


def run_perturbation_scan(
    graph: SparseGraph,
    parameters: LIFParameters,
    *,
    duration_ms: float,
    initial_voltage: np.ndarray,
    node_ids: Sequence[int],
    silence_start_ms: float,
    silence_duration_ms: float,
    deviation_threshold: float = 0.1,
) -> dict:
    """Silence each node in turn and return its perturbation metrics.

    Runs the baseline free-run ONCE from ``initial_voltage``, then one clamped
    free-run per node (same initial state). Single-state; for the multi-state
    ``A_state`` convergence delta use :func:`run_multi_state_perturbation`.
    """
    dt_ms = parameters.dt_ms
    n_steps = int(round(duration_ms / dt_ms))
    s_start, s_end = silence_window(dt_ms, silence_start_ms, silence_duration_ms)

    baseline = run_dynamics(
        graph, parameters, duration_ms=duration_ms, initial_voltage=initial_voltage, record_voltage=True
    )
    nodes: Dict[int, dict] = {}
    for nid in node_ids:
        pert = run_dynamics(
            graph,
            parameters,
            duration_ms=duration_ms,
            initial_voltage=initial_voltage,
            clamp=(int(nid), s_start, s_end),
            record_voltage=True,
        )
        nodes[int(nid)] = perturbation_metrics(
            baseline, pert, silence_end_step=s_end, deviation_threshold=deviation_threshold
        )
    return {
        "n_steps": n_steps,
        "silence_window": [s_start, s_end],
        "nodes": nodes,
    }


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std())
    if std == 0.0 or np.isnan(std):
        return np.zeros_like(values)
    return (values - values.mean()) / std


def a_persist_from_metrics(nodes_metrics: Dict[int, dict]) -> Dict[int, float]:
    """Combine perturbation outcomes into a raw ``A_persist`` scalar per node.

    Each of the three components (max deviation, recovery time, persistent shift)
    is z-scored across nodes (so differing units are comparable) and averaged.
    Higher ``A_persist`` = silencing the node induced a larger / longer / more
    permanent state disturbance = the node contributes more to persistent state.
    """
    ids = list(nodes_metrics)
    if not ids:
        return {}
    max_dev = np.array([nodes_metrics[i]["max_deviation"] for i in ids])
    rec = np.array([nodes_metrics[i]["recovery_time"] for i in ids])
    shift = np.array([nodes_metrics[i]["persistent_shift"] for i in ids])
    # A node that never recovers is maximally impactful for that component.
    rec = np.nan_to_num(rec, nan=float(rec[np.isfinite(rec)].max()) if np.isfinite(rec).any() else 0.0)
    composite = (_zscore(max_dev) + _zscore(rec) + _zscore(shift)) / 3.0
    return {int(i): float(composite[k]) for k, i in enumerate(ids)}


def convergence_score(trajectories) -> float:
    """Higher = stronger multi-initial-state convergence (1 - mean ratio)."""
    result = pairwise_convergence(trajectories)
    return 1.0 - float(result["mean_convergence_ratio"])


def a_state_delta(baseline_score: float, perturbed_score: float) -> float:
    """``A_state`` = convergence_score_baseline - convergence_score_after_silencing.

    Positive means silencing the node *reduced* convergence, i.e. the node
    supports stable-state formation.
    """
    return baseline_score - perturbed_score


def run_multi_state_perturbation(
    graph: SparseGraph,
    parameters: LIFParameters,
    *,
    duration_ms: float,
    initial_states: Sequence[np.ndarray],
    node_id: int,
    silence_start_ms: float,
    silence_duration_ms: float,
) -> Dict[str, float]:
    """``A_state`` for a single node over multiple initial states.

    Expensive: runs a full multi-state free-run twice (baseline and clamped).
    Returns ``{"baseline_convergence": float, "perturbed_convergence": float,
    "A_state": float}``.
    """
    dt_ms = parameters.dt_ms
    s_start, s_end = silence_window(dt_ms, silence_start_ms, silence_duration_ms)

    def _convergence(clamp: Optional[Tuple[int, int, int]]) -> float:
        trajs = []
        for x0 in initial_states:
            res = run_dynamics(
                graph,
                parameters,
                duration_ms=duration_ms,
                initial_voltage=np.asarray(x0, dtype=np.float32),
                clamp=clamp,
                record_voltage=True,
            )
            trajs.append(res.voltage)
        return convergence_score(trajs)

    base = _convergence(None)
    pert = _convergence((int(node_id), s_start, s_end))
    return {
        "baseline_convergence": float(base),
        "perturbed_convergence": float(pert),
        "A_state": float(a_state_delta(base, pert)),
    }
