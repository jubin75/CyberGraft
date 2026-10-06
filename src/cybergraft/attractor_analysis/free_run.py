"""Multi-initial-state free-run dynamics (PRD §8).

Free-run = stop all external drive (``u(t) = 0``) and let the network evolve
from a captured state. V3 uses multiple initial states so that attractor-like
behavior is supported by *initial-condition convergence* (distinct initial
states entering the same region) together with temporal stability, rather than
by a single trajectory.

Initial states (PRD §8.1) are drawn from three regimes seeded by the intact
baseline state: ``baseline + small noise``, ``scaled baseline`` (scaling the
deviation from rest), and ``low-amplitude random``.
"""

from typing import List, Optional, Sequence

import numpy as np

from cybergraft.graph.sparse_graph import SparseGraph
from cybergraft.sim.lif_reference import LIFParameters

from .dynamics import DynamicsResult, run_dynamics


def generate_initial_states(
    baseline_voltage: np.ndarray,
    *,
    v_rest: float,
    n_states: int = 20,
    noise_scale: float = 1.0,
    scaled_factors: Sequence[float] = (0.5, 0.8, 1.2, 1.5),
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Generate ``n_states`` initial voltage vectors from three regimes.

    The regimes are cycled so that a mix of perturbation types is represented:
      1. ``baseline + small_noise``    (default: noise_scale * standard_normal)
      2. ``scaled_baseline``           (scale the deviation from ``v_rest``)
      3. ``low_amplitude_random``      (rest + small random)
    """
    baseline = np.asarray(baseline_voltage, dtype=np.float32)
    node_count = baseline.shape[0]
    rng = np.random.default_rng(seed)
    deviation = baseline - np.float32(v_rest)

    states: List[np.ndarray] = []
    for k in range(n_states):
        regime = k % 3
        if regime == 0:
            x0 = baseline + np.float32(noise_scale) * rng.standard_normal(node_count).astype(np.float32)
        elif regime == 1:
            factor = float(scaled_factors[(k // 3) % len(scaled_factors)])
            x0 = np.float32(v_rest) + np.float32(factor) * deviation
        else:
            x0 = np.float32(v_rest) + np.float32(noise_scale) * rng.standard_normal(node_count).astype(np.float32)
        states.append(x0.astype(np.float32))
    return states


def run_free_run_multi(
    graph: SparseGraph,
    parameters: LIFParameters,
    *,
    duration_ms: float,
    initial_states: Sequence[np.ndarray],
    record_voltage: bool = True,
) -> List[DynamicsResult]:
    """Run one zero-input free-run from each initial state."""
    return [
        run_dynamics(
            graph,
            parameters,
            duration_ms=duration_ms,
            initial_voltage=np.asarray(x0, dtype=np.float32),
            record_voltage=record_voltage,
        )
        for x0 in initial_states
    ]


def free_run_summary(
    results: Sequence[DynamicsResult],
    *,
    readout_ids: Optional[Sequence[int]] = None,
) -> dict:
    """Aggregate per-state summary statistics (PRD §8.2).

    Returns a list (one entry per initial state) of:
      * ``firing_rate_mean``  — mean population firing rate (Hz)
      * ``mean_membrane``     — mean membrane potential over the run (mV)
      * ``active_fraction``   — fraction of neurons that fired at least once
      * ``readout_trajectory``— (T,) mean readout rate, if ``readout_ids`` given
    """
    states: list = []
    for res in results:
        T = res.spikes.shape[0]
        duration_s = T * res.dt_ms / 1000.0
        rate = res.spikes.sum(axis=0, dtype=np.float32) / duration_s if duration_s > 0 else np.zeros(res.spikes.shape[1], dtype=np.float32)
        entry: dict = {
            "firing_rate_mean": float(rate.mean()),
            "mean_membrane": float(res.mean_membrane_potential.mean()),
            "active_fraction": float((res.spikes.sum(axis=0) > 0).mean()),
        }
        if readout_ids is not None:
            ids = list(readout_ids)
            ro = res.spikes[:, ids].mean(axis=1).astype(np.float32) / (res.dt_ms / 1000.0)
            entry["readout_trajectory"] = ro.tolist()
        states.append(entry)
    return {"n_states": len(states), "states": states}


def trajectories(results: Sequence[DynamicsResult]) -> List[np.ndarray]:
    """Extract per-state voltage trajectories (each ``(T, N)``) for convergence analysis."""
    out: List[np.ndarray] = []
    for res in results:
        if res.voltage is None:
            raise ValueError("voltage trajectories were not recorded; rerun with record_voltage=True")
        out.append(res.voltage)
    return out
