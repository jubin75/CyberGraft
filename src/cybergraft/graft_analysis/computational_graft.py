"""Velocity-only computational graft assay (PRD §13-16, §24).

The computational graft is the V3 core new experiment. It is distinguished from
the V2 readout graft by the *source semantics* (PRD §28):

  * readout graft      : ``source = intact output command`` (relay)
  * computational graft: ``source = upstream velocity`` (recompute the integral)

Here we lesion the integrator, graft the velocity ``v(t)`` into a surviving
candidate site through a **static linear** ``W`` (a scalar gain × uniform
per-neuron weights — no recurrence, no memory), let the surviving connectome
dynamics propagate, and read the *actual circuit output*. A memoryless (static)
readout of that output is compared to the normative position ``p* = ∫ v dt``.
Because a static ``W`` and a static readout cannot integrate, any recovered
position must be produced by the surviving circuit's own dynamics — this is the
anti-cheat guarantee of PRD §29.
"""

from typing import Optional, Sequence

import numpy as np

from cybergraft.attractor_analysis.dynamics import Injection, run_dynamics
from cybergraft.graph.sparse_graph import SparseGraph
from cybergraft.sim.lif_reference import LIFParameters

from .graft_metrics import r2


def build_velocity_probe(
    name: str,
    dt_ms: float,
    n_steps: int,
    *,
    step_start_ms: float = 100.0,
    step_duration_ms: float = 400.0,
    sine_frequency_hz: float = 5.0,
    sine_start_ms: float = 0.0,
    sine_duration_ms: float = 1200.0,
    step_hold_on_ms: float = 500.0,
) -> np.ndarray:
    """Unit-amplitude velocity probe (mirrors V2 ``_build_probe``, plus step-hold)."""
    t = np.arange(n_steps, dtype=np.float64) * dt_ms
    v = np.zeros(n_steps, dtype=np.float64)
    if name == "step":
        v[(t >= step_start_ms) & (t < step_start_ms + step_duration_ms)] = 1.0
    elif name == "step_hold":
        v[t < step_hold_on_ms] = 1.0  # velocity step, then v=0 hold (u=0)
    elif name == "sine":
        mask = (t >= sine_start_ms) & (t < sine_start_ms + sine_duration_ms)
        v[mask] = np.sin(2.0 * np.pi * sine_frequency_hz * (t[mask] - sine_start_ms) / 1000.0)
    else:
        raise ValueError(f"unknown probe: {name}")
    return v


def position_target(velocity: np.ndarray, dt_ms: float) -> np.ndarray:
    """Normative position ``p*(t) = ∫ v(τ) dτ`` from the velocity source alone."""
    return np.cumsum(np.asarray(velocity, dtype=np.float64)) * dt_ms / 1000.0


def injection_signal(velocity: np.ndarray, amplitude: float, name: str) -> np.ndarray:
    """Raw velocity source scaled for injection (signed where bipolar).

    Deliberately NO DC pedestal: a pedestal plus slow tau lets the target
    neurons' membrane low-pass fake integration (oracle fix, PRD §29).
    """
    return np.float32(amplitude) * np.asarray(velocity, dtype=np.float32)


def run_velocity_graft(
    graph: SparseGraph,
    parameters: LIFParameters,
    *,
    site_ids: Sequence[int],
    signal: np.ndarray,
    duration_ms: float,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Graft ``signal`` into ``site_ids`` via a static linear ``W`` and return spikes.

    ``weights`` (default uniform ones) is the static per-neuron mapping; the
    scalar gain is folded into ``signal`` by the caller.
    """
    site = np.asarray(site_ids, dtype=np.int64)
    if weights is None:
        weights = np.ones(site.shape[0], dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
    injection = Injection(target_ids=site, weights=weights, signal=signal, gain=1.0)
    result = run_dynamics(
        graph, parameters, duration_ms=duration_ms, inject=injection, record_voltage=False
    )
    return result.spikes


def smoothed_output_rate(spikes: np.ndarray, output_ids: Sequence[int], rate_window: int) -> np.ndarray:
    """Box-smoothed mean firing rate over the output population, shape (T,)."""
    out = np.asarray(output_ids, dtype=np.int64)
    rate = np.zeros((spikes.shape[0], out.shape[0]), dtype=np.float64)
    for j, node_id in enumerate(out):
        rate[:, j] = np.convolve(
            spikes[:, node_id].astype(float), np.ones(rate_window), mode="same"
        )
    return rate.mean(axis=1)


def static_readout_r2(output_traj: np.ndarray, target: np.ndarray, fit: slice, evl: slice) -> float:
    """Static (memoryless) readout of the output trace -> target, held-out R²."""
    X = np.hstack([output_traj.reshape(-1, 1), np.ones((output_traj.shape[0], 1))])
    W = np.linalg.lstsq(X[fit], target[fit], rcond=None)[0]
    return r2(target[evl], X[evl] @ W)


def run_computational_graft_assay(
    graph: SparseGraph,
    parameters: LIFParameters,
    *,
    site_ids: Sequence[int],
    output_ids: Sequence[int],
    velocity: np.ndarray,
    name: str,
    amplitude: float,
    gain_grid: Sequence[float],
    rate_window: int,
    test_frac: float,
    duration_ms: float,
    weights: Optional[np.ndarray] = None,
) -> dict:
    """Sweep the static-linear graft gain and return the best position recovery.

    ``weights`` is the per-neuron static-linear ``W`` (length = len(site_ids)).
    Default (``None``) is uniform ones — the "no-differential-injection" null.
    For the computational task pass the natural-afferent pattern (net signed
    vestibular input onto each site neuron) so the injection is differential and
    the surviving circuit's own dynamics must integrate it (PRD §14.2-14.3).

    Returns a dict with the per-gain R²_comp, the best gain, and the headline
    ``computational_graftability`` = best ``R²_comp`` (normalized recovery, PRD §15).
    """
    dt_ms = parameters.dt_ms
    n_steps = int(round(duration_ms / dt_ms))
    signal = injection_signal(velocity, amplitude, name)
    p_star = position_target(velocity, dt_ms)
    # Piecewise-constant / plateau targets (step, step_hold) have a zero-variance
    # held-out tail, so evaluate in-sample (mirrors V2's step-probe handling).
    if name in ("step", "step_hold"):
        fit = evl = slice(0, n_steps)
    else:
        n_train = int(n_steps * (1.0 - test_frac))
        fit, evl = slice(0, n_train), slice(n_train, n_steps)

    per_gain = []
    for g in gain_grid:
        spikes = run_velocity_graft(
            graph, parameters, site_ids=site_ids,
            signal=np.float32(g) * signal, duration_ms=duration_ms, weights=weights,
        )
        out = smoothed_output_rate(spikes, output_ids, rate_window)
        rec = static_readout_r2(out, p_star, fit, evl)
        per_gain.append({"gain": float(g), "r2_comp": float(rec)})

    best = max(per_gain, key=lambda d: (0.0 if np.isnan(d["r2_comp"]) else d["r2_comp"]))
    return {
        "n_site": len(list(site_ids)),
        "gain_grid": [float(g) for g in gain_grid],
        "per_gain": per_gain,
        "best_gain": best["gain"],
        "computational_graftability": best["r2_comp"],
    }
