"""V3 dynamics engine: initial-state free-run, node perturbation, graft injection.

V3 needs three dynamics capabilities the V2 simulator does not expose directly:

  1. **Free-run from an arbitrary initial state** — stop external drive
     (``u(t) = 0``) and run the network from a captured state so we can measure
     multi-initial-state convergence and temporal stability.
  2. **Node perturbation** — transiently silence a node for a time window (no
     connectome-edge deletion, no weight change) and release it, to estimate the
     node's contribution to persistent-state support.
  3. **Computational graft injection** — inject a source ``s(t)`` into a
     candidate site through a *static linear* mapping ``I_graft(t) = W s(t)``,
     so the surviving circuit's own dynamics must carry any computation.

This is intentionally a **separate** engine from ``ReferenceLIFSimulator`` and
``_run_fixed_drive``: it records per-node voltage trajectories and accepts an
initial state, a per-neuron injection, and a clamp window. V2 code is untouched.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from cybergraft.graph.sparse_graph import SparseGraph
from cybergraft.sim.lif_reference import LIFParameters
from cybergraft.sim.stimulus import Stimulus


@dataclass
class Injection:
    """Static-linear graft injection into a candidate site.

    ``I_graft(t) = gain * weights * signal[t]``, delivered only to
    ``target_ids``. ``weights`` is a *static* per-neuron mapping (no recurrence,
    no memory); any temporal integration must come from the surviving circuit.
    """

    target_ids: np.ndarray  # (K,) int, candidate-site node ids
    weights: np.ndarray     # (K,) float32, static linear W
    signal: np.ndarray      # (T,) float32, source s(t) (e.g. velocity)
    gain: float = 1.0


@dataclass
class DynamicsResult:
    """Spike raster, optional voltage trajectory, and population-average trace."""

    spikes: np.ndarray                    # (T, N) bool
    voltage: Optional[np.ndarray]         # (T, N) float32 or None if not recorded
    mean_membrane_potential: np.ndarray   # (T,) float32
    final_voltage: np.ndarray             # (N,) float32
    dt_ms: float

    @property
    def total_spikes(self) -> int:
        return int(self.spikes.sum())


def run_dynamics(
    graph: SparseGraph,
    parameters: LIFParameters,
    *,
    duration_ms: float,
    initial_voltage: Optional[np.ndarray] = None,
    inject: Optional[Injection] = None,
    stimulus: Optional[Stimulus] = None,
    clamp: Optional[Tuple[int, int, int]] = None,
    record_voltage: bool = True,
) -> DynamicsResult:
    """Run a deterministic Euler LIF pass with V3 extensions.

    Parameters
    ----------
    clamp:
        ``(node_id, start_step, end_step)`` — silence ``node_id`` for steps in
        ``[start_step, end_step)`` by holding it at rest (voltage reset, no
        synaptic accumulation, no spike) and release afterwards. Does not touch
        edges or weights.
    """
    p = parameters
    step_count = int(round(duration_ms / p.dt_ms))
    if step_count <= 0:
        raise ValueError("duration_ms must include at least one timestep")
    node_count = graph.node_count
    delay_steps = int(round(p.synaptic_delay_ms / p.dt_ms))
    spike_history = np.zeros((delay_steps + 1, node_count), dtype=np.bool_)
    spikes_out = np.zeros((step_count, node_count), dtype=np.bool_)
    if record_voltage:
        voltage_traj = np.zeros((step_count, node_count), dtype=np.float32)
    else:
        voltage_traj = None
    mean_voltage = np.zeros(step_count, dtype=np.float32)

    if initial_voltage is None:
        voltage = np.full(node_count, p.v_rest, dtype=np.float32)
    else:
        voltage = np.asarray(initial_voltage, dtype=np.float32).copy()
        if voltage.shape != (node_count,):
            raise ValueError(
                f"initial_voltage shape {voltage.shape}, expected ({node_count},)"
            )

    synaptic_current = np.zeros(node_count, dtype=np.float32)
    refractory = np.zeros(node_count, dtype=np.int32)
    alpha_m = np.float32(p.dt_ms / p.tau_m_ms)
    alpha_syn = np.float32(np.exp(-p.dt_ms / p.tau_syn_ms))
    refractory_steps = int(np.ceil(p.refractory_ms / p.dt_ms))

    # Pre-allocate graft injection (static linear: weights * signal).
    if inject is not None:
        inj_target = np.asarray(inject.target_ids, dtype=np.int64)
        inj_weights = np.asarray(inject.weights, dtype=np.float32)
        inj_signal = np.asarray(inject.signal, dtype=np.float32)
        inj_gain = np.float32(inject.gain)
        if inj_weights.shape != inj_target.shape:
            raise ValueError("inject.weights and inject.target_ids must have equal length")
        if inj_signal.shape[0] != step_count:
            raise ValueError(
                f"inject.signal length {inj_signal.shape[0]}, expected {step_count}"
            )
    else:
        inj_target = np.zeros(0, dtype=np.int64)
        inj_weights = np.zeros(0, dtype=np.float32)
        inj_signal = np.zeros(step_count, dtype=np.float32)
        inj_gain = np.float32(1.0)

    clamp_node = clamp_start = clamp_end = -1
    if clamp is not None:
        clamp_node, clamp_start, clamp_end = clamp

    for step in range(step_count):
        delayed_spikes = spike_history[(step - delay_steps) % spike_history.shape[0]]
        synaptic_current *= alpha_syn
        synaptic_current += graph.incoming_drive(delayed_spikes.astype(np.float32)) * p.synaptic_gain

        injected_current = np.zeros(node_count, dtype=np.float32)
        injected_current[inj_target] += inj_weights * inj_signal[step] * inj_gain
        if stimulus is not None:
            injected_current += stimulus.current_at(step * p.dt_ms, node_count)

        active = refractory == 0
        voltage[active] += alpha_m * (
            (p.v_rest - voltage[active]) + synaptic_current[active] + injected_current[active]
        )
        voltage[~active] = np.float32(p.v_reset)
        refractory[~active] -= 1

        # Transient node silencing: hold the node at rest inside the window.
        if clamp_node >= 0 and clamp_start <= step < clamp_end:
            voltage[clamp_node] = np.float32(p.v_rest)
            synaptic_current[clamp_node] = np.float32(0.0)
            refractory[clamp_node] = 0

        spikes = active & (voltage >= p.v_th)
        if clamp_node >= 0 and clamp_start <= step < clamp_end:
            spikes[clamp_node] = False
        voltage[spikes] = np.float32(p.v_reset)
        refractory[spikes] = refractory_steps

        spikes_out[step] = spikes
        spike_history[step % spike_history.shape[0]] = spikes
        mean_voltage[step] = np.float32(voltage.mean())
        if voltage_traj is not None:
            voltage_traj[step] = voltage

    return DynamicsResult(
        spikes=spikes_out,
        voltage=voltage_traj,
        mean_membrane_potential=mean_voltage,
        final_voltage=voltage,
        dt_ms=p.dt_ms,
    )


def run_baseline_state(
    graph: SparseGraph,
    parameters: LIFParameters,
    *,
    duration_ms: float,
    stimulus: Stimulus,
) -> np.ndarray:
    """Run the network under its intact drive and return the final voltage state.

    This state seeds free-run trajectories (PRD §8.1 ``x0 = baseline_state``).
    """
    result = run_dynamics(
        graph, parameters, duration_ms=duration_ms, stimulus=stimulus, record_voltage=False
    )
    return result.final_voltage
