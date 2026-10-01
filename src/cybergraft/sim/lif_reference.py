"""Framework-independent float32 LIF reference implementation."""

from dataclasses import dataclass

import numpy as np

from cybergraft.graph.sparse_graph import SparseGraph

from .activity import ActivityRecord
from .stimulus import Stimulus


@dataclass(frozen=True)
class LIFParameters:
    dt_ms: float
    tau_m_ms: float
    tau_syn_ms: float
    v_rest: float
    v_reset: float
    v_th: float
    refractory_ms: float
    synaptic_delay_ms: float
    synaptic_gain: float = 1.0

    @classmethod
    def from_config(cls, config: dict) -> "LIFParameters":
        parameter_names = (
            "dt_ms",
            "tau_m_ms",
            "tau_syn_ms",
            "v_rest",
            "v_reset",
            "v_th",
            "refractory_ms",
            "synaptic_delay_ms",
            "synaptic_gain",
        )
        try:
            parameters = cls(**{key: float(config[key]) for key in parameter_names})
        except KeyError as exc:
            raise ValueError(f"Missing LIF parameter: {exc.args[0]}") from exc
        if parameters.dt_ms <= 0 or parameters.tau_m_ms <= 0 or parameters.tau_syn_ms <= 0:
            raise ValueError("dt_ms, tau_m_ms, and tau_syn_ms must be positive")
        if parameters.refractory_ms < 0 or parameters.synaptic_delay_ms < 0:
            raise ValueError("Delays and refractory duration must be non-negative")
        if parameters.synaptic_gain <= 0:
            raise ValueError("synaptic_gain must be positive")
        if parameters.v_th <= parameters.v_reset:
            raise ValueError("v_th must be greater than v_reset")
        return parameters


class ReferenceLIFSimulator:
    """A deterministic Euler LIF simulator using CSR synaptic propagation."""

    def __init__(self, graph: SparseGraph, parameters: LIFParameters) -> None:
        self.graph = graph
        self.parameters = parameters

    def run(self, duration_ms: float, stimulus: Stimulus) -> ActivityRecord:
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        p = self.parameters
        step_count = int(round(duration_ms / p.dt_ms))
        if step_count <= 0:
            raise ValueError("duration_ms must include at least one timestep")
        node_count = self.graph.node_count
        delay_steps = int(round(p.synaptic_delay_ms / p.dt_ms))
        spike_history = np.zeros((delay_steps + 1, node_count), dtype=np.bool_)
        spikes_out = np.zeros((step_count, node_count), dtype=np.bool_)
        mean_voltage = np.zeros(step_count, dtype=np.float32)
        voltage = np.full(node_count, p.v_rest, dtype=np.float32)
        synaptic_current = np.zeros(node_count, dtype=np.float32)
        refractory = np.zeros(node_count, dtype=np.int32)
        alpha_m = np.float32(p.dt_ms / p.tau_m_ms)
        alpha_syn = np.float32(np.exp(-p.dt_ms / p.tau_syn_ms))
        refractory_steps = int(np.ceil(p.refractory_ms / p.dt_ms))

        for step in range(step_count):
            delayed_spikes = spike_history[(step - delay_steps) % spike_history.shape[0]]
            synaptic_current *= alpha_syn
            synaptic_current += self.graph.incoming_drive(delayed_spikes.astype(np.float32)) * p.synaptic_gain
            time_ms = step * p.dt_ms
            injected_current = stimulus.current_at(time_ms, node_count)
            active = refractory == 0
            voltage[active] += alpha_m * (
                (p.v_rest - voltage[active]) + synaptic_current[active] + injected_current[active]
            )
            voltage[~active] = np.float32(p.v_reset)
            refractory[~active] -= 1
            spikes = active & (voltage >= p.v_th)
            voltage[spikes] = np.float32(p.v_reset)
            refractory[spikes] = refractory_steps
            spikes_out[step] = spikes
            spike_history[step % spike_history.shape[0]] = spikes
            mean_voltage[step] = np.float32(voltage.mean())
        return ActivityRecord(spikes=spikes_out, mean_membrane_potential=mean_voltage, dt_ms=p.dt_ms)
