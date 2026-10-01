"""Shared primitives for the connectome-level neural-repair experiments.

Extracted so the public minimal closed-loop exposes only the helpers needed to
reproduce the paper's three findings (structure-function, temporal credit,
substitutability).
"""

import numpy as np

from cybergraft.sim.stimulus import DirectCurrentStimulus, PoissonStimulus, RhythmicStimulus


def _build_stimulus(config: dict, groups: dict, seed: int, dt_ms: float):
    stimulus_type = config.get("type")
    if stimulus_type == "poisson":
        return PoissonStimulus.from_config(config, groups, seed, dt_ms=dt_ms)
    if stimulus_type == "direct_current":
        return DirectCurrentStimulus.from_config(config, groups)
    if stimulus_type == "rhythmic":
        return RhythmicStimulus.from_config(config, groups)
    raise ValueError(f"unsupported stimulus type: {stimulus_type}")


def _run_fixed_drive(graph, parameters, target_ids, drive, mapping_gain, duration_ms, stimulus=None):
    """Standalone LIF pass injecting a scalar per-step drive into target nodes (no STDP)."""
    p = parameters
    step_count = int(round(duration_ms / p.dt_ms))
    node_count = graph.node_count
    delay_steps = int(round(p.synaptic_delay_ms / p.dt_ms))
    spike_history = np.zeros((delay_steps + 1, node_count), dtype=np.bool_)
    spikes_out = np.zeros((step_count, node_count), dtype=np.bool_)
    voltage = np.full(node_count, p.v_rest, dtype=np.float32)
    synaptic_current = np.zeros(node_count, dtype=np.float32)
    refractory = np.zeros(node_count, dtype=np.int32)
    alpha_m = np.float32(p.dt_ms / p.tau_m_ms)
    alpha_syn = np.float32(np.exp(-p.dt_ms / p.tau_syn_ms))
    refractory_steps = int(np.ceil(p.refractory_ms / p.dt_ms))
    target_arr = np.asarray(target_ids, dtype=np.int64)

    for step in range(step_count):
        delayed_spikes = spike_history[(step - delay_steps) % spike_history.shape[0]]
        synaptic_current *= alpha_syn
        synaptic_current += graph.incoming_drive(delayed_spikes.astype(np.float32)) * p.synaptic_gain
        injected_current = np.zeros(node_count, dtype=np.float32)
        injected_current[target_arr] += np.float32(drive[step]) * np.float32(mapping_gain)
        if stimulus is not None:
            injected_current += stimulus.current_at(step * p.dt_ms, node_count)
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

    return spikes_out


def generate_informative_source(
    n_steps: int,
    n_source: int,
    lambda_: float,
    command_rate: np.ndarray,
    dt_ms: float,
    onset_ms: float,
    duration_ms: float,
    seed: int,
) -> np.ndarray:
    """Source whose correlation with the target is controlled by lambda_ in [0,1].

    ``command_rate`` is the normalized intact output rate (the "command").
    ``lambda_=0`` fires exactly where the target fires; ``lambda_=1`` fires at a
    flat Poisson rate (the mean command rate, i.e. no shared information).
    """
    rng = np.random.default_rng(seed)
    onset_step = int(round(onset_ms / dt_ms))
    end_step = min(n_steps, onset_step + int(round(duration_ms / dt_ms)))
    flat = float(np.mean(command_rate[onset_step:end_step])) if end_step > onset_step else 0.0
    signal = np.zeros((n_steps, n_source), dtype=bool)
    for step in range(onset_step, end_step):
        probability = (1.0 - lambda_) * float(command_rate[step]) + lambda_ * flat
        probability = max(0.0, min(1.0, probability))
        signal[step, :] = rng.random(n_source) < probability
    return signal
