"""Standalone LIF + STDP bypass training loop."""

from dataclasses import dataclass

import numpy as np

from cybergraft.sim.lif_reference import LIFParameters

from .mapping_interface import MappingInterface


@dataclass
class BypassResult:
    spikes: np.ndarray
    output_spikes: int
    mapping: MappingInterface


def run_bypass_simulation(
    *,
    graph,
    parameters: LIFParameters,
    mapping: MappingInterface,
    target_node_ids: list[int],
    source_signal: np.ndarray,
    mapping_gain: float,
    stdp_lr: float,
    duration_ms: float,
    train: bool = True,
    stimulus=None,
) -> BypassResult:
    if source_signal.shape[1] != mapping.n_source:
        raise ValueError(
            f"source_signal has {source_signal.shape[1]} sources; mapping expects {mapping.n_source}"
        )
    if len(target_node_ids) != mapping.n_target:
        raise ValueError(
            f"target_node_ids has {len(target_node_ids)} nodes; mapping expects {mapping.n_target}"
        )
    p = parameters
    step_count = int(round(duration_ms / p.dt_ms))
    if source_signal.shape[0] != step_count:
        raise ValueError(
            f"source_signal has {source_signal.shape[0]} steps; expected {step_count}"
        )
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
    target_ids = np.asarray(target_node_ids, dtype=np.int64)
    output_ids = graph.groups["output"]

    for step in range(step_count):
        delayed_spikes = spike_history[(step - delay_steps) % spike_history.shape[0]]
        synaptic_current *= alpha_syn
        synaptic_current += graph.incoming_drive(delayed_spikes.astype(np.float32)) * p.synaptic_gain
        mapped = mapping.forward(source_signal[step])
        injected_current = np.zeros(node_count, dtype=np.float32)
        injected_current[target_ids] += mapped * np.float32(mapping_gain)
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
        if train:
            mapping.update_stdp(spikes[target_ids], dt_ms=p.dt_ms, lr=stdp_lr)

    output_spikes = int(spikes_out[:, output_ids].sum())
    return BypassResult(spikes=spikes_out, output_spikes=output_spikes, mapping=mapping)


def rate_similarity(current: np.ndarray, baseline: np.ndarray) -> float:
    """Amplitude-sensitive similarity in [0,1]."""
    current = np.asarray(current, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    norm_base = float(np.linalg.norm(baseline))
    if norm_base == 0.0:
        return 1.0 if float(np.linalg.norm(current)) == 0.0 else 0.0
    similarity = 1.0 - float(np.linalg.norm(current - baseline)) / norm_base
    return max(0.0, min(1.0, similarity))


def timing_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Scale-invariant timing/shape similarity in [0,1] (zero-mean cosine).

    Ignores DC offset and amplitude; 1.0 = same temporal pattern,
    0.5 = uncorrelated, 0.0 = anti-correlated.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0 if (norm_a == 0.0 and norm_b == 0.0) else 0.0
    cosine = float(np.dot(a, b) / (norm_a * norm_b))
    return 0.5 * (cosine + 1.0)


def output_rate(spikes: np.ndarray, output_ids, win: int = 12) -> np.ndarray:
    """Smoothed per-step total output spike count (float32)."""
    summed = spikes[:, output_ids].sum(axis=1).astype(np.float32)
    kernel = np.ones(win, dtype=np.float32)
    return np.convolve(summed, kernel, mode="same").astype(np.float32)


def run_rstdp_training(
    *,
    graph,
    parameters: LIFParameters,
    mapping: MappingInterface,
    target_node_ids: list[int],
    source_signal: np.ndarray,
    mapping_gain: float,
    rstdp_lr: float,
    duration_ms: float,
    baseline_rate: np.ndarray,
    reward_baseline: float,
    n_episodes: int,
    stimulus=None,
    tau_elig_ms: float = 20.0,
    track_recovery: bool = False,
    per_unit_weight: np.ndarray | None = None,
    temporal_reward: bool = False,
):
    """Train a mapping via R-STDP.

    Returns ``(final_spikes, mapping)``; if ``track_recovery`` is True, returns
    ``(final_spikes, mapping, recovery_curve)`` where ``recovery_curve`` is the
    per-episode ``rate_similarity(output_rate, baseline_rate)`` list.

    ``per_unit_weight`` (length ``mapping.n_target``) replaces the uniform scalar
    reward with a per-candidate weighted reward when provided.

    ``temporal_reward`` replaces the single episodic scalar reward with a
    per-time-step reward ``reward_t = (baseline_rate[t] - rate[t]) / ||baseline_rate||``
    (the signed missing drive, normalized), accumulated step-wise:
    ``ΔW = η · Σ_t reward_t · eligibility_t``.
    """
    if source_signal.shape[1] != mapping.n_source:
        raise ValueError(
            f"source_signal has {source_signal.shape[1]} sources; mapping expects {mapping.n_source}"
        )
    if len(target_node_ids) != mapping.n_target:
        raise ValueError(
            f"target_node_ids has {len(target_node_ids)} nodes; mapping expects {mapping.n_target}"
        )
    n_steps = source_signal.shape[0]
    dt_ms = parameters.dt_ms
    pre_decay = float(np.exp(-dt_ms / tau_elig_ms))
    p = parameters
    node_count = graph.node_count
    delay_steps = int(round(p.synaptic_delay_ms / p.dt_ms))
    refractory_steps = int(np.ceil(p.refractory_ms / p.dt_ms))
    alpha_m = np.float32(p.dt_ms / p.tau_m_ms)
    alpha_syn = np.float32(np.exp(-p.dt_ms / p.tau_syn_ms))
    target_ids = np.asarray(target_node_ids, dtype=np.int64)
    output_ids = graph.groups["output"]

    recovery_curve: list = [] if track_recovery else []
    final_spikes = np.zeros((n_steps, node_count), dtype=np.bool_)
    for episode in range(n_episodes):
        eligibility = np.zeros((mapping.n_target, mapping.n_source), dtype=np.float32)
        pre_trace = np.zeros(mapping.n_source, dtype=np.float32)
        spike_history = np.zeros((delay_steps + 1, node_count), dtype=np.bool_)
        spikes_out = np.zeros((n_steps, node_count), dtype=np.bool_)
        voltage = np.full(node_count, p.v_rest, dtype=np.float32)
        synaptic_current = np.zeros(node_count, dtype=np.float32)
        refractory = np.zeros(node_count, dtype=np.int32)
        target_spikes_all = np.zeros((n_steps, mapping.n_target), dtype=np.float32)
        pre_trace_all = np.zeros((n_steps, mapping.n_source), dtype=np.float32)

        for step in range(n_steps):
            delayed_spikes = spike_history[(step - delay_steps) % spike_history.shape[0]]
            synaptic_current *= alpha_syn
            synaptic_current += graph.incoming_drive(delayed_spikes.astype(np.float32)) * p.synaptic_gain
            mapped = mapping.forward(source_signal[step])
            injected_current = np.zeros(node_count, dtype=np.float32)
            injected_current[target_ids] += mapped * np.float32(mapping_gain)
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
            target_spikes = spikes[target_ids].astype(np.float32)
            eligibility += np.outer(target_spikes, pre_trace)
            if temporal_reward:
                target_spikes_all[step] = target_spikes
                pre_trace_all[step] = pre_trace
            pre_trace *= pre_decay
            pre_trace[source_signal[step]] += 1.0

        rate = output_rate(spikes_out, output_ids)
        sim = rate_similarity(rate, baseline_rate)
        if track_recovery:
            recovery_curve.append(float(sim))
        if temporal_reward:
            norm_base = float(np.linalg.norm(np.asarray(baseline_rate, dtype=np.float64)))
            scale = norm_base if norm_base > 0.0 else 1.0
            reward_t = (np.asarray(baseline_rate, dtype=np.float32) - rate) * np.float32(1.0 / scale)
            weighted_elig = target_spikes_all.T @ (reward_t[:, None] * pre_trace_all)
            if per_unit_weight is not None:
                weighted = weighted_elig * np.asarray(per_unit_weight, dtype=np.float32)[:, None]
                mapping.W_map += np.float32(rstdp_lr) * weighted
            else:
                mapping.W_map += np.float32(rstdp_lr) * weighted_elig
        else:
            reward = sim - reward_baseline
            if per_unit_weight is not None:
                weighted = np.float32(reward) * np.asarray(per_unit_weight, dtype=np.float32)
                mapping.W_map += np.float32(rstdp_lr) * eligibility * weighted[:, None]
            else:
                mapping.W_map += np.float32(rstdp_lr) * eligibility * np.float32(reward)
        mapping.clip()
        final_spikes = spikes_out

    if track_recovery:
        return final_spikes, mapping, recovery_curve
    return final_spikes, mapping
