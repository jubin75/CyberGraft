"""Stimulus representation, kept independent from graph construction."""

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

import numpy as np


class Stimulus(Protocol):
    """Minimal structural interface satisfied by every stimulus type."""

    def current_at(self, time_ms: float, node_count: int) -> np.ndarray: ...


@dataclass(frozen=True)
class DirectCurrentStimulus:
    """A configurable rectangular direct-current injection."""

    target_nodes: tuple[int, ...]
    amplitude: float
    start_ms: float
    duration_ms: float

    @classmethod
    def from_config(cls, config: dict, groups: dict[str, list[int]]) -> "DirectCurrentStimulus":
        if config.get("type") != "direct_current":
            raise ValueError("M0 supports only stimulus.type=direct_current")
        target_nodes: Iterable[int]
        if "target_nodes" in config:
            target_nodes = config["target_nodes"]
        else:
            target_group = config.get("target_group")
            if not target_group or target_group not in groups:
                raise ValueError("Stimulus requires target_nodes or a valid target_group")
            target_nodes = groups[target_group]
        ids = tuple(int(node_id) for node_id in target_nodes)
        if not ids:
            raise ValueError("Stimulus must target at least one node")
        duration_ms = float(config["duration_ms"])
        if duration_ms <= 0:
            raise ValueError("Stimulus duration_ms must be positive")
        return cls(
            target_nodes=ids,
            amplitude=float(config["amplitude"]),
            start_ms=float(config["start_ms"]),
            duration_ms=duration_ms,
        )

    def current_at(self, time_ms: float, node_count: int) -> np.ndarray:
        current = np.zeros(node_count, dtype=np.float32)
        if self.start_ms <= time_ms < self.start_ms + self.duration_ms:
            current[list(self.target_nodes)] = np.float32(self.amplitude)
        return current


@dataclass
class PoissonStimulus:
    """A seeded Poisson spike train delivered to a target node group."""

    target_nodes: tuple[int, ...]
    amplitude: float
    dt_ms: float
    spike_steps: dict[int, set[int]]

    @classmethod
    def from_config(
        cls, config: dict, groups: dict[str, list[int]], seed: int, *, dt_ms: Optional[float] = None
    ) -> "PoissonStimulus":
        if config.get("type") != "poisson":
            raise ValueError("Poisson stimulus requires stimulus.type=poisson")
        target_group = config.get("target_group")
        if not target_group or target_group not in groups:
            raise ValueError("Poisson stimulus requires a valid target_group")
        target_nodes = tuple(int(node_id) for node_id in groups[target_group])
        if not target_nodes:
            raise ValueError("Poisson stimulus must target at least one node")
        if dt_ms is None:
            raise ValueError("Poisson stimulus requires dt_ms")
        rate_hz = float(config["rate_hz"])
        duration_ms = float(config["duration_ms"])
        if rate_hz < 0 or duration_ms <= 0:
            raise ValueError("Poisson stimulus rate_hz must be non-negative and duration_ms positive")
        dt_ms = float(dt_ms)
        start_ms = float(config["start_ms"])
        amplitude = float(config["amplitude"])
        steps = int(round(duration_ms / dt_ms))
        start_step = int(round(start_ms / dt_ms))
        rng = np.random.default_rng(seed)
        probability = rate_hz * dt_ms / 1000.0
        spike_steps: dict[int, set[int]] = {}
        for node_id in target_nodes:
            draws = rng.random(steps)
            spike_steps[node_id] = {int(s) for s in np.nonzero(draws < probability)[0] + start_step}
        return cls(
            target_nodes=target_nodes,
            amplitude=amplitude,
            dt_ms=dt_ms,
            spike_steps=spike_steps,
        )

    def current_at(self, time_ms: float, node_count: int) -> np.ndarray:
        current = np.zeros(node_count, dtype=np.float32)
        step = int(round(time_ms / self.dt_ms))
        for node_id in self.target_nodes:
            if step in self.spike_steps[node_id]:
                current[node_id] = np.float32(self.amplitude)
        return current


@dataclass(frozen=True)
class RhythmicStimulus:
    """A sinusoidal current injection oscillating between 0 and ``amplitude``."""

    target_nodes: tuple[int, ...]
    amplitude: float
    frequency_hz: float
    start_ms: float
    duration_ms: float

    @classmethod
    def from_config(cls, config: dict, groups: dict[str, list[int]]) -> "RhythmicStimulus":
        if config.get("type") != "rhythmic":
            raise ValueError("Rhythmic stimulus requires stimulus.type=rhythmic")
        target_group = config.get("target_group")
        if not target_group or target_group not in groups:
            raise ValueError("Rhythmic stimulus requires a valid target_group")
        target_nodes = tuple(int(node_id) for node_id in groups[target_group])
        if not target_nodes:
            raise ValueError("Rhythmic stimulus must target at least one node")
        amplitude = float(config["amplitude"])
        frequency_hz = float(config["frequency_hz"])
        duration_ms = float(config["duration_ms"])
        if amplitude <= 0:
            raise ValueError("Rhythmic stimulus amplitude must be positive")
        if frequency_hz <= 0:
            raise ValueError("Rhythmic stimulus frequency_hz must be positive")
        if duration_ms <= 0:
            raise ValueError("Rhythmic stimulus duration_ms must be positive")
        return cls(
            target_nodes=target_nodes,
            amplitude=amplitude,
            frequency_hz=frequency_hz,
            start_ms=float(config["start_ms"]),
            duration_ms=duration_ms,
        )

    def current_at(self, time_ms: float, node_count: int) -> np.ndarray:
        current = np.zeros(node_count, dtype=np.float32)
        if self.start_ms <= time_ms < self.start_ms + self.duration_ms:
            phase = 2.0 * np.pi * self.frequency_hz * (time_ms - self.start_ms) / 1000.0
            current[list(self.target_nodes)] = np.float32(self.amplitude * (0.5 + 0.5 * np.sin(phase)))
        return current
