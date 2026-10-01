"""Learned cross-species source-to-target mapping with trace-based pair STDP."""

from pathlib import Path

import numpy as np


class MappingInterface:
    def __init__(
        self,
        *,
        n_source: int,
        n_target: int,
        weight_min: float = 0.0,
        weight_max: float = 1.0,
        tau_pre_ms: float = 20.0,
        tau_post_ms: float = 20.0,
        init_scale: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.tau_pre_ms = float(tau_pre_ms)
        self.tau_post_ms = float(tau_post_ms)
        self.init_scale = float(init_scale)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.W_map = rng.uniform(0, self.init_scale, (n_target, n_source)).astype(np.float32)
        self.pre_trace = np.zeros(n_source, dtype=np.float32)
        self.post_trace = np.zeros(n_target, dtype=np.float32)
        self.last_source_spikes = np.zeros(n_source, dtype=bool)

    @property
    def n_source(self) -> int:
        return int(self.W_map.shape[1])

    @property
    def n_target(self) -> int:
        return int(self.W_map.shape[0])

    def forward(self, source_spikes: np.ndarray) -> np.ndarray:
        self.last_source_spikes = np.asarray(source_spikes, dtype=bool)
        return np.asarray(self.W_map @ source_spikes.astype(np.float32), dtype=np.float32)

    def update_stdp(self, target_spikes: np.ndarray, *, dt_ms: float, lr: float) -> None:
        pre_decay = np.float32(np.exp(-dt_ms / self.tau_pre_ms))
        post_decay = np.float32(np.exp(-dt_ms / self.tau_post_ms))
        a_plus = np.float32(lr)
        a_minus = np.float32(lr * 0.5)
        source_fired = self.last_source_spikes
        target_fired = np.asarray(target_spikes, dtype=bool)
        if np.any(source_fired):
            self.W_map[:, source_fired] -= a_minus * self.post_trace[:, None]
        if np.any(target_fired):
            self.W_map[target_fired, :] += a_plus * self.pre_trace[None, :]
        self.pre_trace *= pre_decay
        self.post_trace *= post_decay
        self.pre_trace[source_fired] += np.float32(1.0)
        self.post_trace[target_fired] += np.float32(1.0)
        self.clip()

    def clip(self) -> None:
        np.clip(self.W_map, self.weight_min, self.weight_max, out=self.W_map)

    def save(self, path: Path) -> None:
        np.savez(
            path,
            W_map=self.W_map,
            n_source=self.n_source,
            n_target=self.n_target,
            weight_min=self.weight_min,
            weight_max=self.weight_max,
            tau_pre_ms=self.tau_pre_ms,
            tau_post_ms=self.tau_post_ms,
            init_scale=self.init_scale,
            seed=self.seed,
        )

    @classmethod
    def load(cls, path: Path) -> "MappingInterface":
        data = np.load(path)
        obj = cls(
            n_source=int(data["n_source"]),
            n_target=int(data["n_target"]),
            weight_min=float(data["weight_min"]),
            weight_max=float(data["weight_max"]),
            tau_pre_ms=float(data["tau_pre_ms"]),
            tau_post_ms=float(data["tau_post_ms"]),
            init_scale=float(data["init_scale"]),
            seed=int(data["seed"]),
        )
        obj.W_map = np.asarray(data["W_map"], dtype=np.float32)
        return obj
