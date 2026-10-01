"""Memory-conscious activity recording for small M0 toy experiments."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ActivityRecord:
    """Spike raster and population-average membrane trace."""

    spikes: np.ndarray
    mean_membrane_potential: np.ndarray
    dt_ms: float

    @property
    def total_spikes(self) -> int:
        return int(self.spikes.sum())

    def firing_rate_hz(self) -> np.ndarray:
        duration_s = self.spikes.shape[0] * self.dt_ms / 1000.0
        if duration_s <= 0:
            return np.zeros(self.spikes.shape[1], dtype=np.float32)
        return self.spikes.sum(axis=0, dtype=np.int64).astype(np.float32) / duration_s

    def save(self, path: Path) -> None:
        """Save compressed activity for later inspection without any visualization layer."""
        np.savez_compressed(
            path,
            spikes=self.spikes,
            mean_membrane_potential=self.mean_membrane_potential,
            dt_ms=np.float32(self.dt_ms),
        )
