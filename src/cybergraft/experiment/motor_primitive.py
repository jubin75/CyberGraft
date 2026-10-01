"""Motor primitive readout and similarity over population firing-rate traces."""

import numpy as np

from cybergraft.sim.activity import ActivityRecord


class MotorPrimitive:
    def __init__(self, *, name: str, output_nodes: list[int]) -> None:
        self.name = name
        self.output_nodes = list(output_nodes)

    def decode(self, activity: ActivityRecord) -> np.ndarray:
        return np.asarray(activity.spikes[:, self.output_nodes].mean(axis=1), dtype=np.float32)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0 if norm_a == 0.0 and norm_b == 0.0 else 0.0
        cosine = float(np.dot(a, b) / (norm_a * norm_b))
        cosine = max(-1.0, min(1.0, cosine))
        return (cosine + 1.0) / 2.0


class EyeMovementPrimitive(MotorPrimitive):
    def __init__(self, output_nodes):
        super().__init__(name="eye_movement", output_nodes=list(output_nodes))
