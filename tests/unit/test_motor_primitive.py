import numpy as np
import pytest

from cybergraft.experiment.motor_primitive import EyeMovementPrimitive, MotorPrimitive
from cybergraft.sim.activity import ActivityRecord


def _activity(spikes):
    steps = spikes.shape[0]
    return ActivityRecord(
        spikes=spikes,
        mean_membrane_potential=np.zeros(steps, dtype=np.float32),
        dt_ms=0.5,
    )


def test_decode_returns_mean_trace():
    spikes = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=bool,
    )
    activity = _activity(spikes)
    primitive = EyeMovementPrimitive([2, 3])
    decoded = primitive.decode(activity)
    assert decoded.dtype == np.float32
    assert decoded.shape == (3,)
    np.testing.assert_allclose(decoded, np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert primitive.name == "eye_movement"
    assert primitive.output_nodes == [2, 3]


def test_similarity_identical_and_zero_vector():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert MotorPrimitive.similarity(a, a) == pytest.approx(1.0)
    zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    assert MotorPrimitive.similarity(a, zero) == 0.0
    assert MotorPrimitive.similarity(zero, zero) == 1.0


def test_similarity_bounded():
    a = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    b = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    value = MotorPrimitive.similarity(a, b)
    assert 0.0 <= value <= 1.0
