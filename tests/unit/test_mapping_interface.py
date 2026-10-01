import numpy as np
import pytest

from cybergraft.experiment.mapping_interface import MappingInterface


def test_forward_shape_and_value():
    mapping = MappingInterface(n_source=4, n_target=2, seed=0)
    spikes = np.array([1, 0, 1, 0], dtype=bool)
    drive = mapping.forward(spikes)
    assert drive.shape == (2,)
    assert drive.dtype == np.float32
    np.testing.assert_allclose(drive, mapping.W_map @ spikes.astype(np.float32))


def test_stdp_potentiation():
    mapping = MappingInterface(n_source=2, n_target=1, seed=0)
    mapping.W_map[...] = 0.0
    lr = 0.1
    # pre-then-post: source 0 fires, then target 0 fires -> W[0,0] potentiated by lr.
    mapping.forward(np.array([True, False]))
    mapping.update_stdp(np.array([False]), dt_ms=0.5, lr=lr)
    mapping.forward(np.array([False, False]))
    mapping.update_stdp(np.array([True]), dt_ms=0.5, lr=lr)
    assert mapping.W_map[0, 0] == pytest.approx(lr, abs=1e-6)
    assert mapping.W_map[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_stdp_depression():
    mapping = MappingInterface(n_source=2, n_target=1, seed=0)
    mapping.W_map[...] = 0.5
    lr = 0.1
    # post-before-pre: target fires, then source 0 fires -> W[0,0] depressed by lr/2.
    mapping.forward(np.array([False, False]))
    mapping.update_stdp(np.array([True]), dt_ms=0.5, lr=lr)
    mapping.forward(np.array([True, False]))
    mapping.update_stdp(np.array([False]), dt_ms=0.5, lr=lr)
    assert mapping.W_map[0, 0] == pytest.approx(0.5 - 0.5 * lr, abs=1e-6)


def test_clip_bounds_weights():
    mapping = MappingInterface(n_source=2, n_target=1, seed=0, weight_min=0.0, weight_max=1.0)
    mapping.W_map[...] = np.array([[-0.5, 2.0]])
    mapping.clip()
    assert mapping.W_map.min() >= 0.0
    assert mapping.W_map.max() <= 1.0


def test_save_load_roundtrip(tmp_path):
    mapping = MappingInterface(n_source=4, n_target=2, seed=7)
    path = tmp_path / "mapping.npz"
    mapping.save(path)
    loaded = MappingInterface.load(path)
    np.testing.assert_allclose(mapping.W_map, loaded.W_map)
    assert loaded.n_source == 4
    assert loaded.n_target == 2
