import numpy as np

from cybergraft.sim.stimulus import PoissonStimulus, RhythmicStimulus


def _poisson_config():
    return {
        "type": "poisson",
        "target_group": "input",
        "rate_hz": 120.0,
        "amplitude": 25.0,
        "start_ms": 20.0,
        "duration_ms": 200.0,
    }


def _groups():
    return {"input": [0, 1, 2], "output": [5, 6], "lesion": [3, 4]}


def test_poisson_current_is_nonzero_within_window():
    groups = _groups()
    stim = PoissonStimulus.from_config(_poisson_config(), groups, seed=0, dt_ms=0.5)
    node_count = 7
    hits = 0
    for step in range(800):
        time_ms = step * 0.5
        current = stim.current_at(time_ms, node_count)
        if current.sum() > 0:
            hits += 1
    assert hits > 0
    # nothing outside the [start, start+duration] window
    outside = 0
    for step in range(800):
        time_ms = step * 0.5
        if time_ms < 20.0 or time_ms >= 220.0:
            if stim.current_at(time_ms, node_count).sum() > 0:
                outside += 1
    assert outside == 0


def test_poisson_seed_determinism_and_difference():
    groups = _groups()
    cfg = _poisson_config()
    stim_a = PoissonStimulus.from_config(cfg, groups, seed=0, dt_ms=0.5)
    stim_b = PoissonStimulus.from_config(cfg, groups, seed=0, dt_ms=0.5)
    stim_c = PoissonStimulus.from_config(cfg, groups, seed=1, dt_ms=0.5)
    assert stim_a.spike_steps == stim_b.spike_steps
    assert stim_a.spike_steps != stim_c.spike_steps


def test_rhythmic_stimulus_sinusoidal_profile():
    groups = {"input": [0]}
    stim = RhythmicStimulus.from_config(
        {"type": "rhythmic", "target_group": "input", "amplitude": 100.0,
         "frequency_hz": 5.0, "start_ms": 10.0, "duration_ms": 400.0},
        groups,
    )
    node_count = 1
    # Outside the window: zero.
    assert stim.current_at(0.0, node_count)[0] == 0.0
    assert stim.current_at(500.0, node_count)[0] == 0.0
    # First peak at t = start + 1/(4f) = 10 + 50 = 60 ms -> amplitude.
    peak = stim.current_at(60.0, node_count)[0]
    assert peak == np.float32(100.0)
    # Trough at t = start + 3/(4f) = 10 + 150 = 160 ms -> ~0.
    trough = stim.current_at(160.0, node_count)[0]
    assert abs(float(trough)) < 1e-3
