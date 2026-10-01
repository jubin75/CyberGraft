import numpy as np
import pytest

from cybergraft.experiment.bypass_training import (
    output_rate,
    rate_similarity,
    run_bypass_simulation,
    run_rstdp_training,
    timing_similarity,
)
from cybergraft.experiment.mapping_interface import MappingInterface
from cybergraft.graph.edge_schema import Edge
from cybergraft.graph.node_schema import Node
from cybergraft.graph.sparse_graph import SparseGraph
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.stimulus import DirectCurrentStimulus


def _bypass_graph():
    nodes = [
        Node(id=0, species="zebrafish", cell_type="bypass", subsystem="bypass"),
        Node(id=1, species="zebrafish", cell_type="bypass", subsystem="bypass"),
        Node(id=2, species="zebrafish", cell_type="ABD_m", subsystem="motor"),
        Node(id=3, species="zebrafish", cell_type="ABD_m", subsystem="motor"),
    ]
    edges = []
    for source in (0, 1):
        for target in (2, 3):
            edges.append(
                Edge(source=source, target=target, weight=3.0, provenance="test",
                     sign="excitatory", sign_source="model_assumption", delay_ms=1.8)
            )
    return SparseGraph.from_edges(nodes, edges, groups={"output": [2, 3]}, max_nodes=4)


def _params():
    return LIFParameters.from_config(
        dict(
            dt_ms=0.5, tau_m_ms=20.0, tau_syn_ms=5.0, v_rest=-52.0, v_reset=-52.0,
            v_th=-45.0, refractory_ms=2.2, synaptic_delay_ms=1.8, synaptic_gain=20.0,
        )
    )


def _source_burst(n_steps, n_source, onset_step, burst_steps):
    signal = np.zeros((n_steps, n_source), dtype=bool)
    signal[onset_step:onset_step + burst_steps, :] = True
    return signal


def test_trained_mapping_recovers_more_than_untrained():
    graph = _bypass_graph()
    params = _params()
    n_steps = 400
    source = _source_burst(n_steps, 16, onset_step=10, burst_steps=190)
    common = dict(
        graph=graph,
        parameters=params,
        target_node_ids=[0, 1],
        source_signal=source,
        mapping_gain=30.0,
        stdp_lr=0.02,
        duration_ms=200.0,
    )
    trained = MappingInterface(n_source=16, n_target=2, seed=0)
    res_train = run_bypass_simulation(mapping=trained, train=True, **common)
    untrained = MappingInterface(n_source=16, n_target=2, seed=0)
    res_untrained = run_bypass_simulation(mapping=untrained, train=False, **common)

    assert res_train.output_spikes > res_untrained.output_spikes
    assert np.count_nonzero(trained.W_map) > 0
    init = MappingInterface(n_source=16, n_target=2, seed=0)
    assert not np.allclose(trained.W_map, init.W_map)


def test_rstdp_training_improves_rate_similarity():
    graph = _bypass_graph()
    params = _params()
    n_steps = 400
    source = _source_burst(n_steps, 16, onset_step=10, burst_steps=190)

    # Intact baseline: drive the bypass candidates directly so output fires.
    baseline_activity = ReferenceLIFSimulator(graph, params).run(
        200.0, DirectCurrentStimulus((0, 1), amplitude=30.0, start_ms=10.0, duration_ms=180.0)
    )
    baseline_rate = output_rate(baseline_activity.spikes, graph.groups["output"])
    reward_baseline = 0.0  # fully-lesioned state has no output

    mapping = MappingInterface(n_source=16, n_target=2, seed=0)
    final_spikes, mapping = run_rstdp_training(
        graph=graph,
        parameters=params,
        mapping=mapping,
        target_node_ids=[0, 1],
        source_signal=source,
        mapping_gain=30.0,
        rstdp_lr=0.05,
        duration_ms=200.0,
        baseline_rate=baseline_rate,
        reward_baseline=reward_baseline,
        n_episodes=10,
        tau_elig_ms=20.0,
    )
    final_rate = output_rate(final_spikes, graph.groups["output"])
    assert int(final_spikes[:, graph.groups["output"]].sum()) > 0
    assert rate_similarity(final_rate, baseline_rate) > reward_baseline


def test_rstdp_zero_episodes_unchanged_mapping():
    graph = _bypass_graph()
    params = _params()
    n_steps = 400
    source = _source_burst(n_steps, 16, onset_step=10, burst_steps=190)
    baseline_rate = np.zeros(n_steps, dtype=np.float32)

    mapping = MappingInterface(n_source=16, n_target=2, seed=0)
    init = mapping.W_map.copy()
    _, mapping = run_rstdp_training(
        graph=graph,
        parameters=params,
        mapping=mapping,
        target_node_ids=[0, 1],
        source_signal=source,
        mapping_gain=30.0,
        rstdp_lr=0.05,
        duration_ms=200.0,
        baseline_rate=baseline_rate,
        reward_baseline=0.0,
        n_episodes=0,
        tau_elig_ms=20.0,
    )
    assert np.allclose(mapping.W_map, init)


def test_timing_similarity_scale_invariant():
    signal = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    # Identical -> 1.0
    assert timing_similarity(signal, signal) == pytest.approx(1.0)
    # Scaled (same shape, different amplitude) -> 1.0
    assert timing_similarity(signal, 3.0 * signal) == pytest.approx(1.0)
    # DC-shifted (same shape) -> 1.0
    assert timing_similarity(signal, signal + 5.0) == pytest.approx(1.0)
    # Negated (anti-correlated) -> 0.0
    assert timing_similarity(signal, -signal) == pytest.approx(0.0)
    # Uncorrelated orthogonal -> ~0.5
    a = np.array([1.0, -1.0, 1.0, -1.0])
    b = np.array([1.0, 1.0, -1.0, -1.0])
    assert timing_similarity(a, b) == pytest.approx(0.5, abs=1e-6)


def test_timing_similarity_zero_norm():
    zero = np.array([0.0, 0.0, 0.0])
    nonzero = np.array([1.0, 2.0, 3.0])
    assert timing_similarity(zero, zero) == 1.0
    assert timing_similarity(zero, nonzero) == 0.0
    assert timing_similarity(nonzero, zero) == 0.0
