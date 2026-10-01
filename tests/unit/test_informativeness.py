import numpy as np

from cybergraft.experiment.bypass_experiment import generate_informative_source
from cybergraft.experiment.bypass_training import output_rate, rate_similarity, run_rstdp_training
from cybergraft.experiment.mapping_interface import MappingInterface
from cybergraft.graph.edge_schema import Edge
from cybergraft.graph.node_schema import Node
from cybergraft.graph.sparse_graph import SparseGraph
from cybergraft.sim.lif_reference import LIFParameters


def test_generate_informative_source_correlation():
    n_steps = 100
    n_source = 10
    command = np.zeros(n_steps)
    command[30:70] = 1.0
    dt_ms = 0.5
    onset_ms = 0.0
    duration_ms = 50.0

    src0 = generate_informative_source(n_steps, n_source, 0.0, command, dt_ms, onset_ms, duration_ms, seed=0)
    src1 = generate_informative_source(n_steps, n_source, 1.0, command, dt_ms, onset_ms, duration_ms, seed=0)

    mean0 = src0.mean(axis=1)
    mean1 = src1.mean(axis=1)
    corr0 = np.corrcoef(mean0, command)[0, 1]
    corr1 = np.corrcoef(mean1, command)[0, 1]
    if np.isnan(corr1):
        corr1 = 0.0
    assert corr0 > corr1
    assert corr0 > 0.9


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


def test_lam0_source_yields_higher_gain_than_lam1():
    graph = _bypass_graph()
    params = _params()
    n_steps = 400
    dt_ms = params.dt_ms
    # Structured target: two separated output bursts.
    baseline_rate = np.zeros(n_steps, dtype=np.float32)
    baseline_rate[50:100] = 4.0
    baseline_rate[250:300] = 4.0
    command_rate = (baseline_rate / 4.0).astype(np.float64)
    reward_baseline = 0.0
    onset_ms = 0.0
    duration_ms = 200.0

    def run(lam):
        source = generate_informative_source(
            n_steps, 16, lam, command_rate, dt_ms, onset_ms, duration_ms, seed=0
        )
        mapping = MappingInterface(n_source=16, n_target=2, seed=0)
        res, _ = run_rstdp_training(
            graph=graph, parameters=params, mapping=mapping, target_node_ids=[0, 1],
            source_signal=source, mapping_gain=30.0, rstdp_lr=0.05, duration_ms=200.0,
            baseline_rate=baseline_rate, reward_baseline=reward_baseline,
            n_episodes=10, tau_elig_ms=20.0,
        )
        return rate_similarity(output_rate(res, graph.groups["output"]), baseline_rate) - reward_baseline

    gain0 = run(0.0)
    gain1 = run(1.0)
    assert gain0 > gain1
