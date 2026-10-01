import numpy as np

from cybergraft.graph.edge_schema import Edge
from cybergraft.graph.node_schema import Node
from cybergraft.graph.sparse_graph import SparseGraph
from cybergraft.sim.lif_reference import LIFParameters, ReferenceLIFSimulator
from cybergraft.sim.stimulus import DirectCurrentStimulus


def test_lif_direct_current_generates_spikes_and_propagates():
    nodes = [Node(id=index, species="synthetic", cell_type="test", subsystem="unit") for index in range(2)]
    graph = SparseGraph.from_edges(
        nodes,
        [Edge(source=0, target=1, weight=18.0, provenance="test")],
        groups={"input": [0], "output": [1]},
    )
    parameters = LIFParameters(
        dt_ms=0.5,
        tau_m_ms=20.0,
        tau_syn_ms=5.0,
        v_rest=-52.0,
        v_reset=-52.0,
        v_th=-45.0,
        refractory_ms=2.0,
        synaptic_delay_ms=1.0,
    )
    stimulus = DirectCurrentStimulus((0,), amplitude=24.0, start_ms=0.0, duration_ms=40.0)
    activity = ReferenceLIFSimulator(graph, parameters).run(80.0, stimulus)
    assert activity.spikes[:, 0].any()
    assert activity.spikes[:, 1].any()
    assert activity.spikes.dtype == np.bool_
