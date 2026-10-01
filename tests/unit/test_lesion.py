import pytest

from cybergraft.experiment.lesion import Lesion

from fakes import synthetic_zebrafish_graph


def test_node_silence_isolates_target():
    graph = synthetic_zebrafish_graph()
    original_edge_count = graph.edge_count
    lesioned, record = Lesion(method="node_silence").apply(graph, [3])
    assert lesioned.node_count == graph.node_count
    assert lesioned.edge_count < original_edge_count
    assert record["method"] == "node_silence"
    assert record["target_nodes"] == [3]
    for edge in lesioned.edges:
        assert edge.source != 3 and edge.target != 3


def test_outgoing_weight_zero_keeps_incoming():
    graph = synthetic_zebrafish_graph()
    lesioned, _ = Lesion(method="outgoing_weight_zero").apply(graph, [3])
    assert lesioned.node_count == graph.node_count
    for edge in lesioned.edges:
        assert edge.source != 3
    # incoming edges to node 3 survive
    assert any(edge.target == 3 for edge in lesioned.edges)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unsupported lesion method"):
        Lesion(method="bad_method")
