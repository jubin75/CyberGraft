import numpy as np
import pytest

from cybergraft.graph.edge_schema import Edge
from cybergraft.graph.node_schema import Node
from cybergraft.graph.sparse_graph import SparseGraph


def _nodes(count: int):
    return [Node(id=index, species="synthetic", cell_type="test", subsystem="unit") for index in range(count)]


def test_sparse_graph_propagates_target_by_source_drive():
    graph = SparseGraph.from_edges(
        _nodes(3),
        [Edge(source=0, target=2, weight=2.5, provenance="test")],
    )
    drive = graph.incoming_drive(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(drive, np.array([0.0, 0.0, 2.5], dtype=np.float32))
    assert graph.weights.dtype == np.float32


def test_sparse_graph_rejects_out_of_bounds_edge():
    with pytest.raises(ValueError, match="out of bounds"):
        SparseGraph.from_edges(_nodes(2), [Edge(source=0, target=2, weight=1.0, provenance="test")])
