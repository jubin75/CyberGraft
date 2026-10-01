"""Graph constraints shared by construction, loading, and simulation."""

from typing import Iterable

from .edge_schema import Edge
from .node_schema import Node


HARD_NODE_LIMIT = 5400


def validate_graph(nodes: Iterable[Node], edges: Iterable[Edge], max_nodes: int = HARD_NODE_LIMIT) -> None:
    """Validate a bounded graph without constructing a dense adjacency matrix."""

    node_list = list(nodes)
    edge_list = list(edges)
    if max_nodes > HARD_NODE_LIMIT:
        raise ValueError(f"max_nodes cannot exceed hard limit {HARD_NODE_LIMIT}")
    if not node_list:
        raise ValueError("Graph must contain at least one node")
    if len(node_list) > max_nodes:
        raise ValueError(f"node_count {len(node_list)} exceeds configured limit {max_nodes}")
    ids = [node.id for node in node_list]
    if ids != list(range(len(node_list))):
        raise ValueError("Node ids must be contiguous and ordered from 0")
    node_count = len(node_list)
    for edge in edge_list:
        if edge.source >= node_count or edge.target >= node_count:
            raise ValueError(f"Edge endpoint out of bounds: {edge}")
