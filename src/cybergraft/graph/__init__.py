"""Small, validated sparse graph primitives used by the reference backend."""

from .edge_schema import Edge
from .node_schema import Node
from .sparse_graph import SparseGraph

__all__ = ["Edge", "Node", "SparseGraph"]
