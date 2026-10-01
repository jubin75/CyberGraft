"""CSR-backed directed graph, stored as target-by-source weights."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy import sparse

from .edge_schema import Edge
from .node_schema import Node
from .validator import HARD_NODE_LIMIT, validate_graph


@dataclass
class SparseGraph:
    """Validated float32 sparse graph.

    The matrix has shape ``(target, source)`` so ``weights @ spikes`` produces
    incoming synaptic drive for each target node. Dense adjacency matrices are
    never created.
    """

    nodes: List[Node]
    edges: List[Edge]
    weights: sparse.csr_matrix
    drive_weights: sparse.csr_matrix
    groups: Dict[str, List[int]]

    @classmethod
    def from_edges(
        cls,
        nodes: Iterable[Node],
        edges: Iterable[Edge],
        groups: Optional[Dict[str, List[int]]] = None,
        max_nodes: int = HARD_NODE_LIMIT,
    ) -> "SparseGraph":
        node_list = list(nodes)
        edge_list = list(edges)
        validate_graph(node_list, edge_list, max_nodes=max_nodes)
        if edge_list:
            rows = np.fromiter((edge.target for edge in edge_list), dtype=np.int32)
            cols = np.fromiter((edge.source for edge in edge_list), dtype=np.int32)
            data = np.fromiter((edge.weight for edge in edge_list), dtype=np.float32)
            weights = sparse.coo_matrix(
                (data, (rows, cols)), shape=(len(node_list), len(node_list)), dtype=np.float32
            ).tocsr()
            weights.sum_duplicates()
            drive_data = np.fromiter(
                (
                    edge.weight if edge.sign in (None, "excitatory") else -edge.weight
                    for edge in edge_list
                ),
                dtype=np.float32,
            )
            drive_weights = sparse.coo_matrix(
                (drive_data, (rows, cols)), shape=(len(node_list), len(node_list)), dtype=np.float32
            ).tocsr()
            drive_weights.sum_duplicates()
        else:
            weights = sparse.csr_matrix((len(node_list), len(node_list)), dtype=np.float32)
            drive_weights = sparse.csr_matrix((len(node_list), len(node_list)), dtype=np.float32)
        validated_groups = {key: list(value) for key, value in (groups or {}).items()}
        for group_name, node_ids in validated_groups.items():
            if any(node_id < 0 or node_id >= len(node_list) for node_id in node_ids):
                raise ValueError(f"Group '{group_name}' contains an out-of-bounds node id")
        return cls(
            nodes=node_list,
            edges=edge_list,
            weights=weights,
            drive_weights=drive_weights,
            groups=validated_groups,
        )

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return int(self.weights.nnz)

    def incoming_drive(self, spikes: np.ndarray) -> np.ndarray:
        """Return float32 target current induced by a source spike vector."""
        if spikes.shape != (self.node_count,):
            raise ValueError(f"Expected spikes shape ({self.node_count},), got {spikes.shape}")
        return np.asarray(self.drive_weights.dot(spikes), dtype=np.float32).reshape(-1)
