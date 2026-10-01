"""Structural lesions applied to a SparseGraph by silencing nodes or zeroing output weights."""

from cybergraft.graph.sparse_graph import SparseGraph


class Lesion:
    def __init__(self, *, method: str = "node_silence") -> None:
        if method not in ("node_silence", "outgoing_weight_zero"):
            raise ValueError(f"unsupported lesion method: {method}")
        self.method = method

    def apply(self, graph: SparseGraph, target_nodes: list[int]) -> tuple[SparseGraph, dict]:
        target_set = set(target_nodes)
        if self.method == "node_silence":
            lesioned_edges = [
                edge
                for edge in graph.edges
                if edge.source not in target_set and edge.target not in target_set
            ]
        else:
            lesioned_edges = [edge for edge in graph.edges if edge.source not in target_set]
        lesioned_graph = SparseGraph.from_edges(
            graph.nodes, lesioned_edges, groups=graph.groups, max_nodes=graph.node_count
        )
        record = {
            "lesion_id": f"lesion-{self.method}-{hash(tuple(sorted(target_nodes))) % 10**6}",
            "method": self.method,
            "target_nodes": sorted(target_nodes),
            "target_count": len(target_nodes),
        }
        return lesioned_graph, record
