"""Conversion of a resolved pathway dict into a validated SparseGraph."""

from cybergraft.graph.edge_schema import Edge
from cybergraft.graph.node_schema import Node
from cybergraft.graph.sparse_graph import SparseGraph

_STAGE_TYPES = ("GNG232", "DNg67", "DNge080", "MN9")


class SubgraphBuilder:
    def __init__(self, *, max_nodes: int = 500, synaptic_delay_ms: float = 1.8) -> None:
        self.max_nodes = max_nodes
        self.synaptic_delay_ms = synaptic_delay_ms

    def build(self, pathway: dict) -> tuple[SparseGraph, dict]:
        id_map = {n["bodyId"]: i for i, n in enumerate(pathway["nodes"])}

        nodes = [
            Node(
                id=id_map[n["bodyId"]],
                species="drosophila",
                cell_type=n["type"],
                subsystem=n["subsystem"],
                input_class=n["input_class"],
                output_class=n["output_class"],
            )
            for n in pathway["nodes"]
        ]

        edges = [
            Edge(
                source=id_map[e["pre"]],
                target=id_map[e["post"]],
                weight=float(e["weight"]),
                provenance=f"male-cns:v1.0 raw_weight={e['weight']}",
                sign="excitatory",
                sign_source="model_assumption",
                delay_ms=self.synaptic_delay_ms,
            )
            for e in pathway["edges"]
        ]

        groups = {
            name: [id_map[body_id] for body_id in body_ids]
            for name, body_ids in pathway["groups"].items()
        }

        graph = SparseGraph.from_edges(nodes, edges, groups=groups, max_nodes=self.max_nodes)

        stage_counts = {stage: 0 for stage in _STAGE_TYPES}
        for n in pathway["nodes"]:
            if n["type"] in stage_counts:
                stage_counts[n["type"]] += 1

        input_body_ids = set(pathway["groups"].get("input", []))
        input_types = sorted(
            {n["type"] for n in pathway["nodes"] if n["bodyId"] in input_body_ids}
        )
        input_classes = sorted(
            {n.get("class", "") for n in pathway["nodes"] if n["bodyId"] in input_body_ids}
        )

        provenance = {
            "dataset": pathway["dataset"],
            "input_source": pathway["input_source"],
            "input_types": input_types,
            "input_classes": input_classes,
            "stage_counts": stage_counts,
            "query_hashes": pathway["query_hashes"],
        }

        return graph, provenance
