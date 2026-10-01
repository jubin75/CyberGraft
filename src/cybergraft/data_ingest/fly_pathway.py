"""Assembly of the male CNS sugar-feeding pathway from neuPrint queries."""

from .malecns_queries import MaleCNSQueries
from .neuprint_client import DataUnavailableError


def _make_node(body_id, subsystem, input_class, output_class, cell_type, instance, superclass, class_="", predicted_nt=""):
    return {
        "bodyId": int(body_id),
        "type": cell_type,
        "instance": instance,
        "superclass": superclass,
        "class": class_,
        "predictedNt": predicted_nt,
        "subsystem": subsystem,
        "input_class": input_class,
        "output_class": output_class,
    }


class FlyFeedingPathway:
    def __init__(self, queries: MaleCNSQueries, config: dict) -> None:
        self.queries = queries
        self.config = config

    def resolve(self) -> dict:
        pathway_cfg = self.config["pathway"]
        stages = list(pathway_cfg.get("stages", ["GNG232", "DNg67", "DNge080", "MN9"]))
        readout = pathway_cfg.get("readout", "MN9")
        data_cfg = self.config["data"]
        max_input = int(data_cfg["max_input_nodes"])
        min_weight = int(data_cfg["min_edge_weight"])

        neurons = self.queries.resolve_neurons_by_type(stages)
        annots = {n["bodyId"]: n for n in neurons}

        gng232_ids = sorted({n["bodyId"] for n in neurons if n["type"] == "GNG232"})
        dn_ids = sorted({n["bodyId"] for n in neurons if n["type"] in ("DNg67", "DNge080")})
        mn9_ids = sorted({n["bodyId"] for n in neurons if n["type"] == readout})

        if not gng232_ids:
            raise DataUnavailableError("no GNG232 neurons resolved")

        upstream = self.queries.fetch_connections(None, gng232_ids, min_weight=min_weight)
        weight_map: dict[int, int] = {}
        for conn in upstream:
            pre = int(conn["bodyId_pre"])
            weight_map[pre] = max(weight_map.get(pre, 0), int(conn["weight"]))
        all_upstream_ids = sorted(weight_map)

        upstream_annots: dict = {}
        if all_upstream_ids:
            upstream_annots = {
                n["bodyId"]: n for n in self.queries.resolve_neurons_by_body_ids(all_upstream_ids)
            }

        gustatory_ids = sorted(
            b for b in all_upstream_ids if upstream_annots.get(b, {}).get("class") == "gustatory"
        )

        if gustatory_ids:
            input_source = "gustatory_grn"
            ranked = sorted(gustatory_ids, key=lambda b: weight_map[b], reverse=True)
            input_ids = ranked[:max_input]
        elif all_upstream_ids:
            input_source = "upstream_gng232"
            ranked = sorted(all_upstream_ids, key=lambda b: weight_map[b], reverse=True)
            input_ids = ranked[:max_input]
        else:
            input_source = "gng232_direct"
            input_ids = list(gng232_ids)

        nodes: list[dict] = []
        if input_source in ("gustatory_grn", "upstream_gng232"):
            subsystem = "taste_grn" if input_source == "gustatory_grn" else "gng232_upstream"
            for body_id in sorted(input_ids):
                annot = upstream_annots.get(body_id, {})
                nodes.append(
                    _make_node(
                        body_id,
                        subsystem,
                        "stimulus",
                        "interneuron",
                        annot.get("type", "unknown"),
                        annot.get("instance", ""),
                        annot.get("superclass", subsystem),
                        annot.get("class", ""),
                        annot.get("predictedNt", ""),
                    )
                )
        for body_id in gng232_ids:
            annot = annots.get(body_id, {})
            nodes.append(
                _make_node(
                    body_id,
                    "gng232",
                    "interneuron",
                    "interneuron",
                    annot.get("type", "GNG232"),
                    annot.get("instance", ""),
                    annot.get("superclass", ""),
                    annot.get("class", ""),
                    annot.get("predictedNt", ""),
                )
            )
        for body_id in dn_ids:
            annot = annots.get(body_id, {})
            nodes.append(
                _make_node(
                    body_id,
                    "descending_neuron",
                    "interneuron",
                    "interneuron",
                    annot.get("type", "DNg67"),
                    annot.get("instance", ""),
                    annot.get("superclass", ""),
                    annot.get("class", ""),
                    annot.get("predictedNt", ""),
                )
            )
        for body_id in mn9_ids:
            annot = annots.get(body_id, {})
            nodes.append(
                _make_node(
                    body_id,
                    "motor",
                    "interneuron",
                    "readout",
                    annot.get("type", readout),
                    annot.get("instance", ""),
                    annot.get("superclass", ""),
                    annot.get("class", ""),
                    annot.get("predictedNt", ""),
                )
            )

        edges: list[dict] = []
        seen: set[tuple[int, int]] = set()

        def add_edges(conns) -> None:
            for conn in conns:
                edge = {
                    "pre": int(conn["bodyId_pre"]),
                    "post": int(conn["bodyId_post"]),
                    "weight": int(conn["weight"]),
                }
                key = (edge["pre"], edge["post"])
                if key not in seen:
                    seen.add(key)
                    edges.append(edge)

        if input_source in ("gustatory_grn", "upstream_gng232"):
            add_edges(self.queries.fetch_connections(input_ids, gng232_ids, min_weight))
        add_edges(self.queries.fetch_connections(gng232_ids, dn_ids, min_weight))
        add_edges(self.queries.fetch_connections(dn_ids, mn9_ids, min_weight))

        groups = {
            "input": list(input_ids),
            "gng232": gng232_ids,
            "dn": dn_ids,
            "mn9": mn9_ids,
            "output": list(mn9_ids),
        }

        return {
            "dataset": self.queries.dataset,
            "input_source": input_source,
            "nodes": nodes,
            "edges": edges,
            "groups": groups,
            "query_hashes": [entry["hash"] for entry in self.queries.query_log],
        }
