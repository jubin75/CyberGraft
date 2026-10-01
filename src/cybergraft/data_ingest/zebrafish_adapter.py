"""Adapter for the Seung zebrafish hindbrain connectome (.mat)."""

from pathlib import Path

import numpy as np
import scipy.io

from cybergraft.graph.edge_schema import Edge
from cybergraft.graph.node_schema import Node
from cybergraft.graph.sparse_graph import SparseGraph


def _extract_label(element) -> str:
    """Flatten a (possibly nested) numpy object cell to its string label."""
    value = element
    while isinstance(value, np.ndarray) and value.size:
        value = value.ravel()[0]
    return str(value) if not isinstance(value, np.ndarray) else ""


class ZebrafishAdapter:
    def __init__(self, *, mat_path: Path, max_nodes: int = 300, synaptic_delay_ms: float = 1.8) -> None:
        self.mat_path = Path(mat_path)
        self.max_nodes = max_nodes
        self.synaptic_delay_ms = synaptic_delay_ms

    def load(self) -> dict:
        if not self.mat_path.exists():
            raise FileNotFoundError(f"zebrafish .mat not found: {self.mat_path}")
        raw = scipy.io.loadmat(self.mat_path)
        connectome = raw["ZConnectome"][0, 0]
        cell_ids = [int(v) for v in np.asarray(connectome["cellID"]).ravel()]
        cell_types = [_extract_label(v) for v in connectome["cellType"]]
        weights = np.asarray(connectome["connectome"], dtype=np.int64)
        readme = str(np.asarray(connectome["readme"]).ravel()[0]) if np.asarray(connectome["readme"]).size else ""
        return {"cell_ids": cell_ids, "cell_types": cell_types, "weights": weights, "readme": readme}

    def build_subgraph(
        self, *, include_types: list[str], readout_type: str, input_type: str, lesion_type: str
    ) -> tuple[SparseGraph, dict]:
        data = self.load()
        cell_ids = data["cell_ids"]
        cell_types = data["cell_types"]
        weights = data["weights"]
        readme = data["readme"]

        id_by_type: dict[str, list[int]] = {}
        for index, label in enumerate(cell_types):
            id_by_type.setdefault(label, []).append(index)

        for required in list(include_types) + [readout_type, input_type, lesion_type]:
            if required not in id_by_type:
                raise ValueError(f"type {required!r} not present in dataset")

        type_counts = {label: len(id_by_type[label]) for label in include_types}
        largest_type = max(include_types, key=lambda label: type_counts[label])
        non_largest = [label for label in include_types if label != largest_type]
        non_largest_count = sum(type_counts[label] for label in non_largest)
        if non_largest_count > self.max_nodes:
            raise ValueError(
                f"non-largest types require {non_largest_count} nodes, exceeding max_nodes {self.max_nodes}"
            )

        readout_idx = id_by_type[readout_type]
        largest_idx = id_by_type[largest_type]
        onto_readout = weights[np.ix_(readout_idx, largest_idx)].sum(axis=0)
        candidates = [
            (largest_idx[k], int(onto_readout[k]))
            for k in range(len(largest_idx))
            if onto_readout[k] > 0
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        budget = self.max_nodes - non_largest_count
        selected_largest = [item[0] for item in candidates[:budget]]

        selected_set: set[int] = set()
        for label in non_largest:
            selected_set.update(id_by_type[label])
        selected_set.update(selected_largest)

        type_order = [input_type, lesion_type, readout_type]
        type_order += [label for label in include_types if label not in type_order]

        ordered: list[int] = []
        for label in type_order:
            members = [index for index in id_by_type[label] if index in selected_set]
            members.sort(key=lambda index: cell_ids[index])
            ordered.extend(members)

        local_of = {index: i for i, index in enumerate(ordered)}

        nodes = []
        for local_id, index in enumerate(ordered):
            label = cell_types[index]
            nodes.append(
                Node(
                    id=local_id,
                    species="zebrafish",
                    cell_type=label,
                    subsystem=label,
                    input_class="stimulus" if label == input_type else "interneuron",
                    output_class="readout" if label == readout_type else "interneuron",
                )
            )

        edges = []
        for target_index in sorted(selected_set):
            row = weights[target_index]
            for source_index in np.nonzero(row)[0]:
                source_index = int(source_index)
                if source_index not in selected_set:
                    continue
                weight = int(row[source_index])
                edges.append(
                    Edge(
                        source=local_of[source_index],
                        target=local_of[target_index],
                        weight=float(weight),
                        provenance=f"seung-zebrafish raw_weight={weight}",
                        sign="excitatory",
                        sign_source="model_assumption",
                        delay_ms=self.synaptic_delay_ms,
                    )
                )

        groups = {
            "input": [local_of[index] for index in selected_set if cell_types[index] == input_type],
            "output": [local_of[index] for index in selected_set if cell_types[index] == readout_type],
            "lesion": [local_of[index] for index in selected_set if cell_types[index] == lesion_type],
        }

        graph = SparseGraph.from_edges(nodes, edges, groups=groups, max_nodes=self.max_nodes)

        cell_type_counts = {
            label: sum(1 for index in id_by_type[label] if index in selected_set)
            for label in type_order
        }

        provenance = {
            "dataset": "seung-zebrafish-hindbrain",
            "source_url": "https://seunglab.org/zebrafish/",
            "readme": readme,
            "cell_type_counts": cell_type_counts,
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
        }

        return graph, provenance

    def build_full_subgraph(
        self,
        *,
        readout_type: str,
        lesion_type: str,
        input_type: str,
        max_nodes: int = 700,
        inhibitory_fraction: float = 0.0,
        inhibitory_gain: float = 1.0,
    ) -> tuple[SparseGraph, dict]:
        """Build the full-connectome subgraph: readout plus all its presynaptic partners.

        Unlike ``build_subgraph`` (the 609 typed subset), this uses the 2884-neuron
        matrix so redundant presynaptic pathways (``_Axl_``, ``_DOs_``, untyped) are
        retained as bypass candidates.
        """
        from collections import Counter

        full_matrix_path = self.mat_path.parent / "ConnMatrixPre_cleaned.mat"
        cells_path = self.mat_path.parent / "AllCells.mat"

        weights = np.asarray(scipy.io.loadmat(full_matrix_path)["ConnMatrixPre_cleaned"], dtype=np.int64)
        cell_ids = [int(v) for v in np.asarray(scipy.io.loadmat(cells_path)["AllCells"]).ravel()]
        id_to_index = {cid: index for index, cid in enumerate(cell_ids)}

        typed_raw = scipy.io.loadmat(self.mat_path)["ZConnectome"][0, 0]
        typed_ids = [int(v) for v in np.asarray(typed_raw["cellID"]).ravel()]
        typed_labels: dict[int, str] = {}
        for i, cid in enumerate(typed_ids):
            typed_labels[cid] = _extract_label(typed_raw["cellType"][i])

        readout_cids = sorted([cid for cid, label in typed_labels.items() if label == readout_type])
        if not readout_cids:
            raise ValueError(f"readout type {readout_type!r} not present in typed labels")
        readout_idx = [id_to_index[cid] for cid in readout_cids if cid in id_to_index]
        if not readout_idx:
            raise ValueError(f"readout cells not found in full connectome")

        total_onto_readout = weights[readout_idx, :].sum(axis=0)
        partner_idx = [
            int(j) for j in np.nonzero(total_onto_readout)[0] if int(j) not in set(readout_idx)
        ]
        partner_cids = sorted({cell_ids[j] for j in partner_idx} - set(readout_cids))

        selected_cids = readout_cids + partner_cids
        local_of_idx = {id_to_index[cid]: i for i, cid in enumerate(selected_cids)}
        selected_idx = [id_to_index[cid] for cid in selected_cids]
        selected_set = set(selected_idx)

        nodes = []
        for local_id, cid in enumerate(selected_cids):
            label = typed_labels.get(cid, "untyped")
            nodes.append(
                Node(
                    id=local_id,
                    species="zebrafish",
                    cell_type=label,
                    subsystem=label,
                    input_class="stimulus" if label == input_type else "interneuron",
                    output_class="readout" if label == readout_type else "interneuron",
                )
            )

        # Dale's principle: mark a fixed fraction of eligible (non-readout, non-input)
        # neurons as inhibitory; the sign applies to all their outgoing synapses.
        rng = np.random.default_rng(12345)
        eligible_local_ids = [
            i for i, node in enumerate(nodes)
            if node.cell_type != readout_type and node.cell_type != input_type
        ]
        n_inhibitory = int(round(inhibitory_fraction * len(eligible_local_ids)))
        if n_inhibitory > 0:
            chosen = rng.choice(eligible_local_ids, size=n_inhibitory, replace=False)
            inhibitory_local_ids = {int(idx) for idx in chosen}
        else:
            inhibitory_local_ids = set()

        edges = []
        for target_idx in selected_idx:
            row = weights[target_idx]
            for source_idx in np.nonzero(row)[0]:
                source_idx = int(source_idx)
                if source_idx not in selected_set:
                    continue
                raw_weight = int(row[source_idx])
                source_local = local_of_idx[source_idx]
                if source_local in inhibitory_local_ids:
                    sign = "inhibitory"
                    weight = float(raw_weight) * inhibitory_gain
                else:
                    sign = "excitatory"
                    weight = float(raw_weight)
                edges.append(
                    Edge(
                        source=source_local,
                        target=local_of_idx[target_idx],
                        weight=weight,
                        provenance=f"seung-zebrafish raw_weight={raw_weight}",
                        sign=sign,
                        sign_source="model_assumption",
                        delay_ms=self.synaptic_delay_ms,
                    )
                )

        groups = {
            "output": [local_of_idx[id_to_index[cid]] for cid in readout_cids],
            "lesion": [local_of_idx[id_to_index[cid]] for cid in selected_cids if typed_labels.get(cid) == lesion_type],
            "input": [local_of_idx[id_to_index[cid]] for cid in selected_cids if typed_labels.get(cid) == input_type],
        }

        graph = SparseGraph.from_edges(nodes, edges, groups=groups, max_nodes=max_nodes)

        provenance = {
            "dataset": "seung-zebrafish-hindbrain-full",
            "source_url": "https://seunglab.org/zebrafish/",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": dict(Counter(node.cell_type for node in nodes)),
            "readme": "full 2884-neuron connectome, untyped partners included",
        }

        return graph, provenance
