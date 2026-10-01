"""Adapter: Drosophila hemibrain EPG head-direction compass -> SparseGraph.

Builds the ring-attractor compass circuit from ``hemibrain:v1.2.1``:

  * lesion  = EPG + PEG   (the angular-velocity -> heading ring integrator)
  * output  = PFL3        (heading/steering readout, projects to DNa02)
  * input   = PEN_a(PEN1) + PEN_b(PEN2)  (angular-velocity input)

Excitatory-only by default (hemibrain neuPrint exposes no reliable predictedNt;
E/I identity would come from Turner-Evans 2020 / Eckstein 2024). A Dale's-principle
E/I assignment (deterministic rng seed 12345) can be enabled via the
``inhibitory_fraction`` / ``inhibitory_gain`` kwargs, mirroring the zebrafish adapter.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

from cybergraft.data_ingest.neuprint_client import NeuPrintClient
from cybergraft.graph.edge_schema import Edge
from cybergraft.graph.node_schema import Node
from cybergraft.graph.sparse_graph import SparseGraph


def _coerce_type(value) -> str:
    """Coerce a neuPrint ``type`` field to a string; NaN/None/blank -> 'untyped'."""
    if isinstance(value, str) and value.strip():
        return value
    return "untyped"


class HemibrainCompassAdapter:
    """Resolve compass neurons + connectivity from hemibrain and assemble a SparseGraph."""

    def __init__(self, client: NeuPrintClient, synaptic_delay_ms: float = 1.8) -> None:
        self.client = client
        self.synaptic_delay_ms = synaptic_delay_ms

    def build(
        self,
        *,
        include_types: list[str],
        lesion_types: list[str],
        output_type: str,
        input_types: list[str],
        min_weight: int = 1,
        inhibitory_fraction: float = 0.0,
        inhibitory_gain: float = 1.0,
    ) -> tuple[SparseGraph, dict]:
        if not self.client.available:
            raise RuntimeError(f"neuPrint unavailable: {self.client.status().get('reason')}")

        rows = self.client.query_neurons({"type": include_types, "status": "Traced"})
        id_to_type: dict[int, str] = {int(r["bodyId"]): r["type"] for r in rows}
        if not id_to_type:
            raise RuntimeError(f"no neurons resolved for types {include_types}")
        all_ids = sorted(id_to_type)

        conns = self.client.query_connections(all_ids, all_ids, min_weight=min_weight)

        local_of: dict[int, int] = {body: i for i, body in enumerate(all_ids)}

        nodes: list[Node] = []
        for body in all_ids:
            t = id_to_type[body]
            nodes.append(
                Node(
                    id=local_of[body],
                    species="drosophila",
                    cell_type=t,
                    subsystem=t,
                    input_class="stimulus" if t in input_types else "interneuron",
                    output_class="readout" if t == output_type else "interneuron",
                )
            )

        # Dale's principle: mark a fixed fraction of eligible (non-readout,
        # non-input) neurons as inhibitory; the sign applies to all their
        # outgoing synapses (mirrors zebrafish build_full_subgraph).
        rng = np.random.default_rng(12345)
        eligible_local_ids = [
            n.id for n in nodes if n.cell_type != output_type and n.cell_type not in input_types
        ]
        n_inhibitory = int(round(inhibitory_fraction * len(eligible_local_ids)))
        if n_inhibitory > 0:
            chosen = rng.choice(eligible_local_ids, size=n_inhibitory, replace=False)
            inhibitory_local_ids = {int(idx) for idx in chosen}
        else:
            inhibitory_local_ids = set()

        edges: list[Edge] = []
        for conn in conns:
            pre = int(conn["bodyId_pre"])
            post = int(conn["bodyId_post"])
            if pre not in local_of or post not in local_of:
                continue
            raw_weight = int(conn["weight"])
            src_local = local_of[pre]
            if src_local in inhibitory_local_ids:
                sign = "inhibitory"
                weight = float(raw_weight) * inhibitory_gain
            else:
                sign = "excitatory"
                weight = float(raw_weight)
            edges.append(
                Edge(
                    source=src_local,
                    target=local_of[post],
                    weight=weight,
                    provenance=f"hemibrain raw_weight={raw_weight}",
                    sign=sign,
                    sign_source="model_assumption",
                    delay_ms=self.synaptic_delay_ms,
                )
            )

        groups = {
            "input": [local_of[b] for b in all_ids if id_to_type[b] in input_types],
            "output": [local_of[b] for b in all_ids if id_to_type[b] == output_type],
            "lesion": [local_of[b] for b in all_ids if id_to_type[b] in lesion_types],
        }

        graph = SparseGraph.from_edges(nodes, edges, groups=groups, max_nodes=len(all_ids))

        provenance = {
            "dataset": "hemibrain:v1.2.1",
            "source_url": "https://neuprint.janelia.org",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": dict(Counter(n.cell_type for n in nodes)),
            "group_sizes": {k: len(v) for k, v in groups.items()},
        }
        return graph, provenance


    def build_full_subgraph(
        self,
        *,
        output_type: str,
        lesion_types: list[str],
        input_types: list[str],
        min_weight: int = 1,
        inhibitory_fraction: float = 0.0,
        inhibitory_gain: float = 1.0,
    ) -> tuple[SparseGraph, dict]:
        """Build the readout + all its presynaptic partners (mirrors zebrafish build_full_subgraph).

        Includes the readout, every neuron that synapses onto it (so feedforward
        direct-projecting partners such as the goal-direction PFN neurons are
        retained as graft sites), plus the lesion and input populations explicitly
        (PEG and PEN are not necessarily direct presynaptic partners of PFL3).
        """
        if not self.client.available:
            raise RuntimeError(f"neuPrint unavailable: {self.client.status().get('reason')}")

        seed_types = [output_type] + list(lesion_types) + list(input_types)
        seed_rows = self.client.query_neurons({"type": seed_types, "status": "Traced"})
        id_to_type: dict[int, str] = {int(r["bodyId"]): _coerce_type(r["type"]) for r in seed_rows}
        out_ids = [b for b, t in id_to_type.items() if t == output_type]
        if not out_ids:
            raise RuntimeError(f"no {output_type} resolved")

        # All presynaptic partners of the readout.
        pre_conns = self.client.query_connections(None, out_ids, min_weight=min_weight)
        for conn in pre_conns:
            b = int(conn["bodyId_pre"])
            if b not in id_to_type:
                id_to_type[b] = "untyped"

        # Resolve types of the newly-added partners (chunked; stays "untyped" if unresolved).
        unresolved = [b for b, t in id_to_type.items() if t == "untyped"]
        if unresolved:
            for chunk in NeuPrintClient.chunked(unresolved, 200):
                try:
                    rows = self.client.query_neurons({"bodyId": list(chunk), "status": "Traced"})
                except Exception:
                    continue
                for r in rows:
                    id_to_type[int(r["bodyId"])] = _coerce_type(r["type"])

        all_ids = sorted(id_to_type)
        local_of: dict[int, int] = {body: i for i, body in enumerate(all_ids)}

        conns = self.client.query_connections(all_ids, all_ids, min_weight=min_weight)

        nodes: list[Node] = []
        for body in all_ids:
            t = id_to_type[body]
            nodes.append(
                Node(
                    id=local_of[body],
                    species="drosophila",
                    cell_type=t,
                    subsystem=t,
                    input_class="stimulus" if t in input_types else "interneuron",
                    output_class="readout" if t == output_type else "interneuron",
                )
            )

        # Dale's principle: mark a fixed fraction of eligible (non-readout,
        # non-input) neurons as inhibitory; the sign applies to all their
        # outgoing synapses (mirrors zebrafish build_full_subgraph).
        rng = np.random.default_rng(12345)
        eligible_local_ids = [
            n.id for n in nodes if n.cell_type != output_type and n.cell_type not in input_types
        ]
        n_inhibitory = int(round(inhibitory_fraction * len(eligible_local_ids)))
        if n_inhibitory > 0:
            chosen = rng.choice(eligible_local_ids, size=n_inhibitory, replace=False)
            inhibitory_local_ids = {int(idx) for idx in chosen}
        else:
            inhibitory_local_ids = set()

        edges: list[Edge] = []
        for conn in conns:
            pre = int(conn["bodyId_pre"])
            post = int(conn["bodyId_post"])
            if pre not in local_of or post not in local_of:
                continue
            raw_weight = int(conn["weight"])
            src_local = local_of[pre]
            if src_local in inhibitory_local_ids:
                sign = "inhibitory"
                weight = float(raw_weight) * inhibitory_gain
            else:
                sign = "excitatory"
                weight = float(raw_weight)
            edges.append(
                Edge(
                    source=src_local,
                    target=local_of[post],
                    weight=weight,
                    provenance=f"hemibrain raw_weight={raw_weight}",
                    sign=sign,
                    sign_source="model_assumption",
                    delay_ms=self.synaptic_delay_ms,
                )
            )

        groups = {
            "input": [local_of[b] for b in all_ids if id_to_type[b] in input_types],
            "output": [local_of[b] for b in all_ids if id_to_type[b] == output_type],
            "lesion": [local_of[b] for b in all_ids if id_to_type[b] in lesion_types],
        }

        graph = SparseGraph.from_edges(nodes, edges, groups=groups, max_nodes=len(all_ids))

        provenance = {
            "dataset": "hemibrain:v1.2.1",
            "source_url": "https://neuprint.janelia.org",
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": dict(Counter(n.cell_type for n in nodes)),
            "group_sizes": {k: len(v) for k, v in groups.items()},
        }
        return graph, provenance


def build_compass_graph(config: dict, project_root: Path, full: bool = True) -> tuple[SparseGraph, dict]:
    """Convenience: construct the client from config and build the compass graph.

    ``full=True`` builds the readout + presynaptic-partner subgraph (mirrors the
    zebrafish build_full_subgraph); ``full=False`` builds only the typed compass
    types.
    """
    data = config["data"]
    ei = config.get("ei", {})
    inhibitory_fraction = float(ei.get("inhibitory_fraction", 0.0))
    inhibitory_gain = float(ei.get("inhibitory_gain", 1.0))
    client = NeuPrintClient.from_config(config, project_root)
    adapter = HemibrainCompassAdapter(
        client,
        synaptic_delay_ms=float(config["simulation"].get("synaptic_delay_ms", 1.8)),
    )
    if full:
        return adapter.build_full_subgraph(
            output_type=data["output_type"],
            lesion_types=list(data["lesion_types"]),
            input_types=list(data["input_types"]),
            min_weight=int(data.get("min_edge_weight", 1)),
            inhibitory_fraction=inhibitory_fraction,
            inhibitory_gain=inhibitory_gain,
        )
    return adapter.build(
        include_types=list(data["include_types"]),
        lesion_types=list(data["lesion_types"]),
        output_type=data["output_type"],
        input_types=list(data["input_types"]),
        min_weight=int(data.get("min_edge_weight", 1)),
        inhibitory_fraction=inhibitory_fraction,
        inhibitory_gain=inhibitory_gain,
    )
