"""Adapter: C. elegans (White 1986) reversal circuit -> SparseGraph.

Builds the nose-touch reversal (escape) circuit, the closest C. elegans
analogue of an input -> integrator -> readout motif:

  * input    = ASH (ASHL/ASHR) nose-touch nociceptors, plus ALM/AVM/PLM touch
  * lesion   = AVA + AVD + AVE  (gap-junction-coupled persistent-state command
    cluster; laser ablation abolishes backward locomotion, Chalfie 1985)
  * readout  = A-class motor (VA/DA, excitatory) + D-class (DD/VD, GABAergic)

The White 1986 edge list carries both chemical and electrical (gap-junction)
synapses. Gap junctions are non-inverting, so electrical edges are assigned
``sign="excitatory"`` (an approximation: the LIF sim has no dedicated
gap-junction channel, and real gap junctions are bidirectional).
"""

from collections import Counter
from pathlib import Path
from typing import Optional

from cybergraft.graph.edge_schema import Edge
from cybergraft.graph.node_schema import Node
from cybergraft.graph.sparse_graph import SparseGraph

EDGELIST_URL = (
    "https://raw.githubusercontent.com/openworm/CElegansNeuroML/master/"
    "herm_full_edgelist.csv"
)

INPUT_NAMES = {"ASHL", "ASHR", "ALML", "ALMR", "AVM", "PLML", "PLMR"}
LESION_NAMES = {"AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"}
READOUT_PREFIXES = ("VA", "DA", "DD", "VD", "VB", "DB")

# GABAergic inhibitory neurons (consensus: Gendrel 2016 / McIntire 1993).
_INHIBITORY_PREFIXES = ("DD", "VD")
_INHIBITORY_EXACT = {"AVL", "RIS", "DVB", "RME", "RMED", "RMEV", "RMEL", "RMER"}


def is_inhibitory(name: str) -> bool:
    """Dale's-principle sign for a C. elegans neuron name."""
    if name.startswith(_INHIBITORY_PREFIXES):
        return True
    return name in _INHIBITORY_EXACT


def _is_readout(name: str) -> bool:
    return any(name.startswith(p) for p in READOUT_PREFIXES)


def _load_edges(path: Path) -> list[tuple[str, str, float, str]]:
    edges = []
    with path.open("r") as fh:
        fh.readline()  # header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            edges.append((parts[0], parts[1], float(parts[2]), parts[3]))
    return edges


class CelegansAdapter:
    def __init__(self, *, edgelist_path: Optional[Path] = None,
                 synaptic_delay_ms: float = 1.8) -> None:
        self.edgelist_path = edgelist_path
        self.synaptic_delay_ms = synaptic_delay_ms

    def _edges(self) -> list[tuple[str, str, float, str]]:
        if self.edgelist_path is None:
            raise ValueError("edgelist_path is required")
        return _load_edges(self.edgelist_path)

    def build_reversal_subgraph(self, *, inhibitory_gain: float = 1.0) -> tuple[SparseGraph, dict]:
        edges = self._edges()

        seed = set(INPUT_NAMES) | set(LESION_NAMES)
        seed |= {s for s, _, _, _ in edges if _is_readout(s)}
        seed |= {t for _, t, _, _ in edges if _is_readout(t)}

        # Include seed neurons + their immediate partners (captures local wiring).
        adjacency: dict[str, set[str]] = {}
        for s, t, w, typ in edges:
            adjacency.setdefault(s, set()).add(t)
            adjacency.setdefault(t, set()).add(s)

        include: set[str] = set(seed)
        for name in list(seed):
            include |= adjacency.get(name, set())

        names = sorted(include)
        local_of = {n: i for i, n in enumerate(names)}

        nodes = []
        for n in names:
            if n in INPUT_NAMES:
                ic, oc = "stimulus", "interneuron"
            elif _is_readout(n):
                ic, oc = "interneuron", "readout"
            else:
                ic, oc = "interneuron", "interneuron"
            nodes.append(
                Node(
                    id=local_of[n],
                    species="celegans",
                    cell_type=n,
                    subsystem="reversal" if (n in LESION_NAMES or _is_readout(n)) else "sensory",
                    input_class=ic,
                    output_class=oc,
                )
            )

        graph_edges = []
        for s, t, w, typ in edges:
            if s not in local_of or t not in local_of:
                continue
            sign = "inhibitory" if is_inhibitory(s) else "excitatory"
            weight = w * inhibitory_gain if sign == "inhibitory" else w
            graph_edges.append(
                Edge(
                    source=local_of[s],
                    target=local_of[t],
                    weight=weight,
                    provenance=f"white1986 type={typ}",
                    sign=sign,
                    sign_source="model_assumption" if sign == "excitatory" else "gabaergic_consensus",
                    delay_ms=self.synaptic_delay_ms,
                )
            )

        groups = {
            "input": [local_of[n] for n in names if n in INPUT_NAMES],
            "lesion": [local_of[n] for n in names if n in LESION_NAMES],
            "output": [local_of[n] for n in names if _is_readout(n)],
        }

        graph = SparseGraph.from_edges(nodes, graph_edges, groups=groups, max_nodes=len(names))

        provenance = {
            "dataset": "white1986-celegans",
            "source_url": EDGELIST_URL,
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "cell_type_counts": dict(Counter(n.cell_type for n in nodes)),
            "group_sizes": {k: len(v) for k, v in groups.items()},
            "note": "gap junctions treated as excitatory chemical synapses",
        }
        return graph, provenance
