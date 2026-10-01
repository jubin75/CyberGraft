"""Null models for connectome structure: randomized graphs for hypothesis testing.

Each generator returns a NEW ``SparseGraph`` identical to the input except for
one randomized aspect, and is deterministic given ``seed`` (``numpy.random.default_rng``).
"""

from dataclasses import replace

import numpy as np

from cybergraft.graph.node_schema import Node
from cybergraft.graph.sparse_graph import SparseGraph


def weight_shuffle(graph: SparseGraph, seed: int) -> SparseGraph:
    """Permute nonzero edge weights among existing edges (topology + degree preserved).

    Edge endpoints (source/target pairs) and their metadata are unchanged; only the
    scalar weights are reassigned. Assumes one edge per (source, target) pair (true
    for the zebrafish connectome), so ``len(graph.edges) == graph.weights.nnz``.
    """
    rng = np.random.default_rng(seed)
    weights = np.array([edge.weight for edge in graph.edges], dtype=np.float64)
    rng.shuffle(weights)
    new_edges = [replace(edge, weight=float(weights[i])) for i, edge in enumerate(graph.edges)]
    return SparseGraph.from_edges(graph.nodes, new_edges, groups=graph.groups, max_nodes=graph.node_count)


def shuffle_cell_types(graph: SparseGraph, seed: int) -> SparseGraph:
    """Permute cell-type labels across nodes (topology + weights preserved).

    The multiset of type labels is preserved but their assignment to nodes is
    randomized, destroying type identity. ``subsystem`` is kept equal to the
    shuffled ``cell_type`` (they are identical in the zebrafish adapter).
    """
    rng = np.random.default_rng(seed)
    labels = [node.cell_type for node in graph.nodes]
    rng.shuffle(labels)
    new_nodes = [
        Node(
            id=node.id,
            species=node.species,
            cell_type=labels[i],
            subsystem=labels[i],
            input_class=node.input_class,
            output_class=node.output_class,
            position=node.position,
        )
        for i, node in enumerate(graph.nodes)
    ]
    return SparseGraph.from_edges(new_nodes, graph.edges, groups=graph.groups, max_nodes=graph.node_count)


def degree_preserving_rewire(graph: SparseGraph, seed: int, n_swaps: int | None = None) -> SparseGraph:
    """Directed degree-preserving rewire (double-edge swap / Maslov--Sneppen).

    Preserves every node's in-degree and out-degree while randomizing which
    specific source/target pairs are connected. Edge weights stay attached to
    their source, so each node's out-weight is also preserved. Self-loops and
    duplicate edges are rejected.
    """
    rng = np.random.default_rng(seed)
    edges = list(graph.edges)
    n = len(edges)
    if n_swaps is None:
        n_swaps = max(4 * n, 1000)
    pairs = {(e.source, e.target) for e in edges}
    done = 0
    attempts = 0
    max_attempts = 100 * n_swaps
    while done < n_swaps and attempts < max_attempts:
        attempts += 1
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        a, b = edges[i].source, edges[i].target
        c, d = edges[j].source, edges[j].target
        if a == d or c == b:
            continue
        if (a, d) in pairs or (c, b) in pairs:
            continue
        pairs.discard((a, b))
        pairs.discard((c, d))
        pairs.add((a, d))
        pairs.add((c, b))
        edges[i] = replace(edges[i], target=d)
        edges[j] = replace(edges[j], target=b)
        done += 1
    return SparseGraph.from_edges(graph.nodes, edges, groups=graph.groups, max_nodes=graph.node_count)
