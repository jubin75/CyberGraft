"""Structural attractor-supporting features on the connectome graph (PRD §7).

V3's *structural* layer scores, for each neuron, how much the static wiring can
support persistent (attractor-like) activity, without ever simulating the
dynamics. Four quantities are computed:

  1. **Strong connectivity** — the strongly-connected components (SCCs) of the
     directed connectome; any mutually-reachable cycle lives inside one SCC.
  2. **Recurrence** — the fraction of a node's outgoing weight that stays inside
     its own SCC (a direct measure of recurrent feedback).
  3. **Bounded cycle participation** — ``diag(B^L)`` for ``L in {2,3,4}``, i.e.
     the number of length-``L`` closed walks through each node, computed with
     sparse matrix powers (no path enumeration).
  4. **Recurrent spectral radius** — the spectral radius of the recurrent
     backbone ``A_R`` (edges whose source *and* target share an SCC).

All four are *structural* properties of the wiring graph. In particular the
recurrent spectral radius is a structural proxy, **not** an attractor-existence
proof (see the note in :func:`recurrent_spectral_radius`).

Consistent with the rest of V3, ``null != 0``: where a score is undefined (e.g.
recurrence for a node with no outgoing weight) it is ``NaN`` and flagged
``missing``, never silently zero-filled.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as splinalg

from cybergraft.graph.sparse_graph import SparseGraph


def _source_target_binary(graph: SparseGraph) -> sparse.csr_matrix:
    """Return the binary source->target adjacency ``(A != 0)`` as a CSR matrix."""
    return (graph.weights.T != 0).tocsr()


def _source_target_weighted(graph: SparseGraph) -> sparse.csr_matrix:
    """Return the weighted source->target adjacency ``A[s, t] = w(s -> t)``."""
    return graph.weights.T.tocsr()


def strongly_connected_components(graph: SparseGraph) -> dict:
    """Compute the strongly-connected components of the directed connectome.

    Uses ``scipy.sparse.csgraph.connected_components`` on the binary
    source->target adjacency with ``connection='strong'``.

    Returns
    -------
    dict
        ``scc_id``            ``(N,) int``   component label per node
        ``node_to_scc``       ``(N,) int``   == ``scc_id``
        ``component_size``    ``(N,) int``   size of the node's component
        ``is_large_scc``      ``(N,) bool``  component size > 1
        ``n_components``      ``int``        number of distinct components
        ``max_component_size`` ``int``       largest component size
    """
    binary = _source_target_binary(graph)
    n_components, labels = csgraph.connected_components(
        binary, directed=True, connection="strong", return_labels=True
    )
    labels = np.asarray(labels, dtype=np.int64)
    sizes = np.bincount(labels)
    component_size = sizes[labels]
    return {
        "scc_id": labels,
        "node_to_scc": labels,
        "component_size": component_size.astype(np.int64),
        "is_large_scc": component_size > 1,
        "n_components": int(n_components),
        "max_component_size": int(sizes.max()) if sizes.size else 0,
    }


def recurrence_scores(graph: SparseGraph) -> dict:
    """Per-node recurrent out-weight fraction (PRD §7.2).

    Recurrent out-weight is the total weight from node ``i`` to nodes in the
    *same* strongly-connected component as ``i``; total out-weight is the sum of
    all outgoing weights from ``i``. ``recurrence_i = recurrent / total``.

    Nodes with ``total_out_weight == 0`` are flagged ``missing`` and get
    ``recurrence = NaN`` (never zero-filled, ``null != 0``).

    Returns
    -------
    dict
        ``recurrence``        ``(N,) float`` recurrence fraction (``NaN`` where missing)
        ``missing``           ``(N,) bool``  True where ``total_out_weight == 0``
        ``total_out_weight``  ``(N,) float`` total outgoing weight per node
    """
    node_count = graph.node_count
    labels = strongly_connected_components(graph)["scc_id"]

    a = _source_target_weighted(graph)
    coo = a.tocoo()
    src = np.asarray(coo.row, dtype=np.int64)
    tgt = np.asarray(coo.col, dtype=np.int64)
    data = np.asarray(coo.data, dtype=np.float64)

    same_scc = labels[src] == labels[tgt]
    recurrent_out = np.bincount(
        src[same_scc], weights=data[same_scc], minlength=node_count
    ).astype(np.float64)

    total_out = np.asarray(a.sum(axis=1), dtype=np.float64).ravel()

    missing = total_out == 0.0
    recurrence = np.full(node_count, np.nan, dtype=np.float64)
    np.divide(recurrent_out, total_out, out=recurrence, where=~missing)

    return {
        "recurrence": recurrence,
        "missing": missing,
        "total_out_weight": total_out,
    }


def bounded_cycle_participation(graph: SparseGraph, max_cycle_length: int = 4) -> dict:
    """Bounded feedback-path participation via sparse matrix powers (PRD §7.3).

    Let ``B`` be the binary source->target adjacency. For ``L`` in ``{2, 3, 4}``
    compute ``diag(B^L)``: ``cycleL_i`` is the number of length-``L`` closed
    walks through node ``i`` (for ``L=2`` this is the number of 2-cycles through
    ``i``). Uses repeated sparse ``@`` — no path enumeration, no dense ``B^L``.

    ``max_cycle_length`` is accepted for API compatibility with the PRD and is
    currently capped at 4 (the returned fields only cover ``cycle2``..``cycle4``).

    Returns
    -------
    dict
        ``cycle2`` ``(N,) float`` number of length-2 closed walks per node
        ``cycle3`` ``(N,) float`` number of length-3 closed walks per node
        ``cycle4`` ``(N,) float`` number of length-4 closed walks per node
    """
    b = _source_target_binary(graph).astype(np.float64)
    b2 = b @ b
    b3 = b2 @ b
    b4 = b3 @ b
    return {
        "cycle2": np.asarray(b2.diagonal(), dtype=np.float64),
        "cycle3": np.asarray(b3.diagonal(), dtype=np.float64),
        "cycle4": np.asarray(b4.diagonal(), dtype=np.float64),
    }


def recurrent_spectral_radius(graph: SparseGraph) -> dict:
    """Spectral radius of the recurrent backbone ``A_R`` (PRD §7.4).

    ``A_R`` is the subgraph of edges whose source *and* target belong to the
    same strongly-connected component. The spectral radius ``rho_R`` is
    ``max |eigenvalue(A_R)|``, estimated with ``scipy.sparse.linalg.eigs``
    (``k=1, which='LM'``).

    Note: ``rho_R`` is a **structural** property of the wiring only. It is a
    heuristic for recurrent gain, **not** a proof that an attractor exists in
    the (nonlinear, noisy) spiking dynamics.

    Returns
    -------
    dict
        ``rho_R``             ``float`` spectral radius (``0.0`` if empty / non-convergent)
        ``n_recurrent_edges`` ``int``   number of same-SCC edges
        ``n_recurrent_nodes`` ``int``   number of nodes with ``component_size > 1``
    """
    scc = strongly_connected_components(graph)
    labels = scc["scc_id"]
    rec_nodes = np.flatnonzero(scc["is_large_scc"])
    n_recurrent_nodes = int(rec_nodes.size)

    if n_recurrent_nodes == 0:
        return {
            "rho_R": 0.0,
            "n_recurrent_edges": 0,
            "n_recurrent_nodes": 0,
        }

    a = _source_target_weighted(graph)
    a_r = a[np.ix_(rec_nodes, rec_nodes)].astype(np.float64)
    n_recurrent_edges = int(a_r.nnz)

    if n_recurrent_edges == 0 or a_r.shape[0] <= 1:
        rho_r = 0.0
    else:
        try:
            eigenvalues = splinalg.eigs(
                a_r, k=1, which="LM", return_eigenvectors=False
            )
            rho_r = float(np.abs(eigenvalues).max())
        except Exception:
            rho_r = 0.0

    return {
        "rho_R": rho_r,
        "n_recurrent_edges": n_recurrent_edges,
        "n_recurrent_nodes": n_recurrent_nodes,
    }


def structural_scores(graph: SparseGraph) -> dict:
    """Convenience aggregator of all structural attractor-supporting scores."""
    return {
        "scc": strongly_connected_components(graph),
        "recurrence": recurrence_scores(graph),
        "cycles": bounded_cycle_participation(graph),
        "spectral_radius": recurrent_spectral_radius(graph),
        "n_nodes": graph.node_count,
    }


if __name__ == "__main__":
    from cybergraft.graph.edge_schema import Edge
    from cybergraft.graph.node_schema import Node

    # Small synthetic graph: a 0->1->2->0 triangle, a 5<->6 two-cycle, and a
    # feedforward chain 0->3->4->5 tying them together.
    nodes = [Node(i, species="test", cell_type="untyped", subsystem="sub") for i in range(7)]
    edges = [
        Edge(source=0, target=1, weight=1.0, provenance="smoke"),
        Edge(source=1, target=2, weight=1.0, provenance="smoke"),
        Edge(source=2, target=0, weight=1.0, provenance="smoke"),
        Edge(source=0, target=3, weight=0.5, provenance="smoke"),
        Edge(source=3, target=4, weight=0.5, provenance="smoke"),
        Edge(source=4, target=5, weight=0.5, provenance="smoke"),
        Edge(source=5, target=6, weight=1.0, provenance="smoke"),
        Edge(source=6, target=5, weight=1.0, provenance="smoke"),
    ]
    graph = SparseGraph.from_edges(nodes, edges)

    scc = strongly_connected_components(graph)
    print("scc_id           =", scc["scc_id"].tolist())
    print("component_size   =", scc["component_size"].tolist())
    print("n_components     =", scc["n_components"])
    print("max_component_size =", scc["max_component_size"])

    rec = recurrence_scores(graph)
    print("recurrence       =", np.round(rec["recurrence"], 4).tolist())
    print("missing          =", rec["missing"].tolist())

    cyc = bounded_cycle_participation(graph)
    print("cycle2           =", cyc["cycle2"].tolist())
    print("cycle3           =", cyc["cycle3"].tolist())
    print("cycle4           =", cyc["cycle4"].tolist())

    rad = recurrent_spectral_radius(graph)
    print("rho_R            =", rad["rho_R"])
    print("n_recurrent_edges =", rad["n_recurrent_edges"])
    print("n_recurrent_nodes =", rad["n_recurrent_nodes"])

    scores = structural_scores(graph)
    print("n_nodes          =", scores["n_nodes"])
