"""Attractor-support score ``A_i`` (PRD §12).

The universal node score combines the two node-level perturbation components
(convergence contribution ``A_state`` and persistent-state support ``A_persist``)
with *equal weighting* — no hand-tuned weights (PRD §12.2). ``A_manifold`` is a
population-specific validation component that is stored but NOT folded into the
universal score (PRD §12.1), so feedforward relay sites with no well-defined
manifold loading are never discarded.

Missingness is preserved: ``None`` stays ``None`` and is never silently
zero-filled.
"""

from typing import Optional

import numpy as np


def zscore(values: np.ndarray) -> np.ndarray:
    """Standardize ``(x - mean) / std`` with NaN-aware statistics.

    Returns ``NaN`` wherever the input is ``NaN``. If all values are missing the
    result is all-NaN; if the finite standard deviation is zero, finite inputs
    map to ``0.0``.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = ~np.isnan(arr)
    if not finite.any():
        return out
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    if std > 0.0:
        out[finite] = (arr[finite] - mean) / std
    else:
        out[finite] = 0.0
    return out


def attractor_support_score(
    A_state: dict,
    A_persist: dict,
    A_manifold: Optional[dict] = None,
) -> dict:
    """Compute the universal equal-weight attractor-support score ``A_i``.

    Each component is z-scored *across all nodes* (vector-level standardization)
    before averaging. ``A_i = mean(z(A_state_i), z(A_persist_i))``; if one of
    the two is missing for a node the mean is taken over the available
    component(s), and if both are missing ``A_i`` is ``None``. ``A_manifold`` is
    standardized and stored but not averaged into ``A_i``.

    Parameters
    ----------
    A_state:
        ``{node_id: float | None}`` convergence contribution (higher = stronger
        state-support).
    A_persist:
        ``{node_id: float | None}`` persistent-state support.
    A_manifold:
        Optional ``{node_id: float | None}`` population-specific manifold score
        (may be missing/None for feedforward nodes).

    Returns
    -------
    dict with ``node_scores`` (per-node standardized components + final score),
    ``score_version``, and ``component_missingness`` (counts of ``None`` per
    component).
    """
    if not isinstance(A_state, dict) or not isinstance(A_persist, dict):
        raise TypeError("A_state and A_persist must be dicts of {node_id: float|None}")

    nodes = set(A_state.keys()) | set(A_persist.keys())
    if A_manifold is not None:
        if not isinstance(A_manifold, dict):
            raise TypeError("A_manifold must be a dict or None")
        nodes |= set(A_manifold.keys())
    node_ids = sorted(nodes)

    def _column(component: Optional[dict]) -> np.ndarray:
        col = np.full(len(node_ids), np.nan, dtype=np.float64)
        if component is None:
            return col
        for idx, node in enumerate(node_ids):
            value = component.get(node)
            if value is not None:
                col[idx] = float(value)
        return col

    state_vec = _column(A_state)
    persist_vec = _column(A_persist)
    manifold_vec = _column(A_manifold)

    z_state = zscore(state_vec)
    z_persist = zscore(persist_vec)
    z_manifold = zscore(manifold_vec)

    node_scores: dict = {}
    for idx, node in enumerate(node_ids):
        components = [z_state[idx], z_persist[idx]]
        available = [c for c in components if not np.isnan(c)]
        score = float(np.mean(available)) if available else None
        node_scores[node] = {
            "A_state": None if np.isnan(z_state[idx]) else float(z_state[idx]),
            "A_persist": None if np.isnan(z_persist[idx]) else float(z_persist[idx]),
            "A_manifold": None if np.isnan(z_manifold[idx]) else float(z_manifold[idx]),
            "attractor_support_score": score,
        }

    return {
        "node_scores": node_scores,
        "score_version": "v1_equal_weight",
        "component_missingness": {
            "A_state": int(np.isnan(state_vec).sum()),
            "A_persist": int(np.isnan(persist_vec).sum()),
            "A_manifold": int(np.isnan(manifold_vec).sum()),
        },
    }


if __name__ == "__main__":
    A_state = {0: 0.2, 1: 0.5, 2: 0.9, 3: None}
    A_persist = {0: 0.3, 1: None, 2: 0.7, 3: 0.1}
    A_manifold = {0: 1.0, 1: 0.5, 2: None, 3: 2.0}

    res = attractor_support_score(A_state, A_persist, A_manifold)
    print(f"score_version={res['score_version']}")
    print(f"component_missingness={res['component_missingness']}")
    for node, entry in sorted(res["node_scores"].items()):
        print(
            f"node {node}: A_state={entry['A_state']!s:>8} "
            f"A_persist={entry['A_persist']!s:>8} "
            f"A_manifold={entry['A_manifold']!s:>8} "
            f"A_i={entry['attractor_support_score']}"
        )
