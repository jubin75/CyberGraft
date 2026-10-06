"""Attractor<->graft relationship: rank-correlating V3 per-node records.

Connects the attractor-dynamics layer (``attractor_support_score``, ``A_state``,
``A_persist``, ``A_manifold``) to the graft layer (``readout_substitutability``,
``computational_graftability``, ...) on a single per-node ``records`` list (V3
JSON schema, PRD §40). Pure NumPy/SciPy (no sklearn/pandas/networkx).

Missing/None values are treated as missing, never zero-filled; every pairwise
statistic operates on the complete-case subset only.
"""

import numpy as np
from scipy.stats import spearmanr

from cybergraft.analysis.null_tests import permutation_spearman_p

__all__ = [
    "_valid_pairs",
    "spearman_rho",
    "permutation_p",
    "correlate",
    "pairwise_correlations",
]


def _valid_pairs(
    records: list[dict],
    x_field: str,
    y_field: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract complete-case ``(x, y)`` columns, dropping None/NaN rows.

    Returns two ``float64`` arrays of equal length. Any record whose ``x_field``
    or ``y_field`` value is missing (``None``) or non-finite is excluded.
    """
    xs: list = []
    ys: list = []
    for rec in records:
        x = rec.get(x_field)
        y = rec.get(y_field)
        if x is None or y is None:
            continue
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if np.isnan(xf) or np.isnan(yf):
            continue
        xs.append(xf)
        ys.append(yf)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, complete-case and variance-guarded.

    Returns ``float('nan')`` if fewer than 3 valid pairs remain after dropping
    non-finite entries, or if either variable has zero variance (constant).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).correlation)


def permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10000,
    seed: int | None = None,
) -> float:
    """Two-sided permutation p-value for Spearman rho (delegates to null_tests)."""
    return permutation_spearman_p(x, y, n_perm=n_perm, seed=seed)


def correlate(
    records: list[dict],
    x_field: str,
    y_field: str,
    *,
    n_perm: int = 10000,
    seed: int | None = None,
) -> dict:
    """Spearman correlation between two record fields + permutation p-value.

    Returns ``{"x_field", "y_field", "rho", "p_perm", "n", "x", "y"}`` where
    ``x``/``y`` are the complete-case value lists actually used.
    """
    x, y = _valid_pairs(records, x_field, y_field)
    rho = spearman_rho(x, y)
    p = permutation_p(x, y, n_perm=n_perm, seed=seed)
    return {
        "x_field": x_field,
        "y_field": y_field,
        "rho": float(rho),
        "p_perm": float(p),
        "n": int(x.size),
        "x": x.tolist(),
        "y": y.tolist(),
    }


def pairwise_correlations(
    records: list[dict],
    fields: list[str],
    *,
    n_perm: int = 10000,
    seed: int | None = None,
) -> dict:
    """All ordered field pairs -> Spearman correlation (key ``"x|y"``).

    Every ordered pair ``(xf, yf)`` (including ``xf == yf``) is correlated and
    stored under the key ``f"{xf}|{yf}"``.
    """
    out: dict = {}
    for xf in fields:
        for yf in fields:
            out[f"{xf}|{yf}"] = correlate(records, xf, yf, n_perm=n_perm, seed=seed)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 30
    direct = rng.gamma(2.0, 0.5, size=n)
    recurrence = rng.beta(2.0, 5.0, size=n)
    degree = rng.poisson(4, size=n).astype(float)
    attractor = 0.3 + 0.6 * direct / direct.max() + 0.2 * recurrence + rng.normal(0, 0.1, size=n)
    substitutability = 0.5 * direct / direct.max() + 0.3 * attractor + rng.normal(0, 0.1, size=n)
    records = [
        {
            "node_id": i,
            "population": "synth",
            "direct_readout_weight": float(direct[i]),
            "recurrence": float(recurrence[i]),
            "degree": float(degree[i]),
            "attractor_support_score": float(attractor[i]),
            "readout_substitutability": float(substitutability[i]),
        }
        for i in range(n)
    ]
    records[3]["attractor_support_score"] = None
    records[5]["direct_readout_weight"] = None
    records[11]["readout_substitutability"] = None

    corr = correlate(records, "direct_readout_weight", "readout_substitutability", seed=0)
    print(
        f"Spearman(direct_readout_weight, readout_substitutability): "
        f"rho={corr['rho']:.4f} p_perm={corr['p_perm']:.4f} n={corr['n']}"
    )

    fields = ["direct_readout_weight", "recurrence", "attractor_support_score", "readout_substitutability"]
    pw = pairwise_correlations(records, fields, n_perm=2000, seed=0)
    for k in (
        "direct_readout_weight|readout_substitutability",
        "attractor_support_score|readout_substitutability",
        "recurrence|readout_substitutability",
    ):
        r = pw[k]
        print(f"  {k}: rho={r['rho']:.4f} p_perm={r['p_perm']:.4f}")
    print("attractor_graft_relationship OK")
