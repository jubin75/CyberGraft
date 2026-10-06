"""Role-separation index (PRD §5 / §21).

``D_i = z(G_i^C) - z(S_i)`` places each candidate site on a relay-vs-computation
axis: ``D_i < 0`` relay-oriented, ``D_i ≈ 0`` mixed, ``D_i > 0``
computation-oriented. It does not replace any V2 metric — it *summarizes* the
functional-role dissociation.
"""

from typing import Optional, Sequence

import numpy as np


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full_like(values, np.nan)
    mean = values[finite].mean()
    std = values[finite].std()
    if std == 0.0:
        return np.zeros_like(values)
    out = np.full_like(values, np.nan)
    out[finite] = (values[finite] - mean) / std
    return out


def role_index(G_comp: Sequence[Optional[float]], S_i: Sequence[Optional[float]]) -> np.ndarray:
    """Return per-site ``D_i = z(G_i^C) - z(S_i)`` (NaN where either is missing)."""
    G = np.asarray([np.nan if v is None else float(v) for v in G_comp], dtype=np.float64)
    S = np.asarray([np.nan if v is None else float(v) for v in S_i], dtype=np.float64)
    return _zscore(G) - _zscore(S)
