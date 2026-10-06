"""Computational graftability metrics (PRD §14.4, §15).

Evaluation of a computational graft is against the *normative* position target
``p*(t) = ∫ v(τ) dτ`` constructed from the velocity source alone (never the
intact circuit output). A memoryless (static) readout of the surviving circuit's
actual output is compared to ``p*``; a static readout can only reconstruct the
position if the surviving circuit has already performed the integration.
"""

from typing import Optional

import numpy as np


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination; NaN when the target has zero variance."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def temporal_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two time series (scale/offset invariant)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def normalized_graftability(gain: float, headroom: Optional[float] = None) -> float:
    """``G_i^C``.

    If ``headroom`` is provided, ``G = gain / headroom`` (PRD §15); otherwise
    return the already-normalized ``gain`` (R² / recovery ∈ [0, 1]). Returns NaN
    when ``headroom`` is non-positive.
    """
    if headroom is None:
        return float(gain)
    if headroom <= 0.0:
        return float("nan")
    return float(gain) / float(headroom)
