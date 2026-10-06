"""Null tests for Spearman rank correlations (V3 mechanistic-analysis layer).

Permutation and bootstrap utilities shared across the analysis layer. Pure
NumPy/SciPy (no sklearn/pandas/networkx). ``permutation_spearman_p`` is the
canonical two-sided permutation test and is imported by
``attractor_graft_relationship``.

Missing/NaN values are treated as missing (complete-case), never zero-filled.
"""

import numpy as np
from scipy.stats import spearmanr

__all__ = ["permutation_spearman_p", "bootstrap_spearman_ci"]


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, complete-case and variance-guarded.

    Returns ``float('nan')`` when fewer than 3 finite pairs remain or when
    either variable is constant (zero variance).
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


def permutation_spearman_p(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10000,
    seed: int | None = None,
) -> float:
    """Two-sided permutation p-value for the Spearman rank correlation.

    Shuffles ``y`` ``n_perm`` times (keeping ``x`` fixed) and recomputes rho.
    The p-value is the fraction of permutations with ``|rho_perm| >= |rho_obs|``.
    Returns ``float('nan')`` if fewer than 3 valid pairs remain or the observed
    rho is undefined (zero variance).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    rho_obs = _spearman_rho(x, y)
    if np.isnan(rho_obs):
        return float("nan")

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        rho_perm = _spearman_rho(x, y_perm)
        if not np.isnan(rho_perm) and abs(rho_perm) >= abs(rho_obs):
            count += 1
    return float(count / n_perm)


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    seed: int | None = None,
    ci: float = 0.95,
) -> dict:
    """Percentile bootstrap confidence interval for Spearman rho.

    Resamples ``(x, y)`` pairs with replacement, recomputes rho on each bootstrap
    replicate, and returns the empirical ``ci`` interval. Returns NaN bounds if
    fewer than 3 valid pairs remain.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    nan_ci = {
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n_boot": int(n_boot),
        "mean": float("nan"),
    }
    if n < 3:
        return nan_ci

    rng = np.random.default_rng(seed)
    rhos = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rhos[b] = _spearman_rho(x[idx], y[idx])
    rhos = rhos[np.isfinite(rhos)]
    if rhos.size == 0:
        return nan_ci

    alpha = 1.0 - float(ci)
    ci_low = float(np.percentile(rhos, 100.0 * alpha / 2.0))
    ci_high = float(np.percentile(rhos, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_boot": int(n_boot),
        "mean": float(np.mean(rhos)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 30
    x = rng.normal(size=n)
    y = 0.7 * x + rng.normal(0, 0.6, size=n)
    x[5] = np.nan
    y[17] = np.nan

    p = permutation_spearman_p(x, y, n_perm=5000, seed=0)
    ci = bootstrap_spearman_ci(x, y, n_boot=2000, seed=0)
    print(f"permutation_spearman_p = {p:.4f}")
    print(
        f"bootstrap_spearman_ci (95%): [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] "
        f"mean={ci['mean']:.4f} n_boot={ci['n_boot']}"
    )
    print("null_tests OK")
