"""Low-dimensional manifold + circularity metrics (PRD §10).

Population activity ``X ∈ R^(T×N)`` is projected onto its principal components
to reveal whether the population traces a low-dimensional (ring-like) manifold —
the signature of a continuous-attractor population (e.g. EPG/PEG). Circularity
metrics and head-direction alignment are reported *without* forcing a behavioral
interpretation (PRD §10.4).
"""

import numpy as np


def pca_manifold(X: np.ndarray, n_components: int = 3) -> dict:
    """PCA projection of population activity via SVD (no sklearn).

    Centers each column, runs ``numpy.linalg.svd`` with ``full_matrices=False``,
    and returns principal scores ``U * s`` (equivalently ``X_centered @ Vh.T``),
    loading vectors, and per-component explained-variance ratios.

    Parameters
    ----------
    X:
        ``(T, N)`` population activity (voltage or rate).
    n_components:
        Number of principal components to keep (scores are zero-padded if the
        data rank is smaller).

    Returns
    -------
    dict with ``projection`` ``(T, n_components)``, ``components``
    ``(n_components, N)``, ``explained_variance_ratio`` ``(n_components,)``, and
    ``z1``/``z2``/``z3`` (the first three score columns, zero-padded).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D (T, N) array")
    t_steps, n_nodes = X.shape
    if t_steps == 0 or n_nodes == 0:
        raise ValueError("X must have at least one row and one column")
    n_components = int(n_components)
    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    Xc = X - X.mean(axis=0)
    U, s, Vh = np.linalg.svd(Xc, full_matrices=False)
    rank = s.shape[0]
    n = min(n_components, rank)

    scores = U[:, :n] * s[:n]  # (T, n) principal scores
    loadings = Vh[:n]          # (n, N) loading vectors

    total_var = float((s * s).sum())
    evr = (s[:n] * s[:n]) / total_var if total_var > 0.0 else np.zeros(n)

    projection = np.zeros((t_steps, n_components), dtype=np.float64)
    projection[:, :n] = scores
    components = np.zeros((n_components, n_nodes), dtype=np.float64)
    components[:n] = loadings
    explained_variance_ratio = np.zeros(n_components, dtype=np.float64)
    explained_variance_ratio[:n] = evr

    def _score(idx: int) -> np.ndarray:
        return projection[:, idx] if idx < n_components else np.zeros(t_steps)

    return {
        "projection": projection,
        "components": components,
        "explained_variance_ratio": explained_variance_ratio,
        "z1": _score(0),
        "z2": _score(1),
        "z3": _score(2),
    }


def circular_metrics(z1: np.ndarray, z2: np.ndarray, n_bins: int = 32) -> dict:
    """Ring-geometry summary of a 2D projection (PRD §10.3).

    Parameters
    ----------
    z1, z2:
        First two principal scores ``(T,)``.
    n_bins:
        Number of angular bins over ``[-pi, pi]`` for ``angular_coverage``.

    Returns
    -------
    dict with ``radial_variance``, ``angular_coverage``, ``circularity_error``,
    and ``mean_radius``.
    """
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 shape mismatch: {z1.shape} != {z2.shape}")

    r = np.sqrt(z1 * z1 + z2 * z2)
    theta = np.arctan2(z2, z1)
    n_bins = max(1, int(n_bins))

    counts, _ = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))
    angular_coverage = float((counts > 0).sum()) / n_bins

    return {
        "radial_variance": float(np.var(r) / (np.mean(r * r) + 1e-9)),
        "angular_coverage": angular_coverage,
        "circularity_error": float(np.std(r) / (np.mean(r) + 1e-9)),
        "mean_radius": float(np.mean(r)),
    }


def circular_correlation(theta: np.ndarray, heading: np.ndarray) -> float:
    """Circular-circular correlation (Jammalamadaka–Sarma).

    Both inputs are angles in radians. Uses the circular mean direction of each
    (via ``mean(sin)``/``mean(cos)``) and returns a coefficient in ``[-1, 1]``.
    """
    theta = np.asarray(theta, dtype=np.float64)
    heading = np.asarray(heading, dtype=np.float64)
    if theta.shape != heading.shape:
        raise ValueError(f"theta and heading shape mismatch: {theta.shape} != {heading.shape}")
    if theta.size < 2:
        return float("nan")

    def _circ_mean(a: np.ndarray) -> float:
        return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))

    mean_theta = _circ_mean(theta)
    mean_heading = _circ_mean(heading)

    sin_theta = np.sin(theta - mean_theta)
    sin_heading = np.sin(heading - mean_heading)

    numerator = float(np.sum(sin_theta * sin_heading))
    denominator = float(np.sqrt(np.sum(sin_theta * sin_theta) * np.sum(sin_heading * sin_heading)))
    if denominator < 1e-12:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    t_steps = 500
    theta = np.linspace(0.0, 2.0 * np.pi, t_steps, endpoint=False)

    # Embed a noisy ring (radius 1.0) plus a smaller secondary ring (radius 0.6)
    # into a 10-d population, so PCA should recover the dominant ring.
    X = np.zeros((t_steps, 10), dtype=np.float64)
    X[:, 0] = 1.0 * np.cos(theta) + 0.10 * rng.normal(size=t_steps)
    X[:, 1] = 1.0 * np.sin(theta) + 0.10 * rng.normal(size=t_steps)
    X[:, 2] = 0.6 * np.cos(theta) + 0.08 * rng.normal(size=t_steps)
    X[:, 3] = 0.6 * np.sin(theta) + 0.08 * rng.normal(size=t_steps)
    X[:, 4:] = 0.05 * rng.normal(size=(t_steps, 6))

    pca = pca_manifold(X, n_components=3)
    print(f"explained_variance_ratio={np.round(pca['explained_variance_ratio'], 4)}")
    cm = circular_metrics(pca["z1"], pca["z2"], n_bins=32)
    print(f"radial_variance={cm['radial_variance']:.4f}")
    print(f"angular_coverage={cm['angular_coverage']:.4f}")
    print(f"circularity_error={cm['circularity_error']:.4f}")
    print(f"mean_radius={cm['mean_radius']:.4f}")

    # Head-direction alignment: a concentrated direction signal plus small noise
    # should correlate strongly (self-correlation would be exactly 1.0).
    direction = rng.normal(0.0, 0.5, size=t_steps)  # concentrated around 0 rad
    heading = direction + 0.10 * rng.normal(size=t_steps)
    rho = circular_correlation(direction, heading)
    print(f"circular_correlation(direction, heading)={rho:.4f}")
