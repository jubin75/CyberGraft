"""Multi-initial-state pairwise convergence (PRD §8.3).

V3 supports attractor-like behavior with *initial-condition convergence*: distinct
initial states entering the same region of state space over a free-run. This
module compares every pair of recorded trajectories ``(T, N)`` and summarizes how
much they move toward each other, plus how many distinct final-state clusters
remain after thresholding.

The key invariant (PRD §8.3): activity similarity alone is NOT sufficient to
claim an attractor — it must co-occur with initial-condition convergence and
temporal stability. This module only provides the pairwise-convergence evidence.
"""

from typing import List

import numpy as np


def pairwise_convergence(
    trajectories: List[np.ndarray],
    *,
    n_initial: int = 1,
    n_final: int = 10,
    eps: float = 0.05,
) -> dict:
    """Compute pairwise time-resolved convergence over ``K`` initial states.

    For every pair ``(i, j)`` the normalized distance at time ``t`` is::

        d_ij(t) = ||X_i(t) - X_j(t)||_2 / max(1e-9, ||X_i(t)||_2 + ||X_j(t)||_2)

    which is bounded in ``[0, ~1]``. ``initial_distance`` and ``final_distance``
    average ``d_ij`` over the first ``n_initial`` and last ``n_final`` timesteps;
    ``convergence_ratio`` is ``final / max(initial, 1e-9)`` (<1 means the pair
    converged toward each other). ``convergence_cluster`` counts connected
    components of the ``final_distance < eps`` adjacency (simple BFS).

    Parameters
    ----------
    trajectories:
        One ``(T, N)`` float array per initial state (voltage or rate). All
        arrays must share the same shape.
    n_initial:
        Number of leading timesteps averaged for ``initial_distance``.
    n_final:
        Number of trailing timesteps averaged for ``final_distance``.
    eps:
        Final-distance threshold below which two states are considered part of
        the same cluster.

    Returns
    -------
    dict with keys ``n_states``, ``initial_distance``, ``final_distance``,
    ``convergence_ratio`` (all ``(K, K)``), off-diagonal means, and
    ``convergence_cluster``.
    """
    trajs = [np.asarray(t, dtype=np.float64) for t in trajectories]
    if len(trajs) < 2:
        raise ValueError("pairwise convergence requires at least two trajectories")
    shape = trajs[0].shape
    if len(shape) != 2 or shape[0] == 0 or shape[1] == 0:
        raise ValueError("each trajectory must be a non-empty (T, N) array")
    for t in trajs:
        if t.shape != shape:
            raise ValueError(f"trajectory shape mismatch: {t.shape} != {shape}")

    k = len(trajs)
    t_steps = shape[0]
    n_initial = max(1, min(int(n_initial), t_steps))
    n_final = max(1, min(int(n_final), t_steps))

    initial = np.zeros((k, k), dtype=np.float64)
    final = np.zeros((k, k), dtype=np.float64)

    for i in range(k):
        xi_norm = np.linalg.norm(trajs[i], axis=1)
        for j in range(i + 1, k):
            diff_norm = np.linalg.norm(trajs[i] - trajs[j], axis=1)
            denom = np.maximum(1e-9, xi_norm + np.linalg.norm(trajs[j], axis=1))
            d = diff_norm / denom  # (T,) normalized distance
            d_init = float(d[:n_initial].mean())
            d_fin = float(d[-n_final:].mean())
            initial[i, j] = initial[j, i] = d_init
            final[i, j] = final[j, i] = d_fin

    ratio = final / np.maximum(initial, 1e-9)

    off = ~np.eye(k, dtype=bool)
    mean_initial = float(initial[off].mean())
    mean_final = float(final[off].mean())
    mean_ratio = float(ratio[off].mean())

    # Connected components of the `final_distance < eps` adjacency (BFS).
    adjacency = final < eps
    np.fill_diagonal(adjacency, False)
    visited = [False] * k
    clusters = 0
    for start in range(k):
        if visited[start]:
            continue
        clusters += 1
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            for v in range(k):
                if adjacency[u, v] and not visited[v]:
                    visited[v] = True
                    stack.append(v)

    return {
        "n_states": k,
        "initial_distance": initial,
        "final_distance": final,
        "convergence_ratio": ratio,
        "mean_initial_distance": mean_initial,
        "mean_final_distance": mean_final,
        "mean_convergence_ratio": mean_ratio,
        "convergence_cluster": clusters,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t_steps, n_nodes = 100, 20
    base = rng.normal(0.0, 1.0, (t_steps, n_nodes))

    # State 0: baseline. State 1: transient perturbation that decays to baseline
    # (converges). State 2: independent noisy state (does not converge).
    decay = np.linspace(1.0, 0.0, t_steps)[:, None]
    traj0 = base
    traj1 = base + 2.0 * decay * rng.normal(0.0, 1.0, (t_steps, n_nodes))
    traj2 = base + 3.0 * rng.normal(0.0, 1.0, (t_steps, n_nodes))

    res = pairwise_convergence([traj0, traj1, traj2], n_initial=5, n_final=10, eps=0.05)
    print(f"n_states={res['n_states']}")
    print(f"mean_initial_distance={res['mean_initial_distance']:.4f}")
    print(f"mean_final_distance={res['mean_final_distance']:.4f}")
    print(f"mean_convergence_ratio={res['mean_convergence_ratio']:.4f}")
    print(f"convergence_cluster={res['convergence_cluster']}")
    print("final_distance:\n", np.round(res["final_distance"], 4))
