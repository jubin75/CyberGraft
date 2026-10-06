"""CyberGraft V3 — attractor-supporting dynamics analysis.

This package implements the V3 "mechanistic layer" without touching the V2
experiments, metric definitions, or conclusions. It is a *new* analysis layer
that builds ``A_i`` (attractor-support score) from network dynamics and connects
it to function-specific graftability.

Submodules:
  * ``dynamics``   — V3 LIF engine: initial-state free-run, node perturbation
                     (time-windowed silencing), and per-neuron static-linear
                     graft injection, with voltage-trajectory recording.
  * ``structural`` — SCC / recurrence / bounded cycles / recurrent spectral
                     radius on the connectome graph.
  * ``convergence``— multi-initial-state pairwise convergence metrics.
  * ``manifold``   — low-dimensional (PCA) manifold + circularity metrics.
  * ``perturbation`` — node-level perturbation assays (A_state / A_persist).
  * ``scoring``    — attractor-support score ``A_i``.

All modules keep ``null != 0``: missing per-node scores are ``None``/``NaN``,
never silently zero-filled.
"""
