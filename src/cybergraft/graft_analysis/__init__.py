"""CyberGraft V3 — function-specific graftability.

V3 splits "graftability" into two tasks (PRD §3):

  * ``G_i^R`` (readout graftability) — reuses V2's readout substitutability
    ``S_i``; source = intact output command, relayed to the readout.
  * ``G_i^C`` (computational graftability) — NEW velocity-only assay: graft the
    *upstream computational input* (velocity) into a surviving site through a
    static linear ``W`` and ask whether the surviving circuit's own dynamics
    recompute the final output (position).

Submodules:
  * ``computational_graft`` — velocity-only graft assay (``G_i^C``).
  * ``graft_metrics``      — recovery / R² / correlation / normalized graftability.
  * ``role_index``         — ``D_i = z(G_i^C) - z(S_i)`` role-separation axis.
"""
