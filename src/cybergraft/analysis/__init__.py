"""Connectome structural analysis: null models, resilience, rule-wiring, graft-substitutability."""

from .null_models import degree_preserving_rewire, shuffle_cell_types, weight_shuffle
from .resilience_experiment import run_functional_resilience_nulls, run_resilience_nulls, structural_redundancy
from .rule_wiring import run_rule_wiring_sweep
from .grafting import run_graft_node_scan, run_graft_role_intervention, run_graft_site_sweep
from .integrator_check import run_integrator_check
from .computational_graft import (
    run_computational_ablation,
    run_computational_graft,
    run_computational_site_scan,
)

__all__ = [
    "weight_shuffle",
    "shuffle_cell_types",
    "degree_preserving_rewire",
    "structural_redundancy",
    "run_resilience_nulls",
    "run_functional_resilience_nulls",
    "run_rule_wiring_sweep",
    "run_graft_site_sweep",
    "run_graft_node_scan",
    "run_graft_role_intervention",
    "run_integrator_check",
    "run_computational_graft",
    "run_computational_site_scan",
    "run_computational_ablation",
]
