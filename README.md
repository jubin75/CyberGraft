# CyberGraft

A connectome-level neural-repair testbed. We lesion a premotor population in a real
connectome, graft a source signal through a trainable mapping into surviving
neurons, and measure *substitutability* — how well a surviving population can
substitute for the lesioned one.

Built on the larval zebrafish hindbrain connectome (Vishwanathan et al., *Nat.
Neurosci.* 2024), using the oculomotor circuit (readout: abducens motoneurons
`ABD_m`; lesioned: velocity-to-position integrator `_Int_`).

## Three findings

1. **Structure–function dissociation.** The circuit's wiring is structurally
   concentrated (the integrator is a strong, specific projection onto the
   motoneurons) yet functionally robust — established against weight-shuffle,
   cell-type-shuffle, and degree-preserving-rewire null models.
2. **Temporal-credit collapse.** The gap between reward-modulated (R-STDP) and
   supervised learning is closed by a per-time-step reward signal, showing the
   deficit lies in collapsing temporal error into a scalar, not in a fundamental
   limit.
3. **Substitutability is a structural property.** Predicted by a graft site's
   direct synaptic weight onto the readout plus its structural role: feedforward
   neurons that directly project onto the readout are the best graft sites, the
   lesioned recurrent population and upstream inputs the worst.

## Install

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `PyYAML`, `psutil` (`mlx` optional, Apple only).

## Data

The connectome files (`ZConnectome_04292021.mat`, `ConnMatrixPre_cleaned.mat`,
`AllCells.mat`) are publicly available at <https://seunglab.org/zebrafish/>
(CC BY-NC-ND 4.0). Place them under `data/raw/zebrafish/`.

## Run the three analyses

```bash
# A — structure–function dissociation (null models)
python -c "from pathlib import Path; from cybergraft.utils.config import load_config; from cybergraft.analysis import run_resilience_nulls, run_functional_resilience_nulls; c = load_config('configs/p5_resilience.yaml'); run_resilience_nulls(c, Path.cwd()); run_functional_resilience_nulls(c, Path.cwd())"

# B — temporal-credit collapse (rule × wiring sweep)
python -c "from pathlib import Path; from cybergraft.utils.config import load_config; from cybergraft.analysis import run_rule_wiring_sweep; c = load_config('configs/p6_rule_wiring.yaml'); run_rule_wiring_sweep(c, Path.cwd(), n_nulls=30, nulls=['weight_shuffle'])"

# C — substitutability (graft-site sweep + node scan + causal intervention)
python -c "from pathlib import Path; from cybergraft.utils.config import load_config; from cybergraft.analysis import run_graft_site_sweep, run_graft_node_scan, run_graft_role_intervention; c = load_config('configs/p6_rule_wiring.yaml'); run_graft_site_sweep(c, Path.cwd()); run_graft_node_scan(c, Path.cwd()); run_graft_role_intervention(c, Path.cwd())"
```

## Layout

- `src/cybergraft/` — the package: `graph` (sparse graph primitives),
  `data_ingest` (connectome adapter), `sim` (LIF simulator), `experiment`
  (lesion/graft/training primitives), `analysis` (the three findings), `utils`.
- `configs/` — experiment configurations.

## License

MIT (code). The connectome data is CC BY-NC-ND 4.0 (Vishwanathan et al., 2024).
