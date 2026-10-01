# CyberGraft

A connectome-level neural-repair testbed. We lesion a premotor population in a real
connectome, graft a source signal through a trainable mapping into surviving
neurons, and measure **substitutability** — how well a surviving population can
substitute for the lesioned one.

Built on three connectomes — the larval zebrafish hindbrain (oculomotor circuit),
the adult *Drosophila* hemibrain (head-direction compass), and *C. elegans*
(reversal circuit) — the testbed formalizes neural repair as a *computational
grafting* problem and asks a question prior work does not: not *which lesions
hurt* (fragility), but *which lesions can be repaired* (restorability), and what
structural conditions determine that.

## Findings

1. **Structure–function dissociation.** The oculomotor circuit's wiring is
   *structurally concentrated* on the velocity-to-position integrator (a strong,
   specific projection onto the motoneurons) yet *functionally robust* —
   established against weight-shuffle, cell-type-shuffle, and degree-preserving-rewire
   null models.
2. **Temporal-credit collapse.** The gap between reward-modulated (R-STDP) and
   supervised learning is closed by a per-time-step reward signal, showing the
   deficit lies in collapsing temporal error into a scalar, not in a fundamental
   limit.
3. **Substitutability is a structural property.** Predicted by a graft site's
   direct synaptic weight onto the readout (R² = 0.60) plus its structural role:
   feedforward neurons that directly project onto the readout are the best graft
   sites; the lesioned recurrent population and upstream inputs are the worst.
4. **Cross-species generality.** The same structural rule replicates in the
   *Drosophila* hemibrain EPG head-direction compass.
5. **Readout and computational substitutability have opposite prerequisites.**
   The substitutability above is *readout* substitutability — relaying a command,
   since the grafted source encodes the intact output. Assigning
   excitatory/inhibitory (E/I) balance and slow time constants turns the
   excitatory-only "integrator" (a bistable latch) into a genuine gradient
   integrator, and reveals that *computational* substitutability — recomputing the
   velocity→position integral — has the *opposite* structural prerequisite: it
   lives in the integrator population itself, via slow intrinsic dynamics, not in
   the feedforward relay sites.

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.10. Dependencies: `numpy`, `scipy`, `PyYAML`, `psutil`
(`mlx` optional, Apple only).

## Data

| Species | Files / source | License |
|---|---|---|
| Zebrafish hindbrain | `ZConnectome_04292021.mat`, `ConnMatrixPre_cleaned.mat`, `AllCells.mat` — <https://seunglab.org/zebrafish/> | CC BY-NC-ND 4.0 |
| *Drosophila* hemibrain | neuPrint `hemibrain:v1.2.1` — <https://neuprint.janelia.org> | CC BY 4.0 |
| *C. elegans* | `herm_full_edgelist.csv` — <https://github.com/openworm/CElegansNeuroML> | MIT (via OpenWorm) |

Place the zebrafish `.mat` files under `data/raw/zebrafish/`. The fly connectome
is queried live through the bundled neuPrint client; the C. elegans edge list is
downloaded from the OpenWorm repository.

## Quick start

### A — structure–function dissociation (null models)

```bash
python -c "from pathlib import Path; from cybergraft.utils.config import load_config; from cybergraft.analysis import run_resilience_nulls, run_functional_resilience_nulls; c = load_config('configs/p5_resilience.yaml'); run_resilience_nulls(c, Path.cwd()); run_functional_resilience_nulls(c, Path.cwd())"
```

### B — temporal-credit collapse (rule × wiring sweep)

```bash
python -c "from pathlib import Path; from cybergraft.utils.config import load_config; from cybergraft.analysis import run_rule_wiring_sweep; c = load_config('configs/p6_rule_wiring.yaml'); run_rule_wiring_sweep(c, Path.cwd(), n_nulls=30, nulls=['weight_shuffle'])"
```

### C — substitutability (graft-site sweep + node scan + causal intervention)

```bash
python -c "from pathlib import Path; from cybergraft.utils.config import load_config; from cybergraft.analysis import run_graft_site_sweep, run_graft_node_scan, run_graft_role_intervention; c = load_config('configs/p6_rule_wiring.yaml'); run_graft_site_sweep(c, Path.cwd()); run_graft_node_scan(c, Path.cwd()); run_graft_role_intervention(c, Path.cwd())"
```

### Integrator check — does the intact circuit integrate velocity?

```bash
python -c "from pathlib import Path; from cybergraft.utils.config import load_config; from cybergraft.analysis import run_integrator_check; c = load_config('configs/p9_integrator_check.yaml'); run_integrator_check(c, Path.cwd())"
```

### Computational substitutability — readout vs computational headroom

```bash
python -c "from pathlib import Path; from cybergraft.utils.config import load_config; from cybergraft.analysis import run_computational_graft, run_computational_site_scan, run_computational_ablation; c = load_config('configs/p10_computational_graft.yaml'); run_computational_graft(c, Path.cwd()); run_computational_site_scan(c, Path.cwd()); run_computational_ablation(c, Path.cwd())"
```

## The computational-substitutability metric

Finding 3 is *readout* substitutability: because the grafted source encodes the
intact output, it asks whether a surviving site can *relay* a command. To
separate this from *computational* substitutability — whether a surviving circuit
can *recompute* the integrator's velocity→position transformation — we lesion the
integrator, drive the vestibular input with **velocity alone**, and regress
readouts of the surviving neurons' activity onto the normative position target
`p* = ∫ v dt`:

- **relay capacity** — a static (memoryless) readout reconstructing *velocity*;
- **computational headroom** — a static readout reconstructing *position*;
- **relocation** — how much a leaky-integrator readout adds over the static readout.

Under the excitatory-only model the surviving circuit relays but does not
integrate (computational headroom ≈ 0). Assigning E/I balance by Dale's principle
and slow time constants (`f = 0.5`, `g = 1`, `τ_syn = 50 ms`, `τ_m = 100 ms`)
restores a gradient integrator, and the per-site scan *inverts* the readout
ranking: the recurrent integrator remnant is the only site with positive
computational headroom. A causal test localizes the mechanism to slow intrinsic
dynamics, not recurrence — ablating the remnant's recurrent synapses leaves its
headroom unchanged.

A cross-species probe reveals an architecture-dependent limit: the same E/I +
slow-τ treatment that restores the zebrafish's sparse line attractor instead
*saturates* the fly's dense ring attractor and the C. elegans reversal circuit's
gap-junction-coupled command cluster, so the raw-weight LIF model realizes a
sparse line attractor but not dense or electrically coupled attractors.

## Package layout

```
src/cybergraft/
├── graph/        # SparseGraph (CSR), Edge/Node schema, validator
├── data_ingest/  # connectome adapters: zebrafish, hemibrain (fly), C. elegans, neuPrint client
├── sim/          # LIF simulator (reference + MLX), stimulus, activity
├── experiment/   # lesion, graft (bypass), trainable mapping, motor primitive
├── analysis/     # null models, resilience, rule-wiring, graft substitutability, integrator check, computational substitutability
└── utils/        # config, logging, manifest, memory guard
```

`configs/` holds the experiment YAML files (`base.yaml` plus `p0`–`p10`), each
extending `base.yaml`.

## Tests

```bash
pytest tests/
```

Unit tests cover the graph, simulator, adapters, and metrics; integration tests
run the milestone experiments end-to-end.

## License

MIT (code). The connectome data are licensed as listed above (zebrafish
CC BY-NC-ND 4.0; hemibrain CC BY 4.0; C. elegans MIT via OpenWorm).
