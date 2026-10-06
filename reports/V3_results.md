# CyberGraft V3 — Results (zebrafish, config B: E/I + slow tau)

## 1. 核心结果

- **Attractor support A_i strongly predicts causal computational support** (per-node rho=0.856, p<0.0001, n=592).
- **Computational graftability is unique to the surviving _Int_ remnant**: site-level G_comp=0.637 (best_gain=2.0); all other sites G_comp=0.0.
- **Relay (S_i) and computation (G_comp) dissociate**: best relay site (untyped_direct, S_i=0.253) has G_comp=0.0; the integrator remnant has S_i=0.037.

## 2. The ten questions (PRD §41)

**1. 最高 recurrence 的结构** — SCC-based per-node recurrence; recurrent _Int_ remnant (see structural.json).

**2. multi-initial-state convergence** — Free-run decays to rest under u=0 (gradient integrator is leaky); replaced by tau_tangent persistence (positive control).

**3. persistent-state dynamics** — Intact tau_tangent=6935ms, lesioned=9260ms (both >> tau_m=100.0ms).

**4. 最高 A_i 的节点** — node 170 (_DOs_, A=16.77), node 172 (_DOs_, A=10.02), node 35 (_Int_, A=8.34), node 597 (_Int_, A=8.34), node 634 (_Int_, A=6.65)

**5. A_i 与 recurrence 相关性** — rho=+0.240 (p=0.000, n=592) — moderate, NOT equivalent.

**6. A_i 是否对计算提供 recurrence 之外解释力** — YES — Model 2 delta_R²(A)=0.292; recurrence coef=-0.0026 (negative when A included).

**7. A_i 是否预测 G_comp** — YES — per-node A_i vs C_i^supp rho=+0.856 (p<0.0001).

**8. direct-readout weight 是否预测 S_i** — Site-level rho=+0.543; per-node under E/I rho=+0.032, but excitatory-only per-node rho=+0.551 (E/I suppresses single-neuron relay).

**9. A_i 与 S_i 是否解耦** — YES — rho=+0.045 (no relation).

**10. relay vs computation role separation** — YES — surviving_Int is computation-only (role +2.86), untyped_direct relay-only (role -1.82).

## 3. Branch determination (PRD §42)

**Branch A (full support)** — attractor-supporting dynamics explain computation-oriented graftability, and direct output connectivity explains relay-oriented graftability:
- A_i -> G_comp (rho=0.856) and A_i adds delta_R²=0.292 beyond recurrence;
- direct_readout_weight -> S_i (site-level rho=0.543);
- A_i is not equivalent to recurrence (rho=0.240).

## 4. Honest caveats

- **C_i^supp is sparse** (7/592 non-zero): the computation is concentrated in a few integrator-core neurons, so the per-node rho=0.856 is driven by a sparse signal.
- **Per-node relay under E/I is degenerate** (S_i~0); the relay relationship is a population-level property under E/I balance (site rho=0.543; excitatory-only per-node rho=0.551).
- **Drosophila manifold (PRD §25) not yet run**; this is zebrafish-only.
- **Attractor support is an operational model-dynamics metric, not a biological label** (PRD §39).
