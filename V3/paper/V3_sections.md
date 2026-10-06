# V3 论文修订方案（function-specific graftability）

> 本文件是**辩证评审后**对 V2 论文的修订指南：把主线从
> "connectome → readout substitutability" 升级为
> "connectome → structural substrates → **function-specific graftability** → relay vs computation"，
> 并把 attractor 语言按证据强度降为三级措辞。V2 的结构、learning、readout graft、
> Drosophila relay replication 全部保留；只有 computational 部分重写 + attractor 措辞降级。
>
> 关键数值（zebrafish, E/I + slow τ）：
> relay：untyped_direct S_i=0.253、surviving_Int S_i=0.037；
> computation：surviving_Int G_comp=0.637、其余 G_comp=0。

---

## 0. 主线升级（最核心的变更，reviewer §1/§13/§15）

```
旧（V2）：
  connectome structure
      ↓
  readout substitutability
      ↓
  (computational headroom 作为补充)

新（V3）：
  connectome
      ↓
  structural substrates
      ↓
  function-specific graftability
      ↓
  ┌──────────────┴──────────────┐
  relay restoration        computation restoration
      ↓                          ↓
  direct output access     state-supporting substrate
      ↓                          ↓
  readout graftability     computational graftability
      ↓                          ↓
     S_i                        G_comp
                                  ↑
                         persistent-state support (A_i)
```

V3 真正由实验直接支撑的**最强结果**不是"某个 attractor score 与某指标相关"，而是：
**同一损伤 connectome 中，针对不同恢复任务，最优 graft site 系统性分离，且结构依赖方向相反。**

---

## 1. Abstract 修订（reviewer §3）

- 把最后一个 finding 从 "Readout and computational substitutability have opposite
  structural prerequisites" 改为：

> (5) **Graftability is function-specific**: the sites that best relay an externally
> supplied command are not the sites that best support recovery of the circuit's
> internal computation. Direct feedforward access favors readout restoration, whereas
> the surviving integrator supports computation-oriented recovery.

- **Abstract 中不出现 "attractor"**（Q2 已明确：当前模型没有充分证据证明 canonical
  attractor）。

---

## 2. Introduction 修订（reviewer §4）

- 核心问题从 "which surviving neurons are structurally able to substitute?" 升级为：

> **Is graftability a single property of a surviving site, or does it depend on the
> function the graft is asked to restore?**

- 由此自然引出两个 assay（readout relay vs internal computation），使全文结构自洽。

---

## 3. Results 修订（computational 部分重写，reviewer §6）

- 原 V2 的 "Computational substitutability" 两小节**保留但重写叙述逻辑**：
  从 "static readout can't integrate → dynamic readout can → E/I+slow makes headroom
  appear"（像"模型加动力学后能积分"）改为真正的问题：

> **Can the surviving network itself support computational grafting?**
> velocity-only source → candidate site → surviving circuit → target computation。

- 结果直接给出**任务依赖的二分**：

| site | S_i (relay) | G_comp (computation) |
|---|---|---|
| untyped_direct | **0.253** | 0.0 |
| gold_top19 | 0.226 | 0.0 |
| surviving_Int | 0.037 | **0.637** |

- 这**不是** computational headroom，而是真正的 computational graftability（velocity
  graft 的 intervention outcome）。

---

## 4. 删除/修改 V2 的过强表述（reviewer §7/§8）

1. **删除/改写** "The integrator is the worst place to graft, precisely because it is
   a recurrent structure whose remaining neurons re-excite the silenced loop."
   → 改为功能特异性表述：

> Recurrence can impair relay through the damaged loop, while the surviving
> integrator population simultaneously preserves computational capacity under
> slow-dynamics conditions.

2. **降级 "line attractor"**：V2 同时出现 "the model realizes a sparse line attractor"
   与 "not by a recurrent line attractor"（自相矛盾）。实际结果（τ≈6.9–9.3s 但 hold 段
   仍衰减 40→22.5、free-run 塌缩到 rest）说明：**long τ ≠ canonical line attractor**。
   删除 "sparse line attractor" 的强表述。

3. **删除** "attractor score proves computation" / "connectome determines attractor"
   / "attractor neurons"（PRD §43 已禁止）。

---

## 5. 三级措辞（reviewer §9，全篇统一）

- **第一层（可直接说，数据支持）**：persistent-state dynamics、slow-dynamical memory、
  state-supporting dynamics。
- **第二层（条件式，用于解释，不作已证实事实）**：attractor-like integration、
  attractor-supporting dynamics、consistent with attractor-related mechanisms。
- **第三层（目前不说）**：canonical line attractor、canonical ring attractor in the
  model、attractor neurons、connectome determines attractor、attractor score proves
  computation。

---

## 6. Drosophila 修订（reviewer §10）

- 生物回路描述可保留 "EPG head-direction compass (a ring attractor...)"（描述已知生物学）。
- 但模型结果改为 limitation 而非坏结果：

> The biological circuit is associated with a ring-attractor architecture, but the
> present excitatory-only LIF implementation does not reproduce a canonical
> ring-attractor manifold.

- 实际结果：excitatory-only LIF 下 EPG/PEG 活动低维（top-2 PC≈87%）但 PCA 轨迹是
  1-D 弧线非圆环（angular coverage=0.125, circularity error=23.7）。

---

## 7. A_i 保留为 operational score（reviewer §11，不要删 attractor analysis）

- 证据等级：

```
  FUNCTION-SPECIFIC GRAFTABILITY        ← STRONG（site 级二分 0 vs 0.637）
      ↓
  computation branch → surviving integrator → G_comp
      ↑
  persistent-state support (A_i)        ← MODERATE（per-node rho=0.856，但 C_i^supp 稀疏 7/592）
```

- A_i 即使只能作为 "operational state-support score" 仍有价值，**不要因 attractor 未证
  明而删除**。

---

## 8. 核心 Figure 修订（reviewer §12）

- 把 V2 的 "readout vs computational headroom" Figure 升级为 **function-specific
  graftability** 四联图（`figures/fig6_function_specific_graftability.png`）：
  - **Panel A**：S_i（readout graftability）by site（6 类）。
  - **Panel B**：G_comp（computational graftability）by site（6 类）。
  - **Panel C**：二维 x=S_i, y=G_comp —— feedforward-direct → high S / low G_comp，
    surviving_Int → low S / high G_comp。
  - **Panel D**：A_i vs C_i^supp（per-node）——作为机制解释，**不是**主发现。

---

## 9. 最终结论措辞（reviewer §14）

不要写 "Attractor-supporting dynamics explain neural graftability"（过强），改：

> **Neural graftability is function-specific.** Sites that directly access the readout
> are preferentially suited for relay-oriented restoration, whereas surviving
> integrator circuitry is preferentially suited for computation-oriented restoration.
> Dynamical analyses further indicate that computation-oriented sites are associated
> with persistent-state support, although the present model does not establish a
> canonical line or ring attractor.

---

## 10. 保留 V2 不变的部分（reviewer §5）

- structurally concentrated but not functionally fragile（仍成立）
- temporal-credit collapse（仍成立）
- direct synaptic weight → readout substitutability（仍是重要结果）
- Drosophila readout relay replication（仍成立）

这些**不要**为了 V3 推翻。
