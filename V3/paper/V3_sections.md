# V3 Paper Sections（并入 main.tex 的新增内容）

> 本文档给出 V3 相对 V2 论文需要**新增/修改**的文本，严格遵循"不重做 V2 主实验、
> 不重定义 V2 指标、不删除 V2 结论"（PRD §1）。数值来自本次实际运行结果，按
> PRD §42 Branch A 表述，并保留 §39 的诚实限制。

---

## Abstract（只加一句，不动第一主线）

> Dynamical analysis further showed that computation-oriented grafting is
> associated with stronger participation in persistent-state dynamics, whereas
> relay-oriented grafting is associated with direct access to the output channel.

---

## Methods 新增小节（PRD §35）

**Computational-graft assay.** 与 readout graft 唯一的区别是 source semantics
（PRD §28）：source = 上游 velocity `v(t)`（非 intact output），通过**静态线性**
`W`（逐神经元自然传入权重 `φ_j = Σ_{a∈_DOs_} w(a→j)·sign(a→j)` 乘以标量增益）注入
candidate site，让 surviving connectome 自身动力学传播，读取实际 `ABD_m` output，
与规范位置 `p*(t)=∫v dt` 比较。静态 W 无法积分，故任何恢复的 position 必来自
surviving circuit 的循环动力学（反作弊，PRD §29）。探针为 step-and-hold（velocity
step 后 u=0 hold），无 DC 基座。

**Attractor-support analysis.** `A_i = mean(z(A_state), z(A_persist))`。首版
`A_persist` = 在 held state 下短暂静默节点 i，测固定 t* 时刻的**分数记忆偏移**；
`A_state`（多初始态收敛差）暂缺（缺失保留 `null≠0`，非补 0）。

**Multi-initial-state dynamics.** 多初始态 free-run 对 gradient integrator（线吸引子）
会塌缩到 rest；改为 **step-and-hold 持久性检验**：velocity step 后 u=0，测 readout
衰减时间常数 `τ_tangent` 对比 `τ_m`（线吸引子判据 `τ_tangent ≫ τ_m`）。

**Node perturbation.** 时窗节点静默（不删 edge、不改 weight，仅改该时窗神经元活动），
释放后测 readout 偏移。

**Low-dimensional manifold analysis.** PCA(n_components=3) + 环形度（radial variance /
angular coverage / circularity error）。无可靠 head-direction target 时只报 geometry。

**Computational graftability.** `G_i^C` = normalized computational recovery（static
readout of actual output → p* 的 R²）。

**Function-specific graftability analysis.** role index `D_i = z(G_i^C) - z(S_i)`。

**Statistical decomposition.** Spearman rho + permutation p；嵌套 OLS + partial R²/ΔR²；
recurrence 控制（`G_comp ~ A + recurrence + direct + degree`）。

---

## Results 新增两个主体小节（PRD §36）

### 36.1 Attractor-supporting dynamics localize to state-oriented circuit structure

- **正对照**：完好 oculomotor integrator 在 E/I+slow 下 `τ_tangent ≈ 6.9s`（69×τ_m），
  积分 ramp=40.7 → 持续积分器存在。
- **损伤不杀计算**：顶 50% `_Int_`（按 readout weight）静默后 `τ_tangent ≈ 9.3s`
  （93×τ_m），ramp=43 → 积分器功能由存活 `_Int_` 残余的循环动力学承载，对"顶投射者"
  损伤鲁棒。
- **逐节点因果计算支持**集中在 `_Int_` 残余（mean C_i^supp=+0.0216）与 `_DOs_`
  （+0.0112），feedforward（untyped）≈0。

### 36.2 Neural graftability is function-specific

- **relay path**：`direct_readout_weight → S_i`（site 级 rho=+0.543）。
- **computation path**：`A_i → C_i^supp`（per-node rho=**+0.856**, p<0.0001, n=592）；
  `G_comp` 唯一非零在 surviving_Int（0.637），feedforward 位点 G_comp=0。
- **role separation**：`D_i = z(G_comp)-z(S_i)` → surviving_Int +2.86（计算向），
  untyped_direct -1.82（relay 向）。
- **recurrence 控制**：`A_i` 与 recurrence rho=+0.240（中等）；Model 2
  `ΔR²(A)=0.292`，recurrence 系数为负 → A_i 提供 recurrence 之外的信息。

> **The structural substrate of successful intervention depends on what the
> graft is asked to restore.**

---

## Discussion 新增小节（PRD §37）

### Dynamical basis of function-specific graftability

1. V2 已发现 readout substitutability 与 computational capability 的相反结构倾向；
2. V3 直接检验这两个行为是否对应不同的 network roles；
3. direct output connectivity 预测 relay-oriented graftability；
4. recurrent / attractor-supporting dynamics 预测 computation-oriented graftability；
5. 因此 neural intervention 应 task-specific 而非 node-generic。

> **Our results suggest that graftability is not a unitary property of a
> surviving neuron. It is task-specific: the same circuit can contain sites
> optimized for relaying an externally supplied command and different sites that
> retain the dynamical machinery required to restore internal computation.**
>
> **Attractor-supporting dynamics provide a mechanistic interpretation of the
> latter class when supported by multi-initial-state convergence, persistence,
> and perturbation recovery.**

---

## Limitations（PRD §39，必须新增）

1. attractor 是模型动力学性质，非 connectome 单独决定；
2. E/I sign 是模型设定，结论依赖该设定；
3. slow time constants 是模型参数；
4. attractor-support score 是 operational metric，非生物学真值；
5. computational graftability 只证明在当前 model 下可行；
6. Drosophila 不能自动推出 zebrafish 的 computational mechanism 跨物种完全成立；
7. 单个 connectome 仍非跨个体样本。

> **Attractor-support should therefore be interpreted as a dynamical phenotype
> of the model constrained by the connectome, rather than as a directly observed
> biological label.**

---

## 尚未完成（供后续补写）

- **Drosophila EPG/PEG manifold**：drive-based 协议结果明确 —— excitatory-only
  raw-weight LIF 模型的 EPG/PEG 活动低维（top-2 PC ≈ 87%）但 PCA 轨迹是 **1-D 弧线
  而非圆环**（angular coverage=0.125，circularity error=23.7）→ 与 V2"raw-weight LIF
  实现 sparse line attractor 而非 dense ring attractor"一致，作为 model limitation
  如实报告，不强行赋予行为含义。
- **Dynamical null（A_i/G_comp）**：作为 supplementary（PRD §26）。
- **structural null 结果**（real rho_R=12.83 不高于 shuffled nulls）建议写入 Discussion
  或 Supplementary：recurrence 是**特异性稀疏结构**（sparse line attractor），非全局增强。
