# CyberGraft V2.0 研究计划

## 0. 核心命题（唯一主命题）

$$
\boxed{\text{Connectome structure predicts where a lesioned neural circuit can be functionally substituted.}}
$$

- **主语**：neural substitutability —— 一个结构性、可预测的原则。
- **不是**：testbed —— testbed 是 resource，只作为"使该原则可检验"的实验装置出现在 Methods。

论文的科学定位从 **"connectome computational testbed"** 提升为
**"a structural principle for post-lesion neural repair"**。

## 1. 论文结构（单命题，五条故事收敛为一个）

| 层级 | 内容 | 处理 |
|---|---|---|
| **主发现** | substitutability 是结构性原则：direct synaptic weight onto readout + structural role 预测修复位点 | 全部图表围绕它 |
| **第二发现** | readout vs computational substitutability 有相反结构前提 | 保留（最漂亮的结果） |
| **泛化** | Drosophila head-direction compass 复现同一规则 | 保留（cross-species） |
| **支撑** | R-STDP / temporal-credit-collapse | 降级到 Supplementary 或删去 |

引言、摘要、标题全部只讲一个命题；其余作为支撑性 evidence 分层呈现。

## 2. 三个关键补强实验（去内生性）

### 实验 1 —— 去 source 循环（held-out source / target-independent source）

**问题**：移植源编码 intact output command，于是"已有正确答案的 source → surviving neuron → readout"，
direct projection 强的 neuron 更易恢复在逻辑上不意外。

**方案**：
- (a) **held-out source**：用 source 集 A 拟合 oracle 映射，在**从未见过**的 source 集 B 上评估 substitutability；
  证明 direct weight 能预测**未见过**的恢复任务。
- (b) **target-independent source**：source 不由 target output 构造（V1.0 已有 flat-Poisson 控制，需推广）。
- (c) **held-out condition**：在一个 condition（如某刺激频率/范式）上拟合
  "direct weight → substitutability"，在另一个未见过的 condition 上检验该关系是否外推。

**验收**：direct weight 在 held-out source 与 held-out condition 上仍显著预测 substitutability
（交叉验证 R² 显著 > 0，且 site ranking 保持）。

### 实验 2 —— 去 lesion 循环（severity sweep + 多种 lesion 策略）

**问题**：lesion = 按 readout weight 排名前 50% 的 `_Int_`，随后又用同一变量解释恢复 —— 循环。

**方案**：
- **severity sweep**：`10% / 25% / 50% / 75% / 100%`（基建已存在：`lesion_fraction` 参数化 + sweep）。
- **random lesion**：随机子集。
- **weight-matched random lesion**：匹配真实权重分布的随机子集（需新增选择策略）。
- 在每种 lesion 策略下重测 structural rule。

**验收**：direct weight → substitutability 的关系在 3 种 lesion 策略 × 5 种 severity 下
方向一致、斜率显著。若规则只在 weight-ranked lesion 下成立，则原结论被否。

### 实验 3 —— 相关性升级为预测（正面比较 + held-out prediction）

**问题**：`r=0.78 / R²=0.60` 不足以证明 direct weight 是真正的 structural determinant。

**方案**：计算并正面比较以下 structural predictors：

| Predictor | 状态 |
|---|---|
| direct synaptic weight onto readout | 已有（`grafting.py`） |
| in/out degree | 已有 |
| recurrence | 已有 |
| hub share | 已有 |
| path strength | 需新增 |
| structural controllability（Liu 2011） | 需新增 |
| functional observability（Montanari 2022） | 需新增 |

然后做 **held-out prediction / cross-validation**，回答：

> 在一个未知 lesion 的 connectome 中，仅凭结构，能否提前预测哪个 surviving site 最适合修复？

**验收**：direct weight 在 held-out 预测中（至少）不逊于其他 predictor，
并给出可解释的结构原则（不是黑箱最优）。

## 3. 附加稳健性（审稿意见第 4 条）

- E/I 与 slow-τ 目前是手调（`f=0.5, g=1, τ_syn=50ms, τ_m=100ms`）。
- 需证明 opposite-structure result 对参数扰动 robust，并用更 biological-informed 的 E/I 约束。
- 否则 reviewer 会问："你是在发现 biological mechanism，还是在调参制造 integrator？"

## 4. 可行性评估（复用 vs 新增）

**V1.0 已具备，直接复用**：
- connectome adapter（zebrafish / hemibrain / C. elegans）
- LIF 仿真器（reference + MLX）
- oracle 映射（closed-form least squares）
- 三套 null models（weight-shuffle / cell-type-shuffle / degree-preserving rewire）
- 逐神经元扫描（592 neurons）+ per-site 结构特征（direct_weight / recurrence / degree / hub_share）
- `lesion_fraction` sweep 基建（`bypass_experiment.py` 已有 sweep 循环）
- temporal held-out（`recompute_graft.py` / `computational_graft.py` 已有 fit-first-80% / eval-tail）

**V2.0 需新增**：
- held-out source / held-out condition 切分逻辑（新）
- random / weight-matched-random lesion 选择策略（新）
- path-strength / controllability / observability 图论指标（新，`numpy/scipy` 可实现）
- cross-validated prediction 流程（新）

**结论**：框架**能实现**，风险**低-中**。三个实验都是对现有 pipeline 的增量扩展，
无需推翻仿真器或 connectome 管线。

## 5. 里程碑（建议执行顺序 = 风险从低到高）

| 里程碑 | 内容 | 风险 | 依赖 |
|---|---|---|---|
| M-V2.0 | 目录 + 研究计划 + 论文骨架 | — | 本步 |
| M-V2.1 | 实验 2：lesion robustness（severity + random/weight-matched） | 低 | 基建已存在 |
| M-V2.2 | 实验 1：de-circularization（held-out source/condition） | 中 | M-V2.1 |
| M-V2.3 | 实验 3：structural predictor benchmark + controllability/observability | 中-高 | M-V2.1 |
| M-V2.4 | E/I 参数稳健性（审稿意见第 4 条） | 中 | M-V2.2 |
| M-V2.5 | 论文单命题重写 | — | 全部 |

**为什么先做实验 2**：它风险最低（基建已存在）、且最能直接回应"lesion 循环"的质疑；
同时它的结果会影响实验 1/3 的实验设计（例如 held-out condition 的选择）。

## 6. 风险与应对

| 风险 | 等级 | 应对 |
|---|---|---|
| 去循环后效应缩小 | 中 | 这是科学事实，如实报告；V1.0 的 flat-Poisson 控制已显示 ranking 存活（best site 0.87→0.75），是积极信号 |
| direct weight 未显著优于其他 predictor | 中 | 若如此，结论降级为"结构角色（feedforward vs recurrent）而非单一权重"仍是可发表的结构性原则 |
| controllability/observability 计算开销 | 低 | 子图仅 ~680 节点，Gramian 用 `scipy` 稀疏求解可行 |
| E/I 参数鲁棒性不成立 | 中 | 该结果是"第二发现"，即使受限于 assigned E/I，也不动摇主命题（readout substitutability 是 excitatory-only 下即成立的） |

## 7. 每个实验的"通过"判据（总览）

- **实验 1 通过**：direct weight 在 held-out source/condition 上显著预测（cross-val R² > 0，ranking 保持）。
- **实验 2 通过**：structural rule 在 3 策略 × 5 severity 下方向一致、斜率显著。
- **实验 3 通过**：direct weight 在 held-out prediction 中不逊于/优于替代 predictor，且给出可解释原则。
- **M-V2.4 通过**：opposite-structure result 在 E/I 与 τ 的合理扰动范围内保持。

三者全部通过 → 论文科学定位实现从 testbed 到 structural principle 的跃迁。
