# CyberGraft V2.0 — Neural Substitutability as a Structural Principle

> V1.0 = "connectome-level computational-grafting **testbed**"（把 testbed 当主要贡献）
> V2.0 = "connectome structure predicts where a lesioned circuit can be **functionally substituted**"（把 neural substitutability 当主要贡献）

## 一句话定位

论文的主命题只有一个：

$$
\boxed{\text{Connectome structure predicts where a lesioned neural circuit can be functionally substituted.}}
$$

主语是 **neural substitutability**（一个可预测的、跨病灶/跨来源/跨回路成立的结构性原则），
不是 **testbed**（一个 resource / 方法学贡献）。

testbed 仍然是研究载体，但它从"主要 scientific contribution"降级为"使该原则可被检验的实验装置"。

## 为什么重构（审稿意见要点）

审稿人指出当前论文的三个内生性弱点，本质是"相关性"被实验设计人为制造出来：

1. **source 循环** — 移植源编码了 intact output command → "direct weight 强 → 易恢复" 近乎同义反复。
2. **lesion 循环** — lesion 按"对 readout 的 synaptic weight 排名前 50%"定义 → 又用同一变量解释恢复。
3. **相关性 ≠ 预测** — `r=0.78 / R²=0.60` 只是相关，未与已有 structural metrics 正面比较，未做 held-out prediction。

对应三个补强实验（详见 `RESEARCH_PLAN.md`）：

| # | 补强实验 | 要消除的循环 | 状态 |
|---|---|---|---|
| 1 | held-out source / target-independent source / held-out condition | source 循环 | 待做 |
| 2 | lesion severity sweep + random / weight-matched-random lesion | lesion 循环 | 待做（基建已存在） |
| 3 | 与 degree / recurrence / path-strength / controllability / observability 正面比较 + cross-validation | 相关≠预测 | 待做（需新增图论指标） |

## 目录

```
V2/
├── README.md           # 本文件：V2.0 定位与导航
├── RESEARCH_PLAN.md    # 详细研究计划（命题、实验、里程碑、风险、验收）
└── paper/
    └── main.tex        # V2.0 论文骨架（单命题重构）
```

## 与 V1.0 的关系

- **不混杂**：V1.0 的代码（`src/cybergraft/`）、配置（`configs/`）、论文（`paper/`）保持不动。
- **复用**：V2.0 直接复用 V1.0 已具备的 connectome adapter、LIF 仿真器、oracle 映射、null models、
  逐神经元扫描（592 neurons）、temporal held-out 机制。
- **新增**：V2.0 新增 held-out source/condition 切分、lesion 选择策略（random / weight-matched）、
  path-strength / controllability / observability 图论指标、以及 cross-validated prediction 流程。

## 状态

- [x] V2.0 目录 + 研究计划 + 论文骨架（本步）
- [ ] 实验 2（lesion robustness）— 风险最低，优先
- [ ] 实验 1（de-circularization：held-out source/condition）
- [ ] 实验 3（structural predictor benchmark）
- [ ] E/I 参数稳健性（审稿意见第 4 条）
- [ ] 论文单命题重写
