# CyberGraft V3 — Function-specific Graftability（机制解释层）

> 在**完全保持 V2 主实验、指标定义与主结论不变**的前提下，新增一个
> **function-specific graftability** 机制层：relay branch（复用 V2 readout
> graft / `S_i`）与 computation branch（新增 velocity-only computational graft
> → `G_i^C`），并用 attractor-support score `A_i` 连接二者。

## 定位与不变式（Invariant）

- **不改 V2**：不重做 V2 主实验、不重定义 V2 指标、不删除 V2 结论。
- **不动 V2 论文与实验数据**：`V2/paper/`、`paper/`、`results/`（实验数据）
  均不在本分支改动范围。
- **V3 是补全，不是替换**：V2 主命题 "Connectome structure predicts neural
  substitutability after circuit lesion" 保持不变，V3 只增加机制解释层。

## 规范文档

- 权威规格：`PRD/chatgpt_review_V3.md`（含 5 层动力学分析、computational graft
  assay、统计策略、falsification tests、验收标准、实现顺序 STEP 1–16）。

## 新增模块（规划，详见 PRD §6 / §49）

```
src/cybergraft/
  attractor_analysis/   # structural / free_run / convergence / manifold / perturbation / scoring
  graft_analysis/       # computational_graft / graft_metrics / role_index
  analysis/             # attractor_graft_relationship / multivariate_models / null_tests / report_v3
```

## 复用（V1.0 核心已存在，V3 直接调用）

- connectome adapters（zebrafish / hemibrain compass / C. elegans）
- LIF simulator（reference + MLX，含 E/I 与 slow-τ）
- lesion（`node_silence`）、oracle graft（static linear `W`）、
  velocity-only source（`analysis/computational_graft.py` 已有）
- null models、config + manifest + seed 系统

## 状态

- [x] V3 分支开辟 + 模块层 + 统计层（提交 `abb7f0a`）
- [x] 正对照：config B 产生持续积分器（τ_tangent 6.9s），损伤不杀计算（9.3s）
- [x] site 级修正实验：surviving_Int 唯一 G_comp=0.637，relay/计算解耦
- [x] 逐节点分析：A_i → C_i^supp rho=0.856（n=592）
- [x] recurrence 控制：ΔR²(A)=0.292，A_i 非 recurrence 换名
- [x] direct_vs_S 排查：E/I 抑制单神经元 relay（excitatory-only rho=0.551）
- [x] Figure 5/6/7 + `reports/V3_results.md`（提交 `bbb18ee`）
- [x] `reports/V3_CHANGELOG.md` + `manifest_v3.json` + `V3/paper/V3_sections.md`
- [x] Null 模型（structural）：real recurrence 不高于 shuffled nulls（特异性稀疏结构）
- [ ] Drosophila EPG/PEG manifold：drive-based ring 协议修正中
- [ ] Dynamical null（A_i/G_comp）：supplementary（PRD §26）

## 关键结果速览

| 结果 | 数值 |
|---|---|
| A_i → C_i^supp（per-node Spearman） | +0.856（p<0.0001, n=592） |
| Model 2 ΔR²(A)（recurrence 控制） | 0.292（recurrence 系数为负） |
| surviving_Int G_comp | 0.637（唯一非零位点） |
| 正对照 τ_tangent（intact / lesioned） | 6.9s / 9.3s（τ_m=100ms） |
| role index（surviving_Int / untyped_direct） | +2.86 / -1.82 |

判定：**Branch A（完整支持，PRD §42）**。
