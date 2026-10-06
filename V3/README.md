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

- [x] V3 分支开辟 + 规范对齐（本步）
- [ ] STEP 1–2：读取 V2 manifest/configs，复用 subgraph / simulator / seed
- [ ] STEP 3–6：structural（SCC/recurrence/bounded cycles）+ free-run + perturbation + `A_i`
- [ ] STEP 7：Drosophila manifold validation
- [ ] STEP 8–10：readout graft（`S_i`）复用 + velocity-only computational graft（`G_i^C`）
- [ ] STEP 11–13：`A` vs `S`/`C`/`G_comp`、nested models、role index `D_i`
- [ ] STEP 14–16：Figure 5/6/7、`V3_results.md`、`main.tex`（仅在数据支持时）
