# CyberGraft V3 — CHANGELOG

> V3 在完全保持 V2 主实验、指标定义与主结论不变的前提下，新增一个
> "function-specific graftability" 机制解释层。本文件记录 V3 相对 V2 的新增内容、
> 关键结果、以及尚未完成/降级为 supplementary 的部分（诚实记录）。

## 新增模块

- `src/cybergraft/attractor_analysis/` — V3 LIF 动力学引擎（初始态 free-run、时窗
  节点静默、逐神经元静态线性 graft 注入 + 电压轨迹记录）、SCC/recurrence/bounded
  cycles/循环子图谱半径、多初始态收敛、PCA 流形 + 环形度、节点扰动 A_state/A_persist、
  A_i 评分。
- `src/cybergraft/graft_analysis/` — velocity-only computational graft assay（静态
  线性 W）、graft 指标、role-separation index `D_i = z(G_comp)-z(S_i)`。
- `src/cybergraft/analysis/` 新增 3 文件 — Spearman + permutation、嵌套 OLS + ΔR²、
  bootstrap/permutation null。

## 关键结果（zebrafish，config B：E/I + slow τ）

1. **正对照**：完好积分器 τ_tangent ≈ 6.9s（69×τ_m），损伤后 ≈ 9.3s（93×τ_m）——
   顶 50% `_Int_` 损伤**不杀**积分计算功能（积分器分布式/循环承载）。
2. **计算 graft 反转修正**：逐神经元自然传入差分 W 使 surviving_Int 成为唯一
   G_comp 非零位点（0.637），feedforward 位点 G_comp=0。
3. **relay 与 computation 解耦**：best relay（untyped_direct, S_i=0.253）G_comp=0；
   surviving_Int S_i=0.037。
4. **逐节点（n=592）**：A_i → C_i^supp（因果计算支持）rho=**+0.856**（p<0.0001）；
   A_i 与 S_i 解耦（rho=+0.045）。
5. **recurrence 控制**：A_i 与 recurrence 仅 rho=+0.240（中等）；Model 2 ΔR²(A)=**0.292**，
   recurrence 系数为负 → **A_i 非 recurrence 的换名**。
6. **Branch 判定（PRD §42）**：function-specific graftability 解离 = **Branch A 级强证据**（site 级二分 0 vs 0.637）；attractor-supporting 作为机制 = **Branch B（提示性，非 canonical attractor）**。

## 尚未完成 / 降级为 supplementary

- **Drosophila EPG/PEG manifold**：drive-based 协议（drive-time bump rotation）→ 结果
  **明确为 "no ring"**：excitatory-only raw-weight LIF 模型的 EPG/PEG 活动低维
  （top-2 PC ≈ 87%）但 PCA 轨迹是 **1-D 弧线而非圆环**（angular coverage=0.125，
  circularity error=23.7，mean radius≈0.01）。与 V2 "raw-weight LIF 实现 sparse line
  attractor 而非 dense ring attractor" 一致 —— 是 valid 的模型限制，非 assay 伪影。
- **Drosophila computational graft**：无可靠 angular-velocity→heading target，按 PRD
  §25.4 谨慎处理，**不做**（fly 只承担 attractor replication + readout graft replication）。
- **Null 模型（structural）**：real rho_R=12.83 不高于 shuffled nulls（weight_shuffle
  z=-2.57，degree-preserving z=-5.76）→ recurrence 是**特异性稀疏结构**，非全局增强。
- **Dynamical null（A_i / G_comp）**：未做（按 PRD §26 计算量优先保留 real-network
  result，作为 supplementary）。
- **A_state（多初始态收敛差）**：暂缺（缺失保留 null≠0）。

## 诚实限制（PRD §39）

- attractor-support 是模型动力学 phenotype，非生物学标签；
- E/I 符号与慢 τ 是模型设定，非 connectome 直接提供的生物学事实；
- 单 connectome 仍非跨个体样本。
