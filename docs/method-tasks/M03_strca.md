# M03 — sTRCA（Time-filter + Similarity-constrained TRCA）

## 论文
- 标题: Task-related component analysis based on time filter and similarity constraint for SSVEP-based BCI
- 期刊: Measurement, 2024
- DOI: 10.1016/j.measurement.2024.114959

## Curated 包索引
- **未提取**（需先补提取：DOI 10.1016/j.measurement.2024.114959，或按需从 Zotero 获取）
- 相关已有包: `03_2021_liu_tdca`（TRCA 基线）、`28_2024_xu_rhythmic_entrain`（RESS，同族）

## 核心算法
scTRCA（相似性约束 TRCA）在 TRCA 目标函数上加入**相似性约束**（模板与单试次相关最大化）；本文进一步加**时间滤波**（局部时间加权，避免整窗计算忽视局部样本内部结构）：

1. TRCA 基座（最大化试次间协方差）
2. 相似性约束项：加入空间滤波后模板-试次相似性
3. 时间局部权重：对时间样本加权（局部窗口）
4. 求解：广义特征分解（约束优化）

## 接口与验证
- 对齐 `TRCA` 类，新增参数（相似性权重 λ、时间窗宽）
- Benchmark 9ch + BETA，多校准块（1-4 block）
- 与 TRCA / scTRCA 对比

## 验收标准
- Benchmark 上优于/持平 TRCA（论文报告改进）
- 结果写 `tasks/methods/M03_strca/results/`
