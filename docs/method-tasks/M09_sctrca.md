# M09 — scTRCA（Similarity-constrained TRCA）

## 论文
- 标题: Similarity-constrained task-related component analysis for enhancing SSVEP detection
- 作者: Sun et al.
- 期刊: Journal of Neural Engineering 18:046022, 2021
- DOI: 10.1088/1741-2552/abfdfa

## Curated 包索引
- 包: `39_2021_sun_sctrca`
- 位置: `curated/papers/03-vep-algorithm/11-spatial-filter/39_2021_sun_sctrca/`

## 核心算法
scTRCA 在 TRCA 目标上加入**相似性约束**：
- 原始 TRCA：最大化试次间协方差（可重复性）
- scTRCA：约束条件使空间滤波后的信号与**模板更相似**（判别性）
- 数学：广义特征分解，其中约束矩阵来自模板-试次相关

## 接口与验证
- 对齐 `TRCA`，新增相似性约束项
- Benchmark 9ch + BETA；多校准块（1-4 block）
- 与 TRCA、sTRCA（M03，加时间滤波版）对照

## 验收标准
- Benchmark 上优于 TRCA（论文报告）；结果写入 `tasks/methods/M09_sctrca/results/`
