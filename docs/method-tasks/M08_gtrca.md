# M08 — gTRCA（Group TRCA）

## 论文
- 标题: Group task-related component analysis (gTRCA): A multivariate method for inter-trial reproducibility analysis
- 作者: Tanaka T
- 期刊: Scientific Reports 10:84, 2020
- DOI: 10.1038/s41598-019-56962-2

## Curated 包索引
- 包: `38_2020_tanaka_gtrca`
- 位置: `curated/papers/03-vep-algorithm/11-spatial-filter/38_2020_tanaka_gtrca/`

## 核心算法
gTRCA 把 TRCA 从单被试扩展到**多被试/群组**：
- 最大化"组内试次间可重复性"（跨被试的联合协方差）
- 得到**群体共享空间滤波器** → 可用于新被试（跨被试迁移基础）
- 数学：对多个被试的试次协方差做联合广义特征分解

## 接口与验证
- 新增 `gtrca.py`：`fit` 接受多被试数据（或被试列表），输出共享滤波器
- Benchmark 9ch：单被试 TRCA vs 群体 gTRCA 对照；跨被试场景验证
- 注意数据组织：需要多被试训练集（BL01 的 35 被试可分训练/测试）

## 验收标准
- Benchmark 上群体滤波器 + 个体模板组合优于单被试 TRCA（论文核心结论）
- 结果写入 `tasks/methods/M08_gtrca/results/`
