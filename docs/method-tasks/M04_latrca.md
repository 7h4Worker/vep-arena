# M04 — LA-TRCA（Latency Aligning TRCA）

## 论文
- 标题: Latency aligning task-related component analysis using wave propagation for enhancing SSVEP-based BCIs
- 期刊: IEEE TNSRE, 2022
- DOI: 10.1109/TNSRE.2022.3162029

## Curated 包索引
- **未提取**（DOI 10.1109/TNSRE.2022.3162029，需补提取）

## 核心算法
LA-TRCA 在 TRCA 基础上对齐**通道间视觉潜伏期**：
1. 用波传播模型估计各通道的视觉延迟
2. 对齐后构造任务相关成分（修正的协方差）
3. 获得更精确的相位信息 → 提高模板匹配精度

## 接口与验证
- 对齐 `TRCA`，新增延迟估计/对齐步骤
- Benchmark 9ch（论文主数据集），0.2-1.0 s 窗
- 对照：TRCA、eTRCA、CCA

## 验收标准
- Benchmark 结果与论文 Table 对比；写 `tasks/methods/M04_latrca/results/`
