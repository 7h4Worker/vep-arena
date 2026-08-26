# M05 — Multi-Objective Optimization High-Pass Spatial Filtering

## 论文
- 标题: Multi-objective optimization-based high-pass spatial filtering for SSVEP-based brain-computer interfaces
- 期刊: IEEE TIM, 2022
- DOI: 10.1109/TIM.2022.3146950

## Curated 包索引
- **未提取**（DOI 10.1109/TIM.2022.3146950）

## 核心算法
多目标优化高通空间滤波：
1. 目标 1：增强与目标模板的相关
2. 目标 2：降低与其他刺激信号的关联（跨刺激约束）
3. 目标 3：抑制容积传导（空间高通）
- 多目标求解（加权和或 Pareto），得到空间滤波器

## 接口与验证
- 对齐 `TRCA`/`SSCOR` 类；Benchmark 9ch
- 写 `tasks/methods/M05_mohp/results/`
