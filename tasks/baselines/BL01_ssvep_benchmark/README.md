# BL01 — Tsinghua SSVEP Benchmark Baseline

Wang et al. (2017) "A Benchmark Dataset for SSVEP-Based Brain-Computer Interfaces"

## 数据集

- 40 目标 (8–15.8 Hz, 0.2 Hz 间隔)
- 35 被试, 6 blocks
- 64 通道 (默认 9ch 后枕选通道)

## 评估方法

| 方法 | 脚本 |
|------|------|
| CCA, FBCCA, ECCA, TRCA, ETRCA, SSCOR, ESSCOR | `run.py` |
| TDCA | `run_tdca.py` |

## 协议

- 被试内, leave-one-block-out 交叉验证
- 时间窗: 0.2–1.5 s (0.1 s 步进)
- 预处理: 0.5 s cue 跳过, 0.14 s 视觉潜伏期, 50 Hz notch, filterbank (5 sub-bands)

## 产出

`results/` 下:
- `trials.csv` — 逐试次结果
- `summary.csv` — 方法 × 窗口汇总 (accuracy, ITR, SEM)
- `subject.csv` — 被试级汇总
- `figures/` — accuracy curve, ITR curve, heatmap, boxplot

## 历史结果

第一代脚本 (`scripts/run_traditional_benchmark.py`) 的产出在 `results/benchmark_9ch/`。
