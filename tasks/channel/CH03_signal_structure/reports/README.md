# Benchmark 9ch SSVEP 信号信道分析

与 `benchmark_decision_channel_capacity`（决策层分析）互补，本 task 从**原始 EEG 信号**出发，
刻画每个被试在 40 个刺激频率上的信号质量、空间滤波器的影响，以及信号层属性与分类/信道容量的关联。

## 分析内容

### Phase 1: 信号层 profile（`analyze_signal.py`）
- 对 35 人 × 40 频率计算 SNR profile（窄带、谐波累积）和 PLV profile
- 输出 `signal_profile.csv`（1400 行：subject × target）
- 生成 SNR/PLV 热力图、频谱全景图

### Phase 2: 空间滤波器影响（`analyze_filters.py`，待建）
- 比较 CCA/TRCA/ETRCA 空间滤波前后的 SNR 增益
- 空间滤波器 topography 对比
- 滤波后特征相似性矩阵 vs 实际混淆矩阵

### Phase 3: 信号层码本优化（`analyze_codebook.py`，待建）
- 谐波干扰矩阵（纯计算，不依赖分类器）
- 干扰矩阵 vs 实际混淆矩阵的相关性
- 基于 SNR + 干扰的先验码本选择 vs 后验裁剪对比

## 运行

```bash
cd vep_arena
python tasks/benchmark_signal_channel_analysis/analyze_signal.py
python tasks/benchmark_signal_channel_analysis/analyze_signal.py --subjects 1-5 --window 1.0
```

## 依赖

- 数据路径：`D:/ProjData/datasets/ssvep_benchmark`（同 config.py）
- 模块：`vep_arena.signal`（spectrum, snr, plv, utils）
- 可选联动：`vep_arena.channel`（读取 capacity 结果做交叉分析）

## 输出

所有输出到 `results/`（git-ignored）：
- `signal_profile.csv` — subject × target 的 SNR、PLV、谱集中度
- `signal_summary_by_subject.csv` — 被试级聚合
- `signal_summary_by_target.csv` — 频率级聚合
- `harmonic_interference.csv` — 40×40 谐波干扰矩阵
- `figures/` — 分析图表
