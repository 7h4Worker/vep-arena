# CH01 数据源与跨任务依赖

> 本任务的主体分析基于自身 runner 产出的预测结果。通道配置对比分析额外引用 BL05 (HD200) 数据。

---

## 自身数据

### 标准 9ch 运行 (run.py)

- **产出**: `results/input/predictions.csv`
- **数据集**: Tsinghua SSVEP Benchmark (Wang et al. 2017), 35 subjects, 40 targets, 9ch occipital
- **方法**: CCA, FBCCA, ECCA, TRCA, ETRCA (可选 SSCOR, ESSCOR)
- **窗口**: 0.2–1.0 s

### 扩展时间轴运行 (run_extended.py)

- **产出**: `results/extended/{coarse_0.1s,fine_8ms,combined}/`
- **窗口**: coarse 0.1–5.0 s (0.1s 步长) + fine 0.100–0.404 s (8ms 步长)
- **方法**: CCA, ETRCA (默认); 可扩展至 5 方法全集
- **注意**: 固定 5.0 s 上下文预处理后切取前缀，避免 filtfilt 边界效应

### 多通道配置运行 (run_extended.py --multichannel)

- **产出**: `results/extended/multichannel/decision_channel/capacity_by_subject_method_channels_window.csv`
- **通道配置**: occipital9, posterior21, posterior32, full64, wholehead32
- **用途**: 通道削减效应分析的基础数据

---

## 跨任务引用

以下数据源被 CH01 的通道对比分析脚本引用：

### BL05 — HD200 离线 TDCA 网格搜索

- **来源**: `baselines/BL05_ssvep_hd_200t/results/offline_tdca_grid/decision_channel/`
- **文件**:
  - `capacity_by_subject_targets_channels_window.csv` — 逐受试者容量
  - `capacity_aggregate.csv` — 聚合容量
- **数据集**: HD-SSVEP 200 target (Chen et al. 2024), 14 subjects, 40/80/120/160/200 targets, TDCA
- **通道配置**: 9, 21, 32, 66
- **窗口**: 100–500 ms
- **引用脚本**: `compare_channel_reduction.py`, `plot_channel_comparison.py`, `show_channel_effect.py` (BL05 自身)

---

## 分析脚本清单

| 脚本 | 类型 | 数据源 | 说明 |
|------|------|--------|------|
| `run.py` | runner | Benchmark 9ch | 标准评估 |
| `run_extended.py` | runner | Benchmark 多窗口/多通道 | 扩展时间轴 + 多通道配置 |
| `analyze.py` | 分析 | 自身 predictions.csv | 容量计算 |
| `plot.py` | 可视化 | 自身 analysis/ | 标准图表 |
| `check_multichannel.py` | 验证 | CH01 multichannel | 数据完整性检查 |
| `compare_channel_reduction.py` | 分析 | CH01 + BL05 | 跨数据集通道削减综合分析 |
| `investigate_accuracy_vs_capacity.py` | 分析 | CH01 multichannel | C_BA vs accuracy 悖论调查 |
| `plot_channel_comparison.py` | 可视化 | CH01 + BL05 | 通道削减效应 6 图 |
