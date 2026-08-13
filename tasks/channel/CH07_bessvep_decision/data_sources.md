# CH07 数据源与跨任务依赖

> 本任务汇集 7 个范式/数据集的决策信道容量结果，进行跨范式信息积累对比分析。
> 所有 CSV 均为各 baseline/task 已产出的聚合容量表，本任务不重新计算 MI。

---

## 自身数据

### JBHI 16t + 35t 容量分析 (analyze_jbhi_capacity.py, analyze_full_decision_channel.py)

- **JBHI16 预测**: `bessvep/results/BS02_16t/full/predictions.csv`
  - 数据集: BesSSVEP 16 target, EFUSIONCA 等方法
- **JBHI35 预测**: `bessvep/results/BS03_35t/final_five_execution_20260730/`
  - 子目录: `tdca_full/`, `periodic_receivers_full/`, `etrca_2s/`
  - 数据集: BesSSVEP 35 target, 多方法对比
- **产出**: `combined_aggregate_capacity.csv`, `jbhi16_capacity.csv`, `jbhi35_capacity.csv`

---

## 跨任务引用

以下数据源被 CH07 的跨范式分析脚本引用：

### CH01 — Benchmark 扩展容量

- **来源**: `channel/CH01_dmc_benchmark/results/extended/combined/analysis/capacity_by_method_window_aggregate.csv`
- **内容**: Tsinghua Benchmark 40 target, CCA/ETRCA, 87 窗口点
- **MI 列名**: `i_uniform`
- **引用脚本**: crossparadigm_comparison, crossparadigm_envelope, crossparadigm_envelope_extended

### CH01 — Benchmark 多通道容量

- **来源**: `channel/CH01_dmc_benchmark/results/extended/multichannel/decision_channel/capacity_by_subject_method_channels_window.csv`
- **内容**: 5 种通道配置 × 5 方法 × 多窗口
- **引用脚本**: crossparadigm_envelope_extended

### BL05 — HD200 离线 TDCA 网格搜索

- **来源**: `baselines/BL05_ssvep_hd_200t/results/offline_tdca_grid/decision_channel/capacity_by_subject_targets_channels_window.csv`
- **内容**: 14 subjects, 40–200 targets × 4 通道配置, TDCA
- **MI 列名**: `c_ba`
- **引用脚本**: crossparadigm_envelope_extended

### BL06 — Binocular AR 决策信道

- **来源**: `baselines/BL06_ssvep_binocular_ar/analysis/decision_channel/tables/capacity_by_condition_method_window.csv`
- **内容**: 14–17 subjects, 8 targets, 9 条件, 4 方法
- **MI 列名**: `MI_uniform`
- **C0**: log2(8) = 3.0 bits
- **引用脚本**: crossparadigm_comparison, crossparadigm_envelope, crossparadigm_envelope_extended

### BL08 — Dual Alpha 决策信道

- **来源**: `baselines/BL08_ssvep_dual_alpha/analysis/decision_channel/tables/capacity_by_paradigm_method_window.csv`
- **内容**: 35 subjects, 40 targets, 双频编码
- **MI 列名**: `MI_uniform`
- **C0**: log2(40) = 5.322 bits
- **引用脚本**: crossparadigm_comparison, crossparadigm_envelope, crossparadigm_envelope_extended

### BL11 — WN-BCI (宽带 c-VEP TDCA)

- **来源**: `baselines/BL11_cvep_wn_tdca/analysis/decision_channel/tables/capacity_for_crosstask.csv`
- **内容**: 160 targets, 宽带白噪声码分多址
- **MI 列名**: `MI_uniform`
- **C0**: log2(160) = 7.322 bits
- **引用脚本**: crossparadigm_comparison

### BL12 — JFPM/NBRS 决策信道

- **来源**: `baselines/BL12_cvep_nbrs_jfpm/analysis/decision_channel_coding/tables/capacity_by_paradigm_method_window.csv`
- **内容**: 100 subjects, JFPM-8 (40 cls) / NBRS-15 / NBRS-8
- **MI 列名**: `I_uniform`
- **C0**: log2(40) = 5.322 bits (JFPM-8)
- **引用脚本**: crossparadigm_comparison, crossparadigm_envelope, crossparadigm_envelope_extended

---

## 分析脚本清单

| 脚本 | 数据源 | 说明 |
|------|--------|------|
| `analyze_jbhi_capacity.py` | BS02, BS03 预测 → 自身 CSV | JBHI 16t/35t 容量计算 |
| `analyze_full_decision_channel.py` | BS02, BS03 预测 → 自身 CSV | 完整决策信道分析 + 图表 |
| `crossparadigm_comparison.py` | CH01, BL06, BL08, BL11, BL12 + 自身 | 7 范式跨范式对比 (5 图) |
| `crossparadigm_envelope.py` | CH01, BL06, BL08, BL12 + 自身 | 容量包络线对比 (5 图) |
| `crossparadigm_envelope_extended.py` | CH01, BL05, BL06, BL08, BL12 + 自身 | 扩展包络线 + HD200 + 多通道 |
