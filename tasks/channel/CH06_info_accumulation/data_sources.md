# 跨数据集数据源引用

> 本分析从 4 个已完成的决策信道分析任务中读取容量表 CSV，不重新计算 MI。

---

## 1. Benchmark (Tsinghua SSVEP Benchmark)

- **来源**: `channel/CH01_dmc_benchmark/results/extended/combined/analysis/capacity_by_method_window_aggregate.csv`
- **数据集**: Tsinghua SSVEP Benchmark (Wang et al. 2017), 35 subjects, 40 targets
- **方法**: CCA, ETRCA
- **窗口**: coarse 0.1–5.0 s（0.1s 步长）+ fine 0.100–0.404 s（2 samples / 8ms 步长），去除重叠后共 87 点
- **预测规模**: 1,461,600 rows；35 subjects × 2 methods × 87 windows × 6 blocks × 40 classes
- **预处理**: 两阶段均固定使用 5.0 s canonical epoch 上下文，再切取目标前缀
- **MI 列名**: `i_uniform`
- **分组**: `method`, `window`, `subject` (本分析使用 subject="all" 聚合行)
- **C0**: log2(40) = 5.322 bits
- **选用条件**: ETRCA (aggregate), CCA (aggregate)
- **编码方案**: 等间距单频 FDMA, 8.0–15.8 Hz, Δf=0.2 Hz

## 2. Binocular AR (Ke 2025)

- **来源**: `baselines/BL06_ssvep_binocular_ar/analysis/decision_channel/tables/capacity_by_condition_method_window.csv`
- **数据集**: Ke 2025 Binocular AR, 14–17 subjects, 8 targets
- **方法**: CCA, FBCCA, TRCA, ETRCA
- **窗口**: 0.1–3.0 s, step 0.1 s (30 points)
- **MI 列名**: `MI_uniform`
- **分组**: `method`, `task` (9 conditions), `window`
- **C0**: log2(8) = 3.0 bits
- **选用条件**: SFSP (单频基线), DFDP (最佳双频), DFDP3 (最优间距)
- **编码方案**: 单频/双频双目编码

## 3. Dual Alpha (GigaDB 102557)

- **来源**: `baselines/BL08_ssvep_dual_alpha/analysis/decision_channel/tables/capacity_by_paradigm_method_window.csv`
- **数据集**: GigaDB 102557, 35 subjects, 40 targets
- **方法**: ETRCA, FBDCCA
- **窗口**: 0.2–2.0 s, step 0.2 s (10 points)
- **MI 列名**: `MI_uniform`
- **分组**: `method`, `paradigm`, `window`
- **C0**: log2(40) = 5.322 bits
- **选用条件**: CA (Checkerboard Arrangement), paradigm="Checkerboard_Arrangment"
- **编码方案**: 双频编码, Freq1 + Freq2

## 4. JFPM / NBRS (Zheng 2024)

- **来源**: `baselines/BL12_cvep_nbrs_jfpm/analysis/decision_channel_coding/tables/capacity_by_paradigm_method_window.csv`
- **数据集**: Zheng 2024 JFPM+NBRS, 100 subjects, 40 targets (JFPM-8) / 8 targets (NBRS-8) / 15 targets (NBRS-15)
- **方法**: FBCCA-CODE, TRCA, MSTRCA
- **窗口**: 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0, 4.0 s (8 points, 非均匀)
- **MI 列名**: `I_uniform`
- **分组**: `paradigm`, `method`, `window`
- **C0**: log2(40) = 5.322 (JFPM-8, NBRS-15) 或 log2(8) = 3.0 (NBRS-8)
- **选用条件**: JFPM-8 / TRCA
- **编码方案**: 频率×相位联合调制 (JFPM) / m序列码分 (NBRS)

---

## 列名映射

| 字段 | Benchmark | Binocular AR | Dual Alpha | JFPM |
|------|-----------|-------------|------------|------|
| MI | `i_uniform` | `MI_uniform` | `MI_uniform` | `I_uniform` |
| BA capacity | `c_ba` | `C_BA` | `C_BA` | `C_BA` |
| C1 (Wolpaw) | `c1` | `C1` | `C1` | `C1` |
| C0 | `c0` | `C0` | `C0` | `C0` |
| accuracy | `accuracy` | `accuracy` | `accuracy` | `accuracy` |
| method | `method` | `method` | `method` | `method` |
| window | `window` | `window` | `window` | `window` |

## 注意事项

- Benchmark 的 aggregate 行: `subject="all"`
- Benchmark extended 的 coarse/fine 在 0.1s 和 0.3s 共 33,600 条重叠预测，已逐条验证完全一致
- Benchmark 的 40×40 plug-in MI 在近机会水平存在有限样本正偏；短窗单点和 8ms 数值导数需谨慎解释
- Binocular AR 的方法名中 ETRCA 对应 Benchmark 的 ETRCA (同一实现)
- JFPM 的 FBCCA-CODE 使用码序列作为参考，不是标准正弦参考的 FBCCA
- 窗口范围差异显著：最细 0.1s 步长 (Binocular AR) → 最粗非均匀 (JFPM)
- 8 类 vs 40 类的 MI 绝对值不可直接比较，需用利用率 MI/C0 归一化
