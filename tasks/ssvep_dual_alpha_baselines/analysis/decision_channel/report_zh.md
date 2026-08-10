# Dual-Alpha SSVEP 决策信道分析

> **日期**: 2026-07-09
> **脚本**: `analysis/decision_channel/plot_decision_channel.py`
> **数据**: GigaDB 102557 Dual-Alpha 数据集，official_baselines 全量结果
> **分析**: 混淆矩阵 → P(Y|X) 转移矩阵 → MI (uniform) + BA capacity
> **方法**: ETRCA + FBDCCA，窗口 0.2-2.0s (10 步)

---

## 1. 问题定位

Dual-Alpha 数据集使用双频编码方案：每个目标由两个频率 (Freq1, Freq2) 联合定义，40 个目标覆盖 8-16 Hz 频率范围。三种显示范式提供了相同编码方案在不同视觉呈现方式下的对比：

- **Checkerboard Arrangement (CA)**：左右棋盘格分别闪烁 Freq1 和 Freq2，9 通道
- **Binocular Vision (BV)**：左右眼分别呈现 Freq1 和 Freq2，9 通道
- **Binocular-Swap Vision (BsV)**：类似 BV 但频率对被随机配对（不同 codebook），64 通道（本分析使用 occipital9 子集）

CA 和 BV 共享相同的 codebook，而 BsV 使用独立的随机配对 codebook。这提供了**同 codebook 不同显示**和**不同 codebook 不同显示**两层对比。

### 数据规模

| 范式 | 被试 | 目标 | 方法 | 窗口 | 通道 | 块数 |
|------|------|------|------|------|------|------|
| CA | 35 | 40 | ETRCA + FBDCCA | 10 | 9 | 5 |
| BV | 35 | 40 | ETRCA + FBDCCA | 10 | 9 | 5 |
| BsV | 35 | 40 | ETRCA | 10 | 9 (从64取) | 5 |

总计 350,000 prediction 行。

---

## 2. 双频编码方案

![Codebook](figures/fig01_codebook.png)

### CA / BV 共享 Codebook

40 个目标的频率对覆盖 Freq1 ∈ [8.2, 15.4] Hz, Freq2 ∈ [9.0, 16.2] Hz。关键特征：

- **所有目标 Freq2 > Freq1**（点全在对角线上方），最小间距 0.8 Hz
- 频率对在 (Freq1, Freq2) 平面上有规则的网格结构
- 多个目标共享同一 Freq1 值（如 8.2 Hz 对应 5 个不同 Freq2），说明 Freq2 是区分同 Freq1 目标的关键维度

### BsV Codebook

与 CA/BV 完全不同的随机配对方案：
- 频率对散布于整个 (Freq1, Freq2) 平面
- 存在 Freq2 < Freq1 的目标（点在对角线下方）
- 无明显网格结构，目标在频率空间中聚集更密集

---

## 3. 40×40 混淆矩阵

![混淆矩阵](figures/fig02_confusion.png)

ETRCA, 2.0s, 35 subjects:

| 范式 | Acc% | MI | C_BA | 利用率 |
|------|------|-----|------|--------|
| CA | 94.0% | 3.938 | 3.992 | 74.0% |
| BV | 88.3% | 3.589 | 3.601 | 67.4% |
| BsV | 78.6% | 3.061 | 3.098 | 57.5% |

**CA > BV > BsV**，与原文报告一致。

混淆矩阵的结构差异：
- **CA**：对角线主导，off-diagonal 混淆稀疏
- **BV**：类似 CA 但对角线稍弱，部分目标有可见混淆
- **BsV**：对角线明显更弱，off-diagonal 混淆更广泛，特别是低编号目标之间

---

## 4. MI 与 BA Capacity 曲线

![MI 曲线](figures/fig03_mi_curves.png)

### 范式排序

CA ETRCA > BV ETRCA > BsV ETRCA，在所有窗口下一致。MI 随窗口单调递增但增速递减，2.0s 时曲线趋于饱和。

### FBDCCA 表现

- **CA FBDCCA**：MI = 1.063 (2.0s)，远低于 ETRCA 的 3.938。Acc 仅 27.7%。
- **BV FBDCCA**：MI = 2.390 (2.0s)，与 ETRCA 差距较小。Acc = 64.6%。
- **BsV**：无 FBDCCA 公开参考结果。

FBDCCA 在 CA 上的极差表现是重要发现：尽管 FBDCCA 专门为双频 SSVEP 设计（使用 Freq1 和 Freq2 的正弦参考），其准确率仅略高于随机 (1/40 = 2.5%)。这可能是因为棋盘格刺激产生的 EEG 响应模式与双眼独立刺激不同——空间叠加而非双眼分离。

### MI vs C_BA 差距

与 Binocular AR (8 类) 不同，40 类信道的 BA capacity 略高于 MI uniform：
- CA: C_BA = 3.992 vs MI = 3.938 (差距 +0.054)
- BV: C_BA = 3.601 vs MI = 3.589 (差距 +0.012)
- BsV: C_BA = 3.098 vs MI = 3.061 (差距 +0.037)

BA 优化通过调整输入分布（降低高混淆目标的权重）可以额外获得 0.01-0.05 bits。效果不大，说明 40 类混淆矩阵的错误结构仍相对对称。

---

## 5. 方法比较：ETRCA vs FBDCCA

![方法比较](figures/fig05_method_comparison.png)

### 聚合结果

| 范式 | ETRCA MI | FBDCCA MI | 差距 |
|------|----------|-----------|------|
| CA | 3.938 | 1.063 | -2.875 |
| BV | 3.589 | 2.390 | -1.199 |

ETRCA 在两个范式上都大幅领先 FBDCCA。

### 被试级分析

Per-subject scatter (panel b) 显示 ETRCA 几乎对所有被试都优于 FBDCCA：
- **CA**（蓝点）：全部在对角线下方，ETRCA 绝对优势
- **BV**（绿点）：大部分在对角线下方，少数接近对角线

### 解读

FBDCCA 使用 Freq1 和 Freq2 的正弦参考信号，只能匹配预设的谐波成分。ETRCA 使用数据驱动模板，能够学习双频刺激实际诱发的完整响应模式——包括两个频率的谐波、互调产物、以及个体特异的空间模式。

BV 上 FBDCCA 表现较好的原因可能是：双眼独立呈现 Freq1 和 Freq2 时，视觉皮层确实产生了较为分离的两个频率响应，FBDCCA 的参考模型更匹配。而 CA 的棋盘格刺激使两个频率在空间上叠加，产生更复杂的响应模式。

---

## 6. 频率间距与目标准确率

![频率间距](figures/fig06_freq_gap_analysis.png)

### |Freq2 - Freq1| vs 目标准确率

对每个范式，计算各目标的 |Freq2 - Freq1| 并与该目标的准确率关联：

- **CA**：频率间距与目标准确率无显著线性关系（r 较小）
- **BV**：类似 CA
- **BsV**：同样无强相关

频率间距不是决定单个目标准确率的主要因素。更可能的因素包括：
- 与其他目标频率的码间距离（编码空间中的最近邻距离）
- 两个频率各自在个体 SSVEP 频率响应函数中的位置

---

## 7. 被试级分析

![被试级](figures/fig04_per_subject.png)

### 范式间个体差异

| 范式 | MI 均值 | MI 范围 |
|------|---------|---------|
| CA | ~5.1 | 4.1-5.3 |
| BV | ~4.9 | 3.0-5.3 |
| BsV | ~4.6 | 2.9-5.2 |

CA 的被试级 MI 方差最小，BsV 最大。

### CA vs BV（同 codebook，不同显示）

54% 的被试 CA 优于 BV——差异不大，但聚合结果 CA 明显优于 BV。说明 CA 的优势在高表现被试中更大。

### CA vs BsV（不同 codebook，不同显示）

74% 的被试 CA 优于 BsV，差距比 CA vs BV 更显著。codebook 差异和通道配置差异共同作用。

---

## 8. 核心发现

1. **显示方式影响信道质量**：同一 codebook 下 CA (74.0%) > BV (67.4%)。棋盘格显示优于双眼分离显示。

2. **Codebook 结构影响信道质量**：BsV 的随机配对 codebook (57.5%) 低于 CA/BV 的结构化 codebook (74.0%/67.4%)，但这一对比混杂了通道配置差异。

3. **ETRCA >> FBDCCA**：模板方法在双频编码中的决定性优势。FBDCCA 的正弦参考模型无法捕获双频刺激的完整神经响应。

4. **40 类信道利用率**：CA ETRCA 2.0s 达到 74% 利用率 (3.94/5.32 bits)。与 JFPM 数据集 (FBCCA-CODE 达 76.3%) 处于同一水平。

5. **BA 优化空间有限**：C_BA - MI_uniform < 0.05 bits，说明 40 类错误结构相当对称。

---

## 9. 限制与下一步

1. **BsV 的 codebook 和通道配置均不同**：无法分离 codebook 效应和通道数效应。需要 BsV 使用 CA/BV codebook 的对照实验（数据集不包含）。

2. **FBDCCA 实现差异**：当前 FBDCCA 使用 MNE-FIR 滤波后端，与原文可能有细微差异。CA 上的极差表现需要进一步排查是否有实现问题。

3. **无原始 EEG 分析**：无法验证频域响应模式，特别是 CA 的空间叠加 vs BV 的双眼分离在 EEG 层面的差异。

4. **Per-subject MI 估计**：40×40 矩阵每行仅 5 个样本（5 块），使用极小 Laplace 平滑 (alpha=1e-6)。估计不可避免地有较大方差。

5. **下一步**：
   - 与 Binocular AR 的跨数据集比较
   - Bootstrap 置信区间
   - 码间距离与混淆概率的关系（类似 JFPM 分析中的 code similarity vs confusion 分析）

---

## 附录：图表清单

| 文件 | 说明 |
|------|------|
| `fig01_codebook.png` | 3 范式的双频 codebook 散点图 |
| `fig02_confusion.png` | 3 范式 40×40 混淆矩阵 (ETRCA, 2.0s) |
| `fig03_mi_curves.png` | MI 和 BA capacity vs 窗口 (3 范式 × 方法) |
| `fig04_per_subject.png` | 被试级 MI 分布 + 范式间散点 |
| `fig05_method_comparison.png` | ETRCA vs FBDCCA: 柱状图 + 被试散点 |
| `fig06_freq_gap_analysis.png` | |Freq2-Freq1| vs 目标准确率 |

| 表格 | 说明 |
|------|------|
| `capacity_by_paradigm_method_window.csv` | 完整容量表 (50 行) |
| `per_subject_mi.csv` | 被试级 MI (2 方法, 2 窗口) |
