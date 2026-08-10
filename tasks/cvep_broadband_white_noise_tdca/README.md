# Broadband White Noise — Preliminary Code Sweep

> **重要**：Zenodo 公开数据仅包含论文的 **preliminary experiment（单目标码扫描）**，
> 不是论文的主实验（20人 offline WN vs JFPM BCI 比较）。

## 论文信息

- **论文**: Shi et al. 2024, "Estimating and Approaching the Maximum Information Rate of Noninvasive Visual Brain-Computer Interface", NeuroImage 289, 120548
- **Zenodo**: `https://zenodo.org/records/8300517`, DOI `10.5281/zenodo.8300517`
- **License**: CC-BY-4.0

## 实验性质（来自原文 Section 2.6）

论文包含三个实验，Zenodo 只公开了第一个：

| 实验 | 被试 | 范式 | 目标 | 用途 | Zenodo |
|------|------|------|------|------|--------|
| **Preliminary** | 10人 | **单目标**注视，160 WN codes 逐一呈现 | 估计信道容量上下界 + 选码 | **有** |
| Offline comparison | 20人 | 40目标 BCI 阵列，WN vs JFPM | 宽带 vs 窄带 BCI 性能对比 | 缺失 |
| Online speller | 部分人 | 40目标在线拼写 | 在线验证 | 缺失 |

**Preliminary 实验的关键特征**：

- 被试注视屏幕中央 **一个大目标**（50×50 cm），不是目标阵列
- 每个 trial 呈现 160 条 WN 码中的一条，1s 刺激 + 0.5s 间隔
- 6 blocks × 160 trials/block = 每人 960 trials
- 62 导 EEG，1000 Hz（pkl 文件中降采样至 250 Hz）
- 用途：评估 160 条白噪声码的**可区分性**，以及估计视觉通路的信息率上下界

**这不是 BCI 选择任务**——没有目标阵列，没有注意力选择，没有 gaze shift。
TDCA 分类结果反映的是**单目标条件下码序列的神经可区分性**，
不等同于多目标 BCI 系统的在线选择性能（后者还涉及注意竞争、目标间干扰等）。

## 论文核心贡献

论文的主要贡献不是分类准确率，而是**信息论框架**：

1. **视觉通路建模为通信信道**：source (stimulus) → channel (RGC-LGN-V1) → receiver (EEG)
2. **频谱资源 SNR(f)** 决定信道容量：I = ∫ log₂(1+SNR(f)) df
3. **上界** 63±20 bps（evoked response SNR）、**下界** 25±3 bps（stimulus reconstruction / TRF）
4. **宽带 WN 比窄带 SSVEP 激活更多频谱资源**（alpha+beta 全覆盖 vs 仅 8–15.8 Hz）
5. **FDMA vs CDMA**：SSVEP = 频分多址，WN = 码分多址，后者因覆盖更宽频带而信息率更高

## 本地数据

```text
D:/ProjData/datasets/cvep_broadband_white_noise_bci_zenodo8300517
```

| 路径 | 内容 |
|------|------|
| `data/seperate/sweep/S_*.pkl` | 10 名被试的 preliminary 数据 |
| `data/stimulation/sweep/STI.mat` | 160×60 白噪声刺激矩阵 |
| `data/seperate/compare/` | **空**（离线比较数据未上传） |
| `core/spatialFilters.py` | Zenodo TDCA 实现 |
| `info/bounds.py`, `info/mutualINFO.py` | 信息率上下界计算 |

S1 有 2 个 WN session (共 1920 trials)，S2–S10 各 1 个 session (960 trials)。
每个 session: 960 × 64ch × 250Hz，160 classes × 6 reps。

## 已完成的分析

### Reference TDCA 码区分性分析

160 码的 leave-one-rep-out 分类（论文 Fig 3g 的复现）：

| Window | Acc | 说明 |
|-------:|----:|------|
| 0.1s | 49.9% | 远超 chance (0.625%)，短窗口已有强区分性 |
| 0.2s | 85.1% | |
| 0.3s | 95.6% | 论文报告 91%（参数可能有差异） |
| 0.4s | 98.8% | |

**注意**：这些准确率是单目标注视条件下的码区分性，不等于 BCI 选择准确率。

结果目录：`results/reference_tdca_full/`

### 频谱资源与信息率分析（论文核心框架复现）

`analysis/spectrum_resources/` — 直接调用 Zenodo 原始 core 代码，忠实复现论文信息率上下界：

| 指标 | 本复现 | 论文 |
|------|-------|------|
| 上界（evoked SNR） | **60.8 ± 21.7 bps** | 63 ± 20 bps |
| 下界（stimulus reconstruction） | **25.6 ± 2.9 bps** | 25 ± 3 bps |

关键发现：白噪声码的判别信息主要落在 **beta 段（13–30 Hz）**而非 alpha；上/下界比 ≈ 2.4
（线性 TRF 解码器仅提取约 42% 的理论信息）；TRF 潜伏期 ~125 ms。详见 `spectrum_resources/report_zh.md`。

### 分析目录

- `analysis/spectrum_resources/` — ✅ 频谱资源/信息率上下界（论文核心框架，单目标数据的正确分析）
- `analysis/reference_tdca/` — 分类结果可视化（准确率、混淆矩阵、session 热图）
- `analysis/decision_channel/` — ⚠️ 已弃用，决策信道分析对单目标数据不适用

## 适合此数据的分析方向

1. ✅ **信息率上下界复现** — 已完成，见 `analysis/spectrum_resources/`
2. ✅ **频谱资源分析（SNR(f)、频段贡献）** — 已完成，同上
3. ✅ **TRF 估计** — 已完成，同上（h(τ) 及频率响应 H(f)）
4. **码区分性结构**：160×160 混淆结构、码间相关与混淆的关系（reference_tdca 已部分覆盖）
5. **码子集选择**：从 160→40 的最优子集策略（待作者提供主实验数据后更有意义）
