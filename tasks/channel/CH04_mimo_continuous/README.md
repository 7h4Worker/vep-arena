# Channel Capacity Analysis: Continuous MIMO Extension

## 前因后果 (Background & Motivation)

### 已完成的离散信道容量分析

当前 vep_arena 项目已完成了基于 **决策层离散无记忆信道 (DMC)** 的 Blahut-Arimoto 容量分析框架：

```
stimuli X ∈ {1,...,M} → [BCI system] → decisions Y ∈ {1,...,M}
                              ↓
              confusion matrix C[i,j] = count(Y=j | X=i)
                              ↓
              P(Y|X) = normalize_confusion(C)
                              ↓
              C_BA = max_q I(X;Y)  (Blahut-Arimoto)
```

#### 已有的分析任务及产出位置：

| Task | 内容 | 核心输出 |
|------|------|---------|
| `benchmark_decision_channel_capacity/` | Benchmark 40-target DMC 容量 (5 methods × 50 windows) | `results/extended/combined/analysis/` |
| `benchmark_multichannel_decision_channel_capacity/` | Benchmark 多通道配置 DMC (5 configs × 5 methods × 50 windows) | → 输出到上者的 `results/extended/multichannel/` |
| `ssvep_hd_200target_tdca_sample/` | HD200 DMC (5 targets × 4 channels × 5 windows × 14 subjects) | `results/offline_tdca_grid/decision_channel/` |
| `ssvep_jbhi_decision_channel/` | 跨范式 DMC envelope 对比 (6-9 paradigms) | `figures_envelope_*/`, `tables/` |

#### 关键发现（推动连续信道分析的动因）：

1. **Benchmark occipital 9ch > full 64ch (在 C_BA 意义下)**
   - 不是统计噪声：35 个被试中 23 个在 9ch 下 C_BA 更高
   - 离散模型给出结果但无法解释机制（为什么减少维度反而提高信息量？）

2. **HD200 表现相反：66ch > 9ch，且差距随 target 数增大**
   - r(targets, loss_9ch) = 0.91-0.99 跨窗口一致
   - 离散模型可以量化差异但无法区分：是信号子空间维度不够？还是噪声增益？

3. **Benchmark τ(full64)=0.351s vs τ(occ9)=0.125s**
   - 64ch 的"慢收敛"到底是参数估计的问题还是物理信道的固有特性？
   - 连续模型可以分离：信号协方差秩 vs 噪声协方差条件数

4. **C_BA ≠ f(accuracy) 的非单调关系**
   - 需要连续层面的信息论量来解释混淆矩阵结构差异的来源

### 离散分析的局限性

```
C_BA = max_q I(X; Y_decision) ≤ I(X; Y_eeg)
```

C_BA 是 **data processing inequality** 下的信息下界——经过分类决策后只能损失信息。
真正的 BCI 信道容量应在 EEG 信号层面度量：

- 离散模型无法区分"信号未被采集"和"信号被采集但未被解码器利用"
- 无法回答"最优解码器能做到多好"（只知道当前解码器有多好）
- 无法解释空间维度变化对信息量的机制性影响

---

## 连续 MIMO 信道容量框架设计

### 物理模型

```
EEG 信号模型:
    Y(t) = H · s(t|x) + n(t)
    
    Y(t) ∈ ℝ^K        : K 通道 EEG 观测
    H ∈ ℝ^{K×D}       : 空间混合矩阵（头模型 leadfield 的投影）
    s(t|x) ∈ ℝ^D      : D 维源信号（条件于刺激 x）
    n(t) ∈ ℝ^K        : 传感器噪声
```

### MIMO 互信息

对于高斯模型：
```
I(X; Y) = h(Y) - h(Y|X)
         = ½ log det(Σ_Y) - ½ E_x[log det(Σ_{Y|x})]

其中:
    Σ_Y = E[Y·Y^T]                           (total covariance)
    Σ_{Y|x} = E[(Y - μ_x)(Y - μ_x)^T | X=x]  (within-class covariance)
    
    => I(X;Y) = ½ log det(Σ_total) - ½ log det(Σ_within)
              = ½ log det(Σ_within^{-1} · Σ_total)
              = ½ Σ_k log(1 + λ_k)
              
    where λ_k are eigenvalues of Σ_within^{-1} · Σ_between
```

### 与 MIMO 通信的对应关系

| BCI | MIMO 通信 | 数学量 |
|-----|-----------|--------|
| 刺激集合 | 发射星座 | X ∈ {1,...,M} |
| EEG 时间段 | 接收向量 | Y ∈ ℝ^{K×T} |
| 信道（头皮-源） | 信道矩阵 | H ∈ ℝ^{K×D} |
| 背景 EEG + 仪器噪声 | 加性高斯噪声 | n ~ N(0, Σ_n) |
| 电极子集选择 | 天线选择 | 选 K' < K 行 of H |
| 空间滤波器 (TRCA) | 接收波束成形 | W^T · Y |

### 分析层级

```
Level 0: 原始 EEG 互信息
    I(X; Y_raw)  -- 存在于 K 通道 × T 时间点的全部信息
    ↓ 空间滤波 (lossy if rank(W) < K)
Level 1: 滤波后互信息  
    I(X; W^T·Y)  ≤  I(X; Y_raw)
    ↓ 决策/量化
Level 2: 离散决策互信息
    I(X; Y_decision) = C_BA  ≤  I(X; W^T·Y)  ≤  I(X; Y_raw)
```

**通道减少的信息论解释：**
- 如果 rank(H·Σ_s·H^T) = r < K：信号子空间只有 r 维，多余的 K-r 维是纯噪声
- 选择 K'=r 个电极（且位于信号子空间投影最大的位置）→ 去掉了纯噪声维度
- 此时 I(X; Y_{K'}) ≈ I(X; Y_K) 但 estimation error 更小（因为参数更少）
- 这精确解释了 Benchmark 9ch > 64ch 现象：信号秩 ≈ 3-5，9ch 枕区电极恰好覆盖

---

## 计算方案

### 方案 A: 基于 epoch 协方差的高斯互信息 (推荐起步)

```python
# 对每个 (subject, channel_config, window) 条件：
#   1. 提取所有 epoch: shape (M*n_trials, K, T)
#   2. 按 class 计算 class-conditional mean: μ_x, shape (M, K, T)
#   3. 计算 within-class covariance: Σ_w = mean_x[ Cov(Y|x) ]
#   4. 计算 total covariance: Σ_t = Cov(Y)
#   5. I_gauss = ½ log det(Σ_w^{-1} · Σ_t)  (bits, /log2)
#   6. 或者用 eigenspectrum: λ_k = eig(Σ_w^{-1} · Σ_t), I = ½ Σ log(1+λ_k)
```

优点：
- 直接用现有 epoch 数据，不需要新实验
- 与 C_BA 可以在相同条件下对比（信息损失 = I_gauss - C_BA）
- 特征值谱直接给出信号子空间维度（有效 rank）

### 方案 B: 基于频域的 per-frequency MIMO 容量

```python
# SSVEP 信号集中在离散频率，可以在频域分解：
#   I(X; Y) ≥ Σ_f I(X; Y(f))  (独立频率分量的和)
#   
# 对每个刺激频率 f_x：
#   SNR(f_x) = |S(f_x)|^2 / PSD_noise(f_x)
#   I(X; Y(f_x)) 可用单频 SISO 模型
#   MIMO 版本: 对每个频率做 K×K 协方差分解
```

优点：
- 利用了 SSVEP 的频域稀疏性
- 可以解释"为什么增加 target 数降低 η"：频率间距减小 → 频域泄漏 → 有效 SNR 下降

### 方案 C: 基于 Fisher 信息的局部容量界

```python
# Fisher 信息矩阵给出参数估计的 Cramér-Rao 下界
# 对于离散刺激集，Fisher 判别分析的 trace/det 给出类间可分性
# J = Σ_w^{-1} · Σ_b 的 eigenspectrum 直接对应 Level 1 的互信息
```

---

## 此 task 的具体目标

### Phase 1: 高斯互信息基线 (Gaussian MI upper bound)

1. 对 Benchmark 和 HD200 的现有 epoch 数据计算 I_gauss
2. 在 5 种通道配置 × 50 个窗口条件下与 C_BA 配对对比
3. 信息损失 = I_gauss - C_BA → 量化解码器效率
4. 特征值谱 → 有效信号维度 vs 通道数关系

**预期回答的问题：**
- Benchmark 64ch 的信号子空间有效秩是多少？（预测 3-8）
- 9ch 的信息损失 vs 64ch 的信息损失谁大？（预测 9ch 损失更小）
- HD200 200-target 的有效秩是否随 target 数增长？

### Phase 2: 频域分解

1. Per-frequency SNR spectrum across channel configs
2. 频率间干扰矩阵（SSVEP harmonics 间的串扰）
3. MIMO capacity per frequency bin → aggregate

### Phase 3: 信息流瀑布图 (Information Cascade)

```
I(X; Y_raw) → [空间滤波损失] → I(X; W^T·Y) → [决策损失] → C_BA
```

在每个 (dataset, config, window) 条件下分解：
- 总可用信息 I_gauss
- 空间滤波后保留的信息
- 决策后保留的信息 (C_BA)
- 各步损失的通道/窗口依赖性

---

## 数据依赖

### 需要的数据（已有）：
- Benchmark epoch 数据：`data_adapters` 可直接加载，~35 subjects × 40 targets × 6 blocks × 64ch
- HD200 epoch 数据：`data_adapters` 可直接加载，~14 subjects × 200 targets × 66ch

### 需要的数据（需确认）：
- Benchmark 原始连续 EEG（用于噪声协方差估计的 pre-stimulus baseline）
- HD200 的 raw 数据路径

### 不需要的：
- 不需要重跑分类实验
- 不需要 MATLAB
- 不需要新数据采集

---

## 与现有分析产物的关系

```
tasks/
├── benchmark_decision_channel_capacity/         ← 离散 DMC (Benchmark, single + multi-channel)
│   ├── results/extended/multichannel/           ← multichannel C_BA 结果
│   ├── figures_channel_comparison_v20260807/    ← 通道对比图（本 session 产出）
│   └── figures_channel_configs_v20260804/       ← MNE 通道位置图
│
├── ssvep_hd_200target_tdca_sample/              ← 离散 DMC (HD200)
│   └── results/offline_tdca_grid/decision_channel/
│
├── ssvep_jbhi_decision_channel/                 ← 跨范式 envelope 对比
│   └── figures_envelope_extended_v20260808/     ← 含 HD200 的 envelope（本 session 产出）
│
├── continuous_mimo_channel_capacity/            ← 【本 task：连续 MIMO 分析】
│   ├── README.md                                 (本文件)
│   ├── run_gaussian_mi.py                        Phase 1 主脚本
│   ├── run_frequency_mimo.py                     Phase 2 频域 MIMO
│   ├── plot_information_cascade.py               Phase 3 信息瀑布
│   ├── results/
│   │   ├── gaussian_mi/                          高斯互信息 CSV + eigenspectra
│   │   ├── frequency_mimo/                       频域 MIMO 容量
│   │   └── cascade/                              信息流分解
│   └── figures/
│       ├── eigenspectrum/                        特征值谱图
│       ├── mi_vs_cba/                            连续 MI vs 离散 C_BA 对比
│       └── cascade/                              信息瀑布图
│
└── benchmark_signal_channel_analysis/           ← 早期探索（已冻结）
```

---

## 当前状态

- [x] 离散 DMC 分析：Benchmark (single-channel) 完成
- [x] 离散 DMC 分析：Benchmark (multi-channel, 5 configs) 完成
- [x] 离散 DMC 分析：HD200 (4 channels × 5 targets) 完成
- [x] 跨数据集通道对比分析完成
- [x] 跨范式 envelope 分析（含 HD200）完成
- [ ] **Phase 1: 高斯互信息** ← next
- [ ] Phase 2: 频域 MIMO
- [ ] Phase 3: 信息瀑布分解

---

## 参考

- Cover & Thomas, *Elements of Information Theory*, Ch. 9 (Gaussian channel)
- Telatar, "Capacity of multi-antenna Gaussian channels", 1999
- Müller-Gerking et al., "Designing optimal spatial filters for single-trial EEG classification", 1999
- Blankertz et al., "Optimizing spatial filters for robust EEG single-trial analysis", 2008
- Wolpaw & Wolpaw, *BCI Principles and Practice*, Ch. 7 (Information transfer)
