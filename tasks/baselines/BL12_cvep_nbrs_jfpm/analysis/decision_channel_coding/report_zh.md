# cVEP NBRS/JFPM 编码空间与决策信道初步分析

> 状态：2026-07-07 支线分析草稿。  
> 本报告只使用当前 cVEP task 的正式复现结果和原始 supplementary codebook，不重新跑分类器。

## 1. 问题定位

Zheng et al. 2024 的数据集不是普通频率标记 SSVEP 的简单重复，而是把 40 个目标映射到三类刺激序列：

- `NBRS-15`：15-25 Hz 窄带随机序列。
- `NBRS-8`：8-16 Hz 窄带随机序列。
- `JFPM-8`：8.0-15.8 Hz 的 joint frequency-phase modulation，频率间隔 0.2 Hz，相邻相位间隔 0.5 pi。

因此这里有两个层面的分析对象：

1. 编码空间：40 个目标的刺激序列本身是否可分、相似结构如何。
2. 决策信道：经过 FBCCA-CODE / TRCA / MSTRCA 后，真实目标到预测目标的 40x40 混淆矩阵保留了多少信息。

这份支线报告先把二者放在同一套图里，作为后续接入完整 Costa / BA / 码本裁剪分析的 cVEP 版本入口。

## 2. 数据与协议

- 数据：Tsinghua cVEP NBRS/JFPM 2024。
- 被试：100。
- 目标：40。
- 通道：`occipital9`，即 PO/O 区 9 通道。
- 每个 paradigm：3 blocks，leave-one-block-out。
- 窗口：0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0, 4.0 s。
- 方法：`FBCCA-CODE`, `TRCA`, `MSTRCA`。
- ITR 时间：`window + 0.5 s`，与当前 cVEP task 复现约定一致。

当前复现主结果已经对齐原文关键数值：MSTRCA 在 NBRS-15 / NBRS-8 / JFPM-8 的 4s accuracy 分别为 83.73%, 89.56%, 87.07%，对应原文 83.85%, 90.27%, 87.65%。FBCCA-CODE 的 4s accuracy 分别为 90.60%, 86.21%, 91.35%，也与原文 FBCCA 基本一致。

## 3. 编码序列原型

![编码原型与码间相关](figures/fig01_code_prototypes_and_similarity.png)

图 1 左侧显示每个 paradigm 的前三个目标在一个 120-frame 周期内的编码原型；右侧显示 40 个目标之间的相关矩阵。

NBRS 的设计逻辑是：先生成窄带随机序列，再用码间互相关阈值筛选可保留码字。当前 supplementary codebook 的统计也反映了这一点：

| paradigm | mean abs offdiag r | p95 abs offdiag r | max abs offdiag r |
|---|---:|---:|---:|
| NBRS-15 | 0.1266 | 0.2367 | 0.2496 |
| NBRS-8 | 0.1479 | 0.2842 | 0.3000 |
| JFPM-8 | 0.1115 | 0.5002 | 0.5657 |

NBRS-15 和 NBRS-8 的最大绝对相关分别接近 0.25 和 0.30，这与原文描述的筛选阈值一致。JFPM-8 则呈现明显的邻近频率/相位带状结构，说明它的“码间相似性”主要来自规则频率相位网格，而不是随机码本筛选。

## 4. 基础性能复现

![ACC 与 ITR](figures/fig02_acc_itr_by_paradigm_method.png)

图 2 给出三种 paradigm 下三种方法的 accuracy 和 ITR。几个现象比较清楚：

- `FBCCA-CODE` 在长窗口上最稳定，4s accuracy 在三个 paradigm 上都接近或超过 86%。
- `MSTRCA` 的最佳 ITR 往往出现在更短窗口：NBRS-8 为 0.8s，NBRS-15/JFPM-8 为 1.2s。
- `TRCA` 作为非 multi-stimulus 版本明显低于 MSTRCA，说明这个数据集里跨邻近目标学习 spatial filter 是关键。

最佳 ITR 摘要如下：

| paradigm | method | best window | accuracy | ITR |
|---|---|---:|---:|---:|
| NBRS-15 | FBCCA-CODE | 1.6s | 69.02% | 85.00 |
| NBRS-15 | MSTRCA | 1.2s | 59.29% | 87.22 |
| NBRS-8 | FBCCA-CODE | 2.0s | 65.96% | 67.23 |
| NBRS-8 | MSTRCA | 0.8s | 56.92% | 104.88 |
| JFPM-8 | FBCCA-CODE | 2.0s | 74.58% | 79.86 |
| JFPM-8 | MSTRCA | 1.2s | 61.60% | 91.59 |

这里需要注意：最佳 ITR 点不等于最高 accuracy 点。对在线系统来说，短窗口中等 accuracy 可能比长窗口高 accuracy 提供更高吞吐。

## 5. 决策信道容量

![BA 容量](figures/fig03_ba_capacity_by_paradigm_method.png)

图 3 把每个 `paradigm x method x window` 的聚合混淆矩阵视为离散决策信道，计算 BA capacity。这里的单位是 bit/selection，不除以时间。

当前第一版用 0.5 的 Laplace 平滑避免稀疏行导致数值不稳定。`C0 = log2(40) = 5.322` bit/selection 是理想 40 类无噪声上限。

最高 BA 容量都出现在 4.0s：

| paradigm | method | C_BA | I_uniform | accuracy |
|---|---|---:|---:|---:|
| NBRS-15 | FBCCA-CODE | 4.003 | 3.990 | 90.60% |
| NBRS-8 | MSTRCA | 3.928 | 3.920 | 89.56% |
| JFPM-8 | FBCCA-CODE | 4.061 | 4.045 | 91.35% |

一个重要现象是：部分高准确率条件下，传统对称错误公式 `C1` 会高于 BA 容量。这不是 BA 的问题，而是 `C1` 只看平均 accuracy 并假设错误均匀分布；真实混淆矩阵存在非均匀、非对称和目标依赖错误时，`C1` 可能高估实际决策信道容量。

![容量时间权衡](figures/fig04_capacity_time_tradeoff.png)

图 4 使用 `60 / (T + 0.5) * C_BA(T)` 做容量时间权衡。它和传统 ITR 曲线相似，但更直接来自 40x40 混淆矩阵，而不是只来自平均 accuracy。

## 6. 混淆矩阵与编码相似性

![最佳 ITR 混淆矩阵](figures/fig05_best_itr_confusion_matrices.png)

图 5 显示每个 paradigm/method 的最佳 ITR 窗口下混淆矩阵。MSTRCA 虽然在短窗口 ITR 高，但对应 accuracy 仍处在 56%-62% 区间，因此 off-diagonal 混淆仍然可见。FBCCA-CODE 在 2s 左右有更高 accuracy，但时间代价更高。

![码间相似性与错误](figures/fig06_code_similarity_vs_confusion.png)

图 6 把 off-diagonal 的码间相关与 off-diagonal 混淆概率配对。当前 MSTRCA 最佳 ITR 窗口下：

- NBRS-15: r = 0.417
- NBRS-8: r = 0.417
- JFPM-8: r = 0.240

这说明在 NBRS 两个随机窄带码本中，码间相似性已经能解释一部分错误结构。JFPM-8 的错误结构与码间相关的线性对应较弱，可能是因为它的可分性更多受频率分辨率、相位锁定和解码器对相位信息利用方式影响。

## 7. 对本地理论体系的含义

这组结果给 cVEP 分析提供了一个很明确的分层：

1. 刺激设计层：NBRS 是“受频段约束的随机码本 + 互相关筛选”，JFPM 是“规则频率相位网格”。
2. 神经响应层：相同码本经过个体 EEG、通道选择、窗口长度和预处理后，形成可测的 response space。
3. 解码决策层：分类器把响应压缩为 40 类预测，形成 40x40 决策信道。
4. 信息论层：BA capacity、uniform MI、传统 `C1` 的差异，反映输入分布优化、真实错误结构和传统 ITR 假设之间的偏差。

因此 cVEP 不是只看“哪种方法 accuracy 高”，而是可以进一步问：

- 哪些码字在物理刺激空间中接近？
- 哪些码字在 EEG 决策空间中仍然接近？
- BA 最优输入分布是否会压低某些高混淆目标？
- 40-target 真实 EEG 和 1000-codebook 仿真之间，码本容量是否一致外推？

这些问题正好可以作为后续 Costa-style codebook pruning 和 HD-200 / Benchmark 决策信道分析的连接点。

## 8. 限制与下一步

- 本报告是 task 内支线分析，当前只使用聚合混淆矩阵，没有做 subject-level bootstrap 置信区间。
- BA 容量当前基于 Laplace 平滑后的聚合矩阵，后续应补 subject-level capacity、SEM 和显著性比较。
- 目前没有重新跑 1000-target 仿真，只读取 40-target 真实 EEG codebook；1000-codebook 可先做纯码间距离和参考模板可分性分析。
- 码间相似性这里只用了 120-frame 第一个周期的 Pearson correlation；后续应加入频谱距离、循环移位相关、reference-template CCA distance。
- 如果要和 Benchmark 决策信道报告统一，下一版应直接复用 `vep_arena/channel` 的完整输出 schema，并补 `q_star`、`C_BA - I_uniform`、codebook pruning 曲线。
