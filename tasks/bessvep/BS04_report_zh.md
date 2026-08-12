# EMBC/JBHI 结构化双目码本决策信道综合分析

> 数据日期：2026-07-28  
> Study：`ssvep_embc_jbhi_binocular_codebook_analysis`  
> 输入：三套已完成、匿名化、trial-level prediction bundle  
> 定位：当前接收机与协议条件下的经验决策信道分析，不是脑-EEG 物理信道容量上界

## 0. 研究问题

EMBC9、JBHI16 和 JBHI35 不是三个普通的独立类别分类任务。每个目标都对应一个有序双目码字：

\[
m=(u_L,u_R),
\]

其中左右眼单元分别可以是 void 或一个刺激频率。三个码本构成清晰的规模阶梯：

- EMBC9：`{void, 8.5, 9.5}²`，完整 3×3 笛卡尔积；
- JBHI16：`{void, 11, 12, 13}²`，完整 4×4 笛卡尔积；
- JBHI35：`{void, 11, 12, 13, 14, 15}² - {(void,void)}`，6×6 网格去掉全空码字。

因此，本分析不只问“准确率是多少”，而是问：

1. 这些组合码字经过 EEG 与接收机后保留了多少可恢复信息；
2. 左眼与右眼两个编码维度是否都被保留，是否发生跨眼泄漏或眼别交换；
3. bPRCA/FusionCA 恢复的是频率单元信息、组合信息，还是一般模板信息；
4. 当码本从 9 扩展到 16、35 时，容量、利用率、错误结构和个体差异如何变化；
5. 当前结果对后续实验补充和解码器设计提出什么具体要求。

## 1. 数据、协议与证据边界

| 数据集 | 目标 | 被试 | block | 窗口 | CV | 证据角色 |
|---|---:|---:|---:|---|---|---|
| EMBC9 | 9 | 8 | 20 | 0.5–4.0 s | Arena leave-one-block-out | 论文数据扩展；论文未说明 leave-one-out 单元 |
| JBHI16 | 16 | 13 | 6 | 0.2–2.0 s | 论文六折 leave-one-block-out | 论文复现主证据 |
| JBHI35 | 35 | 6 | 6 | 0.2–2.0 s | 诊断性六折 leave-one-block-out | 未发表扩展 |

三个源 manifest 均为 `complete`：共 1380/1380 units、193680/193680 predictions、0 errors。本 Study 只读取 predictions 和 manifest，不重新读取 EEG，不重新运行分类，也不接触参与者身份映射。

EMBC9 的跨数据集数值只能作为当前 Arena 协议下的分析；JBHI35 只能作为探索性扩展。只有 JBHI16 同时具有明确论文 CV、论文性能锚点和完整 4×4 码本。

## 2. 分析方法

### 2.1 经验决策信道

对每个 `dataset × method × window`，由真实目标和预测目标构造：

\[
P(\hat m\mid m).
\]

计算以下指标：

- `C0 = log2(M)`：无误码码本上限；
- `C1`：只使用平均准确率、假设对称均匀错误的 Wolpaw 单次选择信息；
- `MI_uniform`：真实混淆矩阵在均匀目标先验下的互信息；
- `C_BA`：对同一个经验决策信道优化输入分布后的 Blahut-Arimoto 容量；
- `C_BA/log2(M)`：码本利用率；
- `C_BA-MI_uniform`：仅靠调整输入先验可获得的空间。

主估计使用跨被试 pooled confusion。原因是 JBHI16/35 每名被试每类只有 6 个测试 trial，直接平均被试级插件 MI/C_BA 会产生明显有限样本上偏。被试层结果只用于准确率异质性和配对接收机检验。

### 2.2 两种容量—时间口径

本报告把 `C_T` 明确拆成两种：

\[
C_{T,EEG}=\frac{60C_{BA}}{T_{EEG}},
\]

\[
C_{T,protocol}=\frac{60C_{BA}}{T_{EEG}+T_{shift}}.
\]

前者只衡量 EEG 证据积累速度；后者采用各数据 task 的 ITR 时间分母。EMBC 使用 2.0 s cue/shift，JBHI 使用 0.5 s。`C_T` 是容量—时间折中，不等同于论文基于 `C1` 的 ITR。

### 2.3 双目因子与错误拓扑

每个目标还原为 `(left_freq, right_freq)`，计算：

- 左眼单元准确率与 `I(L; L_hat)`；
- 右眼单元准确率与 `I(R; R_hat)`；
- 跨眼泄漏 `I(L; R_hat)`、`I(R; L_hat)`；
- 联合目标信息 `I((L,R); (L_hat,R_hat))`；
- `joint - left - right` 描述性非加和项；它不是正式 PID synergy；
- 错误分类：眼别交换、void 结构错误、保留一只眼、左右单元均错误；
- 码字族：void-void、单眼、双眼同频、双眼异频。

## 3. 码本几何

![双目码本](results/full/figures/fig01_codebook_lattices.png)

这张图说明三个任务属于同一个结构化编码族，而不是恰好都叫“双频”：

- EMBC9 与 JBHI16 是完整有序笛卡尔积；
- JBHI35 是完整 6×6 网格只删除 void-void；
- `(f1,f2)` 与 `(f2,f1)` 是两个不同消息，眼别本身就是编码维度；
- 同一频率单元被大量目标复用，纯频率存在性不足以确定目标。

这正是 bPRCA 只增强频率单元仍不够、FusionCA 需要同时保留 target-specific/aperiodic 成分的结构原因。

## 4. 决策信道容量与利用率

![容量与利用率](results/full/figures/fig02_capacity_utilization_curves.png)

### 4.1 2.0 s pooled 结果

| 数据集 | 方法 | Acc | MI_uniform | C_BA | 利用率 | C_BA-MI |
|---|---|---:|---:|---:|---:|---:|
| EMBC9 | eTRCA | 75.07% | 1.904 | 1.971 | 62.2% | 0.067 |
| EMBC9 | ebPRCA | 57.43% | 1.140 | 1.171 | 36.9% | 0.031 |
| EMBC9 | eFusionCA | 72.99% | 1.762 | 1.817 | 57.3% | 0.056 |
| JBHI16 | eTRCA | 77.16% | 2.727 | 2.811 | 70.3% | 0.084 |
| JBHI16 | ebPRCA | 79.41% | 2.759 | 2.837 | 70.9% | 0.079 |
| JBHI16 | eFusionCA | **82.21%** | **2.942** | **3.009** | **75.2%** | 0.067 |
| JBHI35 | eTRCA | 32.62% | 1.593 | 1.704 | 33.2% | 0.111 |
| JBHI35 | ebPRCA | 32.70% | 1.561 | 1.722 | 33.6% | 0.161 |
| JBHI35 | eFusionCA | **35.95%** | **1.678** | **1.827** | **35.6%** | 0.149 |

JBHI16 是当前最完整、最稳定的结构化决策信道：在 16 类、4 bit 理想码本中，eFusionCA 保留 2.94 bit 均匀输入信息，BA 容量达到 3.01 bit，利用率 75.2%。

EMBC9 的 9 类理想上限是 3.17 bit。恢复旧论文代码证实的三子带后，eTRCA 达到 62.2% 利用率，eFusionCA 为 57.3%；这说明此前较低结果主要受接收机协议缺失影响，也说明 PRCA/Fusion 扩展在这套小码本上没有超过标准三子带 eTRCA。

JBHI35 的理想上限是 5.13 bit。eFusionCA 在 2 s 恢复 1.68 bit，利用率只有 35.6%。增加码本规模确实增加了绝对可恢复信息，但远没有按 `log2(M)` 同比例增长。

### 4.2 输入分布优化空间

`C_BA-MI_uniform` 在 EMBC9/JBHI16 只有约 0.05–0.08 bit，均匀目标使用接近合理。JBHI35 的差距增至 0.15 bit，表明其混淆更不对称，理论上可通过非均匀目标使用或结构化子码本提高容量；但这首先是诊断信号，不应直接解释为实际界面应降低某些目标出现概率。

## 5. 容量—时间折中

![容量时间曲线](results/full/figures/fig03_capacity_time_curves.png)

采用 protocol 时间分母时，各方法的 pooled `C_T` 峰值为：

| 数据集 | 方法 | 峰值窗口 | C_T,protocol |
|---|---|---:|---:|
| EMBC9 | eTRCA | 1.0 s | 31.31 bit/min |
| EMBC9 | ebPRCA | 0.5 s | 28.37 bit/min |
| EMBC9 | eFusionCA | 0.5 s | 33.61 bit/min |
| JBHI16 | eTRCA | 0.2 s | 161.94 bit/min |
| JBHI16 | ebPRCA | 0.4 s | 157.86 bit/min |
| JBHI16 | eFusionCA | 0.4 s | **162.17 bit/min** |
| JBHI35 | eTRCA | 0.2 s | 103.97 bit/min |
| JBHI35 | ebPRCA | 0.2 s | 94.40 bit/min |
| JBHI35 | eFusionCA | 0.2 s | 101.18 bit/min |

这些数值高于或不同于论文 ITR 是正常的，因为这里使用真实混淆矩阵的 `C_BA`，而论文 ITR 使用由准确率得到的 `C1`。更有意义的观察是峰值位置：

- JBHI16 的 eTRCA 在 0.2 s 已达到最高速率，eFusionCA 在 0.4 s 用更多证据换来更高单次容量，同时保持几乎相同的峰值速率；
- JBHI35 在 0.2 s 的速率峰值不代表高可靠性，此时码本利用率仍低，属于“大码本带来的低可靠信息流”；
- EMBC 的 2 s cue/shift 使短 EEG 窗口收益被较长固定开销稀释，不能与 JBHI 的 0.5 s 分母直接比较。

## 6. 接收机增益：bPRCA 与 FusionCA

![接收机增益](results/full/figures/fig04_receiver_gain.png)

相对 eTRCA 的 pooled `ΔMI_uniform`：

| 数据集 | 窗口 | ebPRCA - eTRCA | eFusionCA - eTRCA |
|---|---:|---:|---:|
| EMBC9 | 1.0 s | -0.300 | +0.058 |
| EMBC9 | 2.0 s | -0.527 | +0.019 |
| JBHI16 | 1.0 s | -0.036 | +0.176 |
| JBHI16 | 2.0 s | +0.031 | **+0.214** |
| JBHI35 | 1.0 s | -0.051 | +0.076 |
| JBHI35 | 2.0 s | -0.032 | +0.085 |

结论不是“PRCA 普遍优于 TRCA”。当前证据恰好相反：

1. ebPRCA 单独使用时只在 JBHI16 的长窗口略有收益，在 EMBC9 明显弱于 eTRCA；
2. eFusionCA 在三套数据、1 s 和 2 s 均不低于 eTRCA；
3. 最大且最稳定的 Fusion 增益出现在其原生设计对象 JBHI16；
4. 说明频率周期成分与 target-specific/aperiodic 成分是互补信息源，频率单元增强不能替代整段任务模板。

2 s 被试配对准确率检验进一步支持这一点：

- EMBC9：eFusionCA 相对 eTRCA +0.76 个百分点，`p=0.673`；
- JBHI16：+5.05 个百分点，`p=0.00098`；
- JBHI35：+3.33 个百分点，`p=0.254`，受 n=6 和极端异质性限制。

## 7. 左右眼因子与联合信息

![双目因子信息](results/full/figures/fig05_factor_information.png)

eFusionCA、2 s pooled 结果：

| 数据集 | 左眼 Acc | 右眼 Acc | 左眼 MI | 右眼 MI | 联合 MI | 跨眼 MI 范围 |
|---|---:|---:|---:|---:|---:|---:|
| EMBC9 | 78.75% | 78.40% | 0.647 | 0.642 | 1.684 | 0.018–0.020 |
| JBHI16 | 86.86% | 88.46% | 1.272 | 1.334 | 2.942 | 0.0037–0.0039 |
| JBHI35 | 47.94% | 46.59% | 0.389 | 0.359 | 1.678 | 0.025–0.027 |

JBHI16 的左右眼边缘通道都很清晰，跨眼泄漏接近零。这说明模型并不是只识别“出现了哪些频率”，而是保留了有序 `(left,right)` 结构。右眼 MI 略高于左眼，但差值目前只能作为待复现实验现象，不能直接归因于生理眼优势。

联合 MI 高于左右边缘 MI 之和的现象在三套数据均存在，JBHI35 尤其明显。这里不能直接称为神经协同信息：联合 target identity 还包含具体组合、码本约束、模板波形、beat/intermodulation 和非周期成分。正式 synergy 需要 PID 或条件互信息设计。

## 8. 错误拓扑与码字族

![错误拓扑](results/full/figures/fig06_error_topology.png)

eFusionCA、2 s 的错误构成：

| 数据集 | 眼别交换 | void 结构 | 保留一眼 | 两眼均错 |
|---|---:|---:|---:|---:|
| EMBC9 | **31.2%** | 61.8% | 6.0% | 1.0% |
| JBHI16 | 8.1% | **67.1%** | 19.4% | 5.4% |
| JBHI35 | 4.1% | 43.7% | 22.6% | **29.6%** |

EMBC9 的典型困难是眼别交换：错误预测中约三分之一把 `(f1,f2)` 识别为 `(f2,f1)`。这说明它能检测到频率组合，却较难恢复有序眼别，是“组合存在信息强、眼别信息弱”的信道。

JBHI16 的眼别交换已降到 8.1%，主要错误转为 void 结构判别和只保留一个眼单元。这与论文的设计动机一致：bPRCA/FusionCA 的价值不只是频率检测，而是区分 void、单眼、同频双眼和异频双眼条件。

JBHI35 中 29.6% 的错误左右单元均错，显著高于前两套数据。它不是简单的眼别或 void 问题，而是部分被试的整体信号/接收机失效。

![码字族信道](results/full/figures/fig08_category_confusion.png)

eFusionCA、2 s 的码字族对角识别率：

| 数据集 | 单眼 | 双眼同频 | 双眼异频 | void-void |
|---|---:|---:|---:|---:|
| EMBC9 | 79.4% | 83.1% | 91.6% | 61.3% |
| JBHI16 | 79.9% | 83.8% | 93.2% | 64.1% |
| JBHI35 | 49.2% | 45.6% | 69.4% | 不存在 |

双眼异频是三套数据中最容易识别的类别族。void-void 在 EMBC/JBHI16 中反而最弱，说明“无目标刺激”并不是天然容易的类别：外周刺激、全场响应、ERP 和噪声都可能使它被误判为有频率条件。

## 9. 个体差异

![个体准确率](results/full/figures/fig07_subject_accuracy.png)

eFusionCA、2 s：

| 数据集 | 中位准确率 | 范围 |
|---|---:|---:|
| EMBC9 | 74.2% | 42.8–83.3% |
| JBHI16 | 82.3% | 60.4–94.8% |
| JBHI35 | **20.5%** | **1.0–93.8%** |

JBHI35 的群体均值 35.95% 掩盖了最关键事实：它不是“所有人中等偏低”，而是少数高质量被试与多名近机会水平被试的混合。后续若补实验，首要任务不是继续堆分类器，而是建立可解释的采集质量、视觉状态、设备配准和 subject eligibility 指标。

### 9.1 JBHI35：低分是标签错还是信号差

![JBHI35 被试诊断](results/jbhi35_subject_quality_audit/figures/subject_quality_diagnostics.png)

| 被试 | Arena eTRCA 2 s | Oz 目标频率效应 | Oz 目标频率 ITPC | 无序频率集合命中 |
|---|---:|---:|---:|---:|
| S01 | 6.67% | 5.59 dB | 0.339 | 40.48% |
| S02 | 7.14% | 0.22 dB | 0.356 | 14.29% |
| S03 | 2.86% | 1.57 dB | 0.334 | 32.38% |
| S04 | 87.14% | 1.95 dB | 0.617 | 35.24% |
| S05 | 28.10% | 3.66 dB | 0.682 | 32.86% |
| S06 | 63.81% | 3.00 dB | 0.605 | 27.14% |

35 类机会准确率为 2.86%；按目标包含 1 个或 2 个不同活动频率分层后，无序频率集合的机会命中率为 14.29%。S02 的频率效应和频率集合命中都处于机会附近，符合真实低质量或源标签异常。S01/S03 则不同：目标频率效应明确、频率集合命中远高于机会，但跨 6 个 block 的目标频率 ITPC 约 0.33，与随机相位有限样本水平相当。因此两者更接近“频谱峰存在且码本大体对上，但相位/时域模板不稳定”，这会直接压低依赖相位锁定模板的 eTRCA。

这个检验不能证明每个有序左右眼标签都绝对正确，因为无序频率集合不区分 `(f1,f2)` 与 `(f2,f1)`；但它已排除 S01/S03 的整体码本错位。源 `label_list` 重排后又与历史 4D 数据逐元素一致，因此目前没有证据支持 Arena loader 造成了 block/target 置乱。

历史 MATLAB 还存在一个独立实现问题：`train_trca.m` 在 filter-bank 循环中把上一子带继续送入下一子带，训练端形成级联滤波，而测试端仍独立滤波。复刻该行为后，六人 2 s 均值从标准独立子带 eTRCA 的 32.62% 变为 33.65%；S06 从 63.81% 变为 68.57%，接近历史结果 MAT 的 69.52%，但 S01–S03 仍只有 6.67%、7.62%、4.29%。因此历史实现差异能解释少量偏差，不能解释三名近机会被试，也不能作为标准 eTRCA 定义继续沿用。

### 9.2 EMBC9 旧论文结果核对：缺口来自接收机协议

旧备份中还保留了完整的 EMBC9 论文工程，而不只是少量图。当前 8 份 transfer MAT 都能逐数组匹配唯一的旧标注源数据，并分别绑定到一份主 TRCA 结果 MAT；因此这里比较的是同一批数据，不存在被试队列替换。最终绘图脚本和论文 Fig. 5 使用 `0.2-4.0 s`、步长 `0.2 s` 的曲线，并把匿名 `S02` 排除在七人均值之外。论文正文报告的 2 s `79.05% / 43.38 bit/min` 与恢复 MAT 的七人均值 `79.0476% / 43.3778 bit/min` 完全吻合。

旧代码补齐了此前缺失的两个关键细节：先用 `1:4:end` 直接降到 250 Hz，再使用下截止为 6/14/22 Hz 的三个动态 Chebyshev 算法子带。`train_trca.m` 还会把上一子带继续送入下一子带，形成训练端级联，而测试端独立滤波；Arena 不沿用这个不对称错误，只使用标准独立三子带。2 s 对照如下：

| 队列 | 恢复 MATLAB | 旧 Arena 单带 | Arena 独立三子带 | 旧训练级联审计 |
|---|---:|---:|---:|---:|
| 全 8 人 | 75.9028% | 67.0833% | 76.6667% | 76.4583% |
| 论文 7 人 | **79.0476%** | 71.5079% | **79.8413%** | 79.6825% |

在共同的 4 s 窗口，论文七人旧 MAT 为 81.3492%，Arena 独立三子带为 81.9841%，旧单带仅 71.8254%。因此原先约 7.5 个百分点的论文复现缺口，主要由遗漏算法 filter bank 和重采样定义造成，不是标签顺序、源数据版本或被试质量造成。剩余不到 1 个百分点属于 MATLAB/SciPy 滤波系数和广义特征求解等实现差异。

![EMBC9 旧结果与 Arena 协议分解](results/legacy_backup_audit/figures/embc9_legacy_vs_arena.png)

旧工程的结果资产也确实远多于先前已核对部分：最终图目录有 43 个结果 MAT，其中 8 个主 TRCA、8 个 opE、8 个 opG、8 个 opEG、5 个 gamma 曲线变体；另有 17 个频率分析 MAT、8 个平均频率 MAT、11 个 MATLAB `.fig` 源图和 10 份稿件 PDF。它们仍在外接盘只读使用，没有复制进 Git。各结果 MAT 的 `result_label_best` 并非统一窗口预测：不同被试对应不同或多个曲线窗口，因此不能把它们拼成同一窗口的 hard-decision 复现锚点。

整个“已发表实验原始快照”范围更大：JBHI `Results_mat` 下共有 236 个 MAT（含 28 个 TDCA-Bfusion、112 个 gamma 扫描及多组中间修正版本），另一个早期 Frontiers 工程有 123 个数据结果 MAT，eDNN 工程也有多组 14-17 文件的结果目录，其中一组目录名明确标为 `wrong`。本报告只把协议和源数组都能闭合的 EMBC9/JBHI16/JBHI35 结果作为证据；其余目录已完成数量级盘点，但不会仅凭文件存在就并入当前结论。

### 9.3 旧论文工程核对：不是同一批 35 目标会话

旧备份中的论文工程和约 6.8 GB 工程压缩包已做只读检查。JBHI16 的证据链是完整的：当前 13 份 transfer MAT 均能逐样本匹配唯一的旧源数据，旧工程还保留了 eTRCA、ebPRCA、eFusionCA 的 390 个“被试×方法×窗口”结果单元。Arena 与旧 MATLAB hard decision 的平均一致率为 93.66%。关键群体结果如下：

| 方法 | 窗口 | 旧 MATLAB | Arena | 差值 |
|---|---:|---:|---:|---:|
| eTRCA | 0.4 s | 68.0288% | 68.3494% | +0.3205 pp |
| eTRCA | 2.0 s | 77.0833% | 77.1635% | +0.0801 pp |
| ebPRCA | 0.4 s | 67.4679% | 69.3910% | +1.9231 pp |
| ebPRCA | 2.0 s | 79.0865% | 79.4071% | +0.3205 pp |
| eFusionCA | 0.4 s | **71.3942%** | **71.3942%** | 0.0000 pp |
| eFusionCA | 2.0 s | 82.6923% | 82.2115% | -0.4808 pp |

![JBHI16 旧结果与 Arena](results/legacy_backup_audit/figures/jbhi16_legacy_vs_arena.png)

35 目标则不是“同一队列上复现很差”。旧绘图脚本使用 5 个会话，匿名对齐后为 `S01-alt、S03-alt、S04、S05、S06`；当前规范集使用 `S01-S06`。也就是说，旧队列用两个更强的替代会话替换了当前近机会水平的 `S01/S03`，并且完全没有当前低质量的 `S02`。在同一个标准 Arena eTRCA 下：

| 会话 | 0.4 s | 1.0 s | 2.0 s |
|---|---:|---:|---:|
| S01-alt | 54.76% | 70.95% | 78.10% |
| S03-alt | 50.00% | 62.86% | 67.14% |
| S04 | 67.62% | 82.38% | 87.14% |
| S05 | 13.33% | 19.52% | 28.10% |
| S06 | 35.24% | 54.76% | 63.81% |
| 旧 5 会话均值 | **44.19%** | **58.10%** | **64.86%** |
| 当前 6 会话均值 | 22.14% | 29.05% | 32.62% |

![JBHI35 旧队列与当前队列](results/legacy_backup_audit/figures/jbhi35_legacy_cohort_comparison.png)

替代会话还改变了对低分原因的判断。`S01` 当前/替代会话的 Oz 目标频率 ITPC 为 0.339/0.773，2 s eTRCA 为 6.67%/78.10%；`S03` 为 0.334/0.876 和 2.86%/67.14%。两名当前低分会话仍有高于机会水平的无序频率集合命中，因此不像简单的整体 label 置换；主要崩溃发生在跨 block 相位锁定与时域重复性。这更符合采集会话、刺激起始同步、事件延迟或逐 trial 相位稳定性差异，而不是被试固有不可解码。

![JBHI35 当前与替代会话信号诊断](results/legacy_backup_audit/figures/jbhi35_current_vs_alternate_signal.png)

旧 35 目标绘图脚本期望的 5 份结果 MAT 目前只剩 2 份原始版本；另有 1 份同一会话的替代打分版本。因此这里的 5 会话曲线明确标记为按 Arena 标准实现重新计算，不能冒充完整恢复的旧 MATLAB 群体曲线。

## 10. 对该范式的整体理解

当前结果支持把该范式视为一种独立的结构化 SSVEP 编码问题：

1. **消息不是频率标签，而是有序双目频率单元对。** 相同频率集合在不同眼别排列下表示不同消息。
2. **频率复用是核心，而不是副作用。** 每个基础频率出现在多个目标中，单频功率峰无法唯一确定目标。
3. **码字信息来自多层结构。** 基频存在、眼别、void、同/异频、beat/intermodulation、ERP 和 target-specific 模板共同决定可分性。
4. **接收机必须匹配这种结构。** ebPRCA 提取频率周期单元，eTRCA 保留完整任务模板，eFusionCA 的稳定增益证明两类信息互补。
5. **码本扩展不是免费容量。** 从 16 到 35 增加了理想 `C0`，但当前利用率从 75.2% 降至 35.6%；编码自由度只有被接收机和被试稳定恢复时才成为有效信息。

## 11. 后续实验与分析优先级

### 11.1 首要补实验

1. 以 JBHI16 为主协议增加被试，保留完整 4×4 笛卡尔积和明确 block-CV。
2. 对关键成对条件增加 trial：`(f1,f2)` vs `(f2,f1)`、单眼 vs 双眼同频、void vs active。
3. 对 35 目标先做采集质量与 subject eligibility 预注册，记录眼动/注视、头显位置、视觉舒适度和每眼刺激可见性。
4. 对 35 目标增加重复 block；每类 6 个测试 trial 不足以稳定估计个体级 35×35 决策信道。

### 11.2 接收机方向

1. 设计分层接收机：先判定码字族/void 结构，再恢复左右眼频率单元，最后做 target-specific 联合判决。
2. 将 bPRCA 与 TRCA 的等权求和升级为校准得到的 score normalization、概率融合或似然比融合。
3. 直接保存并分析 continuous scores，而不只分析 hard decision；这能区分“正确类略低”与“整个码字簇不可分”。
4. 做结构约束的 codebook pruning：保持左右眼单元使用平衡与笛卡尔对称，不能直接套任意类别贪心删除。

### 11.3 信息论方向

1. 对 pooled confusion 做按被试 bootstrap，给 MI/C_BA 和接收机增益增加置信区间。
2. 若研究左右眼协同，使用条件互信息或 PID，而不是把 `joint-left-right` 直接称为 synergy。
3. 将 `C_T` 与 observed selection time、动态停止和错误代价结合，形成 practical/observed 两级容量—时间曲线。
4. 跨范式比较必须匹配 receiver、窗口、校准量和时间分母；当前结果不能作为物理脑信道容量排名。

## 12. EEG 本体信号：无相位频率组合的空间—频谱结构

本范式的码本不包含相位维度。每个 label 只对应一个有序频率对 `(left_frequency, right_frequency)`，其中 0 表示 void。信号分析严格按该 label 映射，使用 2 s 原始分析 epoch，在每个频率单元的一至三次谐波处计算精确频率投影功率；不再添加接收机 filter bank，也不把固定零相位当作额外码字。

![EMBC9 频率单元地形](results/signal/figures/embc9_frequency_unit_topomaps.png)

![JBHI16 频率单元地形](results/signal/figures/jbhi16_frequency_unit_topomaps.png)

![JBHI35 频率单元地形](results/signal/figures/jbhi35_frequency_unit_topomaps.png)

每张图分别展示：频率出现相对缺失的谱功率差、仅左眼相对仅右眼的差异、以及单通道频率功率与四状态 `{缺失, 左眼, 右眼, 双眼}` 之间的 kNN 互信息。主要观察是：

1. 三套数据的频率出现效应均集中在 Oz/POz 邻域，证明 label—频率对映射在 EEG 频谱层有直接响应；
2. 左右眼差异地形并非零，因此没有相位码并不等于丢失眼别顺序，眼别可以通过枕区空间响应进入信号；
3. EMBC9 的单元特征平均 MI 约 0.23–0.24 bit，JBHI16 约 0.20–0.23 bit，JBHI35 从 11 Hz 的 0.120 bit 降至 15 Hz 的 0.057 bit；35 目标扩展后的高频单元利用明显变弱；
4. 最佳信息导联主要是 Oz、POz、PO3、O1，说明单独固定 Oz 会保留强主响应，但不足以描述眼别和不同频率的空间差异。

以上 topomap 仅由 9 个枕区导联插值，不应解释为全脑源定位。这里的 MI 是“频率单元状态—单通道谱特征”依赖量，也不是物理脑信道容量。

进一步把每个 label 的 `频率 × 导联` 平均签名与 eFusionCA 2 s 混淆比较，off-diagonal Spearman 相关为：EMBC9 `ρ=0.801`、JBHI16 `ρ=0.568`、JBHI35 `ρ=0.190`，均显著。小码本中，错误很大程度可由谱—空间签名接近解释；35 目标时该解释力显著下降，说明更多错误来自有限 trial、个体异质性和高阶组合结构，而不是单纯频谱距离。

## 13. 无相位码本 TDCA 可行性

当前 TDCA 不是论文复现，因为原稿没有 TDCA。这里建立两个 leakage-safe diagnostic receiver：

- `SPECTRAL_TDCA`：每个 label 使用其去重后的活动频率集合，生成固定零相位正余弦三阶谐波 reference；相位不是类别维度；
- `EEG_TDCA`：每折仅从训练 block 的类别平均 EEG 估计时域子空间，不使用解析频率 reference；
- `(void,void)` 没有可定义的频谱 reference，因此 `SPECTRAL_TDCA` 只对该类使用训练折 EEG fallback；
- 两者都按被试内 leave-one-block-out，测试 block 从未进入 reference、模板或空间滤波器。

两名质量正常 smoke 被试、审计后 `delay=2`、`components=4` 的初始单带 discovery 结果为：

| 数据集 | 关键窗口 | eTRCA | EEG-TDCA | 零相位 Spectral-TDCA |
|---|---:|---:|---:|---:|
| EMBC9 | 2.0 s | 75.83% | 85.28% | **87.22%** |
| JBHI16 | 0.4 s | 72.92% | 73.44% | **77.08%** |
| JBHI35 | 2.0 s | 75.48% | 80.48% | **89.05%** |

这说明 TDCA 条件已经成熟到可以做正式的 discovery/validation，而不是成熟到可以直接宣称算法优势。初始 `delay=5, components=8` 在 JBHI35 2 s 出现异常退化；最小参数审计后 `delay=2, components=4` 恢复。因此当前 smoke 存在同一批 S04/S06 用于发现参数的乐观偏差。下一步应固定该配置，在其余匿名被试上做 holdout validation；若继续搜索参数，则必须使用 nested block-CV 或独立被试层验证，不能在全体结果上再挑最优参数。

固定 `delay=2, components=4` 后已完成全部 27 名匿名被试和全部基线窗口：230/230 units、64,560/64,560 TDCA predictions、0 errors。2 s 全量结果为：

下表中的 eTRCA 是 Arena 的标准独立子带实现；EMBC9 已恢复旧论文代码证实的三个算法子带，JBHI35 则没有论文 eTRCA 锚点。两者都不复用历史 MATLAB 的训练端级联滤波错误。

| 数据集 | eTRCA | EEG-TDCA | Spectral-TDCA | Spectral C_BA | Spectral 利用率 |
|---|---:|---:|---:|---:|---:|
| EMBC9 | 75.07% | 78.61% | **81.53%** | 2.180 bit | 68.76% |
| JBHI16 | 77.16% | 74.44% | **79.89%** | 2.961 bit | 74.02% |
| JBHI35 | 32.62% | 32.78% | **37.78%** | 1.847 bit | 36.00% |

全体被试配对增益中，EMBC9 的 Spectral-TDCA 2 s 增益为 +6.46 个百分点，7/8 被试改善，`p=0.0092`；JBHI16 为 +2.72 点，9/13 改善、2 人持平，`p=0.0915`；JBHI35 为 +5.16 点，4/6 改善，`p=0.2202`。窗口级检验是探索性的，未做多重比较校正。

更严格地排除用于参数发现的 S04/S06 后，2 s 的 Spectral-TDCA 增益为：EMBC9 +5.19 点（5/6 改善，`p=0.0628`）；JBHI16 +3.41 点（8/11 改善，`p=0.0495`）；JBHI35 仅 +0.95 点（2/4 改善，`p=0.7862`）。因此 EMBC9 的方向仍为正，但不再达到该探索性被试层检验的 0.05 阈值；JBHI35 的总体提升仍主要来自本来就可解码的 S04/S05/S06，TDCA 不能补救近机会水平的 S01/S02 数据。

从算法结构看，零相位 Spectral-TDCA 尤其适合检验本范式的频谱组合利用：交换眼别但具有相同无序频率集合的目标共享解析频率子空间，最终能否区分取决于训练 EEG 模板与空间滤波是否保留眼别响应。这正好把“频率集合信息”和“有序双目空间信息”分开测量。

## 14. 结论

当前最强证据来自 JBHI16：完整 4×4 双目码本在 eFusionCA 下达到 75.2% 的 BA 利用率，左右眼单元均可稳定恢复，跨眼泄漏接近零，2 s 相对 eTRCA 增加 0.214 bit/selection。它证明了“有序双目单元组合”确实可以成为独立编码维度。

EMBC9 说明小码本也会受眼别交换和 void 判别限制；JBHI35 则说明扩大笛卡尔码本不会自动带来可用容量。EEG 地形与特征 MI 已证明各频率单元和眼别顺序在枕区具有可测结构，零相位 Spectral-TDCA smoke 又表明直接按码本频率组合建模具有潜力。下一阶段应固定 discovery 参数做独立被试验证，同时围绕结构化码字、分层错误和被试质量补实验，而不是在同一批数据上继续无约束挑算法与参数。

## 附录：产物

- `results/full/study_manifest.json`：Study 完整性与解释边界；
- `results/full/tables/channel_metrics_pooled.csv`：主决策信道表；
- `results/full/tables/channel_metrics_subject.csv`：被试级探索表；
- `results/full/tables/factor_metrics_pooled.csv`：左右眼与联合信息；
- `results/full/tables/error_taxonomy_pooled.csv`：错误拓扑；
- `results/full/tables/category_confusion_pooled.csv`：码字族混淆；
- `results/full/tables/receiver_gain_pooled.csv`：pooled 接收机增益；
- `results/full/tables/receiver_gain.csv`：被试配对探索检验；
- `results/full/figures/fig01`–`fig08`：完整图组。
- `results/signal/manifest.json`：EEG 本体信号分析完整性；
- `results/signal/figures/*_frequency_unit_topomaps.png`：三套频率单元 MNE 地形；
- `results/signal/tables/frequency_unit_mi_subject.csv`：频率单元状态—谱特征 MI；
- `results/signal/tables/signal_similarity_confusion.csv`：信号签名与混淆相关；
- `results/jbhi35_subject_quality_audit/subject_quality.csv`：JBHI35 匿名被试信号、相位与标签一致性诊断；
- `results/etrca_scoring_audit/subject.csv`：标准、分数变换与历史级联 eTRCA 审计；
- `results/tdca_smoke/manifest.json`：TDCA reference、参数和 CV 边界；
- `results/tdca_smoke/summary.csv`：两种 TDCA 与 eTRCA smoke；
- `results/tdca_smoke/parameter_audit_jbhi35_2s.csv`：JBHI35 长窗参数异常审计。
- `results/tdca_full/manifest.json`：230-unit 全量 TDCA 完整性；
- `results/tdca_full/comparison_summary.csv`：全窗口 TDCA/eTRCA 比较；
- `results/tdca_full/channel_metrics.csv`：pooled MI、C_BA 与利用率；
- `results/tdca_full/paired_gains.csv`：全体被试配对增益；
- `results/tdca_full/discovery_excluded_gains.csv`：排除 S04/S06 的验证性比较；
- `results/tdca_full/figures/tdca_full_vs_etrca.png`：全量准确率曲线。
