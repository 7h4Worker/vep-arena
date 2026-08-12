# Benchmark 9ch 决策信道容量复现与分析报告

> 状态标注：本报告当前仍是协议审计前的旧结果草稿，不能作为最终汇报版本使用。
> 原因包括 ECCA 曾未按 filter-bank ECCA 计算、raw CCA 预处理未完全对齐
> SSVEP-Analysis-Toolbox、且旧 full 结果未包含 FBCCA。请在
> `benchmark_decision_channel_capacity_full_v2` 重跑并重新执行 `analyze.py`
> 与 `plot.py` 后再使用本文和图片。

## 1. 研究问题

传统 BCI ITR 使用平均准确率和对称错误假设，把分类器输出压缩成一个标量。
本任务把 Benchmark SSVEP 的分类结果显式建模为离散决策信道，比较 Costa 2020 的闭式容量思路和 Arslan-Sinha 2024 的 Blahut-Arimoto 容量估计。

## 2. 数据与协议

- 数据集：Tsinghua Benchmark 9ch occipital preset。
- 被试数：35。
- 方法：CCA, ECCA, ETRCA, TRCA。
- 窗口：0.2s, 0.3s, 0.4s, 0.5s, 0.6s, 0.7s, 0.8s, 0.9s, 1.0s。
- 协议：subject-specific leave-one-block-out；每个 subject/method/window 生成 40x40 混淆矩阵。
- ECCA 定位：使用训练折的个体平均模板与正余弦参考信号，融合标准 CCA、模板 CCA 和模板-参考 spatial filter 相关项，是从无监督 CCA 过渡到 TRCA/eTRCA 这类校准模板方法的经典中间基线。

## 3. 容量指标

- `C0 = log2(M)`：理想无噪声容量。
- `C1`：传统对称错误公式，对应常用 ITR 的单次信息量。
- `C2`：Costa 2020 / Muroga 闭式容量；矩阵不可逆、病态或最优分布不在概率单纯形时标记 invalid。
- `C_BA`：Blahut-Arimoto 数值容量，是本报告主结果。
- `I_uniform`：均匀输入分布下的互信息。
- Costa-style 码本裁剪：当前实现从全 40 类混淆矩阵裁剪子矩阵并重新归一化，做前向选择；代码支持多起点，本次默认使用最高对角准确率起点。它用于分析编码空间，不等同于原文每个候选码本都重训分类器的 wrapper。

## 4. 主要结果图

![容量阶梯](results/figures/fig01_capacity_ladder.png)

图 1 展示 C0、C1、C_BA 和 I_uniform 随窗口变化。C_BA 与 I_uniform 的差距表示输入分布优化空间，C0 与 C_BA 的差距表示决策信道噪声代价。

![方法容量对比](results/figures/fig02_method_capacity_compare.png)

图 2 比较不同解码器的 BA 容量。若某方法在准确率上提升但 C_BA 提升有限，说明它主要改善了均匀使用下的表现，而不是显著改变容量上限。

![平均混淆矩阵](results/figures/fig03_mean_confusion_heatmap.png)

图 3 使用当前最高平均 C_BA 的组合：ETRCA, 1.0s。非对角结构是后续编码空间和频率混淆分析的核心对象。

![错误与频率距离](results/figures/fig04_error_vs_frequency_distance.png)

图 4 检查错误是否集中在邻近频率之间，用于判断混淆结构是否主要来自频率分辨率限制。

![最优输入分布](results/figures/fig05_optimal_input_distribution.png)

图 5 给出 BA 容量实现分布 q*。若 q* 接近均匀，说明 Benchmark 40 类码本对该解码器已经接近均匀最优；若部分频率权重被压低，则提示个体化或非均匀命令先验可能有收益。

![非对称度与容量增益](results/figures/fig06_asymmetry_vs_capacity_gain.png)

图 6 对应 Arslan-Sinha 的核心问题：信道非对称度是否解释 BA 相对传统 C1 的增益。

![C2 与 BA 一致性](results/figures/fig07_c2_vs_ba_consistency.png)

图 7 只使用闭式 C2 有效的组合。偏离对角线通常来自病态矩阵、有限样本稀疏性或闭式解有效性边界。

![CBA-C1 分布](results/figures/fig08_cba_minus_c1_distribution.png)

图 8 直接检查传统公式何时低估或高估 BA 容量。

![容量时间权衡](results/figures/fig09_itr_time_tradeoff.png)

图 9 使用 60/T * C_BA(T) 观察最优工作窗口。该图服务于在线系统中速度与可靠性的折中。

![码本裁剪增益](results/figures/fig10_codebook_pruning_gain.png)

图 10 把全 40 类 BA 输入优化增益与基于混淆子矩阵的码本裁剪增益分开。

![Costa 选择路径](results/figures/fig11_costa_selection_path.png)

图 11 展示各方法代表最佳窗口下的前向码本选择路径。曲线峰值对应当前固定混淆矩阵近似下的最佳 K，随后若继续加入类别，额外类别引入的混淆会抵消类别数增加带来的容量收益。

![Costa 保留频率](results/figures/fig12_costa_selected_frequencies.png)

图 12 把最佳裁剪码本映射回 Benchmark 频率轴。该图用于观察被保留刺激是否覆盖整个频段，还是集中在局部更稳定的频率区间。

![Costa 增益热图](results/figures/fig13_costa_pruning_gain_heatmap.png)

图 13 展示 method × window 下裁剪相对全 40 类 BA 容量的增益。它给出后续在线码本大小选择和 HD-200 迁移的直接入口。

## 5. 数值摘要

- 最高 aggregate C_BA：ETRCA 1.0s，C_BA=4.7910 bit/symbol，accuracy=0.9368。
- C_BA - C1 平均值：1.7034 bit/symbol；最小值：0.0000；最大值：3.4502。
- 最佳裁剪码本：ETRCA 1.0s，K=40，裁剪后 C_BA=4.7910 bit/symbol，相对全 40 类 BA 增益=0.0000。
- 各方法最佳裁剪摘要：CCA 1.0s: K=17, C_BA=2.757；ECCA 1.0s: K=20, C_BA=2.870；ETRCA 1.0s: K=40, C_BA=4.791；TRCA 1.0s: K=40, C_BA=4.554。

## 6. 限制与下一步

- 第一轮使用 Benchmark 9ch，不是 Costa 2020 的 64ch 完整复刻。
- SSCOR 模块已经补入代码层，当前报告中的全量结果若未重新运行 SSCOR/ESSCOR，则仍只反映已有 predictions 中的方法。
- 类别选择是从固定 40 类混淆矩阵裁剪子矩阵，不重训分类器；下一版可补 Costa 原文式 wrapper。
- 下一步可以把同一套 channel 模块迁移到 HD-200、BETA 和个体化码本分析。
# OBSOLETE DRAFT - DO NOT USE FOR FINAL REPORTING

This report was generated before the CCA/FBCCA/ECCA protocol audit. The old
full results used an incorrect ECCA path, did not include FBCCA in the task
default set, and were generated before raw CCA preprocessing was aligned with
SSVEP-Analysis-Toolbox. Re-run `benchmark_decision_channel_capacity_full_v2`,
then run `analyze.py` and `plot.py` to regenerate this report.
