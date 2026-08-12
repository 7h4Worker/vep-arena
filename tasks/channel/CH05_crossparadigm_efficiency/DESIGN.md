# 跨范式信息恢复效率 — Pilot 设计

> 状态: 设计稿 (design)｜执行器 `wsl-sandbox-conda`｜只读上游 predictions, 产出写本 task 目录
> 定位: 这是 `information_accumulation_rate` 的严格版 v2, 也是 Codex `signal_system_main_axis` 主轴的首个落地 pilot。
> 用一个跨范式分析把方法学跑通, 反过来定义 Arena 框架该长什么样 —— 而非先验设计文件治理。

## 0. 这个 pilot 要回答的问题 (主轴, 非分类)

不同**复用范式**(频分/相位/空间/码分)在**相同资源预算**(时间、通道、校准、刺激功率)下,
把实验者控制的自由度转成了多少**可恢复信息**?并把总收益拆成三类,不许糊成一个准确率或一个 ITR:

- **G_dof 单码自由度增益** — 加一个编码维度(频→相→空→码), 一个编码单位多恢复多少独立信息。
- **G_repeat 重复观测增益** — 同一码元重复观测, 靠平均/积分多恢复多少。
- **G_rx 接收机增益** — 编码与观测不变, 换更强算法多恢复多少。

分类器在此仅是"接收机测量步骤"。C_BA/MI_uniform 是**经验决策信道**结果, 不冒充脑-EEG 物理信道容量上界。

## 1. 跨范式数据台账 (均已核实有 trial 预测)

| 范式 | 复用轴 | 数据集 | 类别 | 被试 | block | 窗口 | 方法 | 自由度阶梯 |
|---|---|---|---:|---:|---:|---|---|---|
| FDMA | 频率 | Benchmark | 40 | 35 | 6 | 0.2–1.0s(9) | CCA/ECCA/TRCA/ETRCA/FBCCA | 单频基线 |
| DualFreq | 频+相 | Binocular AR | 8 | 14 | 20 | 0.1–3.0s(30) | CCA/FBCCA/TRCA/ETRCA | SFSP→SFDP→DFSP→DFDP (2×2 因子) |
| DualFreq | 频+相 | Dual-Alpha | 40 | 35 | 5 | 0.2–2.0s(10) | ETRCA/FBDCCA | CA/BV/BSV |
| CDMA | 码序列 | JFPM/NBRS (Zheng24) | 8–40 | 100 | 3 | 0.4–4.0s(8) | FBCCA-CODE/TRCA | JFPM-8/NBRS-8/NBRS-15 |
| SDMA | 频+相+空 | HD-200 | 40–200 | 14 | 18 | 0.1–0.5s(5) | TDCA | **40→80→120→160→200 目标数** |
| MFSC | 多频序列 | 160-MFSC | 160 | 8 | 3 | — | TDCA | calib-free |

## 2. 核心方法学关卡: 时间的两种含义 (Codex §4, 在此数据上是真问题)

窗口变长可能是两件完全不同的事, 现有 MI(T) 把它们混在一条 τ 曲线里:
- **重复观测**: 周期性 SSVEP 窗口变长 = 多看几个相同周期 → 属 G_repeat。
- **码字展开**: 有限长 cVEP / 双相结构窗口变长 = 读取新码片 → 不是重复。

**采用双口径 (不留给用户选):**
- **主口径 = 范式原生编码单位**: FDMA/SDMA 单位≈刺激周期 1/f; CDMA 单位=码序列完整长度; DualFreq 需判定双相是否在展开。区分"1个单位 / 2个单位 / 更多重复"。
- **辅口径 = 固定物理时长**: 相同 0.2/0.5/1.0s 预算下跨范式比可恢复信息。
- 无法从刺激定义识别完整编码单位的数据集 → 只进固定时长比较, 不报重复增益。
- 一个窗口只覆盖有限码字的一部分 → 标"码字展开", 禁止标"重复"。

## 3. 每个分析结果必须携带 comparability_key (Codex §8)

主表不是横向排名总榜, 是**带可比性键的研究目录**。key 至少含:
`(receiver, R_policy, duration_caliber, n_classes_norm, calibration)`。
同 key 才可横向比; 跨 key 只能定性对照。8类 vs 40类的 MI 绝对值不可直接比 → 一律用 MI/C0 (C0=log2(类别数)) 归一化。

## 4. 执行步骤与产出

- **S1 统一接收机测量**: 对每范式, 从 predictions.csv 构经验决策信道 P(m̂|m), 算 C1/MI_uniform/C_BA (走 `vep_arena/channel/capacity.py`, 不重实现)。固定接收机=ETRCA/TRCA 系一档, 保证跨范式接收机可比。
- **S2 G_repeat + 独立性检验**: 用 block 结构构 I_1(单block) vs I_R(累积), 出边际 ΔI_R 曲线与饱和点; 检验多次观测近似独立性 (block 间预测相关)。
- **S3 G_dof**: Binocular 2×2 因子算频率/相位的 I_1 增量与可加性; HD-200 目标数阶梯算空间自由度的边际信息; 高度耦合无配对时报"联合增益", 不强行分摊。
- **S4 时间双口径**: 同范式分别按编码单位与固定物理时长出 MI 曲线, 检验 FDMA(纯重复) vs CDMA(展开) 的 τ 差异是否为真机制而非伪影。
- **S5 跨范式效率表**: 固定物理时长口径下, 各范式 MI/C0 与 R_info=I_R/T_total, 带 comparability_key。
- **S6 结论边界声明**: 每项分析"改了什么/固定了什么/观测到什么增益/支持什么结论/不能支持什么"。现有"FDMA快饱和 CDMA线性"结论**降级为待验证假说**, 由 S4 裁决。

**产出**: `figures/` 三类增益 + 双口径曲线; `tables/crossparadigm_efficiency.csv` (带 comparability_key); `REPORT.md` 含边界声明。

## 5. 与框架的关系 (pilot 反哺)

跑通后, 由本分析真实需要的字段反推 manifest/study schema: 编码单位、重复次数、固定条件、comparability_key。
即"方法学被一个真实分析逼出来", 而不是先写 T1–T7 文件治理。

## 6. 边界 (不做)
- 不重训 DNN、不跑 MNE 官方复现 (归 `.venv-windows`)。
- 不改上游 predictions; 不动 capacity/confusion 干净核心。
- 无实测注视周期的数据集, itr_observed 置 NaN, 不回退 0.5 (见 RESOLUTION)。
