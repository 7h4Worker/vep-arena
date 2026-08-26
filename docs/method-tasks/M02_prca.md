# M02 — PRCA（Periodically Repeated Component Analysis）

## 论文
- 标题: Enhancing SSVEP Identification With Less Individual Calibration Data Using Periodically Repeated Component Analysis
- 作者: Ke Yufeng, Liu Shuang, Ming Dong
- 期刊: IEEE TBME 71(4), 2024
- DOI: 10.1109/TBME.2023.3333435

## Curated 包索引（复现核对原文用）
- 包: `19_2024_ke_bprca`
- 位置: `curated/papers/03-vep-algorithm/11-spatial-filter/19_2024_ke_bprca/`
- 阅读副本: `paper_2024_ke_bprca.md`
- 记录: `paper_record.json`（6 图核验 + evidence）
- 注意: 与 vep_arena 现有 `bprca.py`（binocular PRCA，BessVEP 私有）**不同**——这是单体 PRCA 论文，需独立实现

## 核心算法
PRCA 针对**单试次校准**（每频率仅 1 trial）设计：

1. 周期重复性约束：与传统 TRCA 最大化"试次间协方差"不同，PRCA 利用**单试次的周期重复结构**（SSVEP 的周期片段可重复切分）
2. 构造空间滤波器：最大化重复片段间的一致成分
   - 单试次 x ∈ R^{C×T}，按刺激周期切分为 P 个片段 → 计算片段间协方差矩阵
   - 求解广义特征分解（与 TRCA 同构，但协方差来源 = 片段 vs 试次）
3. 模板：重复片段平均
4. 分类：相关分析（可选 filterbank/集成）

## 与现有骨架的差异
- 基于 `TRCA` 类改造：把"试次间"协方差换成"周期片段间"协方差
- 关键新参数：周期片段长度 P（由刺激频率/采样率决定）
- 接口对齐：`class PRCA` 实现 `fit/predict`，数据形状同 TRCA

## 验证协议
- 数据集：Benchmark 9ch + BETA
- **核心场景：1 block 校准**（每目标 1 trial）；对照 2/3/4 blocks
- 时间窗：0.5-1.0 s
- leave-one-block-out 或固定 block 训练（按论文）

## 期望结果
- 论文核心结论：单试次下 PRCA > TRCA/eCCA（差异随校准减少而增大）
- 具体数字见 curated 包 tables/（Table 2-4：acc/ITR × 校准块数）

## 验收标准
1. 实现 `vep_arena/methods/prca.py` 并注册到 `methods/traditional.py` 聚合或独立导出
2. Benchmark 9ch 1-block 结果与论文 Table 对比 diff ≤ 2%
3. 结果写 `tasks/methods/M02_prca/results/`
