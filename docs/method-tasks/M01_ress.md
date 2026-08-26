# M01 — RESS（Rhythmic Entrainment Source Separation）

## 论文
- 标题: Improving the Performance of Individually Calibrated SSVEP Classification by Rhythmic Entrainment Source Separation
- 作者: Xu Wei, Ke Yufeng, Ming Dong
- 期刊: IEEE TNSRE 32:4284-4293, 2024
- DOI: 10.1109/TNSRE.2024.3503772

## Curated 包索引（复现核对原文用）
- 包: `28_2024_xu_rhythmic_entrain`
- 位置: `curated/papers/03-vep-algorithm/11-spatial-filter/28_2024_xu_rhythmic_entrain/`
- 阅读副本: `paper_2024_xu_rhythmic_entrain.md`
- 记录: `paper_record.json`（含 8 图核验 + 40 引用 + evidence）
- 关键数字（evidence 已提取）: 单块校准下 RESS 显著优于 TRCA；ITR 最高 367.83 bit/min（具体设置见原文 Table 3-4）

## 核心算法
RESS 通过**频域滤波构造空间滤波器**，最大化"节律夹带成分"相对背景的比率：

1. 频域带通：对每个目标频率构造两个频带滤波（SSVEP 相关带 + 背景带）
   - 参数 `d_{p,s}`（相关带通带宽度）、`d_{s,r}`（背景带通带宽度）
2. 空间滤波器求解：最大化
   $$w^T \Sigma_{SSVEP} w \;/\; w^T \Sigma_{background} w$$
   其中 Σ_SSVEP 来自带通滤波后的协方差，Σ_background 来自背景频带的协方差
3. 模板匹配分类（与 TRCA 相同 pipeline：模板相关 + 集成可选）

## 与现有骨架的差异（vep_arena TRCA 基础上改）
- 复用 `trca_core._trca_filter` 的广义特征分解模式，但输入协方差换成**频域滤波后**的 Σ_SSVEP / Σ_background
- 需要新增：频域带通滤波（`scipy.signal.butter` + `sosfilt`），参数 d_{p,s}/d_{s,r} 按论文网格搜索（Fig.2 的 grid：在 Dataset I 上调，迁移到 Dataset II）
- 接口对齐：`class RESS` 实现 `fit(train_x, train_y)` / `predict(x)`，数据形状 `trials × subbands × channels × samples`

## 验证协议
- 数据集：Benchmark 9ch（BL01 preset）+ BETA（BL02 preset）
- 交叉验证：leave-one-block-out（与 BL01 一致）
- 时间窗：0.5/0.6/0.7/0.8/0.9/1.0 s
- 校准块：1 block（论文核心场景 = 单块校准）+ 2/3/4 blocks 对照
- 预处理：与 BL01 一致（cue 跳过、0.14s 潜伏期、notch、9ch）

## 期望结果（论文报告，供复现 diff）
- 单块校准（1 block）：RESS 优于 TRCA、eCCA 等（具体 acc/ITR 见原文 Table 3，需对照 curated 包 tables/）
- 数据集 I/II 上的 acc 对比曲线见原文 Fig. 4-6

## 验收标准
1. `fit/predict` 接口通过 BL01 runner 的 make_model 注册
2. Benchmark 9ch 单块校准结果与论文 Table 3 数字 diff ≤ 2%（acc）
3. 双数据集（Benchmark + BETA）结果写入 `tasks/methods/M01_ress/results/`
4. 更新 `docs/method-tasks/README.md` 状态列
