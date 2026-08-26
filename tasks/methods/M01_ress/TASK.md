# M01_ress — RESS 实现任务

## 目标
在 vep_arena 实现 RESS 方法，并在 Benchmark/BETA 上复现论文结果。

## 论文
Xu W, Ke Y, Ming D, IEEE TNSRE 32:4284-4293 (2024), DOI 10.1109/TNSRE.2024.3503772

## 规格文档（必读）
见 `docs/method-tasks/M01_ress.md` —— 含核心算法、curated 包索引、验证协议、期望结果。

## Curated 论文索引（复现核对原文）
28_2024_xu_rhythmic_entrain
（若标注"未提取"：先向主 agent 请求补提取，或直接从 Zotero/DOI 获取原文核对）

## 实现要求
1. 新增 `vep_arena/methods/ress.py`，实现 `class RESS`：
   - `fit(train_x, train_y) -> self`
   - `predict(x) -> (labels, scores)`
   - 数据形状: `trials x subbands x channels x samples`（与 TRCA/CCA 一致）
2. 注册方法：在 `vep_arena/methods/__init__.py` 或 `traditional.py` 导出
3. 新建任务脚本 `tasks/methods/M01_ress/run.py`（参考 `tasks/baselines/BL01_ssvep_benchmark/run.py` 的 make_model 模式）

## 验证协议
- 数据集: BL01 (Benchmark 9ch) + BL02 (BETA)
- 协议: leave-one-block-out; windows 0.5-1.0 s; calibration blocks 1/2/3/4 (核心=1 block)
- 预处理与 BL01 一致（cue 跳过 0.5s、潜伏期 0.14s、notch 50Hz、filterbank）

## 期望结果（论文报告，curated 包 28 提取）
- 核心结论：单块校准（1 block）下 RESS 显著优于 TRCA/eCCA 等（ANOVA 显著）
- 论文报告 ITR 最高 367.83 bit/min（具体设置见原文 Table 3-4，curated 包 tables/）
- 验收：Benchmark 9ch 单块校准 acc 与论文 Table 3 diff ≤ 2%

## 验收流程（原）
1. 结果写入 `tasks/methods/M01_ress/results/`
2. 与规格文档期望数字 diff ≤ 2%
3. 汇报实现/注册/结果表/对比结论
4. 更新 docs/method-tasks/README.md