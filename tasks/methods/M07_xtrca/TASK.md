# M07_xtrca — xTRCA 实现任务

## 目标
在 vep_arena 实现 xTRCA 方法，并在 Benchmark/BETA 上复现论文结果。

## 论文
Tanaka T, Miyakoshi M, NeuroImage 197:672-687 (2019), DOI 10.1016/j.neuroimage.2019.04.049

## 规格文档（必读）
见 `docs/method-tasks/M07_xtrca.md` —— 含核心算法、curated 包索引、验证协议、期望结果。

## Curated 论文索引（复现核对原文）
37_2019_tanaka_xtrca
包位置: `curated/papers/03-vep-algorithm/11-spatial-filter/37_2019_tanaka_xtrca/`
（阅读副本 paper_*.md + paper_record.json；核对公式/参数/数字后开始实现）

## 实现要求
1. 新增 `vep_arena/methods/xtrca.py`，实现 `class xTRCA`：fit/predict，数据形状 trials x subbands x channels x samples
2. 注册方法（methods/__init__.py 或 traditional.py）
3. 任务脚本 `tasks/methods/M07_xtrca/run.py`（参考 BL01 run.py）

## 验证协议
- 数据集: BL01
- 协议: windows 0.2-1.0 s; 对照 TRCA/eTRCA
- 预处理与 BL01 一致

## 期望结果（论文 Table，curated 包 37 提取）
- 论文是神经科学方法论文（evoked/induced 响应），表格为被试级分类误差
- 表2 分类误差：Subject #1 0.23(0.049)、#2 0.23(0.066)、#3 0.28(0.087)
- 验收：SSVEP 场景下 xTRCA ≥ TRCA（Benchmark），神经科学复现可选

## 验收流程（原）
1. 结果写入 `tasks/methods/M07_xtrca/results/`
2. 与规格文档期望数字 diff ≤ 2%
3. 汇报实现/注册/结果表/对比结论
4. 更新 docs/method-tasks/README.md