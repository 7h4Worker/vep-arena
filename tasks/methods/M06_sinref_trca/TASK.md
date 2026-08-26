# M06_sinref_trca — Sinusoidal-Referenced TRCA 实现任务

## 目标
在 vep_arena 实现 Sinusoidal-Referenced TRCA 方法，并在 Benchmark/BETA 上复现论文结果。

## 论文
IEEE Infocom Workshops (2023), DOI 10.1109/INFOCOMWKSHPS57453.2023.10226001

## 规格文档（必读）
见 `docs/method-tasks/M06_sinref_trca.md` —— 含核心算法、curated 包索引、验证协议、期望结果。

## Curated 论文索引（复现核对原文）
36_2023_wang_sinref_trca
包位置: `curated/papers/03-vep-algorithm/11-spatial-filter/36_2023_wang_sinref_trca/`（已提取 ✅）

> 若标注 10.1109/INFOCOMWKSHPS57453.2023.10226001）
（若标注"未提取"：先向主 agent 请求补提取，或直接从 Zotero/DOI 获取原文核对）

## 实现要求
1. 新增 `vep_arena/methods/sinref_trca.py`，实现 `class SinusoidalReferencedTRCA`：
   - `fit(train_x, train_y) -> self`
   - `predict(x) -> (labels, scores)`
   - 数据形状: `trials x subbands x channels x samples`（与 TRCA/CCA 一致）
2. 注册方法：在 `vep_arena/methods/__init__.py` 或 `traditional.py` 导出
3. 新建任务脚本 `tasks/methods/M06_sinref_trca/run.py`（参考 `tasks/baselines/BL01_ssvep_benchmark/run.py` 的 make_model 模式）

## 验证协议
- 数据集: BL01
- 协议: 对齐 TRCA; Benchmark 9ch
- 预处理与 BL01 一致（cue 跳过 0.5s、潜伏期 0.14s、notch 50Hz、filterbank）

## 期望结果（论文报告）
- 论文为 IEEE Infocom Workshops 短文（6 页），无完整数字表
- 核心结论：正弦参考 TRCA 优于标准 TRCA（Benchmark 上 acc 提升）
- 验收：Benchmark 9ch 上 ≥ TRCA 基线（diff 方向一致，数值以论文图为准）

## 验收流程（原）
1. 结果写入 `tasks/methods/M06_sinref_trca/results/`
2. 与规格文档期望数字 diff ≤ 2%
3. 汇报实现/注册/结果表/对比结论
4. 更新 docs/method-tasks/README.md