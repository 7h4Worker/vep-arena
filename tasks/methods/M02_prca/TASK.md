# M02_prca — PRCA 实现任务

## 目标
在 vep_arena 实现 PRCA 方法，并在 Benchmark/BETA 上复现论文结果。

## 论文
Ke Y, Liu S, Ming D, IEEE TBME 71(4) (2024), DOI 10.1109/TBME.2023.3333435

## 规格文档（必读）
见 `docs/method-tasks/M02_prca.md` —— 含核心算法、curated 包索引、验证协议、期望结果。

## Curated 论文索引（复现核对原文）
19_2024_ke_bprca
（若标注"未提取"：先向主 agent 请求补提取，或直接从 Zotero/DOI 获取原文核对）

## 实现要求
1. 新增 `vep_arena/methods/prca.py`，实现 `class PRCA`：
   - `fit(train_x, train_y) -> self`
   - `predict(x) -> (labels, scores)`
   - 数据形状: `trials x subbands x channels x samples`（与 TRCA/CCA 一致）
2. 注册方法：在 `vep_arena/methods/__init__.py` 或 `traditional.py` 导出
3. 新建任务脚本 `tasks/methods/M02_prca/run.py`（参考 `tasks/baselines/BL01_ssvep_benchmark/run.py` 的 make_model 模式）

## 验证协议
- 数据集: BL01 + BL02
- 协议: 核心=1 block 校准（每目标 1 trial）; 对照 2/3/4 blocks; windows 0.5-1.0 s
- 预处理与 BL01 一致（cue 跳过 0.5s、潜伏期 0.14s、notch 50Hz、filterbank）

## 期望结果（论文 Table，curated 包 19 提取）
ITR (bits/min)：
| 方法 | Dataset I (T_test) | ITR | Dataset II (T_test) | ITR |
|---|---|---|---|---|
| eCCA | 0.5 s | 161.3±67.7 | 0.9 s | 163.3±56.2 |
| ePRCA (T_train=T_test) | 0.5 s | 176.8±60.3 | 0.9 s | 176.9±50.0 |
在线被试 acc/ITR（节选）：S1 97.50%/149.35、S2 91.25%/129.20、S3 72.50%/83.08
验收：Benchmark 1-block acc/ITR 与论文 diff ≤ 2%

## 验收流程（原）
1. 结果写入 `tasks/methods/M02_prca/results/`
2. 与规格文档期望数字 diff ≤ 2%
3. 汇报实现/注册/结果表/对比结论
4. 更新 docs/method-tasks/README.md