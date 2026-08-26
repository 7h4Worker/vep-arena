# M09_sctrca — scTRCA 实现任务

## 目标
在 vep_arena 实现 scTRCA 方法，并在 Benchmark/BETA 上复现论文结果。

## 论文
Sun et al., JNE 18:046022 (2021), DOI 10.1088/1741-2552/abfdfa

## 规格文档（必读）
见 `docs/method-tasks/M09_sctrca.md` —— 含核心算法、curated 包索引、验证协议、期望结果。

## Curated 论文索引（复现核对原文）
39_2021_sun_sctrca
包位置: `curated/papers/03-vep-algorithm/11-spatial-filter/39_2021_sun_sctrca/`
（阅读副本 paper_*.md + paper_record.json；核对公式/参数/数字后开始实现）

## 实现要求
1. 新增 `vep_arena/methods/sctrca.py`，实现 `class scTRCA`：fit/predict，数据形状 trials x subbands x channels x samples
2. 注册方法（methods/__init__.py 或 traditional.py）
3. 任务脚本 `tasks/methods/M09_sctrca/run.py`（参考 BL01 run.py）

## 验证协议
- 数据集: BL01 + BL02
- 协议: 多校准块 1-4; 与 TRCA/sTRCA 对照
- 预处理与 BL01 一致

## 期望结果（论文 Table 2，curated 包 39 提取）
ITR (bits/min) @ 窗口，随训练试次数变化：
| 方法 | Benchmark 2 | 3 | 4 | 5 | BETA 2 | 3 |
|---|---|---|---|---|---|---|
| TRCA | 107.94/1.0 | 141.71/1.0 | 158.16/0.9 | 167.72/0.9 | 66.76/1.0 | 96.20/1.0 |
| msTRCA | 125.72/1.0 | 155.14/0.9 | 169.24/0.9 | 176.67/0.8 | 77.37/1.0 | 109.41/1.0 |
（另有电极数 3-9 的 acc 表：TRCA 0.42-0.60，msTRCA 0.44-0.66）
验收：复现 ITR/acc 与上表 diff ≤ 2%

## 验收流程（原）
1. 结果写入 `tasks/methods/M09_sctrca/results/`
2. 与规格文档期望数字 diff ≤ 2%
3. 汇报实现/注册/结果表/对比结论
4. 更新 docs/method-tasks/README.md