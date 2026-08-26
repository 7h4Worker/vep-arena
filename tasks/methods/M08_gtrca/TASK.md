# M08_gtrca — gTRCA 实现任务

## 目标
在 vep_arena 实现 gTRCA 方法，并在 Benchmark/BETA 上复现论文结果。

## 论文
Tanaka T, Scientific Reports 10:84 (2020), DOI 10.1038/s41598-019-56962-2

## 规格文档（必读）
见 `docs/method-tasks/M08_gtrca.md` —— 含核心算法、curated 包索引、验证协议、期望结果。

## Curated 论文索引（复现核对原文）
38_2020_tanaka_gtrca
包位置: `curated/papers/03-vep-algorithm/11-spatial-filter/38_2020_tanaka_gtrca/`
（阅读副本 paper_*.md + paper_record.json；核对公式/参数/数字后开始实现）

## 实现要求
1. 新增 `vep_arena/methods/gtrca.py`，实现 `class gTRCA`：fit/predict，数据形状 trials x subbands x channels x samples
2. 注册方法（methods/__init__.py 或 traditional.py）
3. 任务脚本 `tasks/methods/M08_gtrca/run.py`（参考 BL01 run.py）

## 验证协议
- 数据集: BL01 (多被试分组)
- 协议: 群体滤波器 vs 单被试 TRCA; 跨被试验证
- 预处理与 BL01 一致

## 期望结果（论文报告）
- Scientific Reports 方法论文，无 SSVEP Benchmark 数字表
- 核心结论：群体滤波器（跨被试）提升组内可重复性；gTRCA 可迁移到新被试
- 验收：Benchmark 上"群体滤波器+个体模板"组合 ≥ 单被试 TRCA；跨被试场景（留一被试）验证

## 验收流程（原）
1. 结果写入 `tasks/methods/M08_gtrca/results/`
2. 与规格文档期望数字 diff ≤ 2%
3. 汇报实现/注册/结果表/对比结论
4. 更新 docs/method-tasks/README.md