# M05_mohp — Multi-Objective High-Pass Filter 实现任务

## 目标
在 vep_arena 实现 Multi-Objective High-Pass Filter 方法，并在 Benchmark/BETA 上复现论文结果。

## 论文
IEEE TIM (2022), DOI 10.1109/TIM.2022.3146950

## 规格文档（必读）
见 `docs/method-tasks/M05_mohp.md` —— 含核心算法、curated 包索引、验证协议、期望结果。

## Curated 论文索引（复现核对原文）
未提取（DOI 10.1109/TIM.2022.3146950）
（若标注"未提取"：先向主 agent 请求补提取，或直接从 Zotero/DOI 获取原文核对）

## 实现要求
1. 新增 `vep_arena/methods/mohp_filter.py`，实现 `class MultiObjectiveHighPassFilter`：
   - `fit(train_x, train_y) -> self`
   - `predict(x) -> (labels, scores)`
   - 数据形状: `trials x subbands x channels x samples`（与 TRCA/CCA 一致）
2. 注册方法：在 `vep_arena/methods/__init__.py` 或 `traditional.py` 导出
3. 新建任务脚本 `tasks/methods/M05_mohp/run.py`（参考 `tasks/baselines/BL01_ssvep_benchmark/run.py` 的 make_model 模式）

## 验证协议
- 数据集: BL01
- 协议: 对齐 TRCA/SSCOR 类; Benchmark 9ch
- 预处理与 BL01 一致（cue 跳过 0.5s、潜伏期 0.14s、notch 50Hz、filterbank）

## 期望结果与验收
1. 结果写入 `tasks/methods/M05_mohp/results/`（trials.csv + summary.csv + figures）
2. 与规格文档"期望结果"的论文数字 diff ≤ 2%（acc）
3. 汇报：实现文件、注册方式、Benchmark/BETA 各窗口 acc/ITR 表、与论文对比结论
4. 更新 `docs/method-tasks/README.md` 状态列 → 已完成

## 环境
- Python: `D:\ProjData\proj_pythonep_arena\.venv\Scripts\python.exe`
- 数据集: `D:/ProjData/datasets`（local_paths.json 已配置）
- 运行示例: `.venv\Scripts\python.exe tasks/methods/M05_mohp/run.py --workers 2 --resume`
