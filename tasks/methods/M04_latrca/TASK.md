# M04_latrca — LA-TRCA 实现任务

## 目标
在 vep_arena 实现 LA-TRCA 方法，并在 Benchmark/BETA 上复现论文结果。

## 论文
IEEE TNSRE (2022), DOI 10.1109/TNSRE.2022.3162029

## 规格文档（必读）
见 `docs/method-tasks/M04_latrca.md` —— 含核心算法、curated 包索引、验证协议、期望结果。

## Curated 论文索引（复现核对原文）
34_2022_huang_latrca
包位置: `curated/papers/03-vep-algorithm/11-spatial-filter/34_2022_huang_latrca/`（已提取 ✅）

> 若标注 10.1109/TNSRE.2022.3162029）
（若标注"未提取"：先向主 agent 请求补提取，或直接从 Zotero/DOI 获取原文核对）

## 实现要求
1. 新增 `vep_arena/methods/la_trca.py`，实现 `class LATRCA`：
   - `fit(train_x, train_y) -> self`
   - `predict(x) -> (labels, scores)`
   - 数据形状: `trials x subbands x channels x samples`（与 TRCA/CCA 一致）
2. 注册方法：在 `vep_arena/methods/__init__.py` 或 `traditional.py` 导出
3. 新建任务脚本 `tasks/methods/M04_latrca/run.py`（参考 `tasks/baselines/BL01_ssvep_benchmark/run.py` 的 make_model 模式）

## 验证协议
- 数据集: BL01
- 协议: windows 0.2-1.0 s; 对照 TRCA/eTRCA/CCA
- 预处理与 BL01 一致（cue 跳过 0.5s、潜伏期 0.14s、notch 50Hz、filterbank）

## 期望结果（论文 Table，curated 包 34 提取）
- 统计表：TRCA vs LA-TRCA 在 Dataset I（0.2-1.4s）和 Benchmark（0.2-1.0s）各窗口的显著性（p 值多 <0.05 或 <0.0001）
- 刺激布局：4×3（垂直间距 100px，水平 500px）/ 3×4 / 2×6 对比
- 验收：Benchmark 上 LA-TRCA ≥ TRCA（各窗口 acc 对比曲线），差异显著性方向一致

## 验收流程（原）
1. 结果写入 `tasks/methods/M04_latrca/results/`
2. 与规格文档期望数字 diff ≤ 2%
3. 汇报实现/注册/结果表/对比结论
4. 更新 docs/method-tasks/README.md