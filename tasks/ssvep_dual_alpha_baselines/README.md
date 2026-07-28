# Dual-Alpha 双频 SSVEP 基线

## 任务定位

本 task 面向 GigaDB 102557 Dual-Alpha 数据集，复现公开代码覆盖的 ETRCA 与 FBDCCA 分类基线，并统一写出 Arena 运行清单、trial 级预测和公开结果表对照。

实现身份：

- ETRCA：Arena ensemble TRCA，与公开脚本的 `meegkit.trca.TRCA(..., ensemble=True)` 调用对齐；
- FBDCCA：分数公式和固定权重来自公开 `fbdcca_process.py`，滤波后端作为复现身份单独记录；
- 不在公开代码范围内的算法或范式组合必须标记为扩展实验，不能混入公开范围结果。

## 数据与论文

- 数据论文：Sun et al., “Dual-Alpha: a large EEG study for dual-frequency SSVEP brain-computer interface,” GigaScience, 2024，doi:10.1093/gigascience/giae041。
- 数据与公开代码：GigaDB 102557，doi:10.5524/102557，CC0 1.0。
- 双频方法论文：Sun et al., IEEE TBME, 2023，doi:10.1109/TBME.2022.3212192。
- 本地默认路径：`D:/ProjData/datasets/ssvep_dual_alpha_gigadb_102557`，可通过 `--root` 覆盖。

## 协议

- 范式：`Checkerboard_Arrangment`、`Binocular_Vision`、`Binocular-Swap_Vision`。
- 目标与 block：40 个目标，每名受试者 5 个 epoch/block。
- 通道：CA/BV 使用公开脚本的 9 通道；BsV 使用 64 通道。
- 时间窗：`0.2:0.2:2.0`。
- 交叉验证：受试者内 5-block leave-one-block-out。
- ITR 时间：`window + 0.5 s`。
- 公开范围：ETRCA 覆盖 CA/BV/BsV；FBDCCA 覆盖 CA/BV。

## 全量运行

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
.venv\Scripts\python.exe tasks\ssvep_dual_alpha_baselines\run.py `
  --subjects 1-35 `
  --paradigms Checkerboard_Arrangment,Binocular_Vision,Binocular-Swap_Vision `
  --windows 0.2:0.2:2.0 `
  --methods ETRCA,FBDCCA `
  --fbdcca-filter-backend mne-fir `
  --workers 8 `
  --resume `
  --out tasks\ssvep_dual_alpha_baselines\results\official_baselines
```

## 当前验证记录

2026-07-08 全量记录：

- 105/105 个受试者/范式单元完成；
- `trials.csv` 8750 行，`predictions.csv` 350000 行，`summary.csv` 50 行；
- ETRCA 最优点与公开表差异不超过 0.16 pp；
- FBDCCA 2.0 s 最优点：CA 差异 +0.36 pp，BV 差异 +0.74 pp。

该轮 MNE-FIR 修复由历史 PsychoPy 环境生成；后续正式重跑必须使用仓库 `.venv`，并在 manifest 记录解释器和后端。

## 预处理与已知限制

- ETRCA/TRCA：使用公开脚本的 Chebyshev filter bank，默认 7 个子带；
- FBDCCA：使用公开代码的频带、固定权重和双频 sine/cosine 参考，默认 5 个子带；
- 正式 FBDCCA 运行必须使用 `--fbdcca-filter-backend mne-fir`；
- 旧 SciPy FIR 近似只能保留在显式命名的 legacy/diagnostic 目录。

公开 FBDCCA 类会先将 epoch 裁剪到目标时间窗，再调用 MNE FIR。短窗下 MNE 可能提示 FIR 长度大于信号长度。Arena 为保持公开代码顺序保留该行为，并在 manifest 记录为已知限制；该警告不能被解释为短窗滤波已经干净有效。

## 产物与边界

生成结果全部位于 `results/`，由 Git 忽略。仓库只提交 runner、测试、协议、NOTES 和报告模板，不提交 EEG、CSV/NPY、图片、下载包或缓存。
