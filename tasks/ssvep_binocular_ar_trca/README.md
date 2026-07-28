# Ke 2025 双目 AR SSVEP 基线

## 任务定位

本 task 面向 Ke et al. 2025 双目 AR SSVEP 数据集，在 Arena 共享方法接口上执行受试者内 leave-one-block-out 的 CCA、FBCCA、TRCA 和 ETRCA 评估。

当前实现属于“参考论文与公开 MATLAB 代码的 Arena 复现”，不是逐位一致的 MATLAB 复刻：

- 事件协议、0.14 s 视觉延迟、双频参考信号和 ITR 分母依据公开代码；
- 连续预处理与滤波由 SciPy 实现，属于明确记录的数值适配；
- 结果不得在未完成 MATLAB/EEGLAB 数值对照前表述为官方代码等价结果。

## 数据与论文

- 论文：Ke et al., “Dataset of binocularly coded steady-state visual evoked potentials recorded with an augmented reality headset,” Scientific Data, 2025，doi:10.1038/s41597-025-05696-0。
- 数据与 MATLAB 代码：Figshare article 26768287。
- 本地默认路径：`D:/ProjData/datasets/ssvep_binocular_ar`，可通过 `--root` 覆盖。
- 数据范围：24 名受试者、8 个目标、三个实验。

本地 zip 清单对应的受试者范围：

- `LF,MF`：`1-14`；
- `SFSP,SFDP,DFSP,DFDP`：`1-13,15`；
- `DFDP1,DFDP3,DFDP5`：`1-8,16-24`。

## 协议

- 主通道：`PO7, PO8, PO5, PO4, PO3, POz, PO6, O1, Oz, O2`。
- block：每个 session 10 个 block；默认合并两个 session，共 20 个 block。
- 裁剪：事件 onset 后增加固定 0.14 s 视觉延迟。
- 交叉验证：受试者内 leave-one-block-out。
- ITR 时间：`window + 1.0 s`，不额外计入 0.14 s 视觉延迟。
- 预处理：SciPy 连续去均值、49–51 Hz notch、5–95 Hz band-pass，再使用公开代码风格的 Chebyshev filter bank。
- 双频参考：每个目标拼接左右眼频率的 sine/cosine 谐波参考。

## 冒烟验证

```powershell
.venv\Scripts\python.exe tasks\ssvep_binocular_ar_trca\run.py `
  --subjects 1 `
  --tasks LF,DFDP,DFDP1 `
  --windows 0.1,0.5,1.0 `
  --methods CCA,FBCCA,TRCA,ETRCA `
  --out tasks\ssvep_binocular_ar_trca\results\smoke_sub001_lf_dfdp
```

## 全量运行

实验 1：

```powershell
.venv\Scripts\python.exe tasks\ssvep_binocular_ar_trca\run.py `
  --subjects 1-14 `
  --tasks LF,MF `
  --windows 0.1:0.1:3.0 `
  --methods CCA,FBCCA,TRCA,ETRCA `
  --workers 6 `
  --resume `
  --out tasks\ssvep_binocular_ar_trca\results\experiment1_trca
```

实验 2：

```powershell
.venv\Scripts\python.exe tasks\ssvep_binocular_ar_trca\run.py `
  --subjects 1-13,15 `
  --tasks SFSP,SFDP,DFSP,DFDP `
  --windows 0.1:0.1:3.0 `
  --methods CCA,FBCCA,TRCA,ETRCA `
  --workers 6 `
  --resume `
  --out tasks\ssvep_binocular_ar_trca\results\experiment2_trca
```

实验 3：

```powershell
.venv\Scripts\python.exe tasks\ssvep_binocular_ar_trca\run.py `
  --subjects 1-8,16-24 `
  --tasks DFDP1,DFDP3,DFDP5 `
  --windows 0.1:0.1:3.0 `
  --methods CCA,FBCCA,TRCA,ETRCA `
  --workers 6 `
  --resume `
  --out tasks\ssvep_binocular_ar_trca\results\experiment3_trca
```

## 产物与边界

每个结果目录应包含：

- `manifest.json`：协议、环境、实现身份和运行状态；
- `trials.csv`、`predictions.csv`、`summary.csv`、`subject.csv`；
- `runtime.csv`、`unit_manifest.csv`；
- accuracy/ITR 曲线、热图和 `confusions/*.npy`。

`results/` 目录由 Git 忽略。原始 EEG、下载包、CSV/NPY、图片和缓存均不得进入提交或 PR。
