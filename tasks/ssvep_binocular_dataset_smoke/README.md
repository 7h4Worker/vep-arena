# 双目 SSVEP 数据集冒烟检查

## 任务定位

本 task 只负责 Dual-Alpha 与 Ke 2025 Binocular AR 两个公开数据集的数据完整性、事件表、时域波形、频谱和码本检查，不运行全量分类器，也不用于声明算法复现精度。

## 数据来源

- Dual-Alpha：35 名受试者、40 个目标、5 个 block、250 Hz，包含 checkerboard、binocular vision 和 binocular-swap 三种范式；论文 doi:10.1093/gigascience/giae041，数据与代码 doi:10.5524/102557。
- Binocular AR：24 名受试者、8 个目标、三个实验，覆盖单/双频与单/双相位双目条件；论文 doi:10.1038/s41597-025-05696-0，数据与代码为 Figshare article 26768287。

证据角色固定为 `dataset smoke`：只验证文件、事件、通道、波形和频谱，不替代正式分类 task。

## 运行

```powershell
.venv\Scripts\python.exe tasks\ssvep_binocular_dataset_smoke\run.py `
  --dataset both `
  --out tasks\ssvep_binocular_dataset_smoke\results\latest
```

常用子集：

```powershell
# 只检查 Dual-Alpha：首名受试者、三个范式
.venv\Scripts\python.exe tasks\ssvep_binocular_dataset_smoke\run.py `
  --dataset dual-alpha `
  --dual-alpha-subject 1

# 只检查 Binocular AR：首个完整受试者、指定任务
.venv\Scripts\python.exe tasks\ssvep_binocular_dataset_smoke\run.py `
  --dataset binocular-ar `
  --ar-tasks LF,DFDP,DFDP1
```

## 产物

`results/latest/` 包含：

- 数据文件状态与 task inventory；
- `sample_summary.csv`、`sample_summary.json`、`run_manifest.json`；
- `smoke_report_zh.md`；
- 时域图、PSD 图和 Dual-Alpha 码本配对图。

`results/` 由 Git 忽略。经过审查后需要长期保留的结论应写入 task REPORT，不复制生成 CSV/图片到版本控制。

## 论文对齐

- `PAPER_ALIGNMENT_KE2025.md` 记录 Ke 2025 的采集、预处理、block、时间窗和 ITR 协议；
- `REPORT_20260708.md` 汇总两个数据集的当前状态，并区分公开结果表与 Arena 本地运行；
- Ke 正式分类使用论文 10 通道、两个 session 合并的 20 blocks、`0.1:0.1:3.0 s` 时间窗、49–51 Hz notch 和 5–95 Hz band-pass。

## 范围边界

- Dual-Alpha 已提供 epoch 级 CSV，可直接按 `condition` 与 `epoch` 组织 block；
- Binocular AR 为连续 BIDS-like EEG，需要根据 event table 切窗；
- smoke 读取器只用于信号检查，正式分类由独立 task 负责；
- 原始数据、下载包、CSV/NPY、图片和缓存不得提交。
