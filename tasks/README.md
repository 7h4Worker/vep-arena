# Task 工作区

`tasks/` 用于沉淀具体的研究问题、复现任务和验证流程。每个 task 独立维护运行入口、协议记录和本地结果目录。

可复用代码保留在 `vep_arena/`：

```text
vep_arena/data/       数据集适配与元数据
vep_arena/methods/    算法模块
vep_arena/plots/      共享绘图组件
vep_arena/neuroviz/   MNE/神经信号可视化
```

每个 task 保持小而明确：

```text
tasks/<dataset>_<scope>_<purpose>/
  README.md
  NOTES_YYYYMMDD.md
  run.py
  plot_<view>.py
  results/
```

文件职责：

- `README.md`：任务定位、数据与论文、协议、运行入口和产物边界；
- `NOTES_YYYYMMDD.md`：按日期记录预处理、通道、滤波、算法与复现判断的修正；
- `run.py`：主实验入口；
- `plot_<view>.py`：从已生成 CSV/NPY 构建 task 图表；
- `results/`：task 本地结果目录，默认由 Git 忽略。

当前公开双目 SSVEP 工作流：

- `ssvep_binocular_dataset_smoke`：检查 Dual-Alpha 与 Ke 2025 Binocular AR 的文件完整性、事件、波形和频谱；
- `ssvep_dual_alpha_baselines`：复现 Dual-Alpha 三种范式的 ETRCA/FBDCCA 公开基线；
- `ssvep_binocular_ar_trca`：评估 Ke 2025 三个实验的 CCA/FBCCA/TRCA/ETRCA。

大型数据集、外部 toolbox、原始 EEG、生成 CSV/NPY、图片、模型和缓存均保留在仓库外或 `results/`，不得提交。
