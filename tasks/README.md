# Tasks

任务按主题分为四个类别，每个类别下的目录以 `{类别ID}_{内容}` 命名。

```text
tasks/
├── baselines/                              公开数据集评估（推 remote）
│   ├── BL01_ssvep_benchmark/               Tsinghua Benchmark 40t (Wang 2017)
│   ├── BL02_ssvep_beta_9ch/                BETA 9ch 标准基线
│   ├── BL03_ssvep_beta_official_grid/      BETA 官方窗口网格
│   ├── BL04_ssvep_mfsc_160t/              160 目标 MFSC (Chen 2021)
│   ├── BL05_ssvep_hd_200t/                HD 200 目标 TDCA/TRCA
│   ├── BL06_ssvep_binocular_ar/           双目 AR SSVEP (Ke 2025)
│   ├── BL07_ssvep_binocular_smoke/        双目数据集 QA
│   ├── BL08_ssvep_dual_alpha/             Dual-Alpha 三范式
│   ├── BL09_ssvep_dual_freq_liang2020/    Liang 双频相位
│   ├── BL10_ssvep_dual_freq_sun2024/      Sun 双频
│   ├── BL11_cvep_wn_tdca/                 WN-BCI cVEP TDCA
│   └── BL12_cvep_nbrs_jfpm/              NBRS/JFPM cVEP
│
├── bessvep/                                BesSSVEP 私有数据集（统一 task，本地）
│   ├── run_embc_9t.py / run_16t.py / ...  BS01-03 baseline 评估
│   ├── analyze_codebook*.py               BS04 码本分析
│   ├── plot_fft_features.py               BS05 FFT 特征
│   ├── build_receiver_matrix.py           BS06 接收算法对比
│   └── results/BS01-BS06_*/               按编号分区的产出
│
├── channel/                                信道分析（理论体系，本地）
│   ├── CH01_dmc_benchmark/                DMC 容量 (Benchmark)
│   ├── CH02_dmc_multichannel/             多通道 DMC 扫描
│   ├── CH03_signal_structure/             信号结构探测
│   ├── CH04_mimo_continuous/              MIMO 连续扩展
│   ├── CH05_crossparadigm_efficiency/     跨范式信息效率
│   ├── CH06_info_accumulation/            dMI/dT 积累率
│   └── CH07_bessvep_decision/             BesSSVEP 决策信道
│
├── probes/                                 探索性分析（本地）
│   └── PB01_benchmark_9ch_features/       Benchmark 频率/TF/通道特征
│
├── _legacy/                                已归档历史任务
└── _shared/                                跨任务共享模块（待建）
```

## 目录规范

每个任务目录包含：
- `run.py` — 主运行入口
- `README.md` — 任务说明
- `results/` — 本地产出（gitignored）

命名：`{类别ID}_{内容}`，如 `BL01_ssvep_benchmark`、`CH03_signal_structure`。

## 清理规范

- 过期结果移入任务自身的 `results/_legacy/`，附 `NOTES_YYYYMMDD.md`
- 过期任务整体移入 `tasks/_legacy/YYYYMMDD_reason/`
