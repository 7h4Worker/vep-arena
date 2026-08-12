# 2026-08-11 tasks/ 目录重组

## 动机

tasks/ 目录从 25+ 扁平目录重组为 4 个主题分类，解决命名混乱和无法从目录名判断内容的问题。

## 分类逻辑

| 类别 | 前缀 | 内容 | 推 remote |
|------|------|------|:---:|
| baselines/ | BL | 公开数据集标准算法评估 | ✅ |
| bessvep/ | BS | BesSSVEP 私有数据集（EMBC 9t / JBHI 16t / 35t） | ❌ |
| channel/ | CH | 信道容量理论分析（DMC → MIMO → 跨范式） | ❌ |
| probes/ | PB | 探索性分析 | ❌ |

## 完整映射表

### baselines/ (12)

| 新名 | 旧名 | git 状态 |
|------|------|---------|
| BL01_ssvep_benchmark | **(新建)** 从 scripts/ 迁移 | 新建 |
| BL02_ssvep_beta_9ch | beta_ssvep_9ch_baselines | git mv |
| BL03_ssvep_beta_official_grid | beta_ssvep_9ch_official_grid | git mv |
| BL04_ssvep_mfsc_160t | ssvep_160target_mfsc_tdca | git mv |
| BL05_ssvep_hd_200t | ssvep_hd_200target_tdca_sample | git mv |
| BL06_ssvep_binocular_ar | ssvep_binocular_ar_trca | git mv |
| BL07_ssvep_binocular_smoke | ssvep_binocular_dataset_smoke | git mv |
| BL08_ssvep_dual_alpha | ssvep_dual_alpha_baselines | git mv |
| BL09_ssvep_dual_freq_liang2020 | ssvep_dual_frequency_phase_liang2020 | git mv |
| BL10_ssvep_dual_freq_sun2024 | ssvep_efficient_dual_frequency_sun2024 | git mv |
| BL11_cvep_wn_tdca | cvep_broadband_white_noise_tdca | git mv |
| BL12_cvep_nbrs_jfpm | cvep_nbrs_jfpm_tsinghua_2024_baselines | git mv |

### bessvep/ (6 + 2 游离文件)

| 新名 | 旧名 | git 状态 |
|------|------|---------|
| BS01_baseline_embc_9t | ssvep_embc_9target_baselines | git mv |
| BS02_baseline_16t | ssvep_jbhi_16target_baselines | git mv |
| BS03_baseline_35t | ssvep_jbhi_35target_baselines | git mv |
| BS04_codebook_analysis | ssvep_embc_jbhi_binocular_codebook_analysis | mv (untracked) |
| BS05_fft_features | ssvep_embc_jbhi_fft_features | mv (untracked) |
| BS06_receiver_matrix | ssvep_embc_jbhi_receiver_matrix | mv (untracked) |
| _shared.py | ssvep_embc_jbhi_shared.py | git mv |
| BS05 内 plot_fft_features.py | plot_embc_jbhi_fft_features.py (游离) | mv |

### channel/ (7)

| 新名 | 旧名 | git 状态 |
|------|------|---------|
| CH01_dmc_benchmark | benchmark_decision_channel_capacity | git mv |
| CH02_dmc_multichannel | benchmark_multichannel_decision_channel_capacity | mv (untracked) |
| CH03_signal_structure | benchmark_signal_channel_analysis | git mv |
| CH04_mimo_continuous | continuous_mimo_channel_capacity | mv (untracked) |
| CH05_crossparadigm_efficiency | crossparadigm_information_efficiency | mv (untracked) |
| CH06_info_accumulation | information_accumulation_rate | mv (untracked) |
| CH07_bessvep_decision | ssvep_jbhi_decision_channel | mv (untracked) |

### probes/ (1)

| 新名 | 旧名 | git 状态 |
|------|------|---------|
| PB01_benchmark_9ch_features | ssvep_benchmark_9ch_feature_analysis | git mv |

## 路径引用修复

因嵌套层级 +1，所有脚本中的 `Path(__file__).resolve().parents[N]` 需要调整：

| 修复类型 | 变更 | 影响文件数 |
|---------|------|-----------|
| `TASK.parents[1]` → `[2]` | TASK = parent, 找项目根 | 10 |
| `Path().parents[2]` → `[3]` | 直接取项目根 | ~43 |
| `Path().parents[1]` → `[2]` | _shared.py（原平级文件） | 1 |
| `Path().parents[1]` → `[3]` | plot_fft_features.py（原平级，移入两层） | 1 |
| 硬编码旧路径 | `tasks/old_name/` → `tasks/category/new_name/` | 6 |

## Benchmark baseline 新建 (BL01)

从 `scripts/run_traditional_benchmark.py` 和 `scripts/run_tdca.py` 创建任务封装：
- `run.py` — CCA/FBCCA/ECCA/TRCA/ETRCA/SSCOR/ESSCOR
- `run_tdca.py` — TDCA 单独运行
- `README.md` — 数据集与协议说明

历史结果在 `results/benchmark_9ch/`（属 T3 清理范围）。

## bessvep/ 合并为统一 task

初始方案将 bessvep 拆为 BS01-BS06 共 6 个独立目录，但 BS04-BS06（分析脚本）强依赖 BS01-BS03（baseline 结果），跨目录硬编码路径导致重命名后断裂。

**合并方案**：bessvep/ 作为一个扁平 task，所有脚本在同级，results/ 下按 BS## 编号分区：
- `run_embc_9t.py`, `run_16t.py`, `run_35t.py` — baseline
- `analyze_codebook*.py`, `run_codebook_tdca.py` — 码本分析
- `plot_fft_features.py` — FFT 特征
- `build_receiver_matrix.py` — 接收算法对比
- `results/{BS01_embc_9t, BS02_16t, BS03_35t, BS04_codebook, BS05_fft, BS06_receiver}/`

分析脚本现在通过 `TASK / "results" / "BS02_16t" / ...` 访问 baseline 结果，纯相对路径。

## 冒烟测试与修复（2026-08-12）

对所有 tasks/ 下 66 个 `parents[N]` 路径引用进行自动化冒烟测试，发现 11 处错误 + 1 处旧 import 路径：

| 类型 | 数量 | 原因 |
|------|------|------|
| `analysis/` 子目录 `parents[4]→[5]` | 4 | 重组 +1 层但 sed 模式只覆盖 `parents[2]` |
| `_legacy/` 深层嵌套 `parents[2]→[5]` | 4 | 搬入 _legacy 时已断裂，重组加深 |
| `tasks/_legacy/` 历史遗留 `TASK.parents[1]→[3]` | 3 | 重组前即已错误 |
| 旧 import 路径 `tasks.ssvep_efficient_...` | 1 | hardcoded grep 未覆盖 import 语句 |

全部修复后重测通过。修复已同步到 main 和 feature/tasks-restructure-baselines 分支。

## 测试整理（2026-08-12）

tests/ 中 7 个测试文件按归属重新组织：

| 文件 | 归属 | 处理 |
|------|------|------|
| test_sscor.py | `vep_arena.methods` 库 | 留 tests/ |
| test_ecca.py | `vep_arena.methods` 库 | 留 tests/ |
| test_bprca.py | `vep_arena.methods` 库 | 留 tests/ |
| test_channel_capacity.py | `vep_arena.channel` 库 | 留 tests/ |
| test_embc_jbhi_data.py | `vep_arena.data` 库 | 留 tests/，修 2 处旧路径 |
| test_benchmark_multichannel_task.py | CH02 专属 | → `channel/CH02_dmc_multichannel/test_task.py` |
| test_extended_runner.py | CH01 专属 | → `channel/CH01_dmc_benchmark/test_runner.py` |

移入 task 后路径简化（`Path(__file__).resolve().parent` 即 task 根），全套 32 tests passed。

## scripts/ 整理（T2，2026-08-12）

scripts/ 目录从 36 个扁平脚本整理为 3 层结构：

### 迁入 baselines task（7 个）

| 脚本 | 目标 task |
|------|----------|
| check_benchmark_offsets.py | BL01_ssvep_benchmark |
| plot_sample_ssvep_grid.py | BL01_ssvep_benchmark |
| plot_beta_w02_20_compare.py | BL02_ssvep_beta_9ch |
| plot_beta_official_grid_compare.py | BL03_ssvep_beta_official_grid |
| generate_hd200_bigfig.py | BL05_ssvep_hd_200t |
| generate_hd200_channel_detail.py | BL05_ssvep_hd_200t |
| generate_hd200_ppt.py | BL05_ssvep_hd_200t |

路径修复：硬编码 `Path("d:/ProjData/...")` → `Path(__file__).resolve().parent` + `TASK.parents[2]`。

### 归档至 scripts/_legacy/（29 个）

第一代评估脚本（DNN、MVMD、report 生成等），已被 tasks/ 体系取代。

### 保留共享 runner（3 个）

- `run_traditional_benchmark.py` — CCA/TRCA 等传统算法
- `run_tdca.py` — TDCA 独立运行
- `run_toolbox_ssvep.py` — toolbox 对比

变更已同步至 `feature/tasks-restructure-baselines` 分支。

## 待处理

- T3: root `results/`（572MB）搬入对应 task 的 results/
- T3: root `runs/`（21GB）归档
- T4: `_` 前缀脚本审查（5 个文件）
- T5: 本地 analysis 内容提交（路径稳定后）
- theory 分支 `feat/theory-p0-p2-survey-v2` 中的 `survey_v2` 任务待合入 `channel/CH08_survey_p0p1/`
- `tasks/_shared/` 跨任务共享模块待建
