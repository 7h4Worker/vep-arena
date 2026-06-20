# VEP Arena 当前项目范围

## 定位

VEP Arena 是一个以视觉诱发电位为中心的 BCI 实验与对比工作区。当前项目不另起新版本、不拆出新仓库，所有新增能力都在现有 `vep_arena` 中连续演进。

项目优先服务以下范式：

- SSVEP：稳态视觉诱发电位，当前主线。
- cVEP / mVEP：编码或运动相关视觉诱发范式。
- RSVP / image-VEP / ERP：事件锁定、图像刺激和快速视觉序列相关分析。

项目不是泛 EEG 大平台，但底层设计要避免把 MI、emotion、ERP 等常见 BCI/神经科学分析完全堵死。MNE 是神经科学分析和可视化层的一等工具；NumPy/PyTorch tensor 是算法输入和模型训练层。

## 当前边界

短期边界是“把 Benchmark 和 BETA 这类 VEP 数据集纳入统一 Arena 口径”，而不是一次性重构成通用 EEG 框架。

当前保留：

- 现有 `vep_arena/data/benchmark.py` 数据路径。
- 现有传统方法、TDCA、DNN 评估脚本和结果输出格式。
- 现有 `results/` 与 `runs/` 生成物策略。

当前新增：

- 中文范围、流程、预处理契约和数据集接入记录。
- MNE 兼容的低侵入桥接与神经科学 QA 图。
- BETA 数据集接入所需的数据事实记录和 loader。

当前避免：

- 不使用 `v2` 命名。
- 不一次性推翻现有 runner。
- 不把当前 `.venv` 中缺少某个包误解为本机没有对应环境。
- 不在图表或报告中隐藏 preprocessing 差异。

## 核心对象

Arena 中每条结果至少要能追溯到以下对象：

```text
dataset -> paradigm -> preprocessing -> protocol -> method -> metrics -> report
```

### Dataset

数据集对象负责描述和读取数据文件，不负责决定算法。

必须记录：

- 数据路径和文件格式。
- subjects / sessions / runs / blocks。
- sampling rate。
- channel names / channel indices / montage。
- event、label 和刺激 metadata。
- trial timing，例如 cue、stim onset、latency、target window。

### Paradigm

范式对象描述任务语义。

首批支持：

- `ssvep`：frequency、phase、target id、window、ITR。
- `erp`：event lock、condition、baseline、evoked、单试次解码。

后续逐步支持：

- `cvep`：code sequence、template correlation、command-level metric。
- `rsvp`：target/non-target、类别不平衡、sequence-level metric。
- `image_vep`：图像类别、condition、ERP/topomap/decoding。

### Preprocessing

预处理必须有名字，并写入 manifest / summary / report。不同方法可以有不同输入契约，但必须显式说明。

当前已知契约：

- `raw_epoch`：直接裁剪目标窗口。
- `ssvep_toolbox_fb5`：Benchmark/传统方法使用的 5 子带 filter bank，含 50 Hz notch。
- `dnn_cheby3_fb`：DNN 使用的 3 子带 Chebyshev-I filter bank，无 notch。
- `erp_mne_basic`：MNE 事件锁定、baseline、evoked/topomap 等分析路径。

### Protocol

协议负责定义 train/test split 和评测范围。

当前主协议：

- subject-specific leave-one-block-out。
- Benchmark 9ch。
- 多窗口评测。

后续协议：

- cross-subject。
- calibration-size sweep。
- cross-dataset transfer。
- online-style simulation。

### Method

方法层保持小接口：

```text
fit(train, context)
predict(test, context)
save(path)
load(path)
```

深度学习方法可以在 adapter 内部保留 device、checkpoint、epoch、seed 和日志。

### Metrics / Report

结果输出保持统一：

- `trials.csv`
- `subject.csv`
- `block.csv`
- `summary.csv`
- `runtime.csv`
- `manifest.json`
- `figures/`
- `report.md`

每份报告必须说明 dataset、paradigm、preprocessing、protocol、method 和环境来源。

## BETA 接入范围

BETA 是当前项目的下一个数据集接入目标。接入顺序如下：

1. 建立 BETA 数据事实记录。
2. 写最小 loader，先输出与 Benchmark 相同风格的 epoch tensor。
3. 生成 MNE QA 图：channel/montage、epoch trace、PSD/SNR、topomap 或 evoked。
4. 跑 FBCCA/TRCA smoke。
5. 跑多窗口 benchmark。
6. 合并进 Arena 统一报告。

## 环境事实

当前已有 MNE/PsychoPy 环境：

```text
D:\ProjData\envs\erp_ssvep_lab\python.exe
mne 1.12.1
psychopy 2022.2.5
```

当前 `vep_arena/.venv` 继续用于已有 benchmark 和脚本。涉及 MNE/PsychoPy 的脚本应明确标注推荐 Python 环境，后续再决定是否将 MNE 作为项目依赖写入 `pyproject.toml`。

