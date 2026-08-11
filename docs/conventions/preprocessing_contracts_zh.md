# 预处理契约

## 目的

Arena 中的结果必须能清楚回答：

```text
这个结果从什么原始数据来？
裁了哪一段？
做了哪些滤波、notch、baseline 或特征变换？
输出给算法的 shape 是什么？
这套预处理适用于哪些方法？
```

不同算法可以使用不同预处理，但不允许在报告中隐藏差异。所有 runner、manifest、summary、report 后续都应记录 `preprocess` 名称。

## 命名规则

预处理名称使用小写 snake_case：

```text
<paradigm_or_family>_<main_operation>_<variant>
```

示例：

- `raw_epoch`
- `ssvep_toolbox_fb5`
- `dnn_cheby3_fb`
- `erp_mne_basic`

如果同一算法为了兼容历史 checkpoint 必须使用特殊输入契约，名称必须保留算法来源，例如 `dnn_cheby3_fb`，不能泛称为 `filterbank`。

## 当前契约总览

| 名称 | 当前状态 | 主要用途 | 输出 |
| --- | --- | --- | --- |
| `raw_epoch` | 已存在 | CCA、MNE QA、ERP 基础分析 | `classes x blocks x channels x samples` |
| `ssvep_toolbox_fb5` | 已存在 | FBCCA、TRCA/FBTRCA、ETRCA、TDCA | `classes x blocks x 5 x channels x samples` |
| `dnn_cheby3_fb` | 已存在 | DNN checkpoint/global/fine-tune | `trials x 3 x channels x samples` |
| `erp_mne_basic` | 规划中 | ERP、RSVP、image-VEP、MNE evoked/topomap | `mne.Epochs` / `mne.Evoked` |

## `raw_epoch`

### 用途

直接裁剪目标时间窗，不做 filter bank。当前 CCA 使用这类输入，后续 MNE QA 和 ERP 基础分析也会从它派生。

### Benchmark 当前实现

代码位置：

- `vep_arena/data/benchmark.py`
- `load_subject_trials(...)`

裁剪规则：

```text
start = cue_seconds + latency_seconds
stop = start + window
```

Benchmark 默认：

```text
cue_seconds = 0.5
latency_seconds = 0.14
sampling_rate = 250
```

所以 2.0 s 窗口对应从 raw trial 的 0.64 s 开始，裁 500 samples。

### 输出 shape

```text
classes x blocks x channels x samples
```

Benchmark 9ch 时：

```text
40 x 6 x 9 x samples
```

### 适用方法

- CCA
- MNE QA 图
- ERP / RSVP / image-VEP 的事件锁定分析基础路径
- 需要原始目标窗的诊断脚本

## `ssvep_toolbox_fb5`

### 用途

SSVEP 传统 spatial-filter / reference-template 方法使用的标准 filter bank 输入。当前 Arena 的 FBCCA、TRCA、ETRCA、TDCA 都走这一类输入。

注意：当前 Arena 中的 `TRCA` 不是裸 raw TRCA，而是 filter-bank TRCA。更准确地说，它属于 FBTRCA 口径；`ETRCA` 是同一 filter bank 输入上的 ensemble 变体。

### Benchmark 当前实现

代码位置：

- `vep_arena/data/benchmark.py`
- `load_subject_filterbank(...)`
- `benchmark_filterbank(...)`
- `notch_50hz(...)`

处理顺序：

1. 从 raw trial 的 cue offset 开始取数据。
2. 保留 `latency + target window + extra_samples`。
3. 对这段较长数据做 50 Hz notch。
4. 对 notch 后数据做 Chebyshev-I bandpass filter bank。
5. 裁掉 latency，只保留目标窗。

这和“先裁最终目标窗再滤波”不同；它能减少短窗边缘效应。

### Notch

```text
50 Hz iircomb notch
Q = 35
scipy.signal.filtfilt
```

### Filter bank

默认 `n_fbs = 5`。

第 `k` 个子带：

```text
passband = [8*k, 90] Hz
stopband = [8*k - 2, 100] Hz
cheb1ord(wp, ws, gpass=3, gstop=40)
cheby1(order, rp=0.5)
filtfilt
```

如果短窗导致滤波器设计失败，当前实现会把 `gstop` 从 40 逐步降到 20。

### 输出 shape

```text
classes x blocks x n_fbs x channels x samples
```

Benchmark 9ch、5 子带时：

```text
40 x 6 x 5 x 9 x samples
```

TDCA 需要 delayed copy 时可以额外保留 `extra_samples`：

```text
classes x blocks x n_fbs x channels x (samples + extra_samples)
```

### 子带融合权重

当前传统方法使用：

```text
weight[idx] = (idx + 1) ** (-1.25) + 0.25
```

代码位置：

- `vep_arena/methods/traditional.py`
- `filterbank_weights(...)`

### 适用方法

- FBCCA
- TRCA / FBTRCA
- ETRCA / FB-eTRCA
- TDCA
- MSCCA / MSETRCA 等复用传统 filter bank 的方法

## `dnn_cheby3_fb`

### 用途

DNN checkpoint、Arena-side DNN global training 和 subject fine-tune 的输入契约。这个契约是为了保持 DNN 历史模型和 checkpoint 的输入兼容。

它也是 filter bank，但不能和 `ssvep_toolbox_fb5` 混称为同一预处理。

### Benchmark 当前实现

代码位置：

- `scripts/evaluate_dnn_checkpoints.py`
- `make_dnn_filter_bank(...)`
- `preprocess_subject_for_dnn(...)`

处理顺序：

1. 使用 `BenchmarkSpec.sample_slice(window)` 直接裁目标窗口，即 cue + latency 后的最终 target window。
2. 在裁好的目标窗上做 3 个 Chebyshev-I order-2 bandpass。
3. 使用 `sosfiltfilt`。
4. 不做 50 Hz notch。

### Filter bank

默认 `subbands = 3`。

第 `i` 个子带：

```text
passband = [8*i, 90] Hz
cheby1(N=2, rp=1, output="sos")
sosfiltfilt
```

### 输出 shape

当前 DNN evaluator 直接展开为 trial rows：

```text
trials x subbands x channels x samples
```

Benchmark 9ch、3 子带时：

```text
(classes * blocks) x 3 x 9 x samples
```

### 适用方法

- DNN global checkpoint inference
- DNN Arena-trained global
- DNN subject fine-tune

### 报告要求

当 DNN 与 FBCCA/TRCA/TDCA 放在同一张图中时，报告必须注明：

- 数据集和 protocol 可以对齐。
- DNN 使用 `dnn_cheby3_fb`。
- FBCCA/TRCA/TDCA 使用 `ssvep_toolbox_fb5`。
- 这不是完全相同的 signal preprocessing。

## `erp_mne_basic`

### 用途

ERP、RSVP、image-VEP 和 MNE 神经科学 QA 的基础契约。当前尚未完全实现，先作为接入 BETA 和事件锁定视觉范式的目标契约。

### 预期输入

数据集 adapter 应提供：

- channel names。
- sampling rate。
- event table。
- event id / condition。
- trial metadata。
- montage 或 channel layout。

### 推荐对象

```text
mne.Raw
mne.Epochs
mne.Evoked
```

### 推荐处理项

具体参数应由数据集和范式决定，但必须写入 manifest：

- re-reference。
- notch。
- bandpass。
- baseline correction。
- reject / annotation。
- epoch tmin / tmax。
- condition labels。

### 输出

神经科学分析层：

```text
mne.Epochs
mne.Evoked
```

算法输入层：

```text
trials x channels x samples
```

或模型需要的派生 tensor。

### 适用图表

- raw / epoch trace。
- PSD。
- ERP / evoked。
- ERP image。
- topomap。
- time-frequency。
- channel montage。

## Manifest 要求

后续 runner 的 `manifest.json` 至少应写入：

```json
{
  "dataset": "...",
  "paradigm": "...",
  "preprocess": {
    "name": "...",
    "version": 1,
    "description": "...",
    "channels": ["..."],
    "sampling_rate": 250,
    "cue_seconds": 0.5,
    "latency_seconds": 0.14,
    "window_seconds": 1.0,
    "notch": "...",
    "filterbank": "...",
    "baseline": null,
    "output_shape": "..."
  },
  "protocol": "...",
  "method": "..."
}
```

如果某个字段不适用，应写 `null` 或明确说明，而不是省略。

## Summary / Report 要求

`summary.csv` 后续应逐步增加或旁路提供以下字段：

```text
dataset
paradigm
preprocess
protocol
method
window
accuracy
itr
subjects
samples
```

图表标题不一定要塞满这些字段，但报告正文必须说明每条曲线对应的预处理契约。

## 当前待办

1. 将现有 runner 的 manifest 逐步改成引用上述契约名称。
2. 为 `ssvep_toolbox_fb5` 和 `dnn_cheby3_fb` 生成固定频响 QA 图。
3. 为 BETA 数据集补充对应的 raw/filterbank/MNE 契约。
4. 在最终 comparison report 中加入 preprocess ledger。

