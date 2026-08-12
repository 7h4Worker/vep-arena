# BETA 数据集接入记录

## 目的

本文件记录 BETA 数据集接入 VEP Arena 前需要固定的数据事实、未确认事项和接入顺序。当前 PR 只做事实记录，不写 loader，不改现有 runner。

Arena 后续接入 BETA 时应遵守：

- 不复制 Benchmark 的硬编码结构。
- 不在未确认原始文件结构前写死 loader。
- 所有 preprocessing、protocol、channel subset 和 timing 都要写入 manifest。
- BETA 的首个接入目标是 smoke + MNE QA，而不是直接全量 benchmark。

## 当前本机状态

已检查本机路径：

```text
D:\ProjData\datasets
```

当前可见数据集目录：

```text
D:\ProjData\datasets\erp_ssvep
D:\ProjData\datasets\mne
D:\ProjData\datasets\ssvep_benchmark
```

当前未找到明确的 BETA 原始数据目录或 BETA `.mat` 文件。因此 BETA loader 暂不应实现为“已可运行”状态。下一步需要先下载或放置原始数据，并确认文件名、变量名和 shape。

建议目标路径：

```text
D:\ProjData\datasets\ssvep_beta
```

## 公开资料口径

BETA 通常指 Liu et al. 2020 的大规模 SSVEP benchmark 数据集。公开工具中常见名称包括：

- `BETA`
- `Liu2020BETA`
- `ssvep_beta`

当前记录参考：

- MOABB `Liu2020BETA` 文档：<https://moabb.neurotechx.com/docs/generated/moabb.datasets.Liu2020BETA.html>
- 论文网页：<https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00627/full>
- PMC 镜像：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7324867/>
- Figshare 数据页：<https://figshare.com/articles/dataset/The_BETA_database/12264401>

这些来源用于确认高层数据事实；具体 loader 仍必须以后续本机原始文件检查结果为准。

已从公开资料和常见工具适配器中确认的高层事实：

| 字段 | 口径 |
| --- | --- |
| paradigm | SSVEP |
| subjects | 70 |
| targets/classes | 40 |
| channels | 64 EEG channels |
| sessions | 1 |
| blocks | 每 subject 4 blocks |
| sampling rate | 采集 1000 Hz，公开处理口径为 downsample 到 250 Hz |
| epoch length | 公开矩阵口径为 3.0 s，750 samples at 250 Hz |
| matrix shape | `[64, 750, 4, 40]`，即 `[channels, time points, block, target]` |
| 数据用途 | 大规模 SSVEP-BCI benchmark |
| Arena 首要任务 | 与 Tsinghua Benchmark 口径对齐，先做 QA 和 smoke |

MOABB 文档还记录了一个重要 timing 差异：S1-S15 为 experienced subjects，stimulation duration 为 2 s；S16-S70 为 naive subjects，stimulation duration 为 3 s；视觉 cue 为 0.5 s。Arena 接入时不能只用一个全局 `stim_seconds` 糊过去，至少要在 spec 或 metadata 中记录 subject group。

需要在原始文件落盘后再次确认的字段：

| 字段 | 待确认内容 |
| --- | --- |
| sampling rate | 本机文件保存的是 1000 Hz 原始数据还是 250 Hz 处理版本 |
| trial length | 文件中是否只含 3.0 s epoch，还是含 cue/pre-stim 额外 samples |
| file format | `.mat` 变量名、维度顺序 |
| event metadata | target id、frequency、phase、run/block |
| channel names | 64 通道名称、是否符合标准 montage |
| artifacts | 是否已有 bad channels / reject 标记 |

## 与 Benchmark 的关系

BETA 和当前 Benchmark 接入目标相似，都是 40-target SSVEP 数据集，但不能直接假设文件结构完全一致。

Benchmark 当前 Arena 口径：

```text
subjects = 35
blocks = 6
classes = 40
sampling_rate = 250
cue_seconds = 0.5
latency_seconds = 0.14
raw shape = channels x samples x classes x blocks
```

BETA 后续应该尽量暴露与 Benchmark 相同的输出接口：

```text
classes x blocks x channels x samples
```

但 adapter 内部必须以 BETA 原始文件为准，不应为了兼容而假装它就是 Benchmark shape。

## 建议通道策略

首批接入建议提供两个 channel preset：

### `beta_all64`

使用 BETA 全部 64 EEG channels。

用途：

- MNE montage QA。
- ERP/topomap/PSD/SNR。
- 后续全通道模型或脑区分析。

### `beta_occipital_9ch`

对齐当前 Benchmark 9ch 的 occipital/parietal subset。

Benchmark 9ch 当前名称约定：

```text
Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
```

后续需要从 BETA channel names 中确认这些通道是否存在，不能只按 index 对齐。

用途：

- 与现有 Benchmark 9ch 结果做更接近的算法口径对比。
- 快速 smoke。

## 建议预处理契约

BETA 首批复用现有契约名称，但要在 manifest 中写 dataset：

| 契约 | 用途 |
| --- | --- |
| `raw_epoch` | MNE QA、CCA、ERP/image-VEP 基础分析 |
| `ssvep_toolbox_fb5` | FBCCA、TRCA/FBTRCA、ETRCA、TDCA |
| `dnn_cheby3_fb` | 只有在 DNN 迁移/重训明确需要时使用 |
| `erp_mne_basic` | ERP/RSVP/image-VEP 扩展 |

BETA 接入时不要默认复用 Benchmark 的 `cue_seconds=0.5` 和 `latency_seconds=0.14`，除非原始资料和文件验证都支持。

## 建议 loader 接口

后续 `vep_arena/data/beta.py` 至少提供：

```python
def subject_file(data_root: Path, subject: int) -> Path: ...
def load_subject_raw(data_root: Path, subject: int) -> np.ndarray: ...
def load_subject_trials(data_root: Path, subject: int, window: float, channels: tuple[str, ...] | tuple[int, ...], spec: BetaSpec) -> np.ndarray: ...
def load_subject_filterbank(data_root: Path, subject: int, window: float, n_fbs: int, channels: ..., spec: BetaSpec) -> np.ndarray: ...
```

若 BETA 原始数据已经有事件表或 run/session 结构，loader 应保留这些信息，并在 MNE bridge 中导出 metadata。

## 首批接入顺序

### Step 1：数据事实确认

确认并记录：

- 数据目录。
- 文件命名。
- `.mat` keys。
- 原始 shape。
- channel names。
- sampling rate。
- blocks/trials。
- target frequency/phase table。

### Step 2：MNE QA smoke

生成：

- epoch trace。
- evoked trace。
- PSD/SNR。
- channel montage。
- topomap 如果本机绘图栈稳定则开启，否则记录跳过原因。

### Step 3：算法 smoke

先跑小范围：

```text
subject 1
block 1
window 1.0 或 2.0
methods: CCA, FBCCA, TRCA/FBTRCA
```

### Step 4：多窗口 benchmark

在 smoke 稳定后再跑：

```text
subjects: 1-70
blocks: all
windows: 0.2-1.0 或按数据集推荐窗口
methods: FBCCA, TRCA/FBTRCA, ETRCA, TDCA
```

### Step 5：报告合并

输出到独立目录：

```text
results/beta_9ch/
results/beta_all64/
```

不要直接混入 `results/benchmark_9ch/`。

## 风险与注意事项

- BETA 文件结构可能与 Tsinghua Benchmark 不同，不能照抄 `benchmark.py`。
- BETA sampling rate 和 trial timing 需要以原始数据为准。
- 若 BETA 使用 64 通道，传统 9ch 结果和 all64 神经科学图应分开报告。
- DNN checkpoint 不能直接假定可迁移到 BETA；如果使用 DNN，需要明确是 cross-dataset inference、retrain 还是 fine-tune。
- MNE topomap 当前在本机环境中默认跳过，后续需要单独处理绘图栈稳定性。

## 下一步 PR

建议后续 PR 顺序：

1. `data/beta-file-inspect`：在 BETA 原始数据落盘后写 inspection 脚本，输出 keys/shape/channel table。
2. `data/beta-adapter`：实现最小 `beta.py` loader。
3. `feature/beta-mne-qa`：复用 MNE bridge 生成 BETA QA 图。
4. `experiment/beta-traditional-smoke`：跑 CCA/FBCCA/TRCA smoke。
