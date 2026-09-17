# Result Artifact Contract

This is the project-level standard for VEP Arena experiment outputs.

## Complete Run Directory

A complete algorithm run directory should contain:

- `manifest.json`: dataset, protocol, methods, subjects, windows, channels, code
  entry point, cache/preprocess notes, status, and timestamp.
- `trials.csv`: one row per evaluated unit used for aggregate metrics, usually
  method/window/subject/block.
- `predictions.csv`: one row per classified trial or command. Required for
  confusion matrices and error-structure analysis.
- `summary.csv`: aggregate method/window metrics.
- `subject.csv`: subject-level aggregate metrics when subjects exist.
- `block.csv` or equivalent split-level table when block/session splits exist.
- `runtime.csv`: timing rows when the runner has meaningful fit/predict stages.
- `confusion_*.npy` or `confusions/*.npy`: confusion matrices generated from
  `predictions.csv`.
- figures generated from CSV/NPY artifacts, not from hidden in-memory state.

`predictions.csv` is mandatory for any result that may be used for scientific
analysis beyond aggregate accuracy/ITR. If a runner cannot write predictions,
the manifest must say why and the result must be marked as aggregate-only.

## Aggregate-Only Directory

Aggregate or comparison directories may intentionally contain only:

- `summary.csv`
- optional `subject.csv`, `block.csv`, `paired_stats.csv`, or source ledgers
- `manifest.json` or an explicit source ledger

These directories should not be treated as raw experimental runs. They are
derived views. Plotting from them is acceptable; confusion/error analysis is
not.

## Required Prediction Columns

Use these columns when the task naturally supports them:

- `method`
- `subject`
- `block` or `session`
- `trial_index`
- `true`
- `pred`
- `correct`
- `window` or `window_ms`
- dataset-specific context such as `targets`, `channels`, `time_samples`

Labels should be stored as the human-facing one-based labels when that matches
the dataset/task output. If zero-based labels are used internally, the manifest
must state the convention.

## HD-200 Online Table 2

The HD-200 online Table 2 runner follows this contract with:

- `online_table2_reproduction.csv`
- `online_table2_predictions.csv`
- `online_table2_manifest.json`
- `confusions/confusion_<subject>_<targets>target.npy`
- `online_table2_reproduction.png`
- `online_table2_confusions.png`
- `online_table2_top_errors.csv`

Filtering caches are not final result artifacts. They live under the dataset
derivatives directory and may be reused by reruns.

## 稳定对应格式 v1：所有 task 共用来源，不强制共用分析逻辑

更新：2026-09-17。以下补充前面的产物规则；不更名已有文件，不要求一次性重写全部历史输出。新接入路径按本节验收；旧路径可先用旁路 manifest/映射适配。这里是文档契约，尚非已覆盖全部 runner 的自动校验器。

### 1. 产物类型决定最低要求

| `artifact_profile` | 最低产物 | 不要求的内容 |
| --- | --- | --- |
| `classification` | manifest、逐试次 predictions、评测单元 trials、summary；有被试/split 则保留对应聚合；可定位的输入与划分；适用的 runtime | 没有提供的 score/probability 不伪造 |
| `analysis` | manifest、来源引用、实际分析参数、task 声明的表/数组及其列/轴/单位说明 | 不强制分类标签、accuracy、ITR 或 predictions |
| `comparison` | manifest、带版本的来源结果清单、对齐/兼容性结论、比较表 | 不重复拷贝全部上游模型/原始数据；可引用受控存储 |
| `aggregate_only` | 聚合表、manifest/来源清单、缺失粒度说明 | 不能冒充完整运行，不能倒推逐试次混淆矩阵 |

前文 Complete Run Directory 与 prediction 要求针对分类实验。纯信号/专题分析遵守 `analysis` 契约，不因为不输出预测而判失败。比较已保存的 aggregate-only 结果时，声明可用统计粒度；没有试次身份证据就不宣称逐试次严格配对。

### 2. 一条可回查的来源链

```text
代码版本 + 源文件索引 + 有效参数
       → 试次/事件与选样记录 → split（如适用）
       → 预处理 / 学习状态 → 预测或分析数组
       → 汇总 / 比较 / 图表
```

最低对应信息如下；可以内联 manifest，也可以引用旁路文件，不要求拆成大量文件：

| 对象 | 必须能确定的内容 | 推荐承载 |
| --- | --- | --- |
| 代码 | git commit、入口、命令；工作区是否 dirty，若 dirty 则保存实际补丁/代码快照引用 | manifest `code` |
| 原始数据 | dataset ID、发布/本地修订、格式版本、源文件 ID/相对路径、内容指纹及缺失记录 | manifest `data` + `sources.csv` 或等价已有索引 |
| 试次/事件 | 稳定 trial ID、源文件/事件、被试/session/block、标签映射、通道/单位、采样率、起止样本 | `trial_index.csv` / loader 元数据；分析输入可用事件索引 |
| 有效参数 | 配置解析后的实际值和默认值、随机种子、软件版本/锁文件指纹；不能只存一条不完整命令 | manifest `resolved_config` / `environment` |
| 协议划分 | train/validation/test 身份、校准/预训练权限、训练组合、时间窗含义 | manifest `protocol` + `splits.json` 或等价确定性划分清单 |
| 预处理 | 名称/版本、操作顺序、滤波/裁窗参数、实际处理范围、学习状态拟合样本 | manifest `preprocess`，遵守预处理契约 |
| 中间状态 | 缓存配方/键、转换参数；拟合滤波器、标准化参数、checkpoint 等状态或可验证重建配方 | manifest `intermediates` / `models/` / `.npz` / 参数 JSON |
| 输出 | profile、schema version、文件用途、列/轴/单位、记录数、来源和校验指纹 | manifest `artifacts` |

源文件指纹可按文件首次登记后复用，不要求每次读取重算所有大文件。仅路径、大小或 mtime 不能证明内容一致；未做内容验证的历史记录应明确 `verification: unverified`，不得冒称已校验来源。

公共 manifest 使用可移植的根目录别名和相对路径。真实根目录、私有被试映射、受限数据和模型在受控本地/私有存储；哈希不是发布许可，不自动将来源清单推送公共仓库。

### 3. 最小 manifest 示例

以下是未执行的格式示意；占位项需要运行时补齐，不是有效验收证据。每个来源路径都相对当前 manifest 或有明确的根目录别名。

```json
{
  "schema_version": "1.0",
  "run_id": "example_run",
  "task_id": "example_task",
  "artifact_profile": "classification",
  "status": "planned",
  "code": {"git_commit": null, "entry_point": null, "command": [], "dirty": false},
  "data": {"dataset_id": null, "revision": null, "sources": "sources.csv", "trial_index": "trial_index.csv"},
  "resolved_config": {},
  "environment": {},
  "protocol": {"id": null, "splits": "splits.json", "training_access": null},
  "preprocess": {"name": null, "version": 1, "parameters": {}, "time_support": null},
  "methods": [],
  "intermediates": [],
  "artifacts": [],
  "acceptance": "acceptance.json"
}
```

`methods` 记录实现、实际参数、seed/重复编号、输入变换和能力；`artifacts` 每项记录 path、role、schema/列含义、记录数和适用的内容指纹。analysis 无 split 或 method 时明确为 null/空并说明原因；任务自有分析参数仍放 `resolved_config`。不强迫分析数据伪装成分类批次。

保留已有 `status=partial/complete` 的运行含义，并允许 `planned/failed`；**执行 complete 不等于验收 accepted**。验收结论和证据另按 [接入流程](integration_workflow_zh.md) 记录。

### 4. 分类结果的身份与覆盖

保留现有 `method, subject, block/session, true, pred, window` 列。新增身份可以是列，也可以有无歧义的旁路映射。一个预测的唯一键是：

```text
(run_id, method_instance_id, split_id, window, trial_id)
```

`method_instance_id` 区分实现/参数/seed/重复运行；`split_id` 区分完整训练/验证/测试身份及训练块组合，而不只是测试 block 的编号。同一 trial 在不同训练组合下的预测不能互相覆盖。若还有决策时间点等粒度，扩展键并写明。

标签 base 与 target → frequency/phase/code 的映射必须明确；子集标签非连续时记录 score 列的 `class_ids`，不默认“标签就是列下标”。跨方法配对按样本和协议身份对齐，不按 CSV 行位置。

完成度检查针对预期唯一键集合，拒绝重复或未知身份并报告缺失。实际样本数来自有效唯一记录；不以完整数据集规格代替本次子集。单个 method/window 的汇总和混淆矩阵只使用声明的范围，不能混入其他窗口。

### 5. 缓存、参数与重建边界

缓存是可丢弃加速层，不是实验结果来源的替代品。键至少覆盖数据内容/adapter 版本、选样身份、通道、时间范围、采样率、预处理名称/版本/实际参数与输出布局；更改相关因素须失效或生成新缓存。训练得到的中间量还必须关联 split，不能跨测试边界复用。

不要求保存每个滤波中间数组。确定性变换保留输入身份、配方、版本和数值容差；不能可靠重建的中间状态必须保存。正式预测用到的随机训练 checkpoint 和学习状态应保存或引用可核验的已有文件。暂不支持状态重放时明确能力限制，不能宣称模型重载已经验证。

resume 必须比对计算相关配置指纹及预期唯一键；与旧结果不匹配时拒绝续跑或生成新 run。增加 subject/window 等计算范围不能静默混入旧 run；尚无显式扩展机制时新建 run。只是线程数等执行参数变化是否允许，也应有规则并记录。

### 6. 兼容与验收

没有 `schema_version` 的旧结果按 legacy 读取，旁路映射写出已知字段、未知字段和来源，不编造缺失的逐试次结果。加字段优先保持兼容；标签、单位、轴或键的语义变化才升级版本并提供映射/重跑说明。

最低核对：文件可读；身份/覆盖正确；从 predictions 或分析数组重算一份指标一致；从记录参数重建一个派生产物；每步能回查代码、输入与中间状态。方法真实性能和论文结论由独立 task 验证，不混入格式/流程的完成条件。
