# Tasks

Tasks are concrete research or validation questions. A task owns its runnable
entry points, notes, and generated results.

## 三层中的 task 边界

本目录属于第三层；[数据与核心的职责](../docs/architecture.md) 不在每个 task 中重新实现。专题 task 可以直接读取数据，保留自己的分析逻辑，不必强行接入分类 runner。是否公开、是否研究某个范式，是属性，不是新的架构层。

README 固定说明：问题；输入数据/上游结果版本；入口和配置；输出表/数组的含义；校验方式和已验证范围。同一问题的参数变化优先用配置，不为每个窗口、通道组合或 seed 新建目录。

采用 [接入回环 T1–T4](../docs/integration_workflow_zh.md) 和 [产物类型契约](../docs/result_artifact_contract.md)。分类任务保留逐试次预测；纯分析任务保留来源、参数和自己的表/数组，不要求虚构 accuracy 或 predictions。新入口逐条适配，现有入口和结果不因文档更新而批量迁移。

Public task layout:

```text
tasks/
  methods/          public method reproductions and benchmark runners
  <dataset>_<scope>_<purpose>/
  _legacy/
```

The package under `vep_arena/` stays reusable:

```text
vep_arena/data/       dataset adapters and metadata
vep_arena/methods/    algorithm modules
vep_arena/plots/      shared plotting primitives
vep_arena/neuroviz/   MNE-oriented views
```

Each task should be small and readable:

```text
tasks/<dataset>_<scope>_<purpose>/
  README.md
  run.py
  plot_acc_itr.py
  results/
```

Naming convention:

- `run.py`: runs the task's main experiment.
- `plot_<view>.py`: builds a task-specific figure from existing outputs.
- `results/`: local outputs owned by the task. This directory is ignored by
  default to prevent accidental commits of predictions, caches, and large logs.

Large datasets and external toolboxes stay outside the repository.

The consolidated method runners are indexed in [`methods/README.md`](methods/README.md).

## Cleanup convention

When a smoke run, partial run, or superseded result is no longer the current
analysis source, keep it under the owning task's `results/_legacy/` directory
instead of deleting it. Add a date-stamped note such as `NOTES_YYYYMMDD.md` in
the task root explaining:

- which result directories are current,
- what issue was validated or fixed,
- which outputs were moved to `_legacy`,
- which outputs should feed the next report or analysis step.

When a whole task folder is historical rather than a current workstream, move
it under `tasks/_legacy/YYYYMMDD_<reason>/` and add a short README in that
legacy folder. Do not move current report/task roots merely because their
`results/` are ignored by git.
