# Tasks

Tasks are concrete research or validation questions. A task owns its runnable
entry points, notes, and generated results.

Current task root layout:

```text
tasks/
  benchmark_decision_channel_capacity/
  benchmark_signal_channel_analysis/
  beta_ssvep_9ch_baselines/
  beta_ssvep_9ch_official_grid/
  cvep_nbrs_jfpm_tsinghua_2024_baselines/
  ssvep_binocular_ar_trca/
  ssvep_binocular_dataset_smoke/
  ssvep_benchmark_9ch_feature_analysis/
  ssvep_hd_200target_tdca_sample/
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
