# Tasks

Tasks are concrete research or validation questions. A task owns its runnable
entry points, notes, and generated results.

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

