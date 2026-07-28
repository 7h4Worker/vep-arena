# VEP Arena

VEP Arena is a local-first Python workspace for visual evoked potential BCI
experiments. The current focus is SSVEP benchmarking across Benchmark, BETA,
and wearable datasets, with room for ERP, cVEP, RSVP, image-VEP, MNE QA, and
online experiment backends.

The project intentionally keeps datasets and large generated artifacts outside
source control. Code should stay small enough to read, while tasks own their
specific results and plots.

## Structure

```text
vep_arena/
  data/       dataset interfaces, presets, Benchmark and toolbox adapters
  methods/    CCA, FBCCA, TRCA/ETRCA, TDCA, MVMD, TRCANet helpers
  plots/      reusable plotting helpers
  neuroviz/   MNE bridge and neuroscience-oriented views

tasks/
  beta_ssvep_9ch_baselines/
  beta_ssvep_9ch_official_grid/

scripts/
  reusable command-line utilities and legacy-compatible entry points

results/      legacy local outputs, ignored by git
runs/         local caches, training outputs, and intermediate artifacts
```

Task directories are the preferred place for new study logic:

```text
tasks/<dataset>_<scope>_<purpose>/
  README.md
  run.py
  plot_acc_itr.py
  results/
```

`results/` under a task is ignored by default so large `predictions.csv`,
runtime logs, and intermediate files are not committed accidentally. Compact
CSV/PNG artifacts can be force-added when we explicitly want to publish them.

Experiment outputs should follow the project artifact contract in
`docs/result_artifact_contract.md`. In short: runnable experimental results
must preserve trial-level `predictions.csv` plus confusion artifacts whenever
classification predictions exist; summary-only directories are treated as
derived aggregate views, not complete experimental runs.

## Environment

The canonical local Arena Python is the repository virtual environment:

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe
```

Keep it synchronized from `pyproject.toml` and `uv.lock`:

```powershell
cd D:\ProjData\proj_python\vep_arena
uv sync --extra all
.venv\Scripts\python.exe scripts\check_neuro_env.py
```

`--extra all` installs the core scientific stack plus torch, MNE/MOABB, and
report-generation dependencies. Do not use bare `python` for official runs.
The older conda environment `D:\ProjData\envs\erp_ssvep_lab` is only a legacy
PsychoPy/online-experiment fallback, not the default Arena execution
environment.

Public contributions must also follow the
[reproduction discipline](docs/reproduction_discipline_zh.md),
[result artifact contract](docs/result_artifact_contract.md), and
[public repository boundary](docs/public_repository_boundary.md).

## Data

Datasets are expected outside this repository. Current local defaults:

```text
D:\ProjData\datasets\ssvep_benchmark
D:\ProjData\datasets\ssvep_beta
D:\ProjData\datasets\ssvep_wearable
```

BETA and wearable datasets are accessed through the local
SSVEP-Analysis-Toolbox adapter. The toolbox repository is treated as an external
reference, not vendored into this repo.

The BETA root can be overridden for task scripts:

```powershell
$env:SSVEP_BETA_ROOT = "D:\ProjData\datasets\ssvep_beta"
```

## Current BETA Baseline Task

Rebuild the `0.2-2.0s` BETA 9ch comparison plot from existing summaries:

```powershell
.venv\Scripts\python.exe tasks\beta_ssvep_9ch_baselines\plot_acc_itr.py
```

Run the full BETA 9ch baseline task:

```powershell
.venv\Scripts\python.exe tasks\beta_ssvep_9ch_baselines\run.py
```

The current methods are:

```text
CCA, FBCCA, TRCA, ETRCA, TDCA
```

The evaluation protocol is subject-specific leave-one-block-out over 70 BETA
subjects, 4 blocks, 40 targets, and the `occipital_9ch` channel preset.

## Compatibility Entry Points

The existing top-level scripts remain usable while the project moves toward
task-owned outputs:

```powershell
.venv\Scripts\python.exe scripts\run_toolbox_ssvep.py --dataset beta --root D:\ProjData\datasets\ssvep_beta --subjects 1-70 --blocks 1-4 --targets 0-39 --channels occipital_9ch --windows 0.2:0.2:2.0 --methods CCA,FBCCA,TRCA,ETRCA,TDCA --workers 6

.venv\Scripts\python.exe scripts\plot_beta_w02_20_compare.py
```

## Notes

- Accuracy is directly comparable across current Arena BETA summaries.
- Current ITR uses the Arena denominator `window + dataset break`. Toolbox
  documentation often includes latency and computation time, so strict ITR
  comparison needs an additional `itr_with_latency` or toolbox-style column.
- MNE/ERP/image-VEP work should reuse `vep_arena.data` and `vep_arena.neuroviz`
  rather than creating dataset-specific plotting forks.
