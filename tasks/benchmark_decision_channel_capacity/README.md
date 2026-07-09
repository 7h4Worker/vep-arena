# Benchmark Decision-Channel Capacity

Purpose: reproduce and extend Costa 2020 and Arslan-Sinha 2024 style
decision-channel capacity analysis on the Tsinghua Benchmark 9-channel SSVEP
setting.

This task treats the classifier decisions as a discrete memoryless channel:

```text
stimulus class C -> SSVEP/EEG + decoder -> predicted class C_hat
```

The first local target is a complete Benchmark 9ch pipeline:

- dataset: Tsinghua Benchmark SSVEP
- subjects: 1-35
- targets: 40
- blocks: 1-6 leave-one-block-out
- default methods: CCA, FBCCA, ECCA, TRCA, ETRCA
- optional added methods after SSCOR integration: SSCOR, ESSCOR
- windows: 0.2-1.0 s

Outputs are local under `results/` and are ignored by git.

## Task Notes

- `NOTES_20260707.md`: current source / result layout, capacity-analysis
  artifacts, and cleanup status for this task.

## Current Validity Note

The previous `benchmark_decision_channel_capacity_full` run was produced
before the CCA/FBCCA/ECCA protocol audit and should be treated as obsolete for
final reporting:

- ECCA was using only the first/raw slot instead of filter-bank ECCA.
- Raw CCA epochs were not fully aligned to SSVEP-Analysis-Toolbox's
  notch-before-latency-crop preprocessing.
- FBCCA was not included in the decision-channel task's default method set.

Keep the old result directory only as a provenance/debug snapshot. Do not run
the new corrected pipeline with the same task name plus `--resume`, because
the runner will skip already-complete method/window rows from the obsolete
CSV files. Use a new task name such as
`benchmark_decision_channel_capacity_full_v2`, or delete the old local result
directory intentionally before rerunning.

## Smoke

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\benchmark_decision_channel_capacity\run.py --smoke --workers 2
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\benchmark_decision_channel_capacity\analyze.py --predictions tasks\benchmark_decision_channel_capacity\results\input\predictions.csv
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\benchmark_decision_channel_capacity\plot.py
```

## Full Run

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\benchmark_decision_channel_capacity\run.py --task-name benchmark_decision_channel_capacity_full_v2 --workers 8
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\benchmark_decision_channel_capacity\analyze.py --predictions tasks\benchmark_decision_channel_capacity\results\input\predictions.csv
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\benchmark_decision_channel_capacity\plot.py
```

`--workers` controls CPU process parallelism in
`scripts/run_traditional_benchmark.py`. The parallel unit is one
subject/window job; each job then runs the requested methods and blocks. On a
shared workstation, use `--workers 4`. For a dedicated run, `--workers 8` is
reasonable if memory and CPU thermals are acceptable.

## Optional SSCOR Extension

ECCA is the calibrated extended CCA bridge between standard CCA and
template/spatial-filter methods. It uses subject-specific class templates from
the leave-one-block-out training folds and sine-cosine references from the
Benchmark stimulus table.

SSCOR and ensemble SSCOR are available through the traditional benchmark
runner. This resume pattern is only valid when the existing result directory
was generated with the same preprocessing and method definitions. It must not
be used to extend the obsolete pre-audit full run noted above.

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\benchmark_decision_channel_capacity\run.py --workers 2 --methods CCA,FBCCA,ECCA,TRCA,ETRCA,SSCOR,ESSCOR --task-name benchmark_decision_channel_capacity_full_v2
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\benchmark_decision_channel_capacity\analyze.py --predictions tasks\benchmark_decision_channel_capacity\results\input\predictions.csv
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\benchmark_decision_channel_capacity\plot.py
```

## Outputs

- `results/input/`: copied runner outputs, especially `predictions.csv`.
- `results/analysis/capacity_by_subject_method_window.csv`
- `results/analysis/qstar_by_class.csv`
- `results/analysis/confusion_counts_subject_method_window.npz`
- `results/analysis/codebook_pruning_by_method_window.csv`
- `results/analysis/costa_selection_path.csv`
- `results/analysis/costa_best_codebooks.csv`
- `results/figures/fig01_*.png` through `fig13_*.png`
- `report_zh.md`
