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
- `NOTES_20260801.md`: fixed-context 0.1-5.0 s extension, completed
  CCA/ETRCA artifacts, validation counts, and updated accumulation findings.

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

## Extended Time-Axis Run

The information-accumulation study uses an extended Benchmark time axis:

- coarse grid: 0.1-5.0 s in 0.1 s increments;
- fine grid: 25-101 samples at 250 Hz in two-sample increments
  (0.100-0.404 s);
- preprocessing context: fixed at 5.0 s for both grids, followed by prefix
  slicing for each requested window;
- checkpoints: one complete CSV part per subject; a cell is resumable only
  after all 6 blocks x 40 classes are present.

The fixed context is required for a valid variable-window comparison. Applying
`filtfilt` independently to each short target window creates strong,
window-dependent boundary artifacts and does not reproduce the canonical epoch
cache protocol used by the corrected `full_v2` run.

The accumulation report currently consumes CCA and ETRCA, so generate those
two curves first:

```powershell
cd D:\ProjData\proj_python\vep_arena
python tasks\benchmark_decision_channel_capacity\run_extended.py --stage coarse --methods CCA,ETRCA --workers 4 --resume
python tasks\benchmark_decision_channel_capacity\run_extended.py --stage fine --methods CCA,ETRCA --workers 4 --resume
python tasks\benchmark_decision_channel_capacity\combine_extended.py
```

To extend the same artifact with all five canonical Benchmark receivers, rerun
both stages with
`--methods CCA,FBCCA,ECCA,TRCA,ETRCA --resume`, then pass the same method list
to `combine_extended.py`. BPRCA, EBPRCA, FusionCA, and EFusionCA remain explicit
diagnostic options through `--methods`; they are not part of the default
single-frequency Benchmark evidence bundle.

For an unattended full-five completion, use the overnight orchestrator:

```powershell
python tasks\benchmark_decision_channel_capacity\run_extended_overnight.py --workers 4 --chunk-size 10
```

It runs the missing-method protocol smoke first, processes fine then coarse in
10-window batches, retries failed batches after 60/180/600 seconds, finalizes
both full-grid manifests, combines all five methods, rebuilds capacity tables
and extended-specific figures, and refreshes the information-accumulation
figures. A PID lock prevents duplicate runners. On Windows it also keeps the
system awake while the pipeline is active. Progress is written to
`results/extended/overnight_full5.log` and
`results/extended/overnight_full5_state.json`.

Extended outputs are kept separate from the corrected 0.2-1.0 s task outputs:

- `results/extended/coarse_0.1s/`
- `results/extended/fine_8ms/`
- `results/extended/combined/`
- `results/extended/combined/analysis/`

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
