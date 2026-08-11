# Sun2024 Efficient Dual-Frequency Smoke

This task starts the local workflow for the public dataset associated with:

Sun et al., "Efficient dual-frequency SSVEP brain-computer interface system exploiting interocular visual resource disparities", Expert Systems with Applications, 2024.

Dataset facts from the downloaded metadata and codebooks:

- Files are ANT Neuro `.cnt` recordings.
- Sampling rate observed by MNE: 1000 Hz.
- Recordings have 64 EEG channels.
- Local root: configured by `ssvep_efficient_dual_frequency_sun2024` in `configs/datasets/local_paths.json`.
- `code of 1Target.xlsx` maps triggers 1-8 to left/right-eye frequency pairs.
- `code of 40Targets.xlsx` maps triggers 1-40 to left/right-eye frequency pairs; trigger 253 marks stimulus end and 254 block end.

The first entry point is an event/codebook/header smoke. It verifies that MNE can read `.cnt` headers and annotations, exports event tables, and creates a time/frequency QA figure before any classifier is added.

## Smoke

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_efficient_dual_frequency_sun2024\run.py --subjects 1 --splits one_target,forty_targets
```

Outputs are written under `results/smoke_inventory/` and are ignored by git:

- `files.csv`
- `events.csv`
- `paired_trials.csv`
- `codebook_one_target.csv`
- `codebook_forty_targets.csv`
- `summary.csv`
- `manifest.json`
- `sun2024_smoke_time_frequency.png`

## CCA Smoke

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_efficient_dual_frequency_sun2024\run.py --subjects 1 --splits one_target,forty_targets --run-cca-smoke --classification-split forty_targets --classification-windows 1.0,2.0 --max-classification-trials 120 --onset-shift 0.14 --harmonics 5 --bandpass 6,90
```

Additional ignored outputs:

- `cca_smoke_trials.csv`
- `cca_smoke_summary.csv`
- `sun2024_cca_smoke_confusion.png`

## Next Step

After the smoke is stable, build epochs from stimulus-start events and pair labels with the Excel codebook. Classification should start with a small 40-target subject/window smoke before any full run.

## Formal TRCA/bTRCA Runner

`run_formal.py` performs block-wise cross-validation on the 40-target offline
split and writes task-owned results under `results/`.

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_efficient_dual_frequency_sun2024\run_formal.py --subjects 1 --windows 0.6 --methods TRCA,ETRCA,BTRCA,EBTRCA --channels occipital9 --n-bands 3 --max-folds 2 --output tasks\ssvep_efficient_dual_frequency_sun2024\results\formal_smoke_20260717
```

Primary outputs:

- `manifest.json`
- `trials.csv`
- `summary.csv`
- `figures/confusion_<method>_<window>s.png`

The formal runner uses the 40-target codebook to recover the 20 swapped
left/right-eye frequency pairs and runs TRCA, eTRCA, bTRCA, and ebTRCA. This is
the task's paper-aligned classification path; CCA/FBCCA remain only smoke or
weak-reference checks.

Current caution: `results/formal_full_20260717_occipital9` is a diagnostic
baseline, not a paper-level reproduction. The reproduction audit and extracted
paper facts are recorded in `REPORT_20260717_REPRO_AUDIT.md`.

Formal result plotting follows the other Arena tasks:

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_efficient_dual_frequency_sun2024\plot_acc_itr.py --run formal_full_20260717_occipital9
```

Outputs are written inside the selected run directory:

- `subject.csv`
- `summary_aggregate.csv`
- `report.md`
- `figures/accuracy_curve.png`
- `figures/itr_curve.png`
- `figures/accuracy_heatmap.png`
- `figures/subject_box_best.png`

Smoke comparisons are diagnostic only and require `--include-smoke`; they are
written under `results/diagnostics/`, not mixed into the formal run figures.

## Comb-Filter Audit Run

The paper states that 40-target offline/online data used comb filtering. The
runner therefore supports an explicit comb-filter mode before the normal
TRCA-style sub-band loop:

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_efficient_dual_frequency_sun2024\run_formal.py --subjects 1 --windows 0.6 --methods TRCA,ETRCA,BTRCA,EBTRCA --channels all --preprocess comb_filterbank --comb-f0 50 --comb-q 35 --n-bands 5 --max-folds 2 --output tasks\ssvep_efficient_dual_frequency_sun2024\results\formal_smoke_all64_comb_20260717
```

Because the public paper does not specify the comb filter parameters, the
assumed base frequency and Q are written into `manifest.json`.
The runner crops to the requested trial range before loading data by default
(`--crop-padding 5`) to keep all-channel runs stable on Windows.

The paper-level offline audit should also restrict local subjects 1-9 to the
first five 40-target blocks and use the paper's Fig. 6 window grid from
0.2 s to 2.0 s:

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_efficient_dual_frequency_sun2024\run_formal.py --subjects all --windows 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0 --methods TRCA,ETRCA,BTRCA,EBTRCA --channels all --preprocess comb_filterbank --comb-f0 50 --comb-q 35 --max-blocks 5 --n-bands 5 --workers 3 --output tasks\ssvep_efficient_dual_frequency_sun2024\results\formal_full_20260718_all64_comb_first5_parallel3
```

The runner checks all 40 classes in each leave-block-out training partition.
Folds with incomplete training coverage are skipped and recorded in
`skipped_folds.csv`; the subject remains usable if at least one valid fold
remains. Parent-process checkpoints are rewritten after every completed
subject, and `--resume` continues only subjects whose complete grid is absent.
