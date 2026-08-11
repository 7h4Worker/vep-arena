# Liang2020 Dual-Frequency Phase Smoke

This task starts the local workflow for the public dataset associated with:

Liang et al., "Optimizing a dual-frequency and phase modulation method for SSVEP-based BCIs", Journal of Neural Engineering, 2020.

Dataset facts captured from the downloaded readme:

- Sampling rate: 1000 Hz.
- Channels: Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2.
- Triggers: 1-12 or 1-40 correspond to target indexes.
- Local root: configured by `ssvep_dual_frequency_phase_liang2020` in `configs/datasets/local_paths.json`.

The first entry point is intentionally an inventory/QA smoke, not a classification run. It verifies MATLAB file orientation, labels, trial counts, and time/frequency content before adding CCA/FBCCA/TRCA/FBDCCA baselines.

## Smoke

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_dual_frequency_phase_liang2020\run.py --subjects 1 --experiments exp1,exp2 --max-files-per-experiment 4
```

Outputs are written under `results/smoke_inventory/` and are ignored by git:

- `files.csv`
- `trials.csv`
- `summary.csv`
- `manifest.json`
- `liang2020_smoke_time_frequency.png`

## Next Step

After the smoke is stable, map each experiment condition to the paper's exact frequency/phase table, then add classification baselines using `vep_arena.methods.traditional.multi_frequency_reference_signals`.

## Formal TRCA/eTRCA Runner

`run_formal.py` runs EEG-template TRCA/eTRCA classification for Experiments
1-4. This follows the paper's recognition path: individual training-data
templates with filter-bank TRCA, 140 ms visual latency removal, and ITR computed
with an added 0.5 s gaze-switching time.

The full command is:

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_dual_frequency_phase_liang2020\run_formal.py --subjects all --experiments exp1,exp2,exp3,exp4 --methods TRCA,ETRCA --windows paper --n-bands 5 --output tasks\ssvep_dual_frequency_phase_liang2020\results\formal_full_20260717
```

Outputs:

- `manifest.json`
- `trials.csv`
- `summary.csv`
- `figures/confusion_exp3_condition*_*.png`
- `figures/confusion_exp4_condition*_*.png`

Generate the primary Accuracy/ITR versus analysis-window curves from a completed
formal summary without rerunning classification:

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_dual_frequency_phase_liang2020\plot_window_curves.py
```

This writes `figures/accuracy_window_curves.png` and
`figures/itr_window_curves.png`, with separate panels for each experiment and
SEM bands across subjects.

## TDCA Diagnostic Smoke

`run_tdca_smoke.py` validates the Figure 1 Method 1 codebook before attempting
reference-based TDCA. The public MAT files do not retain the Figure 1 spatial
ordering, so the first smoke derives a one-to-one MAT-label mapping from
Subject 1 Exp3-condition3 spectral amplitudes. This is a diagnostic calibration,
not an official full reproduction.

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_dual_frequency_phase_liang2020\run_tdca_smoke.py --subject 1 --experiments exp3,exp4 --windows 0.4,0.6 --output tasks\ssvep_dual_frequency_phase_liang2020\results\smoke_tdca_method1_spectral_alignment_20260720
```

Apply that locked map to another subject without deriving a new mapping:

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_dual_frequency_phase_liang2020\run_tdca_smoke.py --subject 2 --experiments exp3 --windows 0.4,0.6 --label-map tasks\ssvep_dual_frequency_phase_liang2020\results\smoke_tdca_method1_spectral_alignment_20260720\digitized_method1_codebook.csv --output tasks\ssvep_dual_frequency_phase_liang2020\results\smoke_tdca_method1_cross_subject_s2_20260720
```

The output includes the mapped codebook, a frequency-amplitude check image,
trial predictions, summaries, and confusion matrices. It must remain separate
from an official table until the paper codebook/order is independently confirmed.

`run_tdca_full.py` scales the locked map to all subjects for Method 1 only.
It writes `diagnostic_full_*`, excluding the calibration subject from the primary
aggregate curve so that the reported cross-subject view does not reuse its map
calibration data.

Experiment mapping from the paper:

- `exp1`: 6-target optimization-method comparison, conditions 1-3.
- `exp2`: 40-target Method 1 versus random-sampling codes, conditions 1-4.
- `exp3`: offline 40-target dual-frequency comparison, conditions 1-3.
- `exp4`: online-style 40-target comparison, conditions 1-2; default split uses
  runs 1-6 for training and 7-9 for testing, matching the paper description.

TDCA/reference-template classifiers are intentionally not part of the formal
default yet because the full target-level frequency/phase codebooks are embedded
as paper figures/appendix panels rather than machine-readable dataset files.
Adding them requires a separate digitized codebook validation step.
