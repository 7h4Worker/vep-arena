# cVEP NBRS/JFPM Tsinghua 2024 Baselines

Arena task for Zheng et al. 2024, "A large dataset for VEP based brain-computer
interfaces employing narrow-band code modulation and frequency-phase modulation".

The dataset is expected at:

`D:/ProjData/datasets/cvep_nbrs_jfpm_tsinghua_2024`

This task reads the released subject-batch zip files directly and writes only
local results under `results/`.

## Task Notes

- `NOTES_20260707.md`: current output structure, 2026-07-06 preprocessing /
  MSTRCA fixes, paper-aligned result checkpoints, and archived legacy runs.
- `analysis/decision_channel_coding/`: branch report for cVEP codebook
  structure and decision-channel analysis.

The released EEG is raw except for 250 Hz down-sampling. The runner performs
the paper-level preprocessing before classification: selected channels, 50 Hz
second-order IIR notch, Chebyshev type-I filter bank, then latency-aware
stimulus-window cropping.

Channel selection is explicit. `all59` keeps the released full-cap montage as a
control, while `occipital9` uses PO3/PO4/PO5/PO6/PO7/PO8/Oz/O1/O2 for the
paper-aligned calibration-free SSVEP/cVEP baseline.

## Smoke

```powershell
.venv\Scripts\python.exe tasks\cvep_nbrs_jfpm_tsinghua_2024_baselines\run.py `
  --subjects 1-2 `
  --paradigms NBRS-15,NBRS-8,JFPM-8 `
  --methods FBCCA-CODE,TRCA,MSTRCA `
  --windows 0.8,1.2 `
  --channels occipital9 `
  --workers 2 `
  --task-name smoke `
  --resume
```

## Full

```powershell
.venv\Scripts\python.exe tasks\cvep_nbrs_jfpm_tsinghua_2024_baselines\run.py `
  --subjects 1-100 `
  --paradigms NBRS-15,NBRS-8,JFPM-8 `
  --methods FBCCA-CODE,TRCA,MSTRCA `
  --windows 0.4,0.8,1.2,1.6,2.0,2.4,3.0,4.0 `
  --channels occipital9 `
  --workers 8 `
  --task-name full_occipital9 `
  --resume
```

## Notes

- Channel presets: `all59`, `posterior17`, `po_o10`, `occipital9`, `o3`, or a
  custom 1-based comma list such as `50,51,52,53,54,55,56,57,58,59`.
- `FBCCA-CODE` uses NBRS code templates with latency-shifted first- and
  second-order references; JFPM uses sinusoidal harmonic references.
- `TRCA` is an Arena TRCA approximation with the paper's paradigm-specific
  filter-bank ranges and weights. It is not yet a strict msTRCA reproduction.
- `MSTRCA` is Arena's formal multi-stimulus TRCA implementation. It estimates
  each target's spatial filters from neighboring target classes in target-index
  order and uses ensemble TRCA scoring by default.
- ITR uses the paper convention `window + 0.5 s` trial time.
