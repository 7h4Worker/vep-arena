# Binocular AR Baselines

Ke et al. 2025 binocular AR SSVEP dataset task. This task runs a
paper-referenced Arena implementation of subject-specific leave-one-block-out
CCA/FBCCA/TRCA/ETRCA experiments on the epoch zip data under
`D:/ProjData/datasets/ssvep_binocular_ar`.

## References And Evidence Role

- Dataset paper: Ke et al., "Dataset of binocularly coded steady-state visual
  evoked potentials recorded with an augmented reality headset," Scientific
  Data, 2025, doi:10.1038/s41597-025-05696-0.
- Public data and MATLAB code: Figshare article 26768287.
- Evidence role: paper-referenced Arena reproduction. The event protocol and
  dual-frequency references follow the public code, while SciPy preprocessing
  is a documented numerical adaptation. Results are not presented as bitwise
  official-code parity.

## Current Scope

- Dataset: `sub-001` to `sub-024` epoch zips.
- Main channels: paper 10-channel preset `PO7, PO8, PO5, PO4, PO3, POz, PO6, O1, Oz, O2`.
- Classes: 8 targets.
- Blocks: 10 blocks per session, two sessions by default, so 20 blocks when both sessions exist.
- Crop: event onset + 0.14 s visual latency.
- ITR denominator: `window + 1.0 s` gaze shift, matching the official MATLAB scripts.
- Preprocessing: SciPy approximation of paper preprocessing, then official-code-style Chebyshev filterbank.
- Methods: CCA, FBCCA, TRCA, ETRCA through Arena's shared traditional method modules.
- Dual-frequency templates: CCA/FBCCA concatenate left/right sine-cosine harmonic references per target, following the public Ke 2025 MATLAB FBCCA code.

Task availability from local zip inventories:

- `LF,MF`: subjects `1-14`.
- `SFSP,SFDP,DFSP,DFDP`: subjects `1-13,15`.
- `DFDP1,DFDP3,DFDP5`: subjects `1-8,16-24`.

## Smoke

```powershell
.venv\Scripts\python.exe tasks\ssvep_binocular_ar_trca\run.py `
  --subjects 1 `
  --tasks LF,DFDP,DFDP1 `
  --windows 0.1,0.5,1.0 `
  --methods CCA,FBCCA,TRCA,ETRCA `
  --out tasks\ssvep_binocular_ar_trca\results\smoke_sub001_lf_dfdp
```

## Full Run Examples

Experiment 1:

```powershell
.venv\Scripts\python.exe tasks\ssvep_binocular_ar_trca\run.py `
  --subjects 1-14 `
  --tasks LF,MF `
  --windows 0.1:0.1:3.0 `
  --methods CCA,FBCCA,TRCA,ETRCA `
  --workers 6 `
  --resume `
  --out tasks\ssvep_binocular_ar_trca\results\experiment1_trca
```

Experiment 2:

```powershell
.venv\Scripts\python.exe tasks\ssvep_binocular_ar_trca\run.py `
  --subjects 1-13,15 `
  --tasks SFSP,SFDP,DFSP,DFDP `
  --windows 0.1:0.1:3.0 `
  --methods CCA,FBCCA,TRCA,ETRCA `
  --workers 6 `
  --resume `
  --out tasks\ssvep_binocular_ar_trca\results\experiment2_trca
```

Experiment 3:

```powershell
.venv\Scripts\python.exe tasks\ssvep_binocular_ar_trca\run.py `
  --subjects 1-8,16-24 `
  --tasks DFDP1,DFDP3,DFDP5 `
  --windows 0.1:0.1:3.0 `
  --methods CCA,FBCCA,TRCA,ETRCA `
  --workers 6 `
  --resume `
  --out tasks\ssvep_binocular_ar_trca\results\experiment3_trca
```

## Outputs

Each result directory contains:

- `trials.csv`: method/task/window/subject/block accuracy and ITR rows.
- `predictions.csv`: target-level true/pred rows.
- `summary.csv`: task/method/window aggregate.
- `subject.csv`: subject-level aggregate.
- `runtime.csv`: fit/predict timing ledger.
- `unit_manifest.csv`: subject/task/window unit status.
- `manifest.json`: protocol and run metadata.
- `figures/accuracy_curve.png`
- `figures/itr_curve.png`
- `figures/accuracy_heatmap.png`
- `confusions/*.npy`

The `results/` directory is ignored by git.
