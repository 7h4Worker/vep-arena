# Dual-Alpha Baselines

Formal Arena task for the GigaDB 102557 Dual-Alpha dual-frequency SSVEP dataset.

## Scope

- Paradigms: `Checkerboard_Arrangment`, `Binocular_Vision`, `Binocular-Swap_Vision`.
- Official classification scope:
  - `ETRCA`: Arena ensemble TRCA, aligned to the public script's `meegkit.trca.TRCA(..., ensemble=True)`.
  - `FBDCCA`: official dual-frequency FBCCA-style classifier for `Checkerboard_Arrangment` and `Binocular_Vision`.
- Default channels: `official`, meaning all channels in the public script. CA/BV use 9 channels, BsV uses 64 channels.
- Windows: `0.2:0.2:2.0`.
- CV: subject-specific 5-block leave-one-block-out; each subject has 40 targets x 5 epochs.

## Run

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
.venv\Scripts\python.exe tasks\ssvep_dual_alpha_baselines\run.py `
  --subjects 1-35 `
  --paradigms Checkerboard_Arrangment,Binocular_Vision,Binocular-Swap_Vision `
  --windows 0.2:0.2:2.0 `
  --methods ETRCA,FBDCCA `
  --fbdcca-filter-backend mne-fir `
  --workers 8 `
  --resume `
  --out tasks\ssvep_dual_alpha_baselines\results\official_baselines
```

Generated results stay under `results/` and are ignored by git.

## Current Full Run

Completed on 2026-07-08:

- Result dir: `tasks/ssvep_dual_alpha_baselines/results/official_baselines`
- Python: generated with `D:\ProjData\envs\erp_ssvep_lab` during the MNE-FIR repair; future reruns should use the canonical `.venv` after `uv sync --extra all`.
- FBDCCA backend: `mne-fir`.
- Units: 105/105 complete.
- Trial rows: 8750.
- Prediction rows: 350000.
- Summary rows: 50.
- Figures: `accuracy_curve.png`, `itr_curve.png`, `accuracy_heatmap.png`, `subject_box_best.png`.

Best mean accuracy:

| Paradigm | Method | Best window | Arena acc | Official acc | Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| Checkerboard_Arrangment | ETRCA | 2.0s | 94.00% | 94.00% | +0.00 pp |
| Checkerboard_Arrangment | FBDCCA | 2.0s | 27.67% | 27.31% | +0.36 pp |
| Binocular_Vision | ETRCA | 2.0s | 88.31% | 88.16% | +0.16 pp |
| Binocular_Vision | FBDCCA | 2.0s | 64.57% | 63.83% | +0.74 pp |
| Binocular-Swap_Vision | ETRCA | 2.0s | 78.61% | 78.54% | +0.07 pp |

## Preprocessing

Dual-Alpha is already distributed as epoch-level CSV. The task crops each epoch from sample 0 to the requested window, then applies method-specific filter banks:

- ETRCA/TRCA: public TRCA pass/stop bands, Chebyshev filter bank, 7 subbands by default.
- FBDCCA: public FBDCCA bands and weights, dual-frequency sine/cosine references, 5 subbands by default. Official runs require the public MNE FIR path (`--fbdcca-filter-backend mne-fir`). The old SciPy FIR approximation has been moved to `results/official_baselines_scipy_fir_legacy_20260708` and is not treated as an official baseline.

This differs from the Ke 2025 Binocular AR task, where the source is continuous EEG and the task must first apply continuous demeaning/notch/bandpass and event-aligned epoching.
