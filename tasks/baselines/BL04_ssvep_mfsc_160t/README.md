# 160-Target MFSC TDCA

This task owns TDCA experiments for the public 160-target MFSC SSVEP dataset associated with:

Chen et al., "Implementing a Calibration-Free SSVEP-Based BCI System with 160 Targets", Journal of Neural Engineering, 2021.

The paradigm is multi-frequency sequential coding (MFSC): each target is encoded by four 1-second symbols selected from eight base frequencies. It is sequence-coded and cVEP-like in how the whole temporal code identifies a class, but it is still an SSVEP/MFSC dataset rather than a conventional binary/m-sequence cVEP dataset.

## Dataset

- Local key: `ssvep_160target_mfsc_chen2021`.
- Sampling rate: 250 Hz.
- Channels: Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2.
- Targets: 160.
- Data file shape: `160 x 9 x 1035`.
- Codebook: `metadata/reqCodeword.mat`, shape `160 x 4`.
- Offline: 8 subjects, 3 blocks except S3 has 2 blocks.
- Online: 12 subjects, 2 blocks.

## Smoke

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_160target_mfsc_tdca\run.py --split offline --subjects 1 --segment-windows 0.4,1.0 --workers 1 --output results\tdca_smoke
```

## Overnight Offline Grid

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
.\.venv\Scripts\python.exe tasks\ssvep_160target_mfsc_tdca\run.py --split offline --subjects all --segment-windows 0.4,0.5,0.6,0.7,0.8,0.9,1.0 --workers 4 --output results\tdca_offline_grid --resume
```

Outputs under `results/` are ignored by git.

Current status:

- Complete: `results/tdca_offline_grid`, offline split, subjects 1-8, segment windows 0.4-1.0 s per symbol, `n_delay=0`.
- Pending: online split, subjects 1-12, same segment windows and `n_delay=0`.
- Pending probe: delay augmentation. This should stay a small probe first because naive delay expansion across concatenated MFSC slots can cross symbol boundaries.

## Expected Next Runs

Online split, same baseline settings:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
.\.venv\Scripts\python.exe tasks\ssvep_160target_mfsc_tdca\run.py --split online --subjects all --segment-windows 0.4,0.5,0.6,0.7,0.8,0.9,1.0 --workers 4 --output results\tdca_online_grid --resume
```

Delay probe, limited scope before any full grid:

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_160target_mfsc_tdca\run.py --split offline --subjects 1,2 --segment-windows 0.4,0.6,0.8 --n-delay 1 --workers 2 --output results\tdca_delay_probe --resume
```
