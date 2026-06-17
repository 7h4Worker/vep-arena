# CCA And FBCCA Reference Notes

Date: 2026-05-22

## Reference Sources

Primary implementation reference:

- `D:\ProjData\_reference\TRCA-SSVEP\src\test_fbcca.m`
- `D:\ProjData\_reference\TRCA-SSVEP\src\filterbank.m`
- `D:\ProjData\_reference\TRCA-SSVEP\tutorial\tutorial_fbcca.m`

The repository describes `test_fbcca.m` as FBCCA-based SSVEP detection and cites:

- X. Chen, Y. Wang, S. Gao, T.-P. Jung, and X. Gao, "Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface", J. Neural Eng., 12, 046008, 2015.

The same file includes CCA reference generation for conventional sine/cosine references and cites the earlier CCA SSVEP references.

## Arena Implementation

Arena files:

- `D:\ProjData\proj_python\vep_arena\vep_arena\methods\cca.py`
- `D:\ProjData\proj_python\vep_arena\scripts\run_cca_fbcca.py`

Output:

- `D:\ProjData\proj_python\vep_arena\runs\cca_fbcca\subject_block.csv`
- `D:\ProjData\proj_python\vep_arena\runs\cca_fbcca\predictions.csv`
- `D:\ProjData\proj_python\vep_arena\runs\cca_fbcca\manifest.json`

## Protocol

Dataset and crop are shared with VEP Arena Benchmark 9ch:

- Tsinghua Benchmark SSVEP
- 35 subjects
- 6 blocks
- 40 targets
- 9 channels: Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, O2
- Crop start: cue 0.5 s + visual latency 0.14 s
- Windows: 0.2 to 1.0 s
- Output rows: method, window, subject, block, accuracy, ITR, samples, seconds

CCA and FBCCA are calibration-free reference methods in this run. They do not use the subject's other blocks for training, but the results are still reported on the same subject/block/window grid so they can be compared as baselines.

## CCA

The Arena CCA baseline is the no-filterbank variant of the FBCCA routine:

- Raw cropped 9-channel trial
- Frequency-only sine/cosine references
- 5 harmonics
- Prediction by maximum first canonical correlation

The Benchmark phase list is not explicitly used because the sine/cosine pair for each harmonic spans the phase-shifted subspace.

## FBCCA

The Arena FBCCA baseline follows `TRCA-SSVEP/src/test_fbcca.m` and `filterbank.m`:

- 5 filter banks
- Filter passbands: 6-90, 14-90, 22-90, 30-90, 38-90 Hz
- Filter stopbands: 4-100, 10-100, 16-100, 24-100, 32-100 Hz
- Chebyshev type I, `cheb1ord`, passband ripple 0.5
- 5 harmonics
- Subband weights: `(idx + 1)^(-1.25) + 0.25`
- Prediction by weighted sum of first canonical correlations

## Current Results

The full run was completed under:

`D:\ProjData\proj_python\vep_arena\runs\cca_fbcca`

The final comparison report and figures include both CCA and FBCCA:

`D:\ProjData\proj_python\vep_arena\results\benchmark_9ch\final_compare`

