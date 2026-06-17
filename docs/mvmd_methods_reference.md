# MVMD SSVEP Method Notes

Date: 2026-05-25

## Papers

- SA-MVMD-TRCA: Jinpeng Lyu et al., "Novel Sinusoidal Signal Assisted Multivariate Variational Mode Decomposition Combined With Task-Related Component Analysis for Enhancing SSVEP-Based BCI Performance", IEEE JBHI, 2024.
- MVMD-CCA line: Kang Wang et al., "An MVMD-CCA Recognition Algorithm in SSVEP-Based BCI and Its Application in Robot Control", IEEE TNNLS, 2022.

## Code Status

No official author code was found during the initial search. The local MVMD core is a NumPy port of the public MATLAB `MVMD.m` reference cloned under:

`D:\ProjData\_reference\MDApproach_PS\MVMD.m`

Arena files:

- `D:\ProjData\proj_python\vep_arena\vep_arena\methods\mvmd.py`
- `D:\ProjData\proj_python\vep_arena\scripts\run_mvmd_methods.py`
- `D:\ProjData\proj_python\vep_arena\scripts\analyze_mvmd_methods.py`

## Local Reproduction Scope

- `MVMD-CCA`: exploratory calibration-free adapter, decomposes/reconstructs each trial with MVMD and then applies CCA.
- `MVMD-TRCA`: exploratory diagnostic adapter, decomposes raw epochs with MVMD, treats modes as filter-bank subbands, and then trains ordinary TRCA. This is not a faithful paper reproduction and currently fails the expected-performance sanity check.
- `SA-MVMD-TRCA`: exploratory local adapter, appending an all-frequency Benchmark sine/cosine bank before MVMD reconstruction, then training TRCA in the usual subject-specific leave-one-block-out protocol.

This implementation is not a faithful paper reproduction. A secondary web note indicates the paper uses a target-frequency sinusoidal auxiliary channel and fuses TRCA scores from IMFs and reconstructed signals. The current adapter should be kept out of the main leaderboard until the original paper settings or official code are obtained.

## Diagnostic Smoke Verification

Command used for a 1.0 s diagnostic pass:

```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
.\.venv\Scripts\python.exe scripts\run_mvmd_methods.py --subjects 1 --blocks 1-5 --windows 1.0 --methods MVMD-TRCA --n-modes 3 --max-iter 10 --tol 1e-3 --output-dir runs\smoke_mvmd_trca_modes_1s_s1_b1-5
```

The run completed and wrote `subject_block.csv`, `predictions.csv`,
`confusion_mvmd_trca.npy`, and `manifest.json`, but the accuracy was much too
low for a reproduction claim. Accuracies for S1 blocks 1-5 were
`0.275, 0.200, 0.250, 0.275, 0.275`; traditional toolbox-style TRCA on S1 1.0 s
is near ceiling in the Arena baseline (`0.9917` subject mean), and the
paper-style SA-MVMD-TRCA path gives S1 1.0 s block accuracies
`0.975, 1.000, 0.975, 1.000, 0.975, 1.000`.

Conclusion: this raw-MVMD-plus-TRCA adapter verifies code wiring only. It should
not be used as an MVMD-TRCA reproduction or leaderboard method. Use
`scripts/run_sa_mvmd_paper.py` for source-aligned SA-MVMD-TRCA validation.
