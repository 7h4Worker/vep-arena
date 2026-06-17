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
- `SA-MVMD-TRCA`: exploratory local adapter, appending an all-frequency Benchmark sine/cosine bank before MVMD reconstruction, then training TRCA in the usual subject-specific leave-one-block-out protocol.

This implementation is not a faithful paper reproduction. A secondary web note indicates the paper uses a target-frequency sinusoidal auxiliary channel and fuses TRCA scores from IMFs and reconstructed signals. The current adapter should be kept out of the main leaderboard until the original paper settings or official code are obtained.
