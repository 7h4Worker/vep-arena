# SA-MVMD-TRCA Paper Reading Notes

Paper: `Novel Sinusoidal Signal Assisted Multivariate Variational Mode Decomposition Combined With Task-Related Component Analysis for Enhancing SSVEP-Based BCI Performance`

Local PDF: `C:\Users\Admin\Desktop\papers-unread\Novel_Sinusoidal_Signal_Assisted_Multivariate_Variational_Mode_Decomposition_Combined_With_Task-Related_Component_Analysis_for_Enhancing_SSVEP-Based_BCI_Performance.pdf`

Date read: 2026-05-25

## Core Algorithm

SA-MVMD appends one sinusoidal assistance channel to the EEG before MVMD. After decomposition, the assistance channel is removed and only the original EEG channels of the IMFs are used.

Assistance signal:

`SA = sum_i A_i * sin(2*pi*f_i^SA*t)`

Important settings:

- `f_i^SA` is based on prior frequency knowledge.
- For SSVEP, `f_i^SA = i * f_n`, where `f_n` is the candidate stimulus frequency.
- `A_i = i^-0.75 + 0.01`.
- `K = Ns + 1`; one extra IMF is used for DC / very low-frequency components.
- The assistance-signal power is scaled by `RP` relative to the original signal. The paper says `RP > 10` after the simulated signal test, and in SSVEP applications RP can be set sufficiently high.
- The MVMD center frequencies should be initialized to the assistance frequencies. Practically: first IMF near DC, remaining IMFs initialized to `f_i^SA`.

For real SSVEP SA-MVMD analysis:

- Benchmark example uses 1 s data, subject 8, block 3, 12 Hz.
- No filtering preprocessing was applied for SA-MVMD decomposition-performance tests.
- `K = 8`, `Ns = 7`; IMFs 1-8.
- IMF 1 is DC / very low frequency.
- IMFs 7 and 8 are discarded because very high-frequency SSVEP components are weak.
- IMFs 2-6 correspond to harmonics 1-5 and are selected.

## SA-MVMD-TRCA

Training phase for target `n`:

1. For known training data `X_train_n`, set assistance fundamental to `f_n`.
2. Run SA-MVMD.
3. Select IMFs `u_train_n,m`, `m = 1..M`; in the paper this corresponds to IMFs 2-6 for harmonics 1-5.
4. Reconstruct wideband signal `Xhat_train_n = sum_m u_train_n,m`.
5. Apply TRCA separately to:
   - reconstructed wideband signal, treated as `m = 0`
   - each selected IMF, `m = 1..M`
6. Store spatial filters `w_n,m` and TRC templates `s_n,m = w_n,m^T * mean_trials(signal_n,m)`.

Feature extraction for one unknown test trial:

1. For every possible target frequency `f_n`, independently run SA-MVMD on the same test trial with assistance fundamental `f_n`.
2. Select IMFs and reconstruct the candidate wideband signal.
3. Compute Pearson correlations:
   - `r_n,0 = corr((Xhat_test_n)^T w_n,0, s_n,0)`
   - `r_n,m = corr((u_test_n,m)^T w_n,m, s_n,m)` for `m = 1..M`
4. Weighted score:
   - `score_n = sum_{m=0..M} a(m) * r_n,m`
   - `a(m) = (m + 1)^-1.25 + 0.25`
5. Predict the target with max score.

This candidate-wise decomposition step is essential and was missing from the earlier exploratory adapter.

## SA-MVMD-eTRCA

The ensemble version stacks spatial filters across targets:

`W_m = [w_1,m, w_2,m, ..., w_Nt,m]`

The test score uses two-dimensional correlation between ensemble-projected candidate test data and ensemble-projected candidate average training data.

## Dataset / Protocol

- Tsinghua Benchmark SSVEP.
- 35 subjects.
- 40 targets.
- Six experimental blocks.
- Stimuli: 5 s effective duration.
- Frequencies: 8-15.8 Hz, step 0.2 Hz.
- Phases: 0 to 1.5*pi, step 0.5*pi.
- Channels: 9 parieto-occipital channels: Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2.
- Sampling rate: 250 Hz.
- Discard first 140 ms after stimulus onset.
- Offline test: six-fold leave-one-block-out CV.
- Five blocks train, one block test.
- Data lengths: 0.4-1.0 s, step 0.2 s.
- For classification, harmonics 1-5 / IMFs 2-6 are used.

## Paper Results Pointers

- Fig. 11: SA-MVMD-TRCA vs TRCA, eCCA, FBCCA, SF-TRCA, TRCA_FDF, and MVMD-TRCA.
- Fig. 12: SA-MVMD-eTRCA vs ensemble controls.
- Supplementary Table IV/V contain detailed numerical results, but the supplement is not present locally yet.
- Online simulation: 0.8 s data, first eight subjects, blocks 1-5 train and block 6 test; delay around 511 ms for SA-MVMD-TRCA and 520 ms for SA-MVMD-eTRCA in MATLAB R2021b on i9-11900K.

## Current Local Implementation Gap

The current `mvmd_methods_20260525` results are exploratory only and should not be compared with the paper.

Major gaps:

- Used all target-frequency references together instead of one target-frequency assistance signal per candidate.
- Did not run 40 candidate-specific SA-MVMD decompositions for each test trial.
- Used `K=5` and `assist_harmonics=2`; paper logic requires `K=8`, `Ns=7`, selected IMFs 2-6.
- Did not use `A_i = i^-0.75 + 0.01` with explicit RP power scaling.
- Did not initialize MVMD center frequencies to the assistance frequencies.
- Did not separately train/use TRCA on each selected IMF plus reconstructed signal.
- Did not implement SA-MVMD-eTRCA ensemble scoring.
- Used 0.5/1.0 s windows instead of paper windows 0.4/0.6/0.8/1.0 s.

## Required Rewrite

Implement separate paper-faithful runners:

- `SA-MVMD-TRCA-paper`
- `SA-MVMD-eTRCA-paper`
- optionally `MVMD-TRCA-paper` as the ablation control

Keep the earlier exploratory outputs out of the final leaderboard.
