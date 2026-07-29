# SSVEP neural communication survey v2

This task builds the P0 unified decoder table from two prediction ledgers:

- the completed Benchmark `full_v2` run for 0.2, 0.3, 0.4, 0.5, and 1.0 s;
- the matching 0.75 s completion run for CCA, FBCCA, TRCA, and ETRCA.

The analysis validates the complete 35-subject, 6-block, 40-class
leave-one-block-out grid before calculating any result. It reports mean subject
accuracy, mean subject Wolpaw ITR (using `window + 0.5 s` per selection), and
pooled-confusion `C_BA` and `MI_uniform` in bits per trial.

Generated CSV, NPZ, JSON, and report files belong in ignored result or exchange
directories and must not be committed.

P1 uses the frozen short-window grid `0.05, 0.1, 0.15, 0.2, 0.3, 0.5,
0.75, 1.0 s`. The analysis reports pooled-confusion and mean subject-level
`C_BA`, plus `MI_uniform`, Fano `C1`, normalized accumulation, marginal
information rate, the between-subject Jensen gap, and ETRCA subject strata.
Short-window epochs must be filtered with a fixed `1.0 s` cache before slicing.

Run with the repository environment:

```powershell
.venv\Scripts\python.exe tasks\ssvep_neural_communication_survey_v2\run_p0_analysis.py `
  --full-v2-predictions <full_v2_predictions.csv> `
  --w075-predictions <w075_predictions.csv> `
  --output-dir <ignored_output_directory>
```

The P1 analysis entry point is:

```powershell
.venv\Scripts\python.exe tasks\ssvep_neural_communication_survey_v2\run_p1_frontload_analysis.py `
  --full-v2-predictions <full_v2_predictions.csv> `
  --short-predictions <0.05_0.1_0.15_predictions.csv> `
  --w075-predictions <0.75_predictions.csv> `
  --style <survey_v2.mplstyle> `
  --output-dir <ignored_output_directory>
```
