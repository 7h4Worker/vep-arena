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

Run with the repository environment:

```powershell
.venv\Scripts\python.exe tasks\ssvep_neural_communication_survey_v2\run_p0_analysis.py `
  --full-v2-predictions <full_v2_predictions.csv> `
  --w075-predictions <w075_predictions.csv> `
  --output-dir <ignored_output_directory>
```
