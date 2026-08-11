# Wong SSVEP Method Notes

Date: 2026-06-17

## Sources

- `https://github.com/edwin465/SSVEP-tlCCA`
- `https://github.com/edwin465/SSVEP-OACCA`
- `https://github.com/pikipity/SSVEP-Analysis-Toolbox`
- Chi Man Wong et al., "Learning across multi-stimulus enhances target
  recognition methods in SSVEP-based BCIs", JNE 2020, DOI `10.1088/1741-2552/ab2373`.
- Chi Man Wong et al., "Online adaptation boosts SSVEP-based BCI performance",
  IEEE TBME 2022, DOI `10.1109/TBME.2021.3133594`.

## Arena Integration Scope

- `MSCCA`: Arena-native adapter. Reuses filter-bank epochs, Benchmark reference
  signals, CCA spatial-filter helpers, class templates, and filter-bank weights.
- `MSETRCA`: Arena-native adapter. Reuses TRCA covariance/filter helpers and
  ordinary Arena prediction scoring, but estimates each target filter from
  neighboring stimulus classes.
- `tlCCA`: not yet included in the standard subject-specific leave-one-block
  runner. The reference code is a stimulus-transfer protocol with source and
  target stimulus frequency sets, so it needs a separate task schema.
- `OACCA`: not included in the standard leave-one-block runner. The reference
  algorithm is simulated online adaptation with sequential updates, agreement
  gates, and online prototype filters, so it needs an online-stream protocol.

## Verification Status

The first implementation target is a 1.0 s S1 smoke run for `MSCCA` and
`MSETRCA`. Results should be treated as adapter validation before broader
comparisons.

Command:

```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"
.\.venv\Scripts\python.exe scripts\run_wong_methods.py --subjects 1 --blocks 1-6 --windows 1.0 --methods MSCCA,MSETRCA,MSCCA+MSETRCA --output-dir runs\wong_methods_s1_b1-6_1s
```

S1 1.0 s leave-one-block results:

| Method | Accuracy |
| --- | ---: |
| MSCCA | 0.9708 |
| MSETRCA | 0.9917 |
| MSCCA+MSETRCA | 0.9917 |

These values are close to the existing TRCA/SA-MVMD sanity checks for S1 1.0 s,
so the adapters are wired plausibly. They are not yet a full benchmark result.

## SA-MVMD Group Refresh

The SA-MVMD result pack was regenerated from existing complete runs under
`runs/sa_mvmd_paper_full` using:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_sa_mvmd_paper.py --run-root runs\sa_mvmd_paper_full --out-dir results\benchmark_9ch\sa_mvmd_paper
.\.venv\Scripts\python.exe scripts\make_final_comparison_plus_sa_mvmd.py
```

This refresh does not rerun the expensive decomposition; it rebuilds CSVs,
figures, and reports from completed run artifacts.
