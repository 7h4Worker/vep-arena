# Sun2024 Dual-Frequency SSVEP

Sun et al. (2024), "Efficient dual-frequency SSVEP brain-computer interface
system exploiting interocular visual resource disparities."

## Current Dataset Boundary

The downloaded public package contains 13 single-target CNT files and 14
forty-target **offline** CNT files. It does not contain the ten-subject online
training/validation data reported in the paper. Online performance must not be
estimated from these files.

The forty-target files have two observed local protocols:

- Subjects 10-14: five recorded repetitions per target (200 trials), matching
  the paper's offline trial count.
- Subjects 1-9: eight recorded repetitions per target (320 trials). The
  paper-comparable adapter uses repetitions 1-5 for the same five-fold
  target-repetition protocol as every other subject; repetitions 6-8 remain
  an explicit extended-data option.

The CNT annotations contain start/end triggers but no usable block boundary.
All current classification folds are therefore target-repetition folds, never
event-order chunks.

## Protocol Smoke

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_efficient_dual_frequency_sun2024\run.py --subjects all
```

This writes the event-derived `trial_ledger.csv`, target repetition counts, a
protocol summary, and one time/frequency QA figure under `results/protocol_smoke/`.

## Correct Offline Classifier Smoke

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_efficient_dual_frequency_sun2024\run_offline.py --subjects 11,13 --repetitions 5 --repetition-start 1 --windows 0.6,1.0 --methods TRCA,ETRCA --preprocess filterbank
```

Each fold holds out repetition `k` for every target and trains on all other
repetitions. The smoke intentionally starts with TRCA/eTRCA. bTRCA/ebTRCA stay
out of the current reproduction path until the paper's O/S group mapping and
matrix-filter scoring are verified against author code or supplementary detail.

Results, figures, CSV, and NPZ files remain task-local and are not committed.

The paper describes comb filtering for the forty-target data but does not
publish its implementation or parameters. The authors' public GitHub repository
contains dataset metadata only. Current work therefore begins from a
target-conditioned spectrum analysis of the published stimulus codebook. No
generic power-line notch is treated as the author comb filter.

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_efficient_dual_frequency_sun2024\analyze_target_comb_spectrum.py --subjects 11,13 --repetitions 5
.\.venv\Scripts\python.exe tasks\ssvep_efficient_dual_frequency_sun2024\run_target_comb_trca.py --subjects 11,13 --windows 0.6 --comb-mode pair_specific --half-width 1.5
```

These are empirical sensitivity tools, not an author-comb reproduction. Their
outputs stay separate from the standard filter-bank baseline until a definition
that improves held-out performance and matches author documentation is found.

The public spreadsheet omits the initial phases shown in the paper's Figure
2(III). The task's shared codebook now includes those digitized phase values.
Several plausible magnitude-, phase-, and periodic-comb definitions can be
compared on the fixed sample protocol with:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
.\.venv\Scripts\python.exe tasks\ssvep_efficient_dual_frequency_sun2024\run_comb_candidates.py --subjects 11,13,12,14 --windows 0.6,1.0 --repetitions 5 --workers 4
```

The completed 2026-07-20 sample found no comb candidate that stably exceeded
the standard two-band TRCA baseline. See `NOTES_20260720.md`; these outputs are
diagnostic negative results and do not justify a full-cohort comb run.
