# SSVEP Binocular Dataset Smoke

Task-owned data-probing workflow for two binocular SSVEP datasets:

- `ssvep_dual_alpha_gigadb_102557`: Dual-Alpha dual-frequency SSVEP.
- `ssvep_binocular_ar`: binocularly coded AR SSVEP.

This task is intentionally a dataset smoke and provenance workflow. It does not
run full classifiers. It checks local file completeness, parses stimulus/event
metadata, loads small sample windows, applies a transparent smoke
preprocessing pass, and writes time-series/PSD figures.

## Dataset Sources

- Dual-Alpha: 35 subjects, 40 targets, 5 blocks, 250 Hz, with checkerboard,
  binocular-vision, and binocular-swap paradigms. Dataset paper
  doi:10.1093/gigascience/giae041; GigaDB data/code doi:10.5524/102557.
- Binocular AR: 24 subjects, 8 targets, three experiments covering
  single-/dual-frequency and single-/dual-phase binocular conditions. Dataset
  paper doi:10.1038/s41597-025-05696-0; public data/code in Figshare article
  26768287.

Evidence role: dataset integrity, event-table, waveform, and spectrum smoke
only. This task does not establish classifier reproduction accuracy.

## Run

```powershell
.venv\Scripts\python.exe tasks\ssvep_binocular_dataset_smoke\run.py `
  --dataset both `
  --out tasks\ssvep_binocular_dataset_smoke\results\latest
```

Useful variants:

```powershell
# Dual-Alpha only, first subject, all three paradigms.
.venv\Scripts\python.exe tasks\ssvep_binocular_dataset_smoke\run.py `
  --dataset dual-alpha `
  --dual-alpha-subject 1

# AR only, first complete local subject, selected task events.
.venv\Scripts\python.exe tasks\ssvep_binocular_dataset_smoke\run.py `
  --dataset binocular-ar `
  --ar-tasks LF,DFDP,DFDP1
```

## Outputs

`results/latest/` contains:

- `dual_alpha_file_status.csv`
- `dual_alpha_codebook_summary.csv`
- `binocular_ar_file_status.csv`
- `binocular_ar_subXXX_task_inventory.csv`
- `sample_summary.csv`
- `sample_summary.json`
- `run_manifest.json`
- `smoke_report_zh.md`
- `figures/*_timeseries.png`
- `figures/*_psd.png`
- `figures/dual_alpha_codebook_pairs.png`

The results directory is ignored by git. Keep durable report material by copying
selected figures/tables into a curated report task after review.

## Status Report

`REPORT_20260708.md` is the current cross-dataset status report. It covers both
Dual-Alpha and Ke 2025 Binocular AR, and separates official baseline CSVs from
Arena-run results.

## Paper Alignment

`PAPER_ALIGNMENT_KE2025.md` records the protocol extracted from Ke et al. 2025
for the binocular AR dataset. The important implications for a formal TRCA run
are:

- use the 10 paper-analysis electrodes: `PO7, PO5, PO3, POz, PO4, PO6, PO8, O1, Oz, O2`;
- reconstruct 20 blocks per condition from two sessions when both sessions are available;
- follow leave-one-block-out cross-validation;
- evaluate data lengths from 0.1 s to 3.0 s in 0.1 s steps;
- apply the paper preprocessing before classification: 49-51 Hz notch and 5-95 Hz band-pass.

## Scope

Dual-Alpha is already epoch-level CSV, so TRCA/ETRCA can be built directly on
`condition` and `epoch`. CCA-family dual-frequency methods must use both
`Freq1` and `Freq2` from `Stimulate_Code.txt`.

Binocular AR is continuous BIDS-like EEG with event tables. The smoke loader
reads one short epoch around an event from `*_eeg.fdt`, using `*_events.tsv`,
`*_channels.tsv`, and `*_eeg.json`. This is sufficient for signal inspection
but not yet a full epoching/classification implementation.
