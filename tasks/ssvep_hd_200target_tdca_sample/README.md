# SSVEP HD 200-Target TDCA / TRCA

Purpose: reproduce and analyze offline / online HD-200 SSVEP results for the
high-density 200-target dataset.

## Task Notes

- `NOTES_20260707.md`: current formal result directories, offline TDCA/TRCA
  grid status, retained scripts, and archived smoke / feature-probe outputs.
- `_legacy/20260707_sample_and_validation_entrypoints/`: earlier small-sample
  TDCA and MATLAB validation entry points.

## Scope

- Dataset: `D:/ProjData/datasets/ssvep_hd_200target`
- Subjects: offline author/released subject set for grid analyses; online
  released subjects for Table 2 reproduction.
- Targets: 200
- Channels: 66
- Blocks: 18
- Protocol: leave-one-block-out
- Offline grid windows: `100`, `200`, `300`, `400`, `500` ms.
- Target sets: `40`, `80`, `120`, `160`, `200`.
- Channel sets: `9`, `21`, `32`, `66`.

## Implementation

- `run_offline_tdca_grid.py`: resumable Python runner for the paper's offline
  `TDCA_Classification.m` grid: target sets 40/80/120/160/200, channel sets
  9/21/32/66, windows 100/200/300/400/500 ms.
- `run_offline_trca_grid.py`: matching offline TRCA grid runner.
- `plot_offline_tdca_grid.py`: offline ACC/ITR, target-count, channel-count,
  and window figures.
- `run_online_table2.py`, `run_online_highest.py`: online reproduction entry
  points.
- `plot_online_table2.py`, `plot_online_table2_confusions.py`: online summary
  and confusion figures.
- `analyze_offline_signal_features.py`: exploratory signal-feature analysis.

Earlier small-sample and MATLAB cache validation scripts are archived under
`_legacy/20260707_sample_and_validation_entrypoints/`.

Filter caches are stored outside the repo:

```text
D:/ProjData/datasets/ssvep_hd_200target/derivatives/tdca_sample/
```

## Results

Current offline grid outputs:

```text
tasks/ssvep_hd_200target_tdca_sample/results/offline_tdca_grid/
tasks/ssvep_hd_200target_tdca_sample/results/offline_trca_grid/
tasks/ssvep_hd_200target_tdca_sample/results/figures/
tasks/ssvep_hd_200target_tdca_sample/results/confusions/
```

Smoke runs and one-off feature probes are kept under `results/_legacy/` after
they have served their validation purpose.

## Commands

Run the paper online Table 2 reproduction:

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_hd_200target_tdca_sample\run_online_table2.py
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_hd_200target_tdca_sample\plot_online_table2.py
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_hd_200target_tdca_sample\plot_online_table2_confusions.py
```

Run or resume the paper offline TDCA grid:

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_hd_200target_tdca_sample\run_offline_tdca_grid.py --subjects author --target-sets 40,80,120,160,200 --channel-sets 9,21,32,66 --windows-ms 100,200,300,400,500
```

The runner skips rows already present in `offline_tdca_grid/trials.csv`, so the
same command resumes after interruption.
