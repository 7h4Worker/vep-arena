# dnn_ssvep_pytorch -- Frozen Snapshot

- **Repo**: `D:\ProjData\proj_python\dnn_ssvep_pytorch`
- **Remote**: none (no `.git` directory; never version-controlled)
- **Last commit**: n/a
- **Status at freeze**: untracked (no git)

## Purpose

PyTorch reproduction of the DNN-based SSVEP classifier from Guney, Oblokulov
& Ozkan (IEEE TBME, 2022). Trains on the Tsinghua SSVEP Benchmark dataset
(35 subjects, 40 classes, leave-one-block-out) and evaluates across signal
windows 0.2--1.0 s.

## Key files

- `dnn_ssvep/model.py` -- `DNNSsvep` network (Conv2d pipeline + linear classifier)
- `dnn_ssvep/data.py` -- Benchmark loader, Chebyshev filter bank, leave-one-block-out split
- `dnn_ssvep/config.py` -- `BenchmarkConfig` dataclass (9 channels, 3 subbands, 250 Hz)
- `dnn_ssvep/download.py` -- dataset downloader
- `dnn_ssvep/metrics.py` -- evaluation helpers
- `scripts/train_benchmark.py` -- main training entry point
- `scripts/run_experiment.py` -- single experiment runner
- `scripts/run_paper_sweep.py` -- signal-window sweep driver
- `scripts/analyze_results.py` -- post-hoc analysis (CSVs, report.md)
- `scripts/evaluate_global_checkpoints.py` -- checkpoint evaluation
- `scripts/make_paper_comparison.py` -- comparison against original MATLAB results
- `scripts/plot_finetune_effect.py` -- fine-tune lift plots
- `scripts/simulate_online_subject.py` -- online simulation script
- `reproduction_notes.md` -- MATLAB-to-PyTorch mapping details

## Model

- **Architecture**: Conv2d pipeline -- subband mix (Kx1x1) -> spatial (Kx9x1, 120 filters) -> dropout -> temporal (Kx1x2, stride 2) -> dropout -> ReLU -> refine (Kx1x10, same-pad) -> heavy dropout -> FC 40-way
- **Parameters**: ~120 conv filters per layer; classifier width depends on signal length
- **Input shape**: `(batch, 3, 9, samples)` -- 3 subbands, 9 channels, variable temporal samples
- **Classes**: 40 (SSVEP frequencies)
- **Init**: subband weights = 1.0; all others N(0, 0.01)
- **Dropout**: 0.1 (early layers), 0.95 (final)

## Dependencies

From `requirements-cu128.txt`:
```
torch==2.7.1+cu128
torchvision==0.22.1+cu128
torchaudio==2.7.1+cu128
```

From `requirements-common.txt`:
```
numpy
scipy
scikit-learn
tqdm
requests
py7zr
h5py
```

Python: `>=3.12, <3.13` (pyproject.toml)

## Outputs preserved

- **63 `.pt` checkpoint files**, ~153 MB total
- Top-level: 2 quick smoke-test models (S1, S1-35 at 0.4 s)
- `outputs/runs/dnn_ssvep_0p4s_full_20260518_overnight/` -- 6 blocks at 0.4 s (full 1000-epoch run)
- `outputs/runs/smoke_reporting_20260518/` -- 1 smoke model
- `outputs/paper_sweeps/paper_windows_20260518_overnight/` -- 54 models across 9 windows (0.2--1.0 s) x 6 blocks
- 2 JSON result summaries at top level
- `results_clean/` -- consolidated CSVs (block_results, subject_block_results, summary_by_window, model_registry), SVG plots (finetune effect, paper comparison), and 1 online-simulation `.pt`

## Migration to Arena

- Model code -> `vep_arena/nn/dnn_ssvep.py`
- Checkpoint evaluation -> `scripts/evaluate_dnn_checkpoints.py`
- Result import -> `scripts/import_dnn_results.py`
- Window sweep driver -> `scripts/run_dnn_window_sweep.py`
- Paper comparison -> `scripts/plot_dnn_comparison.py`
- Multi-algo integration -> `scripts/make_latest_multialgo_with_dnn.py`
- Smoke task -> `tasks/benchmark_dnn_smoke/run.py`

## Notes

- No git history exists; the directory was developed without version control.
  All provenance is reconstructed from file timestamps and contents.
- Original MATLAB reference is at `D:\ProjData\_reference\Deep-SSVEP-BCI`
  (GitHub: `osmanberke/Deep-SSVEP-BCI`).
- Dataset lives at `D:\ProjData\datasets\ssvep_benchmark` (35 subjects, ~2 GB).
- The overnight sweep (2026-05-18) ran all 9 windows x 6 blocks with 1000
  global epochs + 1000 fine-tune epochs on an RTX 5060 Ti.
- `results_clean/paper_comparison/` contains side-by-side accuracy comparison
  with the original MATLAB results.
