# trcanet_benchmark_pytorch — Frozen Snapshot

- **Path**: `D:/ProjData/proj_python/trcanet_benchmark_pytorch`
- **Version control**: None (no .git)
- **Frozen**: 2026-07-02

## Purpose

Deng et al. (2023) TRCA-Net for the Tsinghua Benchmark dataset.
Per-subject TRCA spatial filter fitting + CNN classification on projected features.

## Key files

- `trcanet_benchmark/model.py` — TRCANet CNN classifier
- `trcanet_benchmark/trca.py` — TRCA filter fitting + projection (includes `dnn_filterbank`, `cascade` options)
- `trcanet_benchmark/config.py` — BenchmarkConfig (includes `trcanet_subbands=3`)
- `trcanet_benchmark/data.py` — data loading
- `scripts/run_benchmark.py` — global train + per-subject finetune
- `scripts/run_sweep.py` — window length sweep

## Model

- Architecture: subband conv → filter conv → temporal conv → refine conv → classifier
- Parameters: 778,003
- Input shape: `(batch, subbands, class_filters, samples)`
- Preprocessing: Nakanishi-style Chebyshev-I filterbank, 3 subbands, TRCA spatial filters

## Dependencies

- torch==2.7.1+cu128, numpy, scipy, scikit-learn, h5py

## Outputs preserved

- 147 .pt checkpoint files, ~331 MB total
- Location: `outputs/` (global + per-subject finetuned models)

## Migration to Arena

- Model code → `vep_arena/nn/trcanet.py`
- TRCA filters → `vep_arena/methods/trcanet.py`
- Training script → `scripts/evaluate_trcanet.py`
- Smoke test → `tasks/benchmark_trcanet_smoke/`

## Notes

- Sibling has extra features not in Arena: `cascade` TRCA mode, `dnn_filterbank` projection option
- Arena uses `independent` TRCA + `trca` filterbank (matches sibling defaults)
