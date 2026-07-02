# ssvepformer_benchmark_pytorch — Frozen Snapshot

- **Path**: `D:/ProjData/proj_python/ssvepformer_benchmark_pytorch`
- **Version control**: None (no .git)
- **Frozen**: 2026-07-02

## Purpose

Chen et al. SSVEPFormer adapted for the Tsinghua Benchmark dataset.
Butterworth bandpass 8–64 Hz preprocessing, model performs FFT internally.

## Key files

- `ssvepformer_benchmark/model.py` — SSVEPFormerTH, FBSSVEPFormer
- `ssvepformer_benchmark/config.py` — BenchmarkConfig
- `ssvepformer_benchmark/data.py` — bandpass + split loading
- `scripts/train_benchmark.py` — leave-one-block-out training

## Model

- Architecture: FFT transform → channel combiner → 2× Encoder (conv + attention-like proj) → MLP head
- Parameters: 1,998,064 (SSVEPFormerTH)
- Input shape: `(batch, channels, samples)`
- Preprocessing: Butterworth order-4 bandpass [8, 64] Hz

## Dependencies

- torch==2.7.1+cu128, numpy, scipy, scikit-learn, h5py

## Outputs preserved

- 117 .pt checkpoint files, ~1.1 GB total
- Location: `outputs/` (global models across window lengths and blocks)

## Migration to Arena

- Model code → `vep_arena/nn/ssvepformer.py`
- Training script → `scripts/evaluate_ssvepformer.py`
- Smoke test → `tasks/benchmark_ssvepformer_smoke/`
