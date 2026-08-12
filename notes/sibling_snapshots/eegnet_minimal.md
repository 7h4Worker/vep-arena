# eegnet_minimal — Frozen Snapshot

- **Path**: `D:/ProjData/proj_python/eegnet_minimal`
- **Version control**: None (no .git)
- **Frozen**: 2026-07-02

## Purpose

Minimal CUDA-ready EEGNet-style prototype (Lawhern et al. 2018).
Single-file smoke test validating depthwise separable convolution on synthetic EEG data.

## Key files

- `train_synthetic.py` — model definition + synthetic data + training loop (all-in-one, 141 lines)

## Model

- Architecture: temporal conv → depthwise spatial conv → separable temporal conv → linear classifier
- Parameters: ~28,200
- Input shape: `(batch, 1, channels, samples)`
- Default config: 22 channels, 512 samples, 4 classes

## Dependencies

- torch==2.7.1+cu128, numpy, scikit-learn, tqdm

## Outputs preserved

- No .pt checkpoints (training script does not save models)
- `outputs/smoke_test_2026-05-18.txt` — CUDA validation log only

## Migration to Arena

- Model code → `vep_arena/nn/eegnet.py`
- No Arena training script yet (lowest priority — simplest model)
