# ssvepformer_minimal — Frozen Snapshot

- **Path**: `D:/ProjData/proj_python/ssvepformer_minimal`
- **Version control**: None (no .git)
- **Frozen**: 2026-07-02

## Purpose

Minimal SSVEPFormer-style CUDA prototype. Single-file smoke test validating
FFT feature extraction + MLP-Mixer architecture on synthetic SSVEP data.

## Key files

- `train_synthetic.py` — SSVEPFormerMini model + synthetic data + training loop (all-in-one, 141 lines)

## Model

- Architecture: FFT → channel combiner (Conv1d) → 2× MixerBlock → AdaptiveAvgPool → linear classifier
- Parameters: 536,466
- Input shape: `(batch, eeg_channels, samples)`
- Default config: 8 channels, 512 samples, 4 classes

## Dependencies

- torch==2.7.1+cu128, numpy, scikit-learn, tqdm

## Outputs preserved

- No .pt checkpoints
- `outputs/smoke_test_2026-05-18.txt` — CUDA validation log only

## Migration to Arena

- Concept evolved into the full SSVEPFormerTH at `vep_arena/nn/ssvepformer.py`
- This minimal version was an initial CUDA validation, not a benchmark model
