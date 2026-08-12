# DNN Integration Notes

Author: Liu Haifeng <liuhf@728@gmail.com>  
Created: 2026-06-17  
Last updated: 2026-06-17

## Current Scope

The first Arena-side DNN integration is an external baseline import from the
sibling PyTorch reproduction project:

- Source project: `D:/ProjData/proj_python/dnn_ssvep_pytorch`
- Clean aggregate tables: `results_clean/`
- Arena importer: `scripts/import_dnn_results.py`

This keeps the validated DNN sweep usable inside Arena reports without making
PyTorch a required dependency for the classical SSVEP benchmark code path.

## Imported Result Contract

The importer reads `subject_block_results.csv` and writes the standard Arena
aggregate files:

- `trials.csv`
- `summary.csv`
- `subject.csv`
- `block.csv`
- `runtime.csv`
- `manifest.json`
- `figures/`

The clean DNN result table contains subject/block aggregate accuracy only, so
the importer does not create `predictions.csv` or confusion matrices.

## Preprocessing Boundary

The DNN reproduction preprocesses Benchmark 9-channel data with these observed
settings:

- Canonical crop from cue plus visual latency.
- 9 Benchmark channels.
- 3 Chebyshev-I order-2 subbands with pass bands `[8 * i, 90]` Hz.
- Filtering after crop.
- No 50 Hz notch in the observed preprocessing path.

Arena's classical methods often use the SSVEP-Analysis-Toolbox-style filterbank
and notch path. If DNN should consume that exact Arena preprocessing, the model
must be retrained or fine-tuned on those tensors; imported scores should not be
treated as evidence for that altered pipeline.

## Next Implementation Layers

1. Result import: keep existing clean DNN sweeps comparable inside Arena.
2. Inference adapter: load a trained DNN checkpoint and run Arena-created test
   tensors when the preprocessing contract matches the checkpoint.
3. Training adapter: call the PyTorch training code through a narrow script or
   package boundary, then emit Arena-compatible result files.
4. Adaptation adapter: expose subject fine-tuning or online adaptation as a
   separate protocol, because it is not the same evaluation design as ordinary
   leave-one-block-out classical baselines.

## Smoke Command

```powershell
.\.venv\Scripts\python.exe scripts\import_dnn_results.py
```

The default output is ignored by git:

```text
results/benchmark_9ch/dnn_import_existing
```
