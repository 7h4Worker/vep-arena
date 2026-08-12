# DNN Integration Handoff

Author: Liu Haifeng <liuhf@728@gmail.com>  
Created: 2026-06-18  
Last updated: 2026-06-18

## Current Focus

The current task is to turn DNN support from a historical-result import into an
Arena-side evaluation and adaptation pipeline.

The historical DNN result import is useful only as a reference check. The active
goal is now:

1. Use Arena data loading and DNN-compatible preprocessing.
2. Load saved DNN `.pt` checkpoints.
3. Re-run inference inside Arena.
4. Add subject-specific fine-tuning when reproducing the paper-style final
   DNN scores.
5. Keep the flow reusable for future datasets instead of copying CSV outputs.

## Branch And Git State

Current branch:

```text
dnn-integration
```

Recent commits:

```text
21a1348 Add Arena DNN checkpoint evaluator
0485ee8 Use imported DNN baseline in final comparison
60ba53f Add DNN baseline import adapter
```

The working tree was clean after commit `21a1348`.

## Relevant Repositories

Arena repository:

```text
D:/ProjData/proj_python/vep_arena
```

DNN PyTorch reproduction:

```text
D:/ProjData/proj_python/dnn_ssvep_pytorch
```

Official MATLAB reference:

```text
D:/ProjData/_reference/Deep-SSVEP-BCI
```

## Implemented Arena Files

### Historical result import

```text
scripts/import_dnn_results.py
docs/dnn_integration.md
```

This imports:

```text
D:/ProjData/proj_python/dnn_ssvep_pytorch/results_clean/subject_block_results.csv
```

It writes Arena-style aggregate outputs under:

```text
results/benchmark_9ch/dnn_import_existing
```

This is only a historical validation/reference path. It does not load models or
re-run inference.

### Final comparison integration

```text
scripts/make_final_comparison_plus_sa_mvmd.py
```

The final comparison now prefers:

```text
results/benchmark_9ch/dnn_import_existing/summary.csv
```

for DNN rows when available. This keeps plots connected to the documented DNN
import instead of an older consolidated table.

### Arena checkpoint evaluator

```text
scripts/evaluate_dnn_checkpoints.py
```

This script is the important new path. It performs:

```text
Arena raw Benchmark data
-> DNN-style crop and filterbank preprocessing
-> saved PyTorch checkpoint loading
-> inference
-> Arena-compatible result files
```

It intentionally avoids importing the DNN project's result CSV files.

Arena's own `.venv` does not currently include PyTorch, so run this script with
the DNN project's Python:

```powershell
D:\ProjData\proj_python\dnn_ssvep_pytorch\.venv\Scripts\python.exe scripts\evaluate_dnn_checkpoints.py --window 1.0 --blocks 1-6 --output-dir results\benchmark_9ch\dnn_checkpoint_eval_1s
```

Verified output:

```text
results/benchmark_9ch/dnn_checkpoint_eval_1s
```

## Verified 1.0 s Checkpoint Result

The saved `.pt` models are stage-1 global checkpoints. Arena-side inference
from these checkpoints exactly matches the previous DNN project's global-only
evaluation.

Block-level results at 1.0 s:

| Block | Arena global `.pt` | Previous global-only | Historical fine-tuned |
| ---: | ---: | ---: | ---: |
| 1 | 0.7893 | 0.7893 | 0.9564 |
| 2 | 0.8071 | 0.8071 | 0.9421 |
| 3 | 0.8321 | 0.8321 | 0.9671 |
| 4 | 0.8300 | 0.8300 | 0.9593 |
| 5 | 0.8229 | 0.8229 | 0.9614 |
| 6 | 0.8121 | 0.8121 | 0.9571 |

Mean results:

```text
Arena global .pt inference: 0.8155952381
Previous global-only table: 0.8155952381
Historical fine-tuned result: 0.9572619048
```

This proves that the Arena-side DNN preprocessing and checkpoint loading are
compatible with the saved global checkpoints.

It also proves that the historical high DNN score cannot be recovered by
loading the saved `.pt` files alone, because those `.pt` files are not the
subject-fine-tuned models.

## Saved Model State

Registry:

```text
D:/ProjData/proj_python/dnn_ssvep_pytorch/results_clean/model_registry.csv
```

Current saved checkpoint coverage:

```text
windows: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
blocks per window: 6
total checkpoints: 54
```

There are no saved 2.0 s checkpoints and no saved 2.0 s results.

The registry note is important:

```text
Subject-specific fine-tuned models were used for evaluation but were not saved in the overnight run.
```

The saved `.pt` checkpoints are therefore global stage-1 checkpoints only.

The classifier shape is window-specific, so a 1.0 s model cannot be directly
used for 2.0 s:

```text
0.2 s classifier.weight: (40, 3000)
0.4 s classifier.weight: (40, 6000)
1.0 s classifier.weight: (40, 15000)
```

A 2.0 s DNN result requires training a new 2.0 s model, or training and saving
new 2.0 s checkpoints.

## DNN Preprocessing Contract

The Arena evaluator mirrors the DNN paper/MATLAB preprocessing rather than the
classical Arena filterbank used by FBCCA/TRCA/TDCA.

Confirmed official MATLAB settings from:

```text
D:/ProjData/_reference/Deep-SSVEP-BCI/main.m
D:/ProjData/_reference/Deep-SSVEP-BCI/PreProcess.m
```

Benchmark DNN settings:

```text
subjects: 35
blocks: 6
classes: 40
sampling rate: 250 Hz
visual cue: 0.5 s
visual latency: 0.14 s
channels: [48, 54, 55, 56, 57, 58, 61, 62, 63]
subbands: 3
filter type: Chebyshev-I bandpass
filter order: 2
passband ripple: 1
low cutoffs: 8, 16, 24 Hz
high cutoff: 90 Hz
filtering: filtfilt-style zero-phase filtering
notch: none observed in the DNN path
```

Arena evaluator currently uses:

```text
crop_start_seconds = 0.64
scipy.signal.cheby1(..., output="sos")
scipy.signal.sosfiltfilt(...)
```

This matched the old global-only checkpoint results exactly in the 1.0 s
validation.

## Difference From Classical Arena Preprocessing

Classical Arena methods often use a toolbox-style path:

```text
stimulus-onset segment -> 50 Hz notch -> toolbox filterbank -> latency crop
```

DNN currently uses:

```text
cue + latency crop -> 3 DNN Chebyshev subbands -> model input
```

Do not mix these protocols without retraining. If DNN should use the classical
Arena filterbank tensor, retrain or fine-tune the DNN on that tensor.

## Why Historical Import Is Not Enough

`scripts/import_dnn_results.py` is still useful for quick report comparison,
but it does not answer whether Arena can run DNN on a new dataset.

For new datasets, the needed flow is:

```text
DatasetSpec
-> DNNPreprocessSpec
-> train/evaluate checkpoints
-> optional subject adaptation
-> Arena result writer
```

The checkpoint evaluator is the first real step toward that flow.

## Immediate Next Step

Add a fine-tuning mode to:

```text
scripts/evaluate_dnn_checkpoints.py
```

Recommended CLI shape:

```powershell
D:\ProjData\proj_python\dnn_ssvep_pytorch\.venv\Scripts\python.exe scripts\evaluate_dnn_checkpoints.py `
  --window 1.0 `
  --blocks 1-6 `
  --finetune-epochs 1000 `
  --output-dir results\benchmark_9ch\dnn_checkpoint_eval_1s_finetune
```

Implementation outline:

1. Load the global stage-1 checkpoint for the held-out block.
2. For each subject:
   - clone/load the global checkpoint into a local model;
   - train only on that subject's non-held-out blocks;
   - use dropout `0.6` for the first two dropout layers during fine-tuning;
   - keep final dropout `0.95`;
   - Adam learning rate `1e-4`, weight decay `1e-3`;
   - batch size `classes * train_blocks`, matching MATLAB's second stage.
3. Evaluate the subject's held-out block.
4. Write subject/block rows, predictions, runtime, and manifest.

Expected target for 1.0 s after fine-tuning:

```text
mean accuracy near 0.9572619048
```

Do a shorter smoke first:

```powershell
--subjects 1-2 --blocks 1 --window 1.0 --finetune-epochs 1
```

Then run all subjects/blocks with the full epoch count.

## Open Questions

1. Should fine-tuned subject models be saved this time?
   Recommended: yes, at least optionally, because the previous sweep did not
   save them and that caused the current ambiguity.

2. Should DNN outputs be merged into final comparison only after Arena-side
   inference/fine-tune, instead of historical import?
   Recommended: yes. Keep historical import separate as a reference baseline.

3. Should a dataset abstraction be introduced now?
   Recommended: after fine-tune validation passes on Benchmark 1.0 s, factor the
   evaluator into reusable preprocessing and protocol pieces.

4. Should 2.0 s be trained?
   A 2.0 s point requires new training because no 2.0 s checkpoints exist and
   classifier dimensions are window-specific.
