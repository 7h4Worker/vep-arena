# Result Artifact Contract

This is the project-level standard for VEP Arena experiment outputs.

## Complete Run Directory

A complete algorithm run directory should contain:

- `manifest.json`: dataset, protocol, methods, subjects, windows, channels, code
  entry point, cache/preprocess notes, status, and timestamp.
- `trials.csv`: one row per evaluated unit used for aggregate metrics, usually
  method/window/subject/block.
- `predictions.csv`: one row per classified trial or command. Required for
  confusion matrices and error-structure analysis.
- `summary.csv`: aggregate method/window metrics.
- `subject.csv`: subject-level aggregate metrics when subjects exist.
- `block.csv` or equivalent split-level table when block/session splits exist.
- `runtime.csv`: timing rows when the runner has meaningful fit/predict stages.
- `confusion_*.npy` or `confusions/*.npy`: confusion matrices generated from
  `predictions.csv`.
- figures generated from CSV/NPY artifacts, not from hidden in-memory state.

`predictions.csv` is mandatory for any result that may be used for scientific
analysis beyond aggregate accuracy/ITR. If a runner cannot write predictions,
the manifest must say why and the result must be marked as aggregate-only.

## Aggregate-Only Directory

Aggregate or comparison directories may intentionally contain only:

- `summary.csv`
- optional `subject.csv`, `block.csv`, `paired_stats.csv`, or source ledgers
- `manifest.json` or an explicit source ledger

These directories should not be treated as raw experimental runs. They are
derived views. Plotting from them is acceptable; confusion/error analysis is
not.

## Required Prediction Columns

Use these columns when the task naturally supports them:

- `method`
- `subject`
- `block` or `session`
- `trial_index`
- `true`
- `pred`
- `correct`
- `window` or `window_ms`
- dataset-specific context such as `targets`, `channels`, `time_samples`

Labels should be stored as the human-facing one-based labels when that matches
the dataset/task output. If zero-based labels are used internally, the manifest
must state the convention.

## HD-200 Online Table 2

The HD-200 online Table 2 runner follows this contract with:

- `online_table2_reproduction.csv`
- `online_table2_predictions.csv`
- `online_table2_manifest.json`
- `confusions/confusion_<subject>_<targets>target.npy`
- `online_table2_reproduction.png`
- `online_table2_confusions.png`
- `online_table2_top_errors.csv`

Filtering caches are not final result artifacts. They live under the dataset
derivatives directory and may be reused by reruns.
