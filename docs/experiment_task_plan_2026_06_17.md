# VEP Arena Experiment Task Plan 2026-06-17

This document turns the current VEP Arena state into a task queue for scheduled
experiments. The goal is to keep algorithm work reproducible, cache efficient,
and easy to compare without rerunning completed sweeps.

## Current Stable Baseline

### Data and Preprocessing

- Canonical dataset preset: `benchmark_9ch_cue0.5_latency0.14`.
- Dataset: Tsinghua Benchmark SSVEP.
- Channels: 9-channel occipital/parietal subset `[48, 54, 55, 56, 57, 58, 61, 62, 63]`.
- Timing: skip 0.5 s cue plus 0.14 s visual latency; canonical crop starts at 0.64 s.
- Sampling rate: 250 Hz.
- Classical-method preprocessing: 50 Hz notch plus Benchmark toolbox-style filter bank.
- Cache policy: subject-level max-window canonical epoch cache, then slice shorter windows.
- Cache location: `runs/canonical_epochs`.

The cache rule is important: dataset preprocessing creates the longest requested
epoch for each subject, then all method/window runs reuse that tensor by slicing.
No later algorithm sweep should reread `.mat` files or create separate epoch
files for every shorter window.

### Implemented Arena-Native Methods

- `CCA`: non-training baseline on raw canonical epochs.
- `FBCCA`: filter-bank CCA baseline.
- `TRCA`: subject-specific leave-one-block-out training.
- `ETRCA`: ensemble TRCA variant.
- `TDCA`: subject-specific leave-one-block-out training with temporal-delay augmentation.

### Implemented Outputs

Arena-native runs should produce one shallow result directory under `results/`:

- `manifest.json`
- `trials.csv`
- `predictions.csv`
- `summary.csv`
- `subject.csv`
- `block.csv`
- `runtime.csv`
- `report.md`
- `figures/accuracy_curve.png`
- `figures/itr_curve.png`
- `figures/accuracy_heatmap.png`
- `figures/subject_box_best.png`
- `figures/subject_window_heatmap_<method>.png`
- `figures/block_cv_accuracy_curve.png`

Large optional debugging artifacts are opt-in only:

- `score_matrices/`
- `model_artifacts/`

## Current Gaps

### Task Configuration

Current runners accept command-line arguments and write manifests, but there is
not yet a single task JSON schema that can drive all algorithms. This is the
next cleanup needed before timed or scheduled execution.

Recommended task file location:

```text
configs/tasks/
```

Recommended schema:

```json
{
  "task_name": "tdca_9ch_w02_2s",
  "dataset_preset": "benchmark_9ch_cue0.5_latency0.14",
  "preprocess_preset": "benchmark_toolbox_filterbank_9ch",
  "protocol": "subject_lobo",
  "subjects": "1-35",
  "blocks": "1-6",
  "windows": "0.2:0.2:2.0",
  "methods": ["TDCA"],
  "method_params": {
    "n_fbs": 5,
    "harmonics": 5,
    "n_components": 8,
    "n_delay": 5
  },
  "execution": {
    "workers": 8,
    "blas_threads": 1,
    "stress_first": true,
    "resume": true
  },
  "outputs": {
    "save_score_matrices": false,
    "save_model_artifacts": false
  }
}
```

### Runner Unification

The traditional and TDCA runners share a lot of behavior. They should converge
on common helpers for:

- parsing ranges and windows;
- manifest construction;
- resume/completion checks;
- standardized runtime ledger rows;
- result writing and plotting;
- task JSON loading.

The short-term path can be incremental: keep current scripts working, then add a
`scripts/run_task.py` wrapper that loads JSON and dispatches to method-specific
adapters.

### Stress Test Harness

Every full run should pass a small stress run with the same execution style:

- 2 subjects;
- 2 windows;
- 2 or 3 blocks;
- same worker count style;
- `resume` path checked;
- `runtime.csv` checked;
- figure generation checked;
- cache hit behavior checked.

For multi-process classical methods, set:

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
```

Then use process workers to consume the CPU budget. Avoid nested BLAS thread
oversubscription.

## Algorithm Task Queue

### Task 1: Formalize Task JSON Runner

Deliverables:

- `configs/tasks/*.json` examples for traditional, TDCA, and SA-MVMD tasks.
- A loader that validates dataset/preprocess/protocol/method/execution/output.
- A runner wrapper that writes the resolved task config into `manifest.json`.
- A smoke command and a full command for each task.

Success check:

- The same TDCA smoke test can be run from JSON and from the old CLI with
  matching `summary.csv` values.

### Task 2: Keep TDCA as Standard TDCA

TRCA has a clear ensemble variant (`ETRCA`). TDCA is implemented as the
standard method from Liu et al. 2021, with filter-bank weighting and
temporal-delay augmentation. Do not add a TDCA ensemble variant unless a later
source-backed protocol specifically requires it.

Deliverable:

- Keep TDCA manifests and reports named as standard `TDCA`.

### Task 3: SA-MVMD-TRCA and SA-MVMD-ETRCA

SA-MVMD is more expensive than TDCA and should be integrated with a decomposition
cache before full sweeps.

Subtasks:

- Confirm current paper-note assumptions in `docs/paper_notes/sa_mvmd_trca/`.
- Implement or harden decomposition cache under `runs/`.
- Run a 2-subject stress test.
- Run `SA-MVMD-TRCA` over 0.2-2.0 s.
- Run `SA-MVMD-ETRCA` if the scoring variant is supported by the paper/code.
- Add both to combined plotting without rerunning completed baselines.

Artifact policy:

- Save per-trial predictions and summary CSVs by default.
- Save decomposition internals only when debugging or validating the method.

### Task 4: External Baseline Import Adapters

Existing sibling projects should be imported into Arena schema before being
treated as native methods:

- `dnn_ssvep_pytorch`
- `fbtrca_benchmark_python`
- `ssvepformer_benchmark_pytorch`
- `trcanet_benchmark_pytorch`

First deliverable:

- Convert existing clean result CSVs into Arena-style `trials.csv`,
  `summary.csv`, `subject.csv`, `block.csv`, and `manifest.json`.

Second deliverable:

- Add small inference smoke tests only when checkpoint and preprocessing are
  clear.

### Task 5: DNN Reprocessing and Retraining Decision

The existing DNN preprocessing is not identical to the Arena classical
preprocessing:

- crops the canonical slice using 0.5 s cue plus 0.14 s latency;
- uses 3 Chebyshev-I order-2 subbands;
- each subband is `[8*i, 90]` Hz;
- filtering happens after cropping;
- no 50 Hz notch was observed in `dnn_ssvep/data.py`.

Therefore:

- existing DNN results can be imported as a documented external baseline;
- if DNN is changed to Arena canonical toolbox filter-bank preprocessing, it
  should be retrained;
- comparisons must record the preprocessing preset explicitly.

Recommended DNN task split:

1. `dnn_import_existing_clean_results`
2. `dnn_sample_inference_adapter`
3. `dnn_retrain_arena_preprocess_smoke`
4. `dnn_retrain_arena_preprocess_full`

## Preset Design

Use named presets so scheduled tasks do not hide preprocessing assumptions in
script arguments.

Recommended preset groups:

```text
dataset_preset:
  benchmark_9ch_cue0.5_latency0.14
  benchmark_9ch_cue0.5_latency0.00
  benchmark_9ch_offset_ablation_<latency>

preprocess_preset:
  benchmark_raw_9ch
  benchmark_toolbox_filterbank_9ch
  dnn_cheby3_subband_9ch
  ssvepformer_preprocess_9ch

protocol:
  subject_lobo
  subject_lobo_window_sweep
  calibration_size_sweep
  cross_subject
```

Each task manifest should store the resolved preset content, not only the preset
name. That makes old result folders interpretable even after config files evolve.

## Scheduling Plan

### Near-Term Queue

1. Add task JSON runner and validation.
2. Add stress-test command path.
3. Keep TDCA as standard TDCA; do not schedule ensemble TDCA.
4. Add SA-MVMD-TRCA smoke task.
5. Add SA-MVMD-ETRCA smoke task if supported.
6. Import existing DNN clean results into Arena schema.
7. Add combined plotting that reads multiple result directories only.

### Full Runs After Smoke Tests

Use 0.2-2.0 s windows for the main classical comparison:

```text
0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0
```

Each full run should write one result folder with a short task name and should
not create nested date/task/method/window directory trees.

### Required Run Header

Before scheduled execution, log these values:

- git commit hash;
- task JSON path;
- data root;
- Python executable;
- worker count;
- BLAS thread env vars;
- output result directory;
- cache directory;
- whether score/model artifacts are enabled.

## Acceptance Criteria

A method is considered integrated into VEP Arena only when:

- preprocessing and protocol are named in `manifest.json`;
- a smoke task passes;
- full run can resume from partial output;
- output files match Arena schema;
- plotting can be regenerated from CSVs without rerunning algorithms;
- optional heavy artifacts are disabled by default;
- runtime is captured in `runtime.csv`;
- source code is committed and generated artifacts are ignored.
