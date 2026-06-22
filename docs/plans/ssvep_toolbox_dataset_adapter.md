# SSVEP-Analysis-Toolbox Dataset Ideas

This note records what Arena borrows from SSVEP-Analysis-Toolbox and what it intentionally does not copy.

## Borrowed Ideas

SSVEP-Analysis-Toolbox has a clean dataset contract:

- dataset metadata is explicit: subjects, channels, sampling rate, blocks, targets, frequencies, phases, prestimulus time, break time, latency.
- every dataset can expose subject data in a common shape.
- preprocessing and filter-bank steps can be attached to the dataset path.
- evaluation code consumes trial tensors instead of raw dataset-specific MATLAB shapes.

Arena adopts the same core shape convention:

```text
subject data: blocks x targets x channels x samples
trial batch:  trials x bands x channels x samples
labels:       trials
```

This is enough for Benchmark, BETA, traditional methods, DNN adapters, MNE QA, and future LSL/PsychoPy backends to meet at one boundary.

## What Arena Does Not Copy

Arena does not copy the toolbox evaluator or algorithm implementations at this stage.

Reasons:

- existing Benchmark 9ch results are already available and do not need reruns just to change the interface.
- Arena also needs DNN, MNE, ERP-like views, local study scripts, and online/LSL hooks.
- a lighter Python API is easier to read and operate in this project than a full toolbox framework.

## Current Arena Interface

New lightweight files:

```text
vep_arena/data/interface.py
vep_arena/data/datasets.py
vep_arena/data/beta.py
```

Example:

```python
from vep_arena.data import benchmark_9ch

ds = benchmark_9ch()
batch = ds.get_trials(
    subject=1,
    blocks=[1],
    targets=list(range(40)),
    channels="occipital_9ch",
    window=1.0,
    preprocess="toolbox_fb",
    n_bands=5,
)

assert batch.x.shape == (40, 5, 9, 250)
assert batch.y.shape == (40,)
```

The old functions remain available:

```text
vep_arena/data/benchmark.py
vep_arena/data/epochs.py
```

Existing scripts can keep using them.

## BETA Status

`beta_9ch()` exposes metadata inspired by SSVEP-Analysis-Toolbox, but loading is intentionally not implemented until local BETA files are inspected.

Required before implementation:

- exact local file path.
- `.mat` keys and nested fields.
- shape and dimension order.
- channel names.
- whether files are already downsampled to 250 Hz.

## Result Policy

Changing this interface does not require rerunning existing baseline results.

Existing outputs under `results/benchmark_9ch/` remain valid as historical results. Future studies can use the new dataset interface when adding BETA, LSL hooks, or clearer study scripts.
