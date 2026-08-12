# VEP Arena Architecture

VEP Arena should stay readable first. Advanced options can exist, but the main
path should be easy:

```text
choose dataset -> choose protocol -> choose methods -> run -> read results
```

## Layers

### Data

Dataset loaders expose metadata and trial tensors. They should not know about a
specific algorithm.

Current dataset:

- Tsinghua Benchmark SSVEP
- 35 subjects
- 6 blocks
- 40 targets
- 250 Hz
- 9-channel occipital/parietal subset

### Preprocessing

Preprocessing must be named and recorded in the output. This matters because
neural networks and spatial-filter methods often use different filtering.

Examples:

- `dnn_filterbank`
- `ssvepformer_bandpass`
- `fbtrca_filterbank`
- `notch50_benchmark_filterbank`

### Protocol

Protocols define train/test splits and evaluation scope.

Current protocol:

- subject-specific
- leave-one-block-out
- windows: 0.2 to 1.0 s
- subject/block/window level outputs

Future protocols:

- cross-subject
- calibration-size sweep
- cross-dataset transfer
- online simulation
- session drift or time-order split

### Methods

Each method should expose a small adapter:

```text
fit(train, context)
predict(test, context)
save(path)
load(path)
predict_online(window, context)
```

Method-specific complexity stays inside the adapter. For example, neural
network methods can manage device, epochs, checkpoints, seeds, and logs without
forcing conventional spatial filters to care about those details.

### Evaluation

The evaluator consumes standardized trial rows:

```text
method, dataset, protocol, subject, block, window, accuracy, itr, samples
```

From that, reports can derive:

- group mean
- subject standard deviation
- SEM
- within-subject block standard deviation
- paired statistical tests
- curves and compact tables

### Analysis Tools

These should be optional modules, not part of the core training loop:

- MNE preprocessing and spectral diagnostics
- SNR and phase analysis
- feature embedding export
- t-SNE or UMAP plots
- confusion matrices
- model calibration and deployment checks

## Output Policy

Use short names for clean outputs:

```text
results/benchmark_9ch/report.md
results/benchmark_9ch/summary.csv
results/benchmark_9ch/subject.csv
results/benchmark_9ch/block.csv
results/benchmark_9ch/stats.csv
results/benchmark_9ch/figures/accuracy.svg
```

Raw logs and checkpoints go under `runs/`.

## Current Caveats

The current DNN, SSVEPFormer, and FBTRCA entries are imported from previous
project runs. Their protocols are aligned at the Benchmark 9ch leave-one-block
level, but their preprocessing and training procedures are method-specific.
That is acceptable for the first arena report, but the preprocessing ledger
must stay explicit.

TDCA is the first method implemented directly in VEP Arena. It follows the
paper's Benchmark settings: 5 filter banks, 5 harmonics, 8 components, and
5 temporal delays. For training, the loader keeps the extra post-window samples
needed by TDCA's delayed copies; prediction pads delayed samples with zeros so
the test decision does not use samples beyond the requested window.
