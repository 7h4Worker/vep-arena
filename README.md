# VEP Arena

VEP Arena is a clean workspace for benchmarking visual evoked potential BCI
methods. The first target is SSVEP recognition on the Tsinghua Benchmark
dataset, using the already available DNN, SSVEPFormer, FBTRCA results and a new
TDCA baseline.

The project borrows high-level ideas from SSVEP-Analysis-Toolbox:

- dataset metadata should be explicit
- preprocessing should be registered as a named pipeline
- protocols should generate train/test splits
- algorithms should expose a small fit/predict interface
- evaluation should be centralized

It does not copy the full toolbox structure. Neural networks need checkpoints,
devices, seeds, training logs, and deployment hooks, so this project keeps the
method interface smaller and leaves method-specific training details inside
method adapters.

## Main Ideas

```text
dataset -> preprocess -> protocol -> method -> evaluator -> report
```

Clean outputs should stay short and human-readable:

```text
results/
  benchmark_9ch/
    report.md
    summary.csv
    subject.csv
    block.csv
    stats.csv
    tables/
    figures/
```

Raw method outputs, logs, and checkpoints belong under `runs/`.

## Current Methods

- DNN: imported from `D:/ProjData/proj_python/dnn_ssvep_pytorch`
- SSVEPFormer: imported from `D:/ProjData/proj_python/ssvepformer_benchmark_pytorch`
- FBTRCA: imported from `D:/ProjData/proj_python/fbtrca_benchmark_python`
- TDCA: implemented as a first VEP Arena conventional-method adapter
- TRCA-Net: staged for reproduction from `D:/ProjData/_reference/TRCA-Net`

## Current Benchmark 9ch Report

The first unified report is available at:

```text
D:/ProjData/proj_python/vep_arena/results/benchmark_9ch/report.md
```

To rebuild it from the current imported results and TDCA run:

```powershell
cd D:/ProjData/proj_python/vep_arena
.venv/Scripts/python.exe scripts/build_report.py
```

To rerun TDCA:

```powershell
cd D:/ProjData/proj_python/vep_arena
.venv/Scripts/python.exe scripts/run_tdca.py --subjects 1-35 --blocks 1-6 --windows default --output-dir runs/tdca
```

To verify the TRCA-Net feature adapter:

```powershell
cd D:/ProjData/proj_python/vep_arena
.venv/Scripts/python.exe scripts/check_trcanet_features.py --subject 1 --block 1 --window 0.4
```

## Next Direction

After this first Benchmark 9ch report is stable, the same structure can add:

- FBCCA, eCCA, TRCA variants, TDCA variants
- EEGNet and other neural baselines
- BETA, Nakanishi2015, wearable SSVEP datasets
- MNE-based feature analysis
- spectral/SNR/phase diagnostics
- t-SNE or UMAP feature visualizations
- online-style prediction demos
