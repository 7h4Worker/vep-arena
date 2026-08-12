# Benchmark Multichannel Decision-Channel Capacity

This task extends the fixed-context Tsinghua Benchmark analysis from the
existing occipital 9-channel baseline to two posterior expansions, a
same-count whole-head control, and the full 64-channel recording.

## Experiment contract

- subjects: 35
- methods: CCA, FBCCA, ECCA, TRCA, ETRCA
- windows: 0.1-5.0 s in 0.1 s increments
- preprocessing context: fixed 5.0 s before window slicing
- cross-validation: subject-specific leave-one-block-out
- channel configurations:
  - `occipital9`
  - `posterior21`
  - `posterior32`
  - `wholehead32`
  - `full64`
- confusion matrices: one 40 x 40 count matrix per
  subject/method/channel-config/window cell
- capacity normalization: `alpha=0.0`

The two 32-channel configurations are intentionally separate. The
`posterior32` configuration preserves the nested HD200-style ablation, while
`wholehead32` controls for spatial coverage at the same sensor count. Output
tables therefore include both `channel_config` and `channels`.

## Paths

Generated results remain under the path requested by the parent analysis:

```text
tasks/benchmark_decision_channel_capacity/results/extended/multichannel/
```

Fixed-context 64-channel epoch caches are generated under:

```text
runs/benchmark_multichannel_fixed5_cache/
```

Both locations are ignored by git. Existing 9-channel results are read-only;
their coarse-grid predictions are converted into the per-cell confusion format
without changing the source files.

## Execution

Run the complete resumable pipeline from the repository root:

```powershell
.venv\Scripts\python.exe tasks\benchmark_multichannel_decision_channel_capacity\run_overnight.py --workers 4 --cache-workers 2 --chunk-size 5
```

The runner prevents system sleep while active, records a PID lock and state,
retries failed stages, and relies on atomic per-window confusion files for
resume. Fast methods run before ECCA, and each method processes `full64` first.
