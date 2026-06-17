# VEP Arena Lookback 2026-06-17

## What Went Wrong

- The first TDCA full run mixed three expensive things at once: process-level parallelism, repeated epoch-cache access, and per-block score/model artifact writes.
- Python numerical libraries also used their own BLAS threads inside each worker. That caused oversubscription: `workers * BLAS threads` became much larger than the intended CPU budget.
- The command timeout killed the parent process while workers could keep running, which made progress hard to reason about and briefly left orphaned Python processes.
- The current canonical epoch cache is subject-window based. It avoids repeated `.mat` parsing after prebuild, but it creates many large `.npz` files and is not the cleanest structure for repeated parameter sweeps.
- The runner did not start with a small stress test for the actual full-run mode: resume, workers, artifact policy, cache hits, timeout behavior, and BLAS thread limits together.

## What Is Now Fixed

- TDCA now has an arena-native runner: `scripts/run_tdca.py`.
- TDCA outputs the same core files as traditional methods: `trials.csv`, `predictions.csv`, `summary.csv`, `subject.csv`, `block.csv`, `runtime.csv`, `manifest.json`, and `figures/`.
- TDCA predict caches projected templates and avoids repeated large matrix products. A direct check on S1 0.8 s showed score differences only at floating-point noise level.
- Full TDCA 0.2-2.0 s completed with 35 subjects, 6 blocks, and 10 windows.
- Total comparison with CCA, FBCCA, TRCA, ETRCA, and TDCA is available in `results/traditional_tdca_9ch_w02_2s`.
- The project is now prepared for git source control with generated outputs and large binary artifacts ignored.

## Required Operating Rules

- Use `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` for multi-process classical algorithms.
- Run a small stress test before full sweeps. Minimum stress test:
  - 2 subjects
  - 2 windows
  - 2-3 blocks
  - same worker count style as the planned full run
  - resume path checked
  - runtime ledger checked
- Do not enable score/model artifact writes by default. They are opt-in debugging outputs.
- Prefer one shallow task directory per experiment. Keep raw caches under `runs/`, final reusable outputs under `results/`.
- After a timeout or interrupted run, check and stop orphaned Python processes before resuming.

## Next Architecture Cleanup

- Replace subject-window epoch cache for long-window sweeps with subject-level max-window cache plus slicing.
- Add a small benchmark/stress script that reports:
  - wall time
  - CPU worker count
  - BLAS thread settings
  - rows completed per minute
  - peak-ish process memory from `psutil` if available
- Add an experiment manifest helper so all runners share consistent fields.
- Add one combined plotting script that reads any set of arena-native result directories and redraws without rerunning algorithms.
- For SA-MVMD-TRCA, start with a smoke/stress harness before full evaluation because decomposition is much more expensive than TDCA.

## DNN Adapter Plan

- Treat existing DNN as an imported adapter first, not a retraining target.
- Bring a tiny sample through the VEP Arena schema:
  - load existing DNN subject/block predictions if available;
  - or run a small inference-only sample if checkpoint and preprocessing are clear;
  - output `trials.csv`, `predictions.csv`, `manifest.json`;
  - do not mix it into final figures until preprocessing and protocol are explicitly matched.
