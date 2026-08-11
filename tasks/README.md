# Tasks

Tasks are concrete research or validation questions. A task owns its runnable
entry points, notes, and generated results.

Current task root layout:

```text
tasks/
  ╔══════════════════════════════════════════════════════════════════════╗
  ║  DECISION CHANNEL (DISCRETE DMC / BLAHUT-ARIMOTO)                  ║
  ╚══════════════════════════════════════════════════════════════════════╝
  benchmark_decision_channel_capacity/         Benchmark 40-target DMC
    └── results/extended/multichannel/         (multichannel sweep lives here)
  benchmark_multichannel_decision_channel_capacity/  Scripts-only; outputs → above
  ssvep_hd_200target_tdca_sample/              HD200 DMC (14sub × 5tgt × 4ch × 5win)
  ssvep_jbhi_decision_channel/                 Cross-paradigm DMC envelope + comparison

  ╔══════════════════════════════════════════════════════════════════════╗
  ║  CONTINUOUS CHANNEL (MIMO / GAUSSIAN MI) — NEW                     ║
  ╚══════════════════════════════════════════════════════════════════════╝
  continuous_mimo_channel_capacity/            Signal-level MIMO MI analysis
  benchmark_signal_channel_analysis/           Early exploration (frozen)

  ╔══════════════════════════════════════════════════════════════════════╗
  ║  PARADIGM BASELINES & VALIDATION                                    ║
  ╚══════════════════════════════════════════════════════════════════════╝
  beta_ssvep_9ch_baselines/                    Benchmark 9ch baseline grid
  beta_ssvep_9ch_official_grid/                Benchmark 9ch official reproduction
  ssvep_benchmark_9ch_feature_analysis/        Benchmark signal feature probes
  ssvep_binocular_ar_trca/                     Binocular AR TRCA (3 experiments)
  ssvep_binocular_dataset_smoke/               Binocular dataset smoke test
  ssvep_dual_alpha_baselines/                  Dual-alpha ETRCA/FBDCCA baselines
  ssvep_dual_frequency_phase_liang2020/        Dual-freq phase (Liang 2020)
  ssvep_efficient_dual_frequency_sun2024/      Efficient dual-freq (Sun 2024)
  ssvep_embc_9target_baselines/                EMBC 9-target baselines
  ssvep_jbhi_16target_baselines/               JBHI 16-target baselines
  ssvep_jbhi_35target_baselines/               JBHI 35-target baselines
  cvep_nbrs_jfpm_tsinghua_2024_baselines/      cVEP JFPM coding baselines
  cvep_broadband_white_noise_tdca/             Broadband WN cVEP TDCA
  ssvep_160target_mfsc_tdca/                   160-target MFSC TDCA

  ╔══════════════════════════════════════════════════════════════════════╗
  ║  CROSS-CUTTING ANALYSES                                             ║
  ╚══════════════════════════════════════════════════════════════════════╝
  crossparadigm_information_efficiency/        Info-efficiency meta-analysis
  information_accumulation_rate/               dI/dT accumulation rate
  ssvep_embc_jbhi_binocular_codebook_analysis/ Codebook structure analysis
  ssvep_embc_jbhi_fft_features/                FFT feature comparison
  ssvep_embc_jbhi_receiver_matrix/             Receiver capability matrix

  _legacy/                                     Archived historical tasks
```

The package under `vep_arena/` stays reusable:

```text
vep_arena/data/       dataset adapters and metadata
vep_arena/methods/    algorithm modules
vep_arena/plots/      shared plotting primitives
vep_arena/neuroviz/   MNE-oriented views
```

Each task should be small and readable:

```text
tasks/<dataset>_<scope>_<purpose>/
  README.md
  run.py
  plot_acc_itr.py
  results/
```

Naming convention:

- `run.py`: runs the task's main experiment.
- `plot_<view>.py`: builds a task-specific figure from existing outputs.
- `results/`: local outputs owned by the task. This directory is ignored by
  default to prevent accidental commits of predictions, caches, and large logs.

Large datasets and external toolboxes stay outside the repository.

## Cleanup convention

When a smoke run, partial run, or superseded result is no longer the current
analysis source, keep it under the owning task's `results/_legacy/` directory
instead of deleting it. Add a date-stamped note such as `NOTES_YYYYMMDD.md` in
the task root explaining:

- which result directories are current,
- what issue was validated or fixed,
- which outputs were moved to `_legacy`,
- which outputs should feed the next report or analysis step.

When a whole task folder is historical rather than a current workstream, move
it under `tasks/_legacy/YYYYMMDD_<reason>/` and add a short README in that
legacy folder. Do not move current report/task roots merely because their
`results/` are ignored by git.
