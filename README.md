# VEP Arena

Reproducible evaluation workspace for visual evoked potential (VEP) brain–computer interfaces.
Covers SSVEP and cVEP paradigms across multiple public and private datasets, with both
traditional spatial-filtering methods and DNN baselines.

## Library — `vep_arena/`

```text
vep_arena/
├── data/          Dataset loaders (Benchmark, BETA, HD-200, Dual-Frequency,
│                  Broadband White-Noise cVEP, MFSC-160, EMBC/JBHI binocular, …)
├── methods/       Spatial-filter receivers
│                    CCA · eCCA · FBDCCA · SSCOR
│                    TRCA · TDCA · Multi-Stimulus
│                    bPRCA · bTRCA · FusionCA (binocular)
│                    SA-MVMD-TRCA · MVMD
├── nn/            DNN models — EEGNet, SSVEPformer, TRCANet
├── signal/        Filters (comb, notch), SNR, PLV, spectrum utilities
├── channel/       Discrete channel capacity, confusion-matrix tools
├── plots/         Reusable plotting helpers
├── neuroviz/      MNE bridge and topographic views
└── metrics.py     Accuracy, ITR, per-subject scoring
```

## Tasks — `tasks/`

Each task is a self-contained evaluation directory:

```text
tasks/{paradigm}_{dataset}_{scope}/
  run.py              main entry point
  README.md / NOTES   context and results
  results/            predictions, CSVs, figures (gitignored)
```

Current baseline tasks include evaluations on Benchmark-40, BETA-40,
HD-200, Dual-Frequency (Liang 2020 / Sun 2024), MFSC-160,
Broadband White-Noise cVEP, EMBC-9 / JBHI-16 / JBHI-35 binocular datasets,
and more.

## Quick Start

```bash
# install (requires uv)
uv sync --extra all

# run a baseline evaluation
uv run python tasks/ssvep_jbhi_35target_baselines/run.py

# run tests
uv run pytest tests/
```

## Data

Datasets live outside this repository. Copy `configs/datasets/local_paths.example.json`
to `configs/datasets/local_paths.json` and fill in your local paths.

Public datasets used:

| Dataset | Targets | Source |
|---------|---------|--------|
| Benchmark (Wang 2017) | 40 | [MOABB / Tsinghua](http://bci.med.tsinghua.edu.cn/) |
| BETA (Liu 2020) | 40 | SSVEP-Analysis-Toolbox |
| HD-200 (Chen 2021) | 200 | Tsinghua |
| MFSC-160 (Chen 2021) | 160 | Tsinghua |
| Dual-Freq Phase (Liang 2020) | 12 | Public |
| Efficient Dual-Freq (Sun 2024) | 40 | Public |
| Broadband White-Noise cVEP | 20 | [Zenodo 8300517](https://zenodo.org/records/8300517) |

## Environment

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/) for dependency management.
GPU support: PyTorch with CUDA (optional, for DNN models).

## License

See [LICENSE](LICENSE).
