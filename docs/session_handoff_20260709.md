# 2026-07-09 Session Handoff

## Current Environment

- Canonical Arena Python: `D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe`.
- Repair command completed: `uv sync --extra all`.
- Verified imports in `.venv`: `mne 1.12.1`, `torch 2.11.0+cu128`, `python-pptx 1.0.2`.
- `scripts/check_neuro_env.py` now treats MNE as required and PsychoPy as optional. PsychoPy remains available through the legacy conda environment `D:\ProjData\envs\erp_ssvep_lab` when online experiment work explicitly needs it.

## Result Status

Local result directories are intentionally ignored by git. The current important result families are:

| Workstream | Current result location | Status |
| --- | --- | --- |
| Benchmark decision channel capacity | `tasks/benchmark_decision_channel_capacity/results/input` | complete; `summary=45`, `trials=9450`, `predictions=378000` |
| Benchmark 9ch comparisons | `results/benchmark_9ch/` | historical/top-level results remain available, including `final_compare` and `latest_multialgo_with_arena_dnn` |
| HD-200 offline TRCA/TDCA | `tasks/ssvep_hd_200target_tdca_sample/results/offline_{trca,tdca}_grid` | complete; each has `summary=100`, `trials=1400`, `predictions=3024000` |
| Wearable wet/dry | `results/wearable_{wet,dry}_cca_fbcca_ecca_trca_etrca_w02_10` | complete; analysis report under `tasks/benchmark_signal_channel_analysis/reports/` |
| BETA | `tasks/beta_ssvep_9ch_*` and `results/beta_toolbox_*` | task summaries plus older toolbox-format results available |
| Ke 2025 Binocular AR | `tasks/ssvep_binocular_ar_trca/results/experiment{1,2,3}_trca` | complete; Exp1/2/3 summaries are 240/480/360 rows |
| Dual-Alpha | `tasks/ssvep_dual_alpha_baselines/results/official_baselines` | complete MNE-FIR rerun; `summary=50`, `trials=8750`, `predictions=350000`; old SciPy FIR result moved to legacy |
| cVEP NBRS/JFPM | `tasks/cvep_nbrs_jfpm_tsinghua_2024_baselines/results/full_occipital9_*` | complete FBCCA/TRCA and msTRCA-v2 paper-preproc runs |

## Code And Task Additions

- Added information-theory/channel modules under `vep_arena/channel/`.
- Added ECCA, FBDCCA, SSCOR-related method modules and tests.
- Added signal-analysis helpers under `vep_arena/signal/`.
- Added task-owned workflows for decision-channel analysis, HD-200, BETA, cVEP NBRS/JFPM, Binocular AR, and Dual-Alpha.
- Added `vep_arena/data/binocular.py` for binocular and Dual-Alpha dataset handling.
- Added reproduction discipline documentation: `docs/reproduction_discipline_zh.md`.
- Added root environment rule file: `ENVIRONMENT_REQUIREMENTS.txt`.

## Important Corrections

- Dual-Alpha official FBDCCA now defaults to MNE FIR. The SciPy FIR path is explicitly named `scipy-fir-legacy` and cannot write into an `official_*` output directory.
- Arena `.venv` is now the preferred execution environment; conda is not the default path for offline Arena tasks.
- Generated result folders stay ignored. The ignore rules now also cover task analysis `figures/` and `tables/` folders to avoid accidentally committing generated PNG/CSV outputs.

## Suggested Next Session

1. Start by running `.venv\Scripts\python.exe scripts\check_neuro_env.py`.
2. Use `.venv\Scripts\python.exe` for official offline tasks; avoid bare `python`.
3. If reviewing the commit, focus on script/docs/test changes and verify that result CSV/PNG files are not staged.
4. For further result work, prefer task-owned outputs and update each task README/NOTES when protocol details change.
