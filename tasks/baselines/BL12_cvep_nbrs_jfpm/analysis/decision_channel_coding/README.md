# cVEP Decision-Channel Coding Analysis

Branch analysis for the cVEP NBRS/JFPM task.

This directory does not rerun classifiers. It reads the current formal cVEP
baseline outputs and the released codebooks, then generates a first
decision-channel / encoding-space report.

## Inputs

- `../../results/comparison_occipital9_fbcca_trca_mstrca_v2_20260706/`
- `../../results/full_occipital9_fbcca_trca_w04_40_20260706/`
- `../../results/full_occipital9_mstrca_v2_paper_preproc_w04_40_20260706/`
- `D:/ProjData/datasets/cvep_nbrs_jfpm_tsinghua_2024/metadata/supplementary_extracted/`

## Command

```powershell
.venv\Scripts\python.exe tasks\cvep_nbrs_jfpm_tsinghua_2024_baselines\analysis\decision_channel_coding\plot_coding_channel.py
```

## Outputs

- `figures/`: report figures.
- `tables/`: derived summary, codebook, and capacity tables.
- `report_zh.md`: Chinese report draft.
