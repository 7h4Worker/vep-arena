# BETA SSVEP 9ch Baselines

Purpose: compare standard SSVEP baselines on the BETA dataset using the
SSVEP-Analysis-Toolbox dataset/preprocessing path exposed through Arena.

## Task Notes

- `NOTES_20260707.md`: current scope, output location, and the status of
  historical fallback summaries.

Protocol:

- Dataset: BETA via `vep_arena.data.toolbox_adapter`
- Channels: `occipital_9ch`
- Windows: `0.2, 0.4, ..., 2.0` seconds
- Methods: `CCA`, `FBCCA`, `TRCA`, `ETRCA`, `TDCA`
- Evaluation: subject-specific leave-one-block-out
- Data root: `D:\ProjData\datasets\ssvep_beta`

Run the full task:

```powershell
.venv\Scripts\python.exe tasks\beta_ssvep_9ch_baselines\run.py
```

Rebuild the comparison figure from existing summaries:

```powershell
.venv\Scripts\python.exe tasks\beta_ssvep_9ch_baselines\plot_acc_itr.py
```

Current task outputs are written under:

```text
tasks/beta_ssvep_9ch_baselines/results/
```

The historical source summaries currently live under top-level `results/` and
are used as a fallback so the task can be plotted without rerunning the full
experiment.
