# BETA SSVEP 9ch Official Grid

Purpose: compare Arena's BETA baselines on the common toolbox documentation
grid: `0.25, 0.5, 0.75, 1.0` seconds.

## Task Notes

- `NOTES_20260707.md`: current role and separation from the broader BETA
  baseline task.

This is an analysis/plot task. It reads existing runner summaries and writes a
compact comparison table and figure under this task's `results/` directory.

```powershell
.venv\Scripts\python.exe tasks\beta_ssvep_9ch_official_grid\plot_acc_itr.py
```
