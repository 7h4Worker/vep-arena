# EMBC/JBHI receiver matrix

This task assembles already completed anonymous evaluations into one four-family
receiver view:

- TRCA family: `ETRCA`
- TDCA family: `SPECTRAL_TDCA`
- bPRCA family: `EBPRCA`
- FusionCA family: `EFUSIONCA`

The canonical 9-, 16-, and 35-target results are shown together. The reviewed
JBHI35 historical five-subject set is reported separately and is never pooled
with the six-subject canonical diagnostic set. Generated CSV, JSON, and figures
live below `results/` and are not versioned.

Run with the repository environment:

```powershell
.venv\Scripts\python.exe tasks\ssvep_embc_jbhi_receiver_matrix\build_results.py
```
