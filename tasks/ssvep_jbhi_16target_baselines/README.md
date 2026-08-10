# JBHI 16-target private SSVEP baselines

This task evaluates the final 13-subject JBHI binocular-encoded SSVEP dataset with anonymous Arena IDs.

- Data path key: `ssvep_jbhi_16target_private`.
- Native trial order is verified as six repetitions of labels 1-16; Arena reconstructs target x block explicitly before CV.
- Protocol: subject-specific six-fold leave-one-block-out, 16 test trials per fold.
- Stored epochs already include the 130 ms visual latency and documented EEGLAB notch/bandpass preprocessing.
- The receiver applies the manuscript's fixed-order Chebyshev `[8b-2, 90]` Hz filter-bank rule. The manuscript does not state `B` or ripple; the smoke records its explicit `B=5`, `0.5 dB` Arena assumption.
- Default windows include the paper's 0.4 s peak-ITR window and 2.0 s accuracy endpoint.
- The primary ITR curve averages subject-level ITR values, consistent with the manuscript pairing of 71.39% mean accuracy and 138.50 bits/min at 0.4 s. `summary.csv` also records `itr_from_group_accuracy_bpm` so the nonlinear aggregation alternative remains auditable.
- The default smoke uses anonymous `S04,S06`, whose 0.4 s five-band diagnostic is representative of the manuscript eTRCA scale.
- Supported trained methods are `TRCA`, `ETRCA`, `BPRCA`, `EBPRCA`, `FUSIONCA`, and `EFUSIONCA`. bPRCA uses all three interface frequency units (11, 12, and 13 Hz); FusionCA adds the full-epoch TRCA stream.

Run:

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_jbhi_16target_baselines\run.py --workers 2 --resume
```
