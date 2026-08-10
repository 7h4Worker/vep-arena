# EMBC 9-target private SSVEP baselines

This task evaluates the eight-subject published EMBC binocular SSVEP dataset without copying data into Git.

- Data path key: `ssvep_embc_9target_private` in `configs/datasets/local_paths.json`.
- Tensor: 9 targets x 20 blocks x 9 occipital channels x 4 s.
- Stored epochs already begin 130 ms after stimulus onset and already contain the documented EEGLAB preprocessing.
- Paper analysis resamples 1000 Hz epochs to 250 Hz. The task does not repeat the 48-52 Hz notch or 5-100 Hz FIR.
- The paper says leave-one-out without naming the held-out unit. Arena uses leave-one-block-out and records that assumption in the manifest.
- The default smoke uses anonymous `S04,S06`, selected by a 2 s diagnostic as quality-normal examples (87.8% and 78.3%).
- The shared runner also supports `BPRCA`, `EBPRCA`, `FUSIONCA`, and `EFUSIONCA` with the 8.5 and 9.5 Hz interface units. These are Arena extensions; the paper baseline remains TRCA.
- The recovered paper-result code verifies direct 4x decimation followed by three dynamic Chebyshev receiver bands. Arena applies the three bands independently; the historical MATLAB training-only cascade is retained only in the legacy audit.
- The final paper mean excludes anonymous `S02`. Arena full summaries retain all eight subjects and report the paper-seven subset separately when comparing with the recovered result MAT files.

Run:

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_embc_9target_baselines\run.py --workers 2 --resume
```

Generated results, CSV files, figures, and manifests remain under `results/` and are ignored by Git.
