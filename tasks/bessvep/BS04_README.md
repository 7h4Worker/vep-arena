# EMBC/JBHI binocular codebook study

This cross-task study reads the completed anonymous prediction bundles from the EMBC 9-target, JBHI 16-target, and JBHI 35-target baseline tasks. It does not rerun classifiers or read participant identity metadata.

The study treats each target as an ordered binocular codeword `(left_unit, right_unit)` and reports:

- empirical decision-channel `C1`, `MI_uniform`, and `C_BA`;
- normalized codebook utilization;
- EEG-window and practical capacity-time rates;
- left/right marginal information and cross-eye leakage;
- eye-swap, void-structure, one-eye-preserved, and both-unit error topology;
- receiver gains for eTRCA, ebPRCA, and eFusionCA;
- subject-level distributions and code-family confusion.

The signal-level extension reads the anonymous EEG tensors and reports frequency-unit spectral topographies, left/right contrasts, feature-level mutual information, and target-signature similarity versus receiver confusion. The TDCA smoke uses only class-specific EEG temporal subspaces estimated inside each training fold; it does not assume an external sinusoidal or phase reference.

The JBHI35 receiver audit separates amplitude/codebook agreement from cross-block phase locking and compares the standard independent-subband eTRCA implementation with the historical MATLAB training-side cascade:

```powershell
.\.venv\Scripts\python.exe tasks\ssvep_embc_jbhi_binocular_codebook_analysis\audit_jbhi35_subject_quality.py
.\.venv\Scripts\python.exe tasks\ssvep_embc_jbhi_binocular_codebook_analysis\audit_etrca_scoring.py
.\.venv\Scripts\python.exe tasks\ssvep_embc_jbhi_binocular_codebook_analysis\audit_legacy_backup.py
```

The legacy audit reads `ssvep_embc_9target_legacy_private` and `ssvep_embc_jbhi_legacy_private` from the ignored local-path configuration. It matches source arrays in memory and persists anonymous Arena IDs only. The EMBC branch also separates the current single-band run, the recovered three-band definition, and the historical training-only filter cascade.

Run with the canonical environment:

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_embc_jbhi_binocular_codebook_analysis\analyze.py
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_embc_jbhi_binocular_codebook_analysis\analyze_signal.py
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_embc_jbhi_binocular_codebook_analysis\run_tdca.py
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_embc_jbhi_binocular_codebook_analysis\run_tdca.py --full --workers 4 --resume
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_embc_jbhi_binocular_codebook_analysis\analyze_tdca_full.py
```

Generated CSV, JSON, and figures remain under `results/` and are ignored by Git.
