# JBHI 35-target private SSVEP baselines

This is an Arena diagnostic task for the unpublished six-subject 35-target JBHI extension.

- Data path key: `ssvep_jbhi_35target_private`.
- Canonical source tensor: `data [9, 2000, 35, 6]` at 1000 Hz, stored outside Git as anonymous `S01.mat`-`S06.mat` files.
- The shared loader validates the canonical shape and scalar counts, converts the MATLAB target axis 1-35 to Python labels 0-34, and exposes 35 targets x 6 blocks x 9 channels x 2 s.
- Source filenames are private identity-bearing metadata; the loader and outputs expose only anonymous IDs.
- The manual EEGLAB preprocessing profile is operator-confirmed but lacks per-file history, so results must not be labeled an official paper reproduction.
- Arena uses six-fold leave-one-block-out and the same explicit diagnostic receiver filter bank as the JBHI 16-target smoke.
- `target_mapping_35.csv` is validated in place as the authoritative 35-class left/right frequency codebook; it is not copied into repository outputs.
- The previous signal-correlation order heuristic was invalidated after authoritative labels arrived: single-trial Hungarian matching can overfit noise and is not evidence of block permutation.
- A six-subject 2 s receiver audit selected anonymous `S04,S06` as quality-normal smoke examples. Five-band ETRCA reached 87.1% and 63.8%, while `S01-S03` were near chance and `S05` was weaker.
- Five bands improved the six-subject mean over one band for both TRCA and ETRCA, so the near-chance initial smoke was caused by unrepresentative subject selection rather than label order or duplicate filtering.
- The shared runner also supports `BPRCA`, `EBPRCA`, `FUSIONCA`, and `EFUSIONCA` using the five frequency units recovered from the authoritative 35-target codebook. These remain diagnostic extensions because the 35-target dataset is unpublished.
- This remains an unpublished diagnostic extension rather than a paper reproduction. The verified full 0.2-2.0 s grid is available under the ignored task results.

The bounded handoff smoke compares signed and absolute eTRCA correlation without treating historical `absScore` MAT files as reproduced baselines:

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_jbhi_35target_baselines\run_scoring_smoke.py
```

Run:

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_jbhi_35target_baselines\run.py --workers 2 --resume
```

## Final five-subject historical set

The final historical set is separate from the six-subject canonical diagnostic
dataset above. It contains only `S01,S03,S04,S05,S06`, is resolved through the
ignored `configs/datasets/local_paths.json`, and is exposed through the shared
`jbhi35_historical5` adapter. The adapter checks all 51 package SHA entries,
the private anonymous manifest, package data mapping, `9 x 2000 x 35 x 6`
float64 schema, nine-channel order, 1000 Hz sampling, MATLAB labels 1-35, and
the conversion to Arena labels 0-34 before a runner receives data.

The 2 s eTRCA task is an aggregate-reference comparison, not a MATLAB
equivalence claim. It compares each anonymous subject's total signed correct
count with the package-provided MATLAB count. The package does not provide
per-trial predictions or score matrices, so an aggregate match cannot establish
per-trial or numerical equivalence. Historical `absScore` MAT files remain
reference-only and are not treated as reproduced baselines.

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_jbhi_35target_baselines\run_five_subject_reproduction.py --output <new-local-result-directory>
```

The five-subject TDCA runner is an Arena extension, not a MATLAB or paper TDCA
reproduction. It uses the shared historical adapter and evaluates leakage-safe training-fold EEG references and
zero-phase spectral codebook references over the complete 0.2-2.0 s grid. The
fixed whole-JBHI settings are five filter bands, `n_delay=2`, and
`n_components=4`; they are not tuned on the final five-subject run.

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_jbhi_35target_baselines\run_five_subject_tdca.py --workers 4 --output <new-local-result-directory>
```

The periodic-receiver runner covers standard and ensemble bPRCA/FusionCA over
the same complete window grid through the shared historical adapter. `FusionCA` is an explicit algorithm module that
adds a full-window spatial stream to the frequency-period bPRCA streams. These
results are also Arena extensions rather than MATLAB historical reproductions.

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_jbhi_35target_baselines\run_five_subject_periodic_receivers.py --workers 4 --output <new-local-result-directory>
```

## Historical output status

- Existing ignored result folders are preserved as historical execution artifacts; this adapter integration does not rewrite, rerun, or relabel their files.
- All final-five runners now require an explicit `--output` directory. They refuse a nonempty directory by default; TDCA and periodic runners require explicit `--resume` before writing to one.
- The stored TDCA and periodic-receiver manifests bind the current task configuration and remain Arena extension evidence, but they predate the shared-adapter hash and must not be described as newly regenerated by it.
- The stored 2 s eTRCA comparison has a configuration-hash mismatch with the current config and only tests aggregate correct-count agreement. It remains an unreviewed diagnostic/reference artifact, not an accepted MATLAB-reproduction baseline.
- Any future run, only when separately authorized, records the shared adapter hash together with package and private-manifest hashes. It must still describe the eTRCA comparison as aggregate-count validation unless per-trial package references become available.

The cross-dataset four-family summary is built by
`tasks/ssvep_embc_jbhi_receiver_matrix/build_results.py`. It keeps the canonical
six-subject JBHI35 diagnostic cohort separate from this reviewed historical
five-subject cohort.
