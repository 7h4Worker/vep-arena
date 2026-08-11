# Validation Notes

This task is being promoted from a small sample run toward a paper reproduction
for the high-density 200-target SSVEP dataset.

## Source Anchors

Primary article:

- Ming et al. 2026, *A High-Speed Visual BCI Based on Hybrid
  Frequency-Phase-Space Encoding and High-Density EEG Decoding*.
- DOI: <https://doi.org/10.34133/cbsystems.0555>
- Data/code package: <https://figshare.com/s/c962d29d7752e27176be>

Visible paper checkpoints collected from the article page/PDF snippets:

These values were re-checked against the arXiv PDF text on 2026-06-26.

| checkpoint | paper value | use |
|---|---:|---|
| Online average actual ITR | `472.72 +/- 15.06 bpm` | Full online reproduction target |
| Online highest subject | `S1` | Highest-ITR sanity check |
| Online highest setting | `160 targets`, right/down/left/up, `0.25 s` | Highest-ITR runner setting |
| Online highest accuracy | `96.88%` | Highest-ITR accuracy target |
| Online highest actual ITR | `551.42 bpm` | Highest-ITR ITR target |
| Offline personalized 66/256 peak | `484.76 +/- 12.68 bpm` at `0.2 s` | Offline personalized target |
| Offline dynamic-window comparison | `416.22 +/- 14.18` vs `456.72 +/- 16.87 bpm` | Dynamic-window target |

## Local Completed Checks

### Offline Fixed-Window Sample

Setting:

- Offline files
- 200 targets
- 66 channels
- 18-block leave-one-block-out
- 500 ms
- 5 filter banks
- TDCA latency = 35 samples
- TDCA lag = 5

Output:

```text
tasks/ssvep_hd_200target_tdca_sample/results/summary.csv
```

| subject | accuracy | ITR bpm | status |
|---|---:|---:|---|
| S1 | `0.959722222222222` | `425.562334609305` | complete |
| S2 | `0.915555555555556` | `394.880196864086` | complete |
| S3 | `0.834444444444444` | `343.928068801533` | complete |

The 1000 ms request is explicitly unavailable for the released offline files:
each offline trial has 185 samples, while TDCA needs `250 + 35 + 5 = 290`
samples for a 1000 ms window.

### Online Highest-ITR Sample

Setting:

- Online training `S1.mat` -> online testing `S1.mat`
- 160 targets
- 66 channels
- 18 training blocks
- 5 testing blocks
- 250 ms
- `time_samples = 63`, because 250 ms at 250 Hz gives 62.5 samples and the
  online file length matches `63 + latency35 + lag5 = 103`.

Output:

```text
tasks/ssvep_hd_200target_tdca_sample/results/online_highest_S1_160target_66ch_w250.csv
```

| metric | local | paper | delta |
|---|---:|---:|---:|
| correct trials | `773 / 800` | `775 / 800` | `-2` |
| accuracy | `96.625%` | `96.875%` | `-0.25 pp` |
| actual ITR | `548.9804565689753 bpm` | `551.42 bpm` | `-2.44 bpm` |

This is close enough to validate the setting and target mapping. The remaining
2-trial difference is most likely caused by small preprocessing differences
between SciPy's filter-bank reproduction and MATLAB Signal Processing Toolbox
`cheby1/filtfilt`.

## Implementation Status

- `run.py` is the portable Python TDCA implementation. Its early S1 500 ms
  folds match the validated MATLAB-cache path but it is still too slow for full
  reproduction.
- `run_matlab_cache.m` is the current validated backend for fixed-window
  offline runs. It uses the original TDCA test function, precomputed filter
  caches, `single` precision, and vectorized covariance accumulation to avoid
  the original MATLAB `repmat` memory spike.
- `run_online_highest.py` is the current Python runner for the online highest
  setting.

## Validation Before Full Reproduction

1. Confirm all target-set mappings:
   - 40: up only
   - 80: down/up
   - 120: right/left/up
   - 160: right/down/left/up
   - 200: right/down/left/up/center
2. Confirm ITR denominator is always `stimulus_duration + 0.5 s gaze shift`.
3. Confirm sample rounding:
   - 100/200/300/400/500 ms are exact at 250 Hz.
   - 250 ms online uses 63 samples.
4. Check whether full MATLAB toolbox filtering changes the 2-trial online S1
   mismatch if Signal Processing Toolbox becomes available.
5. Treat Python results as official only after full-row equivalence is verified
   against the MATLAB-cache backend for at least one subject/config matrix.
