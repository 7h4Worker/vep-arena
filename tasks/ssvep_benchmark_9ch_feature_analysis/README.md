# SSVEP Benchmark 9ch Feature Analysis

Purpose: start a task-owned feature-analysis line for the Tsinghua Benchmark
SSVEP 9-channel setting. This task is exploratory and complements classifier
benchmarks by describing response structure, frequency locking, time-frequency
energy, channel profiles, and simple links between signal features and decoding
performance.

The first probe intentionally uses a small subset of subjects and targets. It
is designed to establish useful figures and artifact formats before scaling to
all subjects or additional datasets.

## Task Notes

- `NOTES_20260707.md`: current feature-probe output, archived smoke result,
  and status of this exploratory task.

## Default Probe

```powershell
D:\ProjData\proj_python\vep_arena\.venv\Scripts\python.exe tasks\ssvep_benchmark_9ch_feature_analysis\run.py
```

Defaults:

- dataset: Tsinghua Benchmark SSVEP
- subjects: `1,2,3`
- targets: `1-8,17-20`
- channels: Benchmark occipital 9ch preset
- epoch window: `2.0 s`
- crop start: cue `0.5 s` plus visual latency `0.14 s`
- Welch PSD range: `4-60 Hz`
- SNR harmonics: `1,2,3`
- SNR sideband: exclude `+-0.25 Hz`, use local noise from `0.75-1.75 Hz`
- CWT: Morlet wavelet, `4-60 Hz`
- classifier relation: FBCCA leave-one-block within the selected targets

Outputs go under:

```text
tasks/ssvep_benchmark_9ch_feature_analysis/results/feature_probe/
```

## Signal Definitions

Let `x_{s,t,b,c}[n]` be the latency-corrected epoch for subject `s`, target
`t`, block `b`, and channel `c`, sampled at `Fs = 250 Hz`. Epochs are demeaned
within trial before spectral summaries.

### Welch PSD

Power spectral density is estimated with `scipy.signal.welch`:

```text
Pxx(f) = Welch(x[n], Fs, nperseg=min(N, 512))
```

The probe averages PSD over blocks, subjects, targets, or channels depending on
the figure. Figures are restricted to `4-60 Hz` because the selected SSVEP
targets are `8-15.8 Hz` and the first three harmonics mostly live in this band.

### Harmonic Power

For target frequency `f0` and harmonic `h`, harmonic power is the largest PSD
bin near `h * f0`:

```text
H_h = max Pxx(f), where |f - h*f0| <= peak_band
```

The current probe uses `peak_band = 0.25 Hz`.

### Harmonic SNR

For each harmonic, local spectral SNR is:

```text
SNR_h(dB) = 10 * log10( peak_power(h*f0) / mean_noise_power(h*f0) )
```

where:

```text
peak_power(h*f0) = max Pxx(f), |f - h*f0| <= 0.25 Hz
mean_noise_power(h*f0) = mean Pxx(f), 0.75 <= |f - h*f0| <= 1.75 Hz
```

The final target/channel SNR is the arithmetic mean over valid harmonics
`h = 1, 2, 3`. Harmonics beyond Nyquist or outside the plotted PSD support are
ignored.

### Morlet CWT

The time-frequency view uses `scipy.signal.cwt` with a Morlet wavelet:

```text
W(f, tau) = CWT(x[n], morlet2, width(f))
width(f) = w * Fs / (2*pi*f)
```

The probe uses `w = 6` and frequency samples from `4` to `60 Hz`. The displayed
quantity is `20 * log10(|W| + eps)` averaged over selected trials/channels.
This is a diagnostic view, not a classifier feature.

### Classifier Relation

To connect signal features to decoding without making this a classifier task,
the probe runs FBCCA only on the selected target subset:

```text
train blocks = all blocks except held-out block
test block = held-out block
```

It writes per-subject/target/block correctness and correlates target-level mean
SNR with mean block accuracy. This is a first sanity check for whether harmonic
feature strength tracks decoding difficulty.

## MNE Topomap Notes

The runner tries to import MNE at runtime. If MNE is installed, it exports a
few evoked topomap snapshots for the 9 occipital channels. If MNE is not
available or topomap interpolation fails, the runner records the error in the
manifest and still writes a 3x3 occipital channel grid heatmap.

MNE topomap output is treated as a QA view. The 9-channel occipital montage is
useful for coarse spatial structure but is not a full-head spatial analysis.

## Current Artifact Contract

The task writes:

- `feature_manifest.json`
- `snr_by_subject_target_channel.csv`
- `snr_by_subject_target.csv`
- `harmonic_power.csv`
- `fbcca_selected_predictions.csv`
- `feature_vs_accuracy.csv`
- `figures/target_time_evoked_grid.png`
- `figures/target_psd_grid.png`
- `figures/snr_target_heatmap.png`
- `figures/snr_channel_profile.png`
- `figures/cwt_target01_oz.png`
- `figures/feature_vs_accuracy.png`
- `figures/occipital_grid_snr.png`
- optional `figures/mne_topomap_*.png`
