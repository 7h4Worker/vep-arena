# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy import signal

from vep_arena.config import (
    BENCHMARK_CHANNELS_9,
    BENCHMARK_FREQS,
    BENCHMARK_PHASES_PI,
    BenchmarkSpec,
)


def subject_file(data_root: Path, subject: int) -> Path:
    for name in (f"S{subject}.mat", f"s{subject}.mat", f"S{subject:02d}.mat", f"s{subject:02d}.mat"):
        path = data_root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Subject {subject} not found under {data_root}")


def load_subject_raw(data_root: Path, subject: int) -> np.ndarray:
    raw = loadmat(subject_file(data_root, subject))["data"]
    return np.asarray(raw, dtype=np.float64)


def load_subject_trials(
    data_root: Path,
    subject: int,
    window: float,
    channels: tuple[int, ...] = BENCHMARK_CHANNELS_9,
    spec: BenchmarkSpec | None = None,
) -> np.ndarray:
    """Return trials as classes x blocks x channels x samples."""

    spec = spec or BenchmarkSpec()
    raw = load_subject_raw(data_root, subject)
    channel_idx = np.asarray(channels, dtype=np.int64) - 1
    data = raw[channel_idx, spec.sample_slice(window), :, :]
    return np.transpose(data, (2, 3, 0, 1)).copy()


def notch_50hz(x: np.ndarray, fs: int) -> np.ndarray:
    b, a = signal.iircomb(50, 35, ftype="notch", fs=fs)
    return signal.filtfilt(b, a, x, axis=-1, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1))


def benchmark_filterbank(x: np.ndarray, fs: int, n_fbs: int = 5) -> np.ndarray:
    """Benchmark-style filter bank, returning subbands x channels x samples."""

    out = np.zeros((n_fbs, x.shape[0], x.shape[1]), dtype=np.float64)
    for k in range(1, n_fbs + 1):
        wp = [(8 * k) / (fs / 2), 90 / (fs / 2)]
        ws = [(8 * k - 2) / (fs / 2), 100 / (fs / 2)]
        gstop = 40
        while gstop >= 20:
            try:
                order, wn = signal.cheb1ord(wp, ws, 3, gstop)
                b, a = signal.cheby1(order, 0.5, wn, btype="bandpass")
                out[k - 1] = signal.filtfilt(
                    b,
                    a,
                    x,
                    axis=-1,
                    padtype="odd",
                    padlen=3 * (max(len(b), len(a)) - 1),
                )
                break
            except ValueError:
                gstop -= 1
        if gstop < 20:
            raise ValueError(f"Filterbank failed for subband {k} and signal length {x.shape[1]}")
    return out


def load_subject_filterbank(
    data_root: Path,
    subject: int,
    window: float,
    n_fbs: int,
    channels: tuple[int, ...] = BENCHMARK_CHANNELS_9,
    spec: BenchmarkSpec | None = None,
    extra_samples: int = 0,
) -> np.ndarray:
    """Return classes x blocks x subbands x channels x samples.

    This follows SSVEP-Analysis-Toolbox's Benchmark convention: filter the
    segment from cue offset through latency + target window, then crop away the
    latency samples after filtering. Filtering only the final target window is
    more vulnerable to edge artifacts, especially for short windows.

    `extra_samples` can be used by TDCA training to retain the samples needed
    for temporally delayed copies. The output then has samples + extra_samples
    time points.
    """

    spec = spec or BenchmarkSpec()
    raw = load_subject_raw(data_root, subject)
    channel_idx = np.asarray(channels, dtype=np.int64) - 1
    latency = round(spec.latency_seconds * spec.sampling_rate)
    samples = spec.sample_length(window)
    start = round(spec.cue_seconds * spec.sampling_rate)
    stop = start + latency + samples + extra_samples
    out = np.zeros((spec.classes, spec.blocks, n_fbs, len(channels), samples + extra_samples), dtype=np.float64)
    for cls in range(spec.classes):
        for block in range(spec.blocks):
            trial = raw[channel_idx, start:stop, cls, block]
            trial = notch_50hz(trial, spec.sampling_rate)
            filtered = benchmark_filterbank(trial, spec.sampling_rate, n_fbs)
            out[cls, block] = filtered[:, :, latency : latency + samples + extra_samples]
    return out


def load_subject_toolbox_epochs(
    data_root: Path,
    subject: int,
    window: float,
    n_fbs: int,
    channels: tuple[int, ...] = BENCHMARK_CHANNELS_9,
    spec: BenchmarkSpec | None = None,
) -> np.ndarray:
    """Return classes x blocks x subbands x channels x samples.

    This mirrors SSVEP-Analysis-Toolbox's Benchmark data path:

    1. start at stimulus onset, after the 0.5 s cue;
    2. retain latency + target-window samples;
    3. apply 50 Hz notch and the registered filter bank to that longer segment;
    4. crop away the visual latency.

    Unlike `load_subject_filterbank`, this is intentionally named as a
    toolbox-style epoch loader for conventional reference/template methods.
    """

    return load_subject_filterbank(
        data_root=data_root,
        subject=subject,
        window=window,
        n_fbs=n_fbs,
        channels=channels,
        spec=spec,
        extra_samples=0,
    )


def reference_signals(
    window: float,
    harmonics: int,
    spec: BenchmarkSpec | None = None,
    frequencies: tuple[float, ...] | list[float] | None = None,
    phases_pi: tuple[float, ...] | list[float] | None = None,
) -> list[np.ndarray]:
    spec = spec or BenchmarkSpec()
    frequencies = frequencies or BENCHMARK_FREQS
    phases_pi = phases_pi or BENCHMARK_PHASES_PI
    samples = spec.sample_length(window)
    t = np.linspace(0, (samples - 1) / spec.sampling_rate, samples)[None, :]
    refs: list[np.ndarray] = []
    for freq, phase_pi in zip(frequencies, phases_pi):
        phase = phase_pi * np.pi
        rows = []
        for h in range(1, harmonics + 1):
            rows.append(np.sin(2 * np.pi * h * freq * t + h * phase))
            rows.append(np.cos(2 * np.pi * h * freq * t + h * phase))
        refs.append(np.concatenate(rows, axis=0))
    return refs
