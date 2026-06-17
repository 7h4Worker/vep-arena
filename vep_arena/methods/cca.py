# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.linalg import svd

from vep_arena.config import BENCHMARK_FREQS, BenchmarkSpec


def sine_references(window: float, harmonics: int = 5, spec: BenchmarkSpec | None = None) -> list[np.ndarray]:
    """Frequency-only CCA references used by the FBCCA reference code.

    Returns one matrix per class with shape 2*harmonics x samples.
    """

    spec = spec or BenchmarkSpec()
    samples = spec.sample_length(window)
    t = (np.arange(1, samples + 1, dtype=np.float64) / spec.sampling_rate)[None, :]
    refs: list[np.ndarray] = []
    for freq in BENCHMARK_FREQS:
        rows = []
        for harmonic in range(1, harmonics + 1):
            rows.append(np.sin(2 * np.pi * harmonic * freq * t))
            rows.append(np.cos(2 * np.pi * harmonic * freq * t))
        refs.append(np.concatenate(rows, axis=0))
    return refs


def fbcca_filterbank(eeg: np.ndarray, fs: int, idx_fb: int) -> np.ndarray:
    """Nakanishi/Chen FBCCA filter bank.

    `eeg` shape can be channels x samples or trials x channels x samples.
    """

    passband = np.asarray([6, 14, 22, 30, 38, 46, 54, 62, 70, 78], dtype=np.float64)
    stopband = np.asarray([4, 10, 16, 24, 32, 40, 48, 56, 64, 72], dtype=np.float64)
    fb = idx_fb - 1
    wp = [passband[fb] / (fs / 2), 90 / (fs / 2)]
    ws = [stopband[fb] / (fs / 2), 100 / (fs / 2)]
    order, wn = signal.cheb1ord(wp, ws, 3, 40)
    b, a = signal.cheby1(order, 0.5, wn, btype="bandpass")
    return signal.filtfilt(b, a, eeg, axis=-1)


def _orth(x: np.ndarray) -> np.ndarray:
    x = x.T.astype(np.float64, copy=False)
    x = x - np.mean(x, axis=0, keepdims=True)
    if np.linalg.norm(x) <= 1e-12:
        return np.zeros((x.shape[0], 1), dtype=np.float64)
    q, r = np.linalg.qr(x, mode="reduced")
    keep = np.abs(np.diag(r)) > 1e-10
    if not np.any(keep):
        return q[:, :1] * 0.0
    return q[:, keep]


def first_cca_corr(x: np.ndarray, ref_q: np.ndarray) -> float:
    qx = _orth(x)
    if qx.size == 0 or ref_q.size == 0:
        return 0.0
    vals = svd(qx.T @ ref_q, compute_uv=False, check_finite=False)
    if vals.size == 0:
        return 0.0
    return float(np.clip(vals[0], 0.0, 1.0))


class CCAClassifier:
    """Frequency-reference CCA classifier for SSVEP."""

    def __init__(self, window: float, harmonics: int = 5, spec: BenchmarkSpec | None = None) -> None:
        self.window = window
        self.harmonics = harmonics
        self.spec = spec or BenchmarkSpec()
        self.refs_q = [_orth(ref) for ref in sine_references(window, harmonics, self.spec)]

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict labels for trials shaped trials x channels x samples."""

        scores = np.zeros((x.shape[0], len(self.refs_q)), dtype=np.float64)
        for trial_idx, trial in enumerate(x):
            qx = _orth(trial)
            for cls, ref_q in enumerate(self.refs_q):
                vals = svd(qx.T @ ref_q, compute_uv=False, check_finite=False)
                scores[trial_idx, cls] = float(np.clip(vals[0], 0.0, 1.0)) if vals.size else 0.0
        return np.argmax(scores, axis=1), scores


class FBCCAClassifier:
    """Filter-bank CCA classifier following the public TRCA-SSVEP MATLAB code."""

    def __init__(
        self,
        window: float,
        harmonics: int = 5,
        n_fbs: int = 5,
        spec: BenchmarkSpec | None = None,
    ) -> None:
        self.window = window
        self.harmonics = harmonics
        self.n_fbs = n_fbs
        self.spec = spec or BenchmarkSpec()
        self.refs_q = [_orth(ref) for ref in sine_references(window, harmonics, self.spec)]
        self.fb_weights = np.asarray([(idx + 1) ** (-1.25) + 0.25 for idx in range(n_fbs)], dtype=np.float64)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict labels for trials shaped trials x channels x samples."""

        band_scores = np.zeros((x.shape[0], self.n_fbs, len(self.refs_q)), dtype=np.float64)
        for fb in range(self.n_fbs):
            filtered = fbcca_filterbank(x, self.spec.sampling_rate, fb + 1)
            for trial_idx, trial in enumerate(filtered):
                qx = _orth(trial)
                for cls, ref_q in enumerate(self.refs_q):
                    vals = svd(qx.T @ ref_q, compute_uv=False, check_finite=False)
                    band_scores[trial_idx, fb, cls] = float(np.clip(vals[0], 0.0, 1.0)) if vals.size else 0.0
        scores = np.einsum("f,tfc->tc", self.fb_weights, band_scores)
        return np.argmax(scores, axis=1), scores
