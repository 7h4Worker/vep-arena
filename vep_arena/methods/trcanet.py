# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.signal import cheb1ord, cheby1, filtfilt


def trcanet_filterbank(eeg: np.ndarray, fs: int, idx_fb: int) -> np.ndarray:
    """Official TRCA-Net/Nakanishi-style filter bank for Benchmark.

    Input shape is channels x samples or channels x samples x trials.
    """

    passband = np.asarray([6, 14, 22], dtype=float)
    stopband = np.asarray([4, 10, 16], dtype=float)
    nyq = fs / 2
    idx = idx_fb - 1
    wp = [passband[idx] / nyq, 90 / nyq]
    ws = [stopband[idx] / nyq, 100 / nyq]
    order, wn = cheb1ord(wp, ws, 3, 40)
    b, a = cheby1(order, 0.5, wn, btype="bandpass")
    return filtfilt(b, a, eeg, axis=1)


def trca(eeg: np.ndarray) -> np.ndarray:
    """Task-related component analysis.

    Input shape: channels x samples x trials.
    """

    chans, samples, trials = eeg.shape
    s_mat = np.zeros((chans, chans), dtype=np.float64)
    for i in range(trials - 1):
        x1 = eeg[:, :, i] - np.mean(eeg[:, :, i], axis=1, keepdims=True)
        for j in range(i + 1, trials):
            x2 = eeg[:, :, j] - np.mean(eeg[:, :, j], axis=1, keepdims=True)
            s_mat += x1 @ x2.T + x2 @ x1.T
    ux = np.reshape(eeg, (chans, samples * trials), order="F")
    ux = ux - np.mean(ux, axis=1, keepdims=True)
    q_mat = ux @ ux.T
    q_mat += np.eye(chans) * 1e-8
    vals, vecs = eigh(s_mat, q_mat)
    return vecs[:, np.argsort(vals)[::-1]]


def fit_trca_filters(train: np.ndarray, fs: int, n_fbs: int = 3) -> np.ndarray:
    """Fit per-class TRCA filters.

    train shape: classes x channels x samples x train_blocks.
    output shape: subbands x classes x channels.
    """

    classes, channels, _, _ = train.shape
    weights = np.zeros((n_fbs, classes, channels), dtype=np.float64)
    for cls in range(classes):
        class_eeg = train[cls]
        for fb in range(n_fbs):
            filtered = trcanet_filterbank(class_eeg, fs, fb + 1)
            weights[fb, cls] = trca(filtered)[:, 0]
    return weights


def project_with_trca_filters(eeg: np.ndarray, weights: np.ndarray, fs: int) -> np.ndarray:
    """Project trials through all TRCA filters.

    eeg shape: classes x channels x samples x blocks.
    weights shape: subbands x class_filters x channels.
    output shape: blocks * classes x subbands x class_filters x samples.
    """

    classes, _, samples, blocks = eeg.shape
    n_fbs, filters, _ = weights.shape
    out = np.zeros((blocks * classes, n_fbs, filters, samples), dtype=np.float32)
    row = 0
    for block in range(blocks):
        for cls in range(classes):
            trial = eeg[cls, :, :, block]
            for fb in range(n_fbs):
                filtered = trcanet_filterbank(trial, fs, fb + 1)
                out[row, fb] = weights[fb] @ filtered
            row += 1
    return out


class TRCANetTorchMixin:
    """Factory for the CNN part.  Delegates to ``vep_arena.nn.trcanet``."""

    @staticmethod
    def make_model(filters: int, samples: int, subbands: int, classes: int, dropout: float, final_dropout: float):
        from vep_arena.nn.trcanet import TRCANet

        return TRCANet(
            filters=filters,
            samples=samples,
            subbands=subbands,
            classes=classes,
            dropout=dropout,
            final_dropout=final_dropout,
        )
