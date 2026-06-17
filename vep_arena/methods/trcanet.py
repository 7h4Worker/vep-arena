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
    """Factory for the CNN part, imported lazily to keep torch optional."""

    @staticmethod
    def make_model(filters: int, samples: int, subbands: int, classes: int, dropout: float, final_dropout: float):
        import torch
        from torch import nn

        class TRCANet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.subband_conv = nn.Conv2d(subbands, 1, kernel_size=(1, 1), bias=False)
                self.filter_conv = nn.Conv2d(1, 120, kernel_size=(filters, 1))
                self.dropout1 = nn.Dropout(dropout)
                self.temporal_conv = nn.Conv2d(120, 120, kernel_size=(1, 2), stride=(1, 2))
                self.dropout2 = nn.Dropout(dropout)
                self.relu = nn.ReLU()
                self.refine_conv = nn.Conv2d(120, 120, kernel_size=(1, 10), padding="same")
                self.dropout3 = nn.Dropout(final_dropout)
                with torch.no_grad():
                    dummy = torch.zeros(1, subbands, filters, samples)
                    flat = self._features(dummy).flatten(1).shape[1]
                self.classifier = nn.Linear(flat, classes)
                self.reset_parameters()

            def reset_parameters(self) -> None:
                nn.init.ones_(self.subband_conv.weight)
                for layer in (self.filter_conv, self.temporal_conv, self.refine_conv, self.classifier):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

            def _features(self, x):
                x = self.subband_conv(x)
                x = self.filter_conv(x)
                x = self.dropout1(x)
                x = self.temporal_conv(x)
                x = self.dropout2(x)
                x = self.relu(x)
                x = self.refine_conv(x)
                x = self.dropout3(x)
                return x

            def forward(self, x):
                return self.classifier(self._features(x).flatten(1))

        return TRCANet()
