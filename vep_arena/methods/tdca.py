# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import numpy as np
from scipy.linalg import eig, qr


def _qr_projection(ref: np.ndarray) -> np.ndarray:
    centered = ref.T - np.mean(ref.T, axis=0, keepdims=True)
    q, _, _ = qr(centered, mode="economic", pivoting=True)
    return q @ q.T


def _augment_delay_train(x: np.ndarray, samples: int, padding_len: int) -> np.ndarray:
    channels, points = x.shape
    if points < samples + padding_len:
        raise ValueError("Training data must contain target samples plus padding_len extra samples.")
    return np.concatenate([x[:, delay : delay + samples] for delay in range(padding_len + 1)], axis=0)


def _augment_delay_test(x: np.ndarray, samples: int, padding_len: int) -> np.ndarray:
    channels, points = x.shape
    if points < samples:
        raise ValueError("Test data is shorter than the target sample length.")
    parts = []
    for delay in range(padding_len + 1):
        if delay == 0:
            parts.append(x[:, :samples])
        else:
            parts.append(np.concatenate([x[:, delay:samples], np.zeros((channels, delay))], axis=1))
    return np.concatenate(parts, axis=0)


def _corrcoef_flat(a: np.ndarray, b: np.ndarray) -> float:
    av = np.ravel(a)
    bv = np.ravel(b)
    av = av - np.mean(av)
    bv = bv - np.mean(bv)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def _sorted_eigvec(sb: np.ndarray, sw: np.ndarray) -> np.ndarray:
    sw = sw + np.eye(sw.shape[0]) * 1e-8
    vals, vecs = eig(sb, sw)
    order = np.argsort(np.real(vals))[::-1]
    vecs = np.real(vecs[:, order])
    scale = np.sqrt(np.abs(np.diag(vecs.T @ sw @ vecs)))
    scale[scale <= 1e-12] = 1.0
    return vecs / scale


class TDCA:
    """Task-discriminant component analysis for SSVEP recognition."""

    def __init__(
        self,
        n_components: int = 8,
        n_delay: int = 5,
        fb_weights: np.ndarray | None = None,
    ) -> None:
        self.n_components = n_components
        self.padding_len = n_delay
        self.fb_weights = fb_weights
        self.projections: list[np.ndarray] | None = None
        self.templates: np.ndarray | None = None
        self.projected_templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None
        self.target_samples: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, refs: list[np.ndarray]) -> "TDCA":
        """Fit TDCA.

        x shape: trials x subbands x channels x samples.
        y shape: trials.
        refs: one reference matrix per class, each 2*harmonics x samples.
        """

        classes = len(refs)
        subbands, channels, _ = x.shape[1:]
        samples = refs[0].shape[1]
        self.target_samples = samples
        aug_channels = channels * (self.padding_len + 1)
        self.projections = [_qr_projection(ref) for ref in refs]
        self.templates = np.zeros((classes, subbands, aug_channels, samples * 2), dtype=np.float64)
        self.filters = np.zeros((subbands, aug_channels, self.n_components), dtype=np.float64)
        if self.fb_weights is None:
            self.fb_weights = np.asarray([(i + 1) ** (-1.25) + 0.25 for i in range(subbands)], dtype=np.float64)

        for fb in range(subbands):
            class_trials: list[list[np.ndarray]] = []
            class_means: list[np.ndarray] = []
            for cls in range(classes):
                idx = np.where(y == cls)[0]
                trials = []
                for trial_idx in idx:
                    delayed = _augment_delay_train(x[trial_idx, fb], samples, self.padding_len)
                    trials.append(np.concatenate([delayed, delayed @ self.projections[cls]], axis=1))
                class_trials.append(trials)
                class_mean = np.mean(np.stack(trials, axis=0), axis=0)
                class_means.append(class_mean)
                self.templates[cls, fb] = class_mean

            global_mean = np.mean(np.stack(class_means, axis=0), axis=0)
            sw = np.zeros((aug_channels, aug_channels), dtype=np.float64)
            sb = np.zeros((aug_channels, aug_channels), dtype=np.float64)
            for trials, class_mean in zip(class_trials, class_means):
                trial_count = max(1, len(trials))
                for trial in trials:
                    diff = trial - class_mean
                    sw += diff @ diff.T / trial_count
                diff_mean = class_mean - global_mean
                sb += diff_mean @ diff_mean.T / classes
            self.filters[fb] = _sorted_eigvec(sb, sw)[:, : self.n_components]
        self.projected_templates = np.zeros((classes, subbands, self.n_components, samples * 2), dtype=np.float64)
        for cls in range(classes):
            for fb in range(subbands):
                self.projected_templates[cls, fb] = self.filters[fb].T @ self.templates[cls, fb]
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.projections is None
            or self.templates is None
            or self.projected_templates is None
            or self.filters is None
            or self.fb_weights is None
            or self.target_samples is None
        ):
            raise RuntimeError("TDCA model is not fitted.")
        trials, subbands, _, _ = x.shape
        classes = self.templates.shape[0]
        all_scores = np.zeros((trials, subbands, classes), dtype=np.float64)
        pred = np.zeros(trials, dtype=np.int64)
        for trial_idx in range(trials):
            for fb in range(subbands):
                delayed = _augment_delay_test(x[trial_idx, fb], self.target_samples, self.padding_len)
                base = self.filters[fb].T @ delayed
                filt = self.filters[fb]
                for cls in range(classes):
                    projected = np.concatenate([base, base @ self.projections[cls]], axis=1)
                    all_scores[trial_idx, fb, cls] = _corrcoef_flat(projected, self.projected_templates[cls, fb])
            pred[trial_idx] = int(np.argmax(self.fb_weights @ all_scores[trial_idx]))
        return pred, all_scores

    def predict_online(self, window: np.ndarray) -> int:
        """Predict one online-style window: subbands x channels x samples."""

        pred, _ = self.predict(window[None, ...])
        return int(pred[0])
