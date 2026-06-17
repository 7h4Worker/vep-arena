# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import numpy as np

from vep_arena.config import BenchmarkSpec
from vep_arena.data.benchmark import reference_signals
from vep_arena.methods.trca_core import corr_flat as _corr_flat
from vep_arena.methods.trca_core import trca_filter as _trca_filter
from vep_arena.methods.trca_core import trca_scores


def filterbank_weights(n_fbs: int) -> np.ndarray:
    return np.asarray([(idx + 1) ** (-1.25) + 0.25 for idx in range(n_fbs)], dtype=np.float64)


def _center_rows(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=-1, keepdims=True)


def _orth_rows(x: np.ndarray) -> np.ndarray:
    """Column-orthonormal basis for row signals after sample-wise transpose."""

    z = _center_rows(x).T.astype(np.float64, copy=False)
    if np.linalg.norm(z) <= 1e-12:
        return np.zeros((z.shape[0], 1), dtype=np.float64)
    q, r = np.linalg.qr(z, mode="reduced")
    keep = np.abs(np.diag(r)) > 1e-10
    if not np.any(keep):
        return q[:, :1] * 0.0
    return q[:, keep]


def _cca_corr(x: np.ndarray, y_q: np.ndarray) -> float:
    x_q = _orth_rows(x)
    vals = np.linalg.svd(x_q.T @ y_q, compute_uv=False)
    return float(np.clip(vals[0], 0.0, 1.0)) if vals.size else 0.0


class CCA:
    name = "CCA"

    def __init__(self, window: float, harmonics: int = 5, spec: BenchmarkSpec | None = None) -> None:
        self.refs_q = [_orth_rows(ref) for ref in reference_signals(window, harmonics, spec)]

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "CCA":
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict from epochs shaped trials x subbands x channels x samples.

        CCA uses the first subband/raw slot. The runner supplies a single raw
        slot for this method.
        """

        trials = x[:, 0]
        scores = np.zeros((trials.shape[0], len(self.refs_q)), dtype=np.float64)
        for trial_idx, trial in enumerate(trials):
            for cls, ref_q in enumerate(self.refs_q):
                scores[trial_idx, cls] = _cca_corr(trial, ref_q)
        return np.argmax(scores, axis=1), scores


class FBCCA:
    name = "FBCCA"

    def __init__(
        self,
        window: float,
        harmonics: int = 5,
        n_fbs: int = 5,
        spec: BenchmarkSpec | None = None,
    ) -> None:
        self.refs_q = [_orth_rows(ref) for ref in reference_signals(window, harmonics, spec)]
        self.weights = filterbank_weights(n_fbs)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "FBCCA":
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores_by_band = np.zeros((x.shape[0], x.shape[1], len(self.refs_q)), dtype=np.float64)
        for trial_idx, trial in enumerate(x):
            for fb_idx, band in enumerate(trial):
                for cls, ref_q in enumerate(self.refs_q):
                    scores_by_band[trial_idx, fb_idx, cls] = _cca_corr(band, ref_q)
        scores = np.einsum("f,tfc->tc", self.weights[: x.shape[1]], scores_by_band)
        return np.argmax(scores, axis=1), scores


class TRCA:
    name = "TRCA"

    def __init__(self, n_fbs: int = 5, ensemble: bool = False) -> None:
        self.weights = filterbank_weights(n_fbs)
        self.ensemble = ensemble
        self.templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "TRCA":
        classes = int(np.max(train_y)) + 1
        _, n_fbs, channels, samples = train_x.shape
        self.templates = np.zeros((classes, n_fbs, channels, samples), dtype=np.float64)
        self.filters = np.zeros((n_fbs, classes, channels), dtype=np.float64)
        for cls in range(classes):
            class_trials = train_x[train_y == cls]
            self.templates[cls] = np.mean(class_trials, axis=0)
            for fb_idx in range(n_fbs):
                self.filters[fb_idx, cls] = _trca_filter(class_trials[:, fb_idx])
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.templates is None or self.filters is None:
            raise RuntimeError("TRCA model is not fitted.")
        scores = trca_scores(x, self.templates, self.filters, self.weights, ensemble=self.ensemble)
        return np.argmax(scores, axis=1), scores
