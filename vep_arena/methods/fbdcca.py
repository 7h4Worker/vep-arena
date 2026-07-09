# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-07-08
# Description: Filter-bank dual-frequency CCA for Dual-Alpha SSVEP tasks.
from __future__ import annotations

import numpy as np


FBDCCA_DEFAULT_WEIGHTS = np.asarray([1.0, 0.71, 0.58, 0.50, 0.45, 0.41, 0.38, 0.35, 0.33, 0.31], dtype=np.float64)


def _center_rows(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=-1, keepdims=True)


def _orth_rows(x: np.ndarray) -> np.ndarray:
    z = _center_rows(np.asarray(x, dtype=np.float64)).T
    if np.linalg.norm(z) <= 1e-12:
        return np.zeros((z.shape[0], 1), dtype=np.float64)
    q, r = np.linalg.qr(z, mode="reduced")
    keep = np.abs(np.diag(r)) > 1e-10
    if not np.any(keep):
        return q[:, :1] * 0.0
    return q[:, keep]


def _cca_corr_from_orth(x_q: np.ndarray, y_q: np.ndarray) -> float:
    vals = np.linalg.svd(x_q.T @ y_q, compute_uv=False)
    return float(np.clip(vals[0], 0.0, 1.0)) if vals.size else 0.0


class FBDCCA:
    """Filter-bank dual-frequency CCA classifier.

    The public Dual-Alpha code evaluates a CCA score between each filtered EEG
    trial and a target template containing sine/cosine rows for both alpha
    frequencies and their harmonics. Band scores are squared and combined with
    fixed filter-bank weights.
    """

    name = "FBDCCA"

    def __init__(
        self,
        references: list[np.ndarray] | tuple[np.ndarray, ...],
        weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
        square_scores: bool = True,
    ) -> None:
        self.refs_q = [_orth_rows(ref) for ref in references]
        self.weights = np.asarray(FBDCCA_DEFAULT_WEIGHTS if weights is None else weights, dtype=np.float64)
        self.square_scores = bool(square_scores)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "FBDCCA":
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict epochs shaped trials x subbands x channels x samples."""

        trials, n_fbs, _, _ = x.shape
        classes = len(self.refs_q)
        band_scores = np.zeros((trials, n_fbs, classes), dtype=np.float64)
        for trial_idx, trial in enumerate(x):
            for fb_idx, band in enumerate(trial):
                trial_q = _orth_rows(band)
                for cls, ref_q in enumerate(self.refs_q):
                    band_scores[trial_idx, fb_idx, cls] = _cca_corr_from_orth(trial_q, ref_q)
        if self.square_scores:
            band_scores = band_scores**2
        scores = np.einsum("f,tfc->tc", self.weights[:n_fbs], band_scores)
        return np.argmax(scores, axis=1), scores
