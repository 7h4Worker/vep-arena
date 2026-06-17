# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import numpy as np


def center_rows(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=-1, keepdims=True)


def corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    av = np.ravel(a) - np.mean(a)
    bv = np.ravel(b) - np.mean(b)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlate each row of `a` with vector `b`."""

    aa = np.asarray(a, dtype=np.float64)
    bb = np.ravel(np.asarray(b, dtype=np.float64))
    aa = aa - np.mean(aa, axis=1, keepdims=True)
    bb = bb - np.mean(bb)
    denom = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb)
    out = np.zeros(aa.shape[0], dtype=np.float64)
    keep = denom > 1e-12
    out[keep] = aa[keep] @ bb / denom[keep]
    return out


def _solve_trca(sb: np.ndarray, sw: np.ndarray) -> np.ndarray:
    sw = sw + np.eye(sw.shape[0]) * 1e-8
    try:
        chol = np.linalg.cholesky(sw)
        inv_chol = np.linalg.solve(chol, np.eye(chol.shape[0]))
        mat = inv_chol @ sb @ inv_chol.T
        vals, vecs = np.linalg.eigh((mat + mat.T) * 0.5)
        w = inv_chol.T @ vecs[:, int(np.argmax(vals))]
    except np.linalg.LinAlgError:
        mat = np.linalg.pinv(sw) @ sb
        vals, vecs = np.linalg.eig(mat)
        w = np.real(vecs[:, int(np.argmax(np.real(vals)))])
    norm = np.linalg.norm(w)
    return w / norm if norm > 1e-12 else w


def trca_filter_original_pairwise(trials: np.ndarray) -> np.ndarray:
    """Original pairwise TRCA filter for validation and timing baselines.

    Input shape is trials x channels x samples. This computes the inter-trial
    covariance with explicit pair loops, so it scales quadratically in the
    number of trials.
    """

    x = center_rows(np.asarray(trials, dtype=np.float64))
    n_trials, channels, _ = x.shape
    sb = np.zeros((channels, channels), dtype=np.float64)
    for i in range(n_trials - 1):
        for j in range(i + 1, n_trials):
            sb += x[i] @ x[j].T + x[j] @ x[i].T
    ux = np.transpose(x, (1, 0, 2)).reshape(channels, -1)
    sw = ux @ ux.T
    return _solve_trca(sb, sw)


def trca_filter(trials: np.ndarray) -> np.ndarray:
    """Fit the first TRCA spatial filter with the reformulated covariance.

    Input shape is trials x channels x samples. The covariance form matches
    Chiang et al.'s reformulation: the pairwise inter-trial covariance can be
    obtained from the covariance of the summed trials minus the within-trial
    covariance. This avoids explicit trial-pair matrix multiplications and
    reduces that part from O(N^2) to O(N). NumPy whitening is used instead of
    SciPy generalized eigensolvers for better process stability.
    """

    x = center_rows(np.asarray(trials, dtype=np.float64))
    summed = np.sum(x, axis=0)
    ux = np.transpose(x, (1, 0, 2)).reshape(x.shape[1], -1)
    sw = ux @ ux.T
    sb = summed @ summed.T - sw
    return _solve_trca(sb, sw)


def trca_filter_from_cst(eeg: np.ndarray) -> np.ndarray:
    """Fit TRCA from channels x samples x trials data."""

    return trca_filter(np.transpose(np.asarray(eeg), (2, 0, 1)))


def trca_scores(
    x: np.ndarray,
    templates: np.ndarray,
    filters: np.ndarray,
    weights: np.ndarray,
    *,
    ensemble: bool = False,
) -> np.ndarray:
    """Score filter-bank TRCA/eTRCA trials.

    Shapes:
    - x: trials x filterbanks x channels x samples
    - templates: classes x filterbanks x channels x samples
    - filters: filterbanks x classes x channels
    """

    trials, n_fbs, _, _ = x.shape
    classes = templates.shape[0]
    band_scores = np.zeros((trials, n_fbs, classes), dtype=np.float64)
    for fb_idx in range(n_fbs):
        if ensemble:
            w = filters[fb_idx].T
            projected_trials = np.matmul(np.transpose(x[:, fb_idx], (0, 2, 1)), w).reshape(trials, -1)
            for cls in range(classes):
                projected_template = (templates[cls, fb_idx].T @ w).reshape(-1)
                band_scores[:, fb_idx, cls] = corr_rows(projected_trials, projected_template)
        else:
            for cls in range(classes):
                w = filters[fb_idx, cls]
                projected_trials = x[:, fb_idx].transpose(0, 2, 1) @ w
                projected_template = templates[cls, fb_idx].T @ w
                band_scores[:, fb_idx, cls] = corr_rows(projected_trials, projected_template)
    return np.einsum("f,tfc->tc", weights[:n_fbs], band_scores)
