"""Shared numerical utilities for signal and information-theoretic analysis."""
from __future__ import annotations

import numpy as np

EPS = 1e-15


def safe_log2(x: np.ndarray | float) -> np.ndarray | float:
    """``log₂`` with convention ``0·log(0) = 0``."""
    arr = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(arr)
    mask = arr > 0
    out[mask] = np.log2(arr[mask])
    return float(out) if np.isscalar(x) else out


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p ‖ q) in bits.  Inputs are normalised internally."""
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)
    ps, qs = p.sum(), q.sum()
    if ps <= 0 or qs <= 0:
        return float("nan")
    p, q = p / ps, q / qs
    mask = p > EPS
    return float(np.sum(p[mask] * np.log2(p[mask] / np.clip(q[mask], EPS, None))))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits.  Symmetric, bounded [0, 1]."""
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)
    ps, qs = p.sum(), q.sum()
    if ps <= 0 or qs <= 0:
        return float("nan")
    p, q = p / ps, q / qs
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def pearson_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray | float:
    """Row-wise Pearson correlation.

    Parameters
    ----------
    a, b : (n, m) or (m,)

    Returns
    -------
    r : (n,) or scalar
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    squeeze = a.ndim == 1
    if squeeze:
        a, b = a[np.newaxis], b[np.newaxis]
    ac = a - a.mean(axis=-1, keepdims=True)
    bc = b - b.mean(axis=-1, keepdims=True)
    num = np.sum(ac * bc, axis=-1)
    denom = np.sqrt(np.sum(ac**2, axis=-1) * np.sum(bc**2, axis=-1))
    r = num / np.clip(denom, EPS, None)
    return float(r[0]) if squeeze else r


def spectral_concentration(
    psd: np.ndarray,
    freqs: np.ndarray,
    target_freq: float,
    n_harmonics: int = 5,
    bandwidth: float = 0.5,
) -> float:
    """Fraction of total PSD concentrated at target and its harmonics.

    Higher → more sparse spectrum → easier to separate from other targets.
    """
    psd_1d = np.mean(psd, axis=tuple(range(psd.ndim - 1))) if psd.ndim > 1 else psd
    nyquist = freqs[-1]

    sig_mask = np.zeros(len(freqs), dtype=bool)
    for k in range(1, n_harmonics + 1):
        hf = target_freq * k
        if hf < nyquist:
            sig_mask |= np.abs(freqs - hf) <= bandwidth

    total = float(np.sum(psd_1d))
    if total <= 0:
        return 0.0
    return float(np.sum(psd_1d[sig_mask])) / total


def harmonic_interference_matrix(
    freqs_list: np.ndarray | tuple[float, ...],
    n_harmonics: int = 5,
    resolution_hz: float = 0.25,
) -> np.ndarray:
    """Predicted spectral interference between pairs of target frequencies.

    Entry ``(i, j)`` counts the number of (harmonic of ``f_i``,
    harmonic of ``f_j``) pairs that fall within ``resolution_hz`` of each
    other.  Purely deterministic — depends only on the frequency set.

    Returns
    -------
    interference : (n_targets, n_targets)  — symmetric, zero diagonal
    """
    freqs_list = np.asarray(freqs_list, dtype=np.float64)
    n = len(freqs_list)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        hi = [freqs_list[i] * k for k in range(1, n_harmonics + 1)]
        for j in range(i + 1, n):
            hj = [freqs_list[j] * k for k in range(1, n_harmonics + 1)]
            count = sum(
                1 for a in hi for b in hj if abs(a - b) <= resolution_hz
            )
            mat[i, j] = mat[j, i] = count
    return mat
