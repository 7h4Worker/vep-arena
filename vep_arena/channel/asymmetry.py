from __future__ import annotations

import math

import numpy as np

from vep_arena.channel.capacity import binary_entropy


EPS = 1e-12


def _as_transition(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Transition matrix must be square.")
    row_sums = P.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("Transition matrix has an empty row.")
    return P / row_sums[:, None]


def stationary_distribution(P: np.ndarray, max_iter: int = 10_000, tol: float = 1e-12) -> np.ndarray:
    P = _as_transition(P)
    phi = np.full(P.shape[0], 1.0 / P.shape[0], dtype=np.float64)
    for _ in range(max_iter):
        next_phi = phi @ P
        if np.linalg.norm(next_phi - phi, ord=1) < tol:
            phi = next_phi
            break
        phi = next_phi
    phi = np.clip(phi, EPS, None)
    return phi / phi.sum()


def asymmetry_score(P: np.ndarray) -> float:
    P = _as_transition(P)
    phi = stationary_distribution(P)
    sqrt_phi = np.sqrt(phi)
    inv_sqrt_phi = 1.0 / sqrt_phi
    lap = np.eye(P.shape[0]) - P
    transformed = sqrt_phi[:, None] * lap * inv_sqrt_phi[None, :]
    skew = 0.5 * (transformed - transformed.T)
    values = np.linalg.svd(skew, compute_uv=False)
    return float(values[0]) if values.size else 0.0


def fano_bound(epsilon: float, M: int) -> float:
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    if M <= 1:
        return 0.0
    return float(binary_entropy(epsilon) + epsilon * math.log2(M - 1))
