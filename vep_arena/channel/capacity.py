from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import logsumexp


EPS = 1e-15


@dataclass(frozen=True)
class BAResult:
    capacity: float
    q: np.ndarray
    iterations: int
    gap: float
    converged: bool


@dataclass(frozen=True)
class ClosedCapacityResult:
    capacity: float
    q: np.ndarray
    d: np.ndarray
    valid: bool
    condition: float
    reason: str


def _as_transition(P: np.ndarray, *, require_rows: bool = True) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Transition matrix must be square.")
    if not np.all(np.isfinite(P)):
        raise ValueError("Transition matrix contains non-finite values.")
    if np.any(P < -1e-12):
        raise ValueError("Transition matrix contains negative probabilities.")
    P = np.clip(P, 0.0, None)
    if require_rows:
        row_sums = P.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("Transition matrix has an empty row.")
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError("Transition matrix rows must sum to 1.")
    return P


def _xlog2x(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(arr)
    mask = arr > 0
    out[mask] = arr[mask] * np.log2(arr[mask])
    if np.isscalar(x):
        return float(out)
    return out


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    return float(-_xlog2x(p) - _xlog2x(1.0 - p))


def capacity_c0(M: int) -> float:
    if M <= 0:
        raise ValueError("M must be positive.")
    return math.log2(M)


def capacity_c1(M: int, p: float) -> float:
    """Symmetric-error BCI single-trial information in bits/symbol."""

    if M <= 1:
        return 0.0
    p = float(np.clip(p, 0.0, 1.0))
    if p >= 1.0:
        return math.log2(M)
    if p <= 0.0:
        return math.log2(M) + math.log2(1.0 / (M - 1))
    return math.log2(M) + p * math.log2(p) + (1.0 - p) * math.log2((1.0 - p) / (M - 1))


def conditional_entropy_rows(P: np.ndarray) -> np.ndarray:
    P = _as_transition(P)
    return -np.sum(_xlog2x(P), axis=1)


def mutual_info(P: np.ndarray, q: np.ndarray) -> float:
    P = _as_transition(P)
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or q.size != P.shape[0]:
        raise ValueError("Input distribution shape does not match transition matrix.")
    if np.any(q < -1e-12):
        raise ValueError("Input distribution contains negative probabilities.")
    q = np.clip(q, 0.0, None)
    total = q.sum()
    if total <= 0:
        raise ValueError("Input distribution has zero mass.")
    q = q / total
    r = q @ P
    h_y = -float(np.sum(_xlog2x(r)))
    h_y_given_x = float(q @ conditional_entropy_rows(P))
    return h_y - h_y_given_x


def mutual_info_uniform(P: np.ndarray) -> float:
    P = _as_transition(P)
    return mutual_info(P, np.full(P.shape[0], 1.0 / P.shape[0], dtype=np.float64))


def capacity_ba(P: np.ndarray, max_iter: int = 10_000, tol: float = 1e-5) -> BAResult:
    """Blahut-Arimoto capacity for a discrete memoryless channel.

    Rows are inputs/classes and columns are decoder outputs. The update is
    performed in log-space so sparse confusion matrices remain stable.
    """

    P = _as_transition(P)
    M = P.shape[0]
    q = np.full(M, 1.0 / M, dtype=np.float64)
    gap = math.inf
    converged = False
    D = np.zeros(M, dtype=np.float64)
    for iteration in range(1, max_iter + 1):
        r = np.clip(q @ P, EPS, None)
        log_ratio = np.zeros_like(P)
        mask = P > 0
        log_ratio[mask] = np.log2(P[mask]) - np.log2(np.broadcast_to(r, P.shape)[mask])
        D = np.sum(P * log_ratio, axis=1)
        capacity = float(q @ D)
        gap = float(np.max(D) - capacity)
        if gap < tol:
            converged = True
            break
        log_q = np.log(np.clip(q, EPS, None)) + D * math.log(2.0)
        q = np.exp(log_q - logsumexp(log_q))
    else:
        iteration = max_iter
        capacity = float(q @ D)
    return BAResult(capacity=float(q @ D), q=q, iterations=iteration, gap=gap, converged=converged)


def capacity_c2_closed(P: np.ndarray, cond_max: float = 1e10) -> ClosedCapacityResult:
    """Muroga closed-form capacity used by Costa 2020.

    The returned result is marked invalid when the inverse is unstable or the
    implied capacity-achieving distribution is outside the probability simplex.
    """

    P = _as_transition(P)
    try:
        condition = float(np.linalg.cond(P))
        if not np.isfinite(condition) or condition > cond_max:
            return ClosedCapacityResult(math.nan, np.full(P.shape[0], math.nan), np.full(P.shape[0], math.nan), False, condition, "ill_conditioned")
        inv = np.linalg.inv(P)
    except np.linalg.LinAlgError:
        return ClosedCapacityResult(math.nan, np.full(P.shape[0], math.nan), np.full(P.shape[0], math.nan), False, math.inf, "singular")
    row_entropy = conditional_entropy_rows(P)
    h = -(inv @ row_entropy)
    with np.errstate(over="ignore", invalid="ignore"):
        terms = np.power(2.0, h)
    if not np.all(np.isfinite(terms)):
        return ClosedCapacityResult(math.nan, np.full(P.shape[0], math.nan), np.full(P.shape[0], math.nan), False, condition, "nonfinite_terms")
    total = float(np.sum(terms))
    if total <= 0 or not np.isfinite(total):
        return ClosedCapacityResult(math.nan, np.full(P.shape[0], math.nan), np.full(P.shape[0], math.nan), False, condition, "nonfinite_terms")
    capacity = math.log2(total)
    with np.errstate(over="ignore", invalid="ignore"):
        d = inv.T @ terms
    if not np.all(np.isfinite(d)):
        return ClosedCapacityResult(float(capacity), np.full(P.shape[0], math.nan), d, False, condition, "nonfinite_terms")
    q = d / total
    if np.any(d <= 0) or np.any(q < -1e-10) or not np.all(np.isfinite(q)):
        return ClosedCapacityResult(float(capacity), q, d, False, condition, "simplex_violation")
    q = np.clip(q, 0.0, None)
    q = q / q.sum()
    return ClosedCapacityResult(float(capacity), q, d, True, condition, "ok")


def capacity_binary_closed(p12: float, p21: float) -> float:
    """Closed-form capacity for a binary asymmetric channel."""

    p12 = float(np.clip(p12, 0.0, 1.0))
    p21 = float(np.clip(p21, 0.0, 1.0))
    denom = 1.0 - p12 - p21
    if abs(denom) <= 1e-12:
        return 0.0
    h12 = binary_entropy(p12)
    h21 = binary_entropy(p21)
    z = 2.0 ** ((h21 - h12) / denom)
    output_one = 1.0 / (1.0 + z)
    q1 = ((1.0 - p21) - output_one) / denom
    q1 = float(np.clip(q1, 0.0, 1.0))
    output = q1 * p12 + (1.0 - q1) * (1.0 - p21)
    return float(binary_entropy(output) - q1 * h12 - (1.0 - q1) * h21)
