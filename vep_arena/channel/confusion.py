from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def confusion_counts(true: Iterable[int], pred: Iterable[int], classes: int) -> np.ndarray:
    counts = np.zeros((classes, classes), dtype=np.int64)
    for t, p in zip(true, pred):
        ti = int(t)
        pi = int(p)
        if 0 <= ti < classes and 0 <= pi < classes:
            counts[ti, pi] += 1
    return counts


def normalize_confusion(counts: np.ndarray, alpha: float = 0.0, zero_policy: str = "uniform") -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("Confusion counts must be a square matrix.")
    if np.any(counts < 0):
        raise ValueError("Confusion counts contain negative values.")
    P = counts + float(alpha)
    row_sums = P.sum(axis=1, keepdims=True)
    empty = row_sums[:, 0] <= 0
    if np.any(empty):
        if zero_policy == "raise":
            raise ValueError("Confusion matrix has empty rows.")
        if zero_policy != "uniform":
            raise ValueError(f"Unknown zero_policy: {zero_policy}")
        P[empty] = 1.0
        row_sums = P.sum(axis=1, keepdims=True)
    return P / row_sums


def subset_counts(counts: np.ndarray, subset: list[int] | np.ndarray) -> np.ndarray:
    counts = np.asarray(counts)
    subset = np.asarray(subset, dtype=np.int64)
    return counts[np.ix_(subset, subset)]


def bootstrap_counts(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    out = np.zeros_like(counts)
    for idx, row in enumerate(counts):
        total = int(row.sum())
        if total <= 0:
            continue
        probs = row / total
        out[idx] = rng.multinomial(total, probs)
    return out
