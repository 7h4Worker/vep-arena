from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np

from vep_arena.channel.capacity import (
    capacity_ba,
    capacity_c1,
    capacity_c2_closed,
    mutual_info_uniform,
)
from vep_arena.channel.confusion import normalize_confusion, subset_counts


@dataclass(frozen=True)
class CodebookMetrics:
    size: int
    accuracy: float
    retained_mass: float
    capacity_ba: float
    c1: float
    c2_closed: float
    c2_valid: bool
    c2_condition: float
    c2_reason: str
    i_uniform: float
    ba_minus_c1: float
    ba_minus_uniform: float
    q_min: float
    q_max: float
    q_support_1e3: int


@dataclass(frozen=True)
class SelectionStep:
    size: int
    added_class: int
    capacity: float
    seed_class: int
    accuracy: float
    retained_mass: float
    c1: float
    c2_closed: float
    c2_valid: bool
    c2_condition: float
    c2_reason: str
    i_uniform: float
    ba_minus_c1: float
    ba_minus_uniform: float
    q_min: float
    q_max: float
    q_support_1e3: int
    subset: tuple[int, ...]


@dataclass(frozen=True)
class SelectionResult:
    best_subset: tuple[int, ...]
    best_capacity: float
    seed_class: int
    path: tuple[SelectionStep, ...]


def codebook_metrics(
    counts: np.ndarray,
    subset: Iterable[int],
    alpha: float = 0.01,
    ba_max_iter: int = 3000,
    ba_tol: float = 1e-5,
) -> CodebookMetrics:
    """Evaluate a cropped codebook from a full-class confusion matrix."""

    counts = np.asarray(counts, dtype=np.float64)
    subset_arr = np.asarray(sorted(int(x) for x in subset), dtype=np.int64)
    if subset_arr.ndim != 1 or subset_arr.size == 0:
        raise ValueError("subset must contain at least one class.")
    sub_counts = subset_counts(counts, subset_arr)
    total = float(np.sum(sub_counts))
    correct = float(np.trace(sub_counts))
    accuracy = correct / total if total else 0.0
    original_total = float(np.sum(counts[subset_arr, :]))
    retained_mass = total / original_total if original_total else 0.0
    P = normalize_confusion(sub_counts, alpha=alpha)
    ba = capacity_ba(P, max_iter=ba_max_iter, tol=ba_tol)
    closed = capacity_c2_closed(P)
    i_uniform = mutual_info_uniform(P)
    c1 = capacity_c1(subset_arr.size, accuracy)
    return CodebookMetrics(
        size=int(subset_arr.size),
        accuracy=float(accuracy),
        retained_mass=float(retained_mass),
        capacity_ba=float(ba.capacity),
        c1=float(c1),
        c2_closed=float(closed.capacity) if np.isfinite(closed.capacity) else np.nan,
        c2_valid=bool(closed.valid),
        c2_condition=float(closed.condition),
        c2_reason=closed.reason,
        i_uniform=float(i_uniform),
        ba_minus_c1=float(ba.capacity - c1),
        ba_minus_uniform=float(ba.capacity - i_uniform),
        q_min=float(np.min(ba.q)),
        q_max=float(np.max(ba.q)),
        q_support_1e3=int(np.sum(ba.q >= 1e-3)),
    )


def _step_from_metrics(
    *,
    seed_class: int,
    added_class: int,
    subset: list[int],
    metrics: CodebookMetrics,
) -> SelectionStep:
    return SelectionStep(
        size=metrics.size,
        added_class=int(added_class),
        capacity=metrics.capacity_ba,
        seed_class=int(seed_class),
        accuracy=metrics.accuracy,
        retained_mass=metrics.retained_mass,
        c1=metrics.c1,
        c2_closed=metrics.c2_closed,
        c2_valid=metrics.c2_valid,
        c2_condition=metrics.c2_condition,
        c2_reason=metrics.c2_reason,
        i_uniform=metrics.i_uniform,
        ba_minus_c1=metrics.ba_minus_c1,
        ba_minus_uniform=metrics.ba_minus_uniform,
        q_min=metrics.q_min,
        q_max=metrics.q_max,
        q_support_1e3=metrics.q_support_1e3,
        subset=tuple(int(x) for x in subset),
    )


def _score(metrics: CodebookMetrics, scoring: str) -> float:
    if scoring == "ba":
        return metrics.capacity_ba
    if scoring == "c2":
        return metrics.c2_closed if metrics.c2_valid else -np.inf
    if scoring == "c2_or_ba":
        return metrics.c2_closed if metrics.c2_valid else metrics.capacity_ba
    raise ValueError(f"Unknown scoring mode: {scoring}")


def _seed_order(counts: np.ndarray, seed_classes: Iterable[int] | str | None, n_starts: int | None) -> list[int]:
    M = counts.shape[0]
    if seed_classes is None:
        row_sums = np.maximum(counts.sum(axis=1), 1.0)
        return [int(np.argmax(np.diag(counts) / row_sums))]
    if isinstance(seed_classes, str):
        if seed_classes == "best_diag":
            row_sums = np.maximum(counts.sum(axis=1), 1.0)
            return [int(np.argmax(np.diag(counts) / row_sums))]
        if seed_classes == "all":
            return list(range(M))
        if seed_classes == "top_diag":
            row_sums = np.maximum(counts.sum(axis=1), 1.0)
            order = np.argsort(-(np.diag(counts) / row_sums))
            limit = M if n_starts is None else min(M, int(n_starts))
            return [int(x) for x in order[:limit]]
        raise ValueError(f"Unknown seed mode: {seed_classes}")
    out = [int(x) for x in seed_classes]
    if n_starts is not None:
        out = out[: int(n_starts)]
    if not out:
        raise ValueError("At least one seed class is required.")
    return out


def _greedy_from_seed(
    counts: np.ndarray,
    seed_class: int,
    *,
    alpha: float,
    max_size: int,
    scoring: str,
    ba_max_iter: int,
    ba_tol: float,
) -> SelectionResult:
    counts = np.asarray(counts, dtype=np.float64)
    M = counts.shape[0]
    selected: list[int] = [int(seed_class)]
    remaining = set(range(M)) - {int(seed_class)}
    seed_metrics = codebook_metrics(counts, selected, alpha=alpha, ba_max_iter=ba_max_iter, ba_tol=ba_tol)
    path: list[SelectionStep] = [
        _step_from_metrics(seed_class=seed_class, added_class=seed_class, subset=selected, metrics=seed_metrics)
    ]
    best_step = path[0]
    while remaining and len(selected) < max_size:
        best_candidate: int | None = None
        best_candidate_score = -np.inf
        best_candidate_metrics: CodebookMetrics | None = None
        best_candidate_subset: list[int] = []
        for candidate in sorted(remaining):
            subset = sorted([*selected, candidate])
            metrics = codebook_metrics(counts, subset, alpha=alpha, ba_max_iter=ba_max_iter, ba_tol=ba_tol)
            candidate_score = _score(metrics, scoring)
            if candidate_score > best_candidate_score:
                best_candidate = candidate
                best_candidate_score = candidate_score
                best_candidate_metrics = metrics
                best_candidate_subset = subset
        if best_candidate is None or best_candidate_metrics is None:
            break
        selected = best_candidate_subset
        remaining.remove(best_candidate)
        step = _step_from_metrics(
            seed_class=seed_class,
            added_class=best_candidate,
            subset=selected,
            metrics=best_candidate_metrics,
        )
        path.append(step)
        if step.capacity > best_step.capacity:
            best_step = step
    return SelectionResult(
        best_subset=best_step.subset,
        best_capacity=float(best_step.capacity),
        seed_class=int(seed_class),
        path=tuple(path),
    )


def greedy_codebook_pruning(
    counts: np.ndarray,
    alpha: float = 0.01,
    max_size: int | None = None,
    seed_classes: Iterable[int] | str | None = "best_diag",
    n_starts: int | None = None,
    scoring: str = "ba",
    ba_max_iter: int = 3000,
    ba_tol: float = 1e-5,
) -> SelectionResult:
    """Forward codebook selection from a fixed full-class confusion matrix.

    This is the v1 approximation to Costa-style wrapper selection: the
    classifier is not retrained for each subset; rows/columns are cropped from
    the existing decision channel and renormalized.
    """

    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("counts must be square.")
    M = counts.shape[0]
    max_size = M if max_size is None else min(M, int(max_size))
    starts = _seed_order(counts, seed_classes, n_starts)
    results = [
        _greedy_from_seed(
            counts,
            seed,
            alpha=alpha,
            max_size=max_size,
            scoring=scoring,
            ba_max_iter=ba_max_iter,
            ba_tol=ba_tol,
        )
        for seed in starts
    ]
    return max(results, key=lambda item: item.best_capacity)


def gain_decomposition(
    counts: np.ndarray,
    alpha: float = 0.01,
    selected: SelectionResult | None = None,
    **selection_kwargs: object,
) -> dict[str, float | int]:
    full_P = normalize_confusion(counts, alpha=alpha)
    full_ba = capacity_ba(full_P).capacity
    full_acc = float(np.trace(counts) / max(np.sum(counts), 1.0))
    full_c1 = capacity_c1(counts.shape[0], full_acc)
    selected = selected or greedy_codebook_pruning(counts, alpha=alpha, **selection_kwargs)
    best_step = max(selected.path, key=lambda step: step.capacity)
    return {
        "full_c1": float(full_c1),
        "full_ba": float(full_ba),
        "best_pruned_ba": float(selected.best_capacity),
        "best_size": int(len(selected.best_subset)),
        "best_seed_class": int(selected.seed_class),
        "best_accuracy": float(best_step.accuracy),
        "best_retained_mass": float(best_step.retained_mass),
        "best_i_uniform": float(best_step.i_uniform),
        "best_c2_closed": float(best_step.c2_closed),
        "best_c2_valid": bool(best_step.c2_valid),
        "ba_gain": float(full_ba - full_c1),
        "pruning_gain": float(selected.best_capacity - full_ba),
        "total_gain": float(selected.best_capacity - full_c1),
    }
