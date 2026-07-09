# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Multi-stimulus SSVEP method adapters for VEP Arena.
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import linalg

from vep_arena.config import BENCHMARK_FREQS, BenchmarkSpec
from vep_arena.data.benchmark import reference_signals
from vep_arena.methods.traditional import filterbank_weights
from vep_arena.methods.trca_core import corr_flat, corr_rows, trca_filter


def neighbor_indices(freqs: tuple[float, ...], target: int, n_neighbor: int) -> list[int]:
    order = np.argsort(np.asarray(freqs, dtype=np.float64))
    rank = int(np.where(order == target)[0][0])
    n_neighbor = min(max(1, n_neighbor), len(order))
    left = n_neighbor // 2
    start = max(0, min(rank - left, len(order) - n_neighbor))
    return [int(idx) for idx in order[start : start + n_neighbor]]


def ordered_neighbor_indices(class_order: Sequence[int], target: int, neighbor_width: int) -> list[int]:
    """Return class labels within a centered window in an explicit target order."""

    order = np.asarray(class_order, dtype=np.int64)
    if order.ndim != 1 or order.size == 0:
        raise ValueError("class_order must be a non-empty 1D sequence.")
    matches = np.where(order == int(target))[0]
    if matches.size != 1:
        raise ValueError(f"Target {target} must appear exactly once in class_order.")
    rank = int(matches[0])
    width = max(0, int(neighbor_width))
    start = max(0, rank - width)
    stop = min(order.size, rank + width + 1)
    return [int(idx) for idx in order[start:stop]]


def _center_columns(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) - np.mean(x, axis=0, keepdims=True)


def _cca_weights(x_samples_features: np.ndarray, y_samples_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _center_columns(x_samples_features)
    y = _center_columns(y_samples_features)
    qx, rx, px = linalg.qr(x, mode="economic", pivoting=True)
    qy, ry, py = linalg.qr(y, mode="economic", pivoting=True)
    left, vals, right_t = linalg.svd(qx.T @ qy, full_matrices=False, lapack_driver="gesvd")
    if vals.size == 0:
        return np.zeros(x.shape[1]), np.zeros(y.shape[1])
    ax_perm = linalg.solve_triangular(rx, left[:, 0], lower=False, check_finite=False)
    by_perm = linalg.solve_triangular(ry, right_t.T[:, 0], lower=False, check_finite=False)
    ax = np.zeros(x.shape[1], dtype=np.float64)
    by = np.zeros(y.shape[1], dtype=np.float64)
    ax[px] = ax_perm
    by[py] = by_perm
    ax_norm = np.linalg.norm(ax)
    by_norm = np.linalg.norm(by)
    return ax / ax_norm if ax_norm > 1e-12 else ax, by / by_norm if by_norm > 1e-12 else by


def _project_corr(x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    return corr_flat(x.T @ u, y.T @ v)


class MSCCA:
    """Multi-stimulus CCA following Wong et al. 2020.

    The adapter reuses Arena filter-bank epochs and reference signals. For each
    target, it concatenates neighboring class templates and neighboring
    sine-cosine references sorted by stimulus frequency, learns CCA filters,
    then combines test-vs-reference and test-vs-template correlations.
    """

    name = "MSCCA"

    def __init__(
        self,
        window: float,
        n_neighbor: int = 12,
        harmonics: int = 5,
        n_fbs: int = 5,
        spec: BenchmarkSpec | None = None,
    ) -> None:
        self.window = window
        self.n_neighbor = n_neighbor
        self.harmonics = harmonics
        self.n_fbs = n_fbs
        self.spec = spec or BenchmarkSpec()
        self.refs = reference_signals(window, harmonics, self.spec)
        self.weights = filterbank_weights(n_fbs)
        self.templates: np.ndarray | None = None
        self.u: np.ndarray | None = None
        self.v: np.ndarray | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "MSCCA":
        classes = len(self.refs)
        _, n_fbs, channels, _ = train_x.shape
        harmonic_rows = self.refs[0].shape[0]
        self.templates = np.zeros((classes, n_fbs, channels, train_x.shape[-1]), dtype=np.float64)
        self.u = np.zeros((n_fbs, classes, channels), dtype=np.float64)
        self.v = np.zeros((n_fbs, classes, harmonic_rows), dtype=np.float64)
        for cls in range(classes):
            self.templates[cls] = np.mean(train_x[train_y == cls], axis=0)
        for cls in range(classes):
            neighbors = neighbor_indices(BENCHMARK_FREQS, cls, self.n_neighbor)
            joined_refs = np.concatenate([self.refs[idx] for idx in neighbors], axis=-1)
            for fb in range(n_fbs):
                joined_templates = np.concatenate([self.templates[idx, fb] for idx in neighbors], axis=-1)
                u, v = _cca_weights(joined_templates.T, joined_refs.T)
                self.u[fb, cls] = u[:channels]
                self.v[fb, cls] = v[:harmonic_rows]
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.templates is None or self.u is None or self.v is None:
            raise RuntimeError("MSCCA model is not fitted.")
        trials, n_fbs, _, _ = x.shape
        classes = self.templates.shape[0]
        band_scores = np.zeros((trials, n_fbs, classes), dtype=np.float64)
        for trial_idx in range(trials):
            for fb in range(n_fbs):
                trial = x[trial_idx, fb]
                for cls in range(classes):
                    r1 = _project_corr(trial, self.refs[cls], self.u[fb, cls], self.v[fb, cls])
                    r2 = _project_corr(trial, self.templates[cls, fb], self.u[fb, cls], self.u[fb, cls])
                    band_scores[trial_idx, fb, cls] = np.sign(r1) * r1 * r1 + np.sign(r2) * r2 * r2
        scores = np.einsum("f,tfc->tc", self.weights[:n_fbs], band_scores)
        return np.argmax(scores, axis=1), scores


def _trca_parts(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(trials, dtype=np.float64)
    summed = np.sum(x, axis=0)
    stacked = np.transpose(x, (1, 0, 2)).reshape(x.shape[1], -1)
    return summed, stacked


def _solve_aggregate_trca(summed: np.ndarray, stacked: np.ndarray) -> np.ndarray:
    sb = summed @ summed.T - stacked @ stacked.T
    centered = stacked.T - np.mean(stacked.T, axis=0, keepdims=True)
    sw = centered.T @ centered + np.eye(stacked.shape[0]) * 1e-8
    vals, vecs = linalg.eig(sb, sw, check_finite=False)
    order = np.argsort(np.real(vals))[::-1]
    w = np.real(vecs[:, order[0]])
    norm = np.linalg.norm(w)
    return w / norm if norm > 1e-12 else w


class MSTRCA:
    """Multi-stimulus TRCA with target-order neighborhoods.

    For each target and filter-bank band, the spatial filter is estimated from
    the TRCA covariance parts of the target and its neighboring stimuli in an
    explicit class order. Target templates remain class-specific means, so the
    multi-stimulus step only improves the spatial filter estimate.
    """

    name = "MSTRCA"

    def __init__(
        self,
        neighbor_width: int = 2,
        *,
        weights: np.ndarray | None = None,
        n_fbs: int = 5,
        class_order: Sequence[int] | None = None,
        ensemble: bool = True,
    ) -> None:
        self.neighbor_width = int(neighbor_width)
        self.weights = np.asarray(weights if weights is not None else filterbank_weights(n_fbs), dtype=np.float64)
        self.class_order = list(class_order) if class_order is not None else None
        self.ensemble = bool(ensemble)
        self.templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None
        self.neighbors_: list[list[int]] | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "MSTRCA":
        classes = int(np.max(train_y)) + 1
        _, n_fbs, channels, samples = train_x.shape
        if self.weights.size < n_fbs:
            raise ValueError(f"Need at least {n_fbs} filter-bank weights, got {self.weights.size}.")

        class_order = self.class_order if self.class_order is not None else list(range(classes))
        self.templates = np.zeros((classes, n_fbs, channels, samples), dtype=np.float64)
        self.filters = np.zeros((n_fbs, classes, channels), dtype=np.float64)
        self.neighbors_ = []

        class_trials: list[np.ndarray] = []
        for cls in range(classes):
            trials = train_x[train_y == cls]
            if trials.size == 0:
                raise ValueError(f"No training trials for class {cls}.")
            class_trials.append(trials)
            self.templates[cls] = np.mean(trials, axis=0)

        for cls in range(classes):
            self.neighbors_.append(ordered_neighbor_indices(class_order, cls, self.neighbor_width))

        for fb in range(n_fbs):
            parts = [_trca_parts(trials[:, fb]) for trials in class_trials]
            for cls, neighbors in enumerate(self.neighbors_):
                joined_sum = np.concatenate([parts[idx][0] for idx in neighbors], axis=-1)
                joined_stack = np.concatenate([parts[idx][1] for idx in neighbors], axis=-1)
                self.filters[fb, cls] = _solve_aggregate_trca(joined_sum, joined_stack)
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.templates is None or self.filters is None:
            raise RuntimeError("MSTRCA model is not fitted.")
        trials, n_fbs, _, _ = x.shape
        classes = self.templates.shape[0]
        band_scores = np.zeros((trials, n_fbs, classes), dtype=np.float64)
        for fb in range(n_fbs):
            if self.ensemble:
                ensemble_w = self.filters[fb].T
                projected_trials = np.matmul(np.transpose(x[:, fb], (0, 2, 1)), ensemble_w).reshape(trials, -1)
                for cls in range(classes):
                    projected_template = (self.templates[cls, fb].T @ ensemble_w).reshape(-1)
                    band_scores[:, fb, cls] = corr_rows(projected_trials, projected_template)
            else:
                for cls in range(classes):
                    w = self.filters[fb, cls]
                    projected_trials = x[:, fb].transpose(0, 2, 1) @ w
                    projected_template = self.templates[cls, fb].T @ w
                    band_scores[:, fb, cls] = corr_rows(projected_trials, projected_template)
        scores = np.einsum("f,tfc->tc", self.weights[:n_fbs], band_scores)
        return np.argmax(scores, axis=1), scores


class MSETRCA:
    """Multi-stimulus ensemble TRCA following Wong et al. 2020."""

    name = "MSETRCA"

    def __init__(self, n_neighbor: int = 2, n_fbs: int = 5, class_order: Sequence[int] | None = None) -> None:
        self.n_neighbor = n_neighbor
        self.weights = filterbank_weights(n_fbs)
        self.class_order = list(class_order) if class_order is not None else None
        self.templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "MSETRCA":
        classes = int(np.max(train_y)) + 1
        _, n_fbs, channels, samples = train_x.shape
        self.templates = np.zeros((classes, n_fbs, channels, samples), dtype=np.float64)
        self.filters = np.zeros((n_fbs, classes, channels), dtype=np.float64)
        class_trials = []
        for cls in range(classes):
            trials = train_x[train_y == cls]
            class_trials.append(trials)
            self.templates[cls] = np.mean(trials, axis=0)
        for fb in range(n_fbs):
            parts = [_trca_parts(trials[:, fb]) for trials in class_trials]
            for cls in range(classes):
                if self.class_order is None:
                    neighbors = neighbor_indices(BENCHMARK_FREQS, cls, self.n_neighbor)
                else:
                    neighbors = ordered_neighbor_indices(self.class_order, cls, self.n_neighbor)
                joined_sum = np.concatenate([parts[idx][0] for idx in neighbors], axis=-1)
                joined_stack = np.concatenate([parts[idx][1] for idx in neighbors], axis=-1)
                self.filters[fb, cls] = _solve_aggregate_trca(joined_sum, joined_stack)
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.templates is None or self.filters is None:
            raise RuntimeError("MSETRCA model is not fitted.")
        trials, n_fbs, _, _ = x.shape
        classes = self.templates.shape[0]
        band_scores = np.zeros((trials, n_fbs, classes), dtype=np.float64)
        for fb in range(n_fbs):
            ensemble_w = self.filters[fb].T
            projected_trials = np.matmul(np.transpose(x[:, fb], (0, 2, 1)), ensemble_w).reshape(trials, -1)
            for cls in range(classes):
                projected_template = (self.templates[cls, fb].T @ ensemble_w).reshape(-1)
                band_scores[:, fb, cls] = corr_rows(projected_trials, projected_template)
        scores = np.einsum("f,tfc->tc", self.weights[:n_fbs], band_scores)
        return np.argmax(scores, axis=1), scores
