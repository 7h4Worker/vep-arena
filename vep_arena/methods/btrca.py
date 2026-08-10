from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vep_arena.methods.traditional import filterbank_weights
from vep_arena.methods.trca_core import _solve_trca, corr_rows


def _center_rows(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=-1, keepdims=True)


def _within_cov(trials: np.ndarray, template: np.ndarray) -> np.ndarray:
    channels = int(template.shape[0])
    cov = np.zeros((channels, channels), dtype=np.float64)
    for trial in trials:
        diff = trial - template
        cov += diff @ diff.T
    return cov


class BTRCA:
    """Binocular TRCA for swapped dual-frequency target pairs.

    The Sun2024 task groups two targets that swap the left/right-eye
    frequencies. For each target, the discriminative numerator is the
    symmetrized covariance shared with its swapped partner. The default fit
    pools the denominator across the target's O/S group, following the paper's
    Eq. (9) direction, while still using a leading-vector TRCA approximation.
    """

    name = "BTRCA"

    def __init__(
        self,
        pair_indices: Sequence[tuple[int, int]],
        n_fbs: int = 5,
        *,
        ensemble: bool = False,
    ) -> None:
        self.pair_indices = tuple((int(a), int(b)) for a, b in pair_indices)
        self.weights = filterbank_weights(n_fbs)
        self.ensemble = bool(ensemble)
        self.templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "BTRCA":
        classes = int(np.max(train_y)) + 1
        _, n_fbs, channels, samples = train_x.shape
        self.templates = np.zeros((classes, n_fbs, channels, samples), dtype=np.float64)
        self.filters = np.zeros((n_fbs, classes, channels), dtype=np.float64)

        centered = _center_rows(np.asarray(train_x, dtype=np.float64))
        for cls in range(classes):
            class_trials = centered[train_y == cls]
            if class_trials.size == 0:
                raise ValueError(f"No training trials for class {cls}.")
            self.templates[cls] = np.mean(class_trials, axis=0)

        group_o = tuple(cls_a for cls_a, _ in self.pair_indices)
        group_s = tuple(cls_b for _, cls_b in self.pair_indices)
        seen = set(group_o) | set(group_s)
        missing = sorted(set(range(classes)) - seen)
        if missing:
            raise ValueError(f"BTRCA pair map does not cover classes: {missing}")

        for fb_idx in range(n_fbs):
            sw_o = np.zeros((channels, channels), dtype=np.float64)
            sw_s = np.zeros((channels, channels), dtype=np.float64)
            for cls in group_o:
                sw_o += _within_cov(centered[train_y == cls, fb_idx], self.templates[cls, fb_idx])
            for cls in group_s:
                sw_s += _within_cov(centered[train_y == cls, fb_idx], self.templates[cls, fb_idx])

            for cls_a, cls_b in self.pair_indices:
                if cls_a >= classes or cls_b >= classes:
                    raise ValueError(f"Pair {(cls_a, cls_b)} exceeds fitted class count {classes}.")
                trials_a = centered[train_y == cls_a, fb_idx]
                trials_b = centered[train_y == cls_b, fb_idx]
                sum_a = np.sum(trials_a, axis=0)
                sum_b = np.sum(trials_b, axis=0)
                sb = sum_a @ sum_b.T + sum_b @ sum_a.T
                self.filters[fb_idx, cls_a] = _solve_trca(sb, sw_o)
                self.filters[fb_idx, cls_b] = _solve_trca(sb, sw_s)

        return self

    def fit_target_noise_variant(self, train_x: np.ndarray, train_y: np.ndarray) -> "BTRCA":
        """Fit the first, target-local noise variant retained for diagnostics."""

        classes = int(np.max(train_y)) + 1
        _, n_fbs, channels, samples = train_x.shape
        self.templates = np.zeros((classes, n_fbs, channels, samples), dtype=np.float64)
        self.filters = np.zeros((n_fbs, classes, channels), dtype=np.float64)
        centered = _center_rows(np.asarray(train_x, dtype=np.float64))

        seen: set[int] = set()
        for cls_a, cls_b in self.pair_indices:
            if cls_a >= classes or cls_b >= classes:
                raise ValueError(f"Pair {(cls_a, cls_b)} exceeds fitted class count {classes}.")
            seen.update((cls_a, cls_b))
            trials_a = centered[train_y == cls_a]
            trials_b = centered[train_y == cls_b]
            self.templates[cls_a] = np.mean(trials_a, axis=0)
            self.templates[cls_b] = np.mean(trials_b, axis=0)
            for fb_idx in range(n_fbs):
                template_a = self.templates[cls_a, fb_idx]
                template_b = self.templates[cls_b, fb_idx]
                sum_a = np.sum(trials_a[:, fb_idx], axis=0)
                sum_b = np.sum(trials_b[:, fb_idx], axis=0)
                sb = sum_a @ sum_b.T + sum_b @ sum_a.T
                sw_a = _within_cov(trials_a[:, fb_idx], template_a)
                sw_b = _within_cov(trials_b[:, fb_idx], template_b)
                self.filters[fb_idx, cls_a] = _solve_trca(sb, sw_a)
                self.filters[fb_idx, cls_b] = _solve_trca(sb, sw_b)

        missing = sorted(set(range(classes)) - seen)
        if missing:
            raise ValueError(f"BTRCA pair map does not cover classes: {missing}")
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.templates is None or self.filters is None:
            raise RuntimeError("BTRCA model is not fitted.")
        trials, n_fbs, _, _ = x.shape
        classes = self.templates.shape[0]
        band_scores = np.zeros((trials, n_fbs, classes), dtype=np.float64)
        x = _center_rows(np.asarray(x, dtype=np.float64))

        for fb_idx in range(n_fbs):
            if self.ensemble:
                w = self.filters[fb_idx].T
                projected_trials = np.matmul(np.transpose(x[:, fb_idx], (0, 2, 1)), w).reshape(trials, -1)
                for cls in range(classes):
                    projected_template = (self.templates[cls, fb_idx].T @ w).reshape(-1)
                    band_scores[:, fb_idx, cls] = corr_rows(projected_trials, projected_template)
            else:
                for cls in range(classes):
                    w = self.filters[fb_idx, cls]
                    projected_trials = x[:, fb_idx].transpose(0, 2, 1) @ w
                    projected_template = self.templates[cls, fb_idx].T @ w
                    band_scores[:, fb_idx, cls] = corr_rows(projected_trials, projected_template)

        scores = np.einsum("f,tfc->tc", self.weights[:n_fbs], band_scores)
        return np.argmax(scores, axis=1), scores
