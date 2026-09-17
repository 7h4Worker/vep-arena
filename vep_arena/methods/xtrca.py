from __future__ import annotations

import numpy as np

from vep_arena.methods.trca_core import corr_flat, trca_filter


def _normalize(trials: np.ndarray) -> np.ndarray:
    x = np.asarray(trials, dtype=np.float64)
    x = x - np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return np.divide(x, std, out=np.zeros_like(x), where=std > 1e-12)


def _shift_trials(trials: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    return np.asarray([np.roll(trial, -int(shift), axis=-1) for trial, shift in zip(trials, shifts)])


def _reproducibility(trials: np.ndarray, spatial_filter: np.ndarray) -> float:
    x = _normalize(trials)
    projected = x.transpose(0, 2, 1) @ spatial_filter
    summed = np.sum(projected, axis=0)
    within = float(np.sum(projected**2))
    between = float(summed @ summed - within)
    return between / max(within, 1e-12)


def xtrca_filter(
    trials: np.ndarray,
    *,
    search_samples: int = 5,
    max_cycles: int = 5,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _normalize(trials)
    n_trials = x.shape[0]
    if n_trials < 2:
        raise ValueError("xTRCA requires at least two training trials")
    shifts = np.zeros(n_trials, dtype=np.int64)
    previous = -np.inf
    candidates = np.arange(-int(search_samples), int(search_samples) + 1)
    for _ in range(int(max_cycles)):
        aligned = _shift_trials(x, shifts)
        spatial_filter = trca_filter(aligned)
        for trial_idx in range(n_trials):
            other_template = np.mean(
                np.delete(aligned, trial_idx, axis=0), axis=0
            )
            projected_template = spatial_filter @ other_template
            scores = []
            for candidate in candidates:
                shifted = np.roll(x[trial_idx], -int(candidate), axis=-1)
                scores.append(float((spatial_filter @ shifted) @ projected_template))
            shifts[trial_idx] = int(candidates[int(np.argmax(scores))])
            aligned[trial_idx] = np.roll(x[trial_idx], -shifts[trial_idx], axis=-1)
        value = _reproducibility(aligned, spatial_filter)
        if np.isfinite(previous) and abs(value - previous) / max(abs(value), 1e-12) < tolerance:
            break
        previous = value
    shifts = np.rint(shifts - np.mean(shifts)).astype(np.int64)
    shifts = np.clip(shifts, -search_samples, search_samples)
    aligned = _shift_trials(x, shifts)
    spatial_filter = trca_filter(aligned)
    return spatial_filter, shifts, aligned


class xTRCA:
    """Cross-correlation TRCA adapted to periodic SSVEP epochs."""

    name = "XTRCA"

    def __init__(
        self,
        n_fbs: int = 5,
        *,
        search_samples: int = 5,
        max_cycles: int = 5,
        tolerance: float = 1e-4,
    ) -> None:
        self.weights = np.asarray(
            [(index + 1) ** (-1.25) + 0.25 for index in range(n_fbs)], dtype=np.float64
        )
        self.search_samples = int(search_samples)
        self.max_cycles = int(max_cycles)
        self.tolerance = float(tolerance)
        self.templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None
        self.training_shifts: list[np.ndarray] | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "xTRCA":
        x = _normalize(train_x)
        y = np.asarray(train_y, dtype=np.int64)
        if x.ndim != 4:
            raise ValueError("train_x must have shape trials x subbands x channels x samples")
        classes = int(np.max(y)) + 1
        _, n_fbs, channels, samples = x.shape
        self.templates = np.zeros((classes, n_fbs, channels, samples), dtype=np.float64)
        self.filters = np.zeros((n_fbs, classes, channels), dtype=np.float64)
        self.training_shifts = []
        for cls in range(classes):
            class_trials = x[y == cls]
            _, shifts, _ = xtrca_filter(
                class_trials[:, 0], search_samples=self.search_samples,
                max_cycles=self.max_cycles, tolerance=self.tolerance,
            )
            aligned = _shift_trials(class_trials, shifts)
            self.training_shifts.append(shifts)
            self.templates[cls] = np.mean(aligned, axis=0)
            for fb_idx in range(n_fbs):
                self.filters[fb_idx, cls] = trca_filter(aligned[:, fb_idx])
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.templates is None or self.filters is None:
            raise RuntimeError("xTRCA model is not fitted.")
        epochs = _normalize(x)
        scores = np.zeros((epochs.shape[0], self.templates.shape[0]), dtype=np.float64)
        candidates = range(-self.search_samples, self.search_samples + 1)
        for trial_idx, trial in enumerate(epochs):
            for cls in range(self.templates.shape[0]):
                best_score = -np.inf
                for shift in candidates:
                    shifted = np.roll(trial, -shift, axis=-1)
                    band_score = 0.0
                    for fb_idx in range(trial.shape[0]):
                        spatial_filter = self.filters[fb_idx, cls]
                        correlation = corr_flat(
                            spatial_filter @ shifted[fb_idx],
                            spatial_filter @ self.templates[cls, fb_idx],
                        )
                        band_score += self.weights[fb_idx] * correlation
                    best_score = max(best_score, band_score)
                scores[trial_idx, cls] = best_score
        return np.argmax(scores, axis=1), scores
