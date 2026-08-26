from __future__ import annotations

import numpy as np

from vep_arena.config import BENCHMARK_FREQS
from vep_arena.methods.trca_core import _solve_trca, trca_scores


def _normalize_trials(trials: np.ndarray) -> np.ndarray:
    x = np.asarray(trials, dtype=np.float64)
    x = x - np.mean(x, axis=-1, keepdims=True)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 1e-12)


def _reference(frequency: float, samples: int, sampling_rate: int, harmonics: int) -> np.ndarray:
    time = np.arange(1, samples + 1, dtype=np.float64) / sampling_rate
    rows = []
    for harmonic in range(1, harmonics + 1):
        rows.extend(
            [
                np.sin(2 * np.pi * harmonic * frequency * time),
                np.cos(2 * np.pi * harmonic * frequency * time),
            ]
        )
    return np.asarray(rows)


def sinusoidal_referenced_filter(
    trials: np.ndarray,
    reference: np.ndarray,
    weighting_factor: float = 0.7,
) -> np.ndarray:
    x = _normalize_trials(trials)
    reference = reference - np.mean(reference, axis=-1, keepdims=True)
    n_trials, channels, samples = x.shape
    if n_trials < 2:
        raise ValueError("sinusoidal-referenced TRCA requires at least two training trials")
    summed = np.sum(x, axis=0)
    within = sum(trial @ trial.T for trial in x)
    inter_trial = summed @ summed.T - within
    cross = sum(trial @ reference.T for trial in x)
    q1 = within / (n_trials * samples)
    q2 = reference @ reference.T / samples
    size = channels + reference.shape[0]
    objective = np.zeros((size, size), dtype=np.float64)
    constraint = np.zeros_like(objective)
    objective[:channels, :channels] = weighting_factor * inter_trial
    objective[:channels, channels:] = (1.0 - weighting_factor) * cross / 2.0
    objective[channels:, :channels] = objective[:channels, channels:].T
    constraint[:channels, :channels] = q1
    constraint[channels:, channels:] = q2
    solution = _solve_trca(objective, constraint)
    spatial_filter = solution[:channels]
    norm = np.linalg.norm(spatial_filter)
    return spatial_filter / norm if norm > 1e-12 else spatial_filter


class SinusoidalReferencedTRCA:
    """Sinusoidal-referenced TRCA (srTRCA)."""

    name = "SRTRCA"

    def __init__(
        self,
        n_fbs: int = 1,
        *,
        frequencies: tuple[float, ...] | list[float] = BENCHMARK_FREQS,
        sampling_rate: int = 250,
        harmonics: int = 5,
        weighting_factor: float = 0.7,
        ensemble: bool = False,
    ) -> None:
        if not 0.0 <= weighting_factor <= 1.0:
            raise ValueError("weighting_factor must be in [0, 1]")
        self.weights = np.asarray(
            [(index + 1) ** (-1.25) + 0.25 for index in range(n_fbs)], dtype=np.float64
        )
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.sampling_rate = int(sampling_rate)
        self.harmonics = int(harmonics)
        self.weighting_factor = float(weighting_factor)
        self.ensemble = bool(ensemble)
        self.templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "SinusoidalReferencedTRCA":
        x = np.asarray(train_x, dtype=np.float64)
        y = np.asarray(train_y, dtype=np.int64)
        if x.ndim != 4:
            raise ValueError("train_x must have shape trials x subbands x channels x samples")
        classes = int(np.max(y)) + 1
        _, n_fbs, channels, samples = x.shape
        if classes > len(self.frequencies):
            raise ValueError("frequencies does not cover every training class")
        normalized = _normalize_trials(x)
        self.templates = np.zeros((classes, n_fbs, channels, samples), dtype=np.float64)
        self.filters = np.zeros((n_fbs, classes, channels), dtype=np.float64)
        for cls in range(classes):
            class_trials = normalized[y == cls]
            self.templates[cls] = np.mean(class_trials, axis=0)
            reference = _reference(
                float(self.frequencies[cls]), samples, self.sampling_rate, self.harmonics
            )
            for fb_idx in range(n_fbs):
                self.filters[fb_idx, cls] = sinusoidal_referenced_filter(
                    class_trials[:, fb_idx], reference, self.weighting_factor
                )
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.templates is None or self.filters is None:
            raise RuntimeError("sinusoidal-referenced TRCA model is not fitted.")
        scores = trca_scores(
            _normalize_trials(x), self.templates, self.filters, self.weights,
            ensemble=self.ensemble,
        )
        return np.argmax(scores, axis=1), scores
