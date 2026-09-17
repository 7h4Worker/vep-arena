from __future__ import annotations

import numpy as np

from vep_arena.config import BENCHMARK_FREQS
from vep_arena.methods.trca_core import _solve_trca, corr_flat


def temporal_laplacian(samples: int, local_range: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(samples, dtype=np.float64)
    distance = (indices[None, :] - indices[:, None]) / float(local_range)
    weights = np.zeros_like(distance)
    keep = np.abs(distance) < 1.0
    weights[keep] = (1.0 - np.abs(distance[keep]) ** 3) ** 3
    laplacian = np.diag(np.sum(weights, axis=1)) - weights
    values, vectors = np.linalg.eigh((laplacian + laplacian.T) * 0.5)
    positive = values > 1e-10
    temporal_filter = vectors[:, positive] * np.sqrt(values[positive])[None, :]
    return laplacian, temporal_filter


def sine_references(
    frequencies: np.ndarray, samples: int, sampling_rate: int, harmonics: int
) -> np.ndarray:
    time = np.arange(1, samples + 1, dtype=np.float64) / sampling_rate
    references = np.empty((len(frequencies), 2 * harmonics, samples), dtype=np.float64)
    for cls, frequency in enumerate(frequencies):
        rows = []
        for harmonic in range(1, harmonics + 1):
            rows.extend(
                [
                    np.sin(2 * np.pi * harmonic * frequency * time),
                    np.cos(2 * np.pi * harmonic * frequency * time),
                ]
            )
        references[cls] = np.asarray(rows)
    return references


def strca_filter(
    trials: np.ndarray, reference: np.ndarray, laplacian: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(trials, dtype=np.float64)
    x = x - np.mean(x, axis=-1, keepdims=True)
    reference = reference - np.mean(reference, axis=-1, keepdims=True)
    n_trials, channels, samples = x.shape
    if n_trials < 2:
        raise ValueError("sTRCA requires at least two training trials")
    summed = np.sum(x, axis=0)
    within = sum(trial @ laplacian @ trial.T for trial in x)
    s11 = (summed @ laplacian @ summed.T - within) / (
        n_trials * (n_trials - 1) * samples**2
    )
    s12 = sum(trial @ laplacian @ reference.T for trial in x) / (
        n_trials * samples**2
    )
    s22 = reference @ reference.T / samples
    q1 = within / (n_trials * samples**2)
    q2 = s22.copy()
    size = channels + reference.shape[0]
    objective = np.zeros((size, size), dtype=np.float64)
    constraint = np.zeros_like(objective)
    objective[:channels, :channels] = s11
    objective[:channels, channels:] = s12
    objective[channels:, :channels] = s12.T
    objective[channels:, channels:] = s22
    constraint[:channels, :channels] = q1
    constraint[channels:, channels:] = q2
    spatial_filter = _solve_trca(objective, constraint)
    return spatial_filter[:channels], spatial_filter[channels:]


class sTRCA:
    """Similarity-constrained temporally local TRCA (stTRCA in the paper)."""

    name = "STRCA"

    def __init__(
        self,
        n_fbs: int = 5,
        *,
        frequencies: tuple[float, ...] | list[float] = BENCHMARK_FREQS,
        sampling_rate: int = 250,
        harmonics: int = 5,
        local_range: float = 5.0,
    ) -> None:
        self.weights = np.asarray(
            [(index + 1) ** (-1.25) + 0.25 for index in range(n_fbs)], dtype=np.float64
        )
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.sampling_rate = int(sampling_rate)
        self.harmonics = int(harmonics)
        self.local_range = float(local_range)
        self.templates: np.ndarray | None = None
        self.references: np.ndarray | None = None
        self.eeg_filters: np.ndarray | None = None
        self.reference_filters: np.ndarray | None = None
        self.temporal_filter: np.ndarray | None = None
        self.projected_templates: np.ndarray | None = None
        self.projected_references: np.ndarray | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "sTRCA":
        x = np.asarray(train_x, dtype=np.float64)
        y = np.asarray(train_y, dtype=np.int64)
        if x.ndim != 4:
            raise ValueError("train_x must have shape trials x subbands x channels x samples")
        classes = int(np.max(y)) + 1
        if classes > len(self.frequencies):
            raise ValueError("frequencies does not cover every training class")
        _, n_fbs, channels, samples = x.shape
        laplacian, self.temporal_filter = temporal_laplacian(samples, self.local_range)
        self.references = sine_references(
            self.frequencies[:classes], samples, self.sampling_rate, self.harmonics
        )
        self.templates = np.zeros((classes, n_fbs, channels, samples), dtype=np.float64)
        self.eeg_filters = np.zeros((n_fbs, classes, channels), dtype=np.float64)
        self.reference_filters = np.zeros((n_fbs, classes, 2 * self.harmonics), dtype=np.float64)
        for cls in range(classes):
            class_trials = x[y == cls]
            self.templates[cls] = np.mean(class_trials, axis=0)
            for fb_idx in range(n_fbs):
                eeg_filter, reference_filter = strca_filter(
                    class_trials[:, fb_idx], self.references[cls], laplacian
                )
                self.eeg_filters[fb_idx, cls] = eeg_filter
                self.reference_filters[fb_idx, cls] = reference_filter
        temporal_components = self.temporal_filter.shape[1]
        self.projected_templates = np.zeros(
            (n_fbs, classes, temporal_components), dtype=np.float64
        )
        self.projected_references = np.zeros_like(self.projected_templates)
        for fb_idx in range(n_fbs):
            for cls in range(classes):
                self.projected_templates[fb_idx, cls] = (
                    self.eeg_filters[fb_idx, cls]
                    @ self.templates[cls, fb_idx]
                    @ self.temporal_filter
                )
                self.projected_references[fb_idx, cls] = (
                    self.reference_filters[fb_idx, cls]
                    @ self.references[cls]
                    @ self.temporal_filter
                )
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if any(
            value is None
            for value in (
                self.templates, self.references, self.eeg_filters,
                self.reference_filters, self.temporal_filter,
                self.projected_templates, self.projected_references,
            )
        ):
            raise RuntimeError("sTRCA model is not fitted.")
        epochs = np.asarray(x, dtype=np.float64)
        if epochs.shape[-1] != self.templates.shape[-1]:
            raise ValueError("sTRCA test and training windows must have equal length")
        band_scores = np.zeros(
            (epochs.shape[0], epochs.shape[1], self.templates.shape[0]), dtype=np.float64
        )
        for trial_idx, trial in enumerate(epochs):
            for fb_idx, band in enumerate(trial):
                temporally_filtered = band @ self.temporal_filter
                for cls in range(self.templates.shape[0]):
                    eeg_filter = self.eeg_filters[fb_idx, cls]
                    projected_trial = eeg_filter @ temporally_filtered
                    correlations = (
                        corr_flat(projected_trial, self.projected_templates[fb_idx, cls]),
                        corr_flat(projected_trial, self.projected_references[fb_idx, cls]),
                    )
                    band_scores[trial_idx, fb_idx, cls] = sum(
                        np.sign(value) * value**2 for value in correlations
                    )
        scores = np.einsum("f,tfc->tc", self.weights[: epochs.shape[1]], band_scores)
        return np.argmax(scores, axis=1), scores
