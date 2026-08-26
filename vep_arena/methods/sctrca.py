from __future__ import annotations

import numpy as np

from vep_arena.config import BENCHMARK_FREQS
from vep_arena.methods.traditional import filterbank_weights
from vep_arena.methods.trca_core import _solve_trca, corr_flat


def _standardize_channels(trials: np.ndarray) -> np.ndarray:
    centered = np.asarray(trials, dtype=np.float64) - np.mean(trials, axis=-1, keepdims=True)
    scale = np.std(centered, axis=-1, keepdims=True)
    return centered / np.maximum(scale, 1e-12)


def sine_cosine_reference(
    frequency: float, samples: int, sampling_rate: int, harmonics: int,
) -> np.ndarray:
    time = np.arange(1, samples + 1, dtype=np.float64) / sampling_rate
    rows = []
    for harmonic in range(1, harmonics + 1):
        angle = 2.0 * np.pi * harmonic * frequency * time
        rows.extend((np.sin(angle), np.cos(angle)))
    return np.asarray(rows, dtype=np.float64)


def similarity_constrained_filter(
    trials: np.ndarray, reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = _standardize_channels(trials)
    reference = _standardize_channels(reference[None, ...])[0]
    n_trials, channels, samples = x.shape
    if n_trials < 2:
        raise ValueError("scTRCA requires at least two training trials per class.")
    summed = np.sum(x, axis=0)
    concatenated = x.transpose(1, 0, 2).reshape(channels, -1)
    within = concatenated @ concatenated.T
    s11 = (summed @ summed.T - within) / (n_trials * (n_trials - 1) * samples)
    s12 = (summed @ reference.T) / (n_trials * samples)
    s22 = (reference @ reference.T) / samples
    s = np.block([[s11, s12], [s12.T, s22]])
    q1 = within / (n_trials * samples)
    q = np.zeros_like(s)
    q[:channels, :channels] = q1
    q[channels:, channels:] = s22
    spatial_filter = _solve_trca(s, q)
    return spatial_filter[:channels], spatial_filter[channels:]


class scTRCA:
    name = "scTRCA"

    def __init__(
        self,
        n_fbs: int = 1,
        frequencies: tuple[float, ...] | list[float] = BENCHMARK_FREQS,
        sampling_rate: int = 250,
        harmonics: int = 5,
        ensemble: bool = False,
    ) -> None:
        self.weights = filterbank_weights(n_fbs)
        self.frequencies = tuple(float(value) for value in frequencies)
        self.sampling_rate = int(sampling_rate)
        self.harmonics = int(harmonics)
        self.ensemble = bool(ensemble)
        self.templates: np.ndarray | None = None
        self.spatial_filters: np.ndarray | None = None
        self.reference_filters: np.ndarray | None = None
        self.references: np.ndarray | None = None

    @property
    def filters(self) -> np.ndarray | None:
        return self.spatial_filters

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "scTRCA":
        train_x = np.asarray(train_x, dtype=np.float64)
        train_y = np.asarray(train_y, dtype=np.int64)
        if train_x.ndim != 4:
            raise ValueError("train_x must have shape trials x subbands x channels x samples.")
        classes = int(np.max(train_y)) + 1
        if classes > len(self.frequencies):
            raise ValueError("A stimulation frequency is required for every class.")
        _, n_fbs, channels, samples = train_x.shape
        reference_rows = 2 * self.harmonics
        self.references = np.asarray([
            sine_cosine_reference(
                self.frequencies[cls], samples, self.sampling_rate, self.harmonics
            )
            for cls in range(classes)
        ])
        self.templates = np.zeros((classes, n_fbs, channels, samples), dtype=np.float64)
        self.spatial_filters = np.zeros((n_fbs, classes, channels), dtype=np.float64)
        self.reference_filters = np.zeros((n_fbs, classes, reference_rows), dtype=np.float64)
        for cls in range(classes):
            class_trials = train_x[train_y == cls]
            for fb_idx in range(n_fbs):
                normalized_trials = _standardize_channels(class_trials[:, fb_idx])
                self.templates[cls, fb_idx] = np.mean(normalized_trials, axis=0)
                spatial, reference = similarity_constrained_filter(
                    normalized_trials, self.references[cls]
                )
                self.spatial_filters[fb_idx, cls] = spatial
                self.reference_filters[fb_idx, cls] = reference
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if any(value is None for value in (
            self.templates, self.spatial_filters, self.reference_filters, self.references
        )):
            raise RuntimeError("scTRCA model is not fitted.")
        epochs = np.asarray(x, dtype=np.float64)
        if epochs.ndim != 4:
            raise ValueError("x must have shape trials x subbands x channels x samples.")
        trials, n_fbs, _, _ = epochs.shape
        classes = self.templates.shape[0]
        band_scores = np.zeros((trials, n_fbs, classes), dtype=np.float64)
        for fb_idx in range(n_fbs):
            normalized_epochs = _standardize_channels(epochs[:, fb_idx])
            for cls in range(classes):
                if self.ensemble:
                    spatial = self.spatial_filters[fb_idx].T
                    reference_filter = self.reference_filters[fb_idx].T
                else:
                    spatial = self.spatial_filters[fb_idx, cls][:, None]
                    reference_filter = self.reference_filters[fb_idx, cls][:, None]
                projected_template = (spatial.T @ self.templates[cls, fb_idx]).reshape(-1)
                projected_reference = (reference_filter.T @ self.references[cls]).reshape(-1)
                for trial_idx in range(trials):
                    projected_trial = (spatial.T @ normalized_epochs[trial_idx]).reshape(-1)
                    correlations = (
                        corr_flat(projected_trial, projected_template),
                        corr_flat(projected_trial, projected_reference),
                    )
                    band_scores[trial_idx, fb_idx, cls] = sum(
                        np.sign(value) * value * value for value in correlations
                    )
        scores = np.einsum("f,tfc->tc", self.weights[:n_fbs], band_scores)
        return np.argmax(scores, axis=1), scores
