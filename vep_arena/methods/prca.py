from __future__ import annotations

import numpy as np

from vep_arena.config import BENCHMARK_FREQS
from vep_arena.methods.trca_core import trca_filter, trca_scores


def periodic_components(
    trials: np.ndarray, target_frequency: float, sampling_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return period trials and their mean PRC.

    Input is trials x channels x samples; output periods is
    (trials * complete periods) x channels x period samples.
    """

    x = np.asarray(trials, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("trials must have shape trials x channels x samples")
    period_samples = int(round(sampling_rate / float(target_frequency)))
    periods_per_trial = int(np.floor(x.shape[-1] * float(target_frequency) / sampling_rate))
    if periods_per_trial < 2:
        raise ValueError("PRCA requires at least two complete stimulation periods")
    usable = periods_per_trial * period_samples
    if usable > x.shape[-1]:
        periods_per_trial -= 1
        usable = periods_per_trial * period_samples
    periods = x[..., :usable].reshape(x.shape[0], x.shape[1], periods_per_trial, period_samples)
    periods = periods.transpose(0, 2, 1, 3).reshape(-1, x.shape[1], period_samples)
    return periods, np.mean(periods, axis=0)


def synthetic_template(period_template: np.ndarray, samples: int) -> np.ndarray:
    repeats = int(np.ceil(samples / period_template.shape[-1]))
    return np.tile(period_template, (1, repeats))[:, :samples]


class PRCA:
    """Periodically repeated component analysis for SSVEP classification."""

    name = "PRCA"

    def __init__(
        self,
        n_fbs: int = 5,
        *,
        frequencies: tuple[float, ...] | list[float] = BENCHMARK_FREQS,
        sampling_rate: int = 250,
        ensemble: bool = False,
    ) -> None:
        self.weights = np.asarray(
            [(index + 1) ** (-1.25) + 0.25 for index in range(n_fbs)], dtype=np.float64
        )
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.sampling_rate = int(sampling_rate)
        self.ensemble = bool(ensemble)
        self.templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "PRCA":
        x = np.asarray(train_x, dtype=np.float64)
        y = np.asarray(train_y, dtype=np.int64)
        if x.ndim != 4:
            raise ValueError("train_x must have shape trials x subbands x channels x samples")
        classes = int(np.max(y)) + 1
        if classes > len(self.frequencies):
            raise ValueError("frequencies does not cover every training class")
        _, n_fbs, channels, samples = x.shape
        self.templates = np.zeros((classes, n_fbs, channels, samples), dtype=np.float64)
        self.filters = np.zeros((n_fbs, classes, channels), dtype=np.float64)
        for cls in range(classes):
            class_trials = x[y == cls]
            if not len(class_trials):
                raise ValueError(f"training data is missing class {cls}")
            for fb_idx in range(n_fbs):
                periods, period_template = periodic_components(
                    class_trials[:, fb_idx], float(self.frequencies[cls]), self.sampling_rate
                )
                self.filters[fb_idx, cls] = trca_filter(periods)
                self.templates[cls, fb_idx] = synthetic_template(period_template, samples)
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.templates is None or self.filters is None:
            raise RuntimeError("PRCA model is not fitted.")
        epochs = np.asarray(x, dtype=np.float64)
        if epochs.shape[-1] != self.templates.shape[-1]:
            templates = np.empty((*self.templates.shape[:-1], epochs.shape[-1]), dtype=np.float64)
            for cls in range(self.templates.shape[0]):
                period_samples = int(round(self.sampling_rate / float(self.frequencies[cls])))
                for fb_idx in range(self.templates.shape[1]):
                    templates[cls, fb_idx] = synthetic_template(
                        self.templates[cls, fb_idx, :, :period_samples], epochs.shape[-1]
                    )
        else:
            templates = self.templates
        scores = trca_scores(
            epochs, templates, self.filters, self.weights, ensemble=self.ensemble
        )
        return np.argmax(scores, axis=1), scores
