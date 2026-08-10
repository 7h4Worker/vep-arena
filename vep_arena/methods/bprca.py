from __future__ import annotations

import numpy as np

from vep_arena.methods.traditional import filterbank_weights
from vep_arena.methods.trca_core import _solve_trca, corr_flat


def _period_segments(trials: np.ndarray, period_samples: int) -> np.ndarray:
    n_trials, channels, samples = trials.shape
    periods_per_trial = samples // period_samples
    if periods_per_trial < 1:
        return trials.copy()
    used = periods_per_trial * period_samples
    return (
        trials[:, :, :used]
        .reshape(n_trials, channels, periods_per_trial, period_samples)
        .transpose(0, 2, 1, 3)
        .reshape(n_trials * periods_per_trial, channels, period_samples)
    )


def _prca_filter(trials: np.ndarray, period_samples: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(trials, dtype=np.float64)
    segments = _period_segments(data, period_samples)
    centered_segments = segments - np.mean(segments, axis=-1, keepdims=True)
    summed = np.sum(centered_segments, axis=0)
    segment_matrix = centered_segments.transpose(1, 0, 2).reshape(data.shape[1], -1)
    within_segments = segment_matrix @ segment_matrix.T
    between_segments = summed @ summed.T - within_segments

    spatial_filter = _solve_trca(between_segments, within_segments)
    prototype = np.mean(segments, axis=0)
    return spatial_filter, prototype


def _repeat_template(prototype: np.ndarray, samples: int) -> np.ndarray:
    repeats = int(np.ceil(samples / prototype.shape[-1]))
    return np.tile(prototype, (1, repeats))[:, :samples]


class BPRCA:
    name = "BPRCA"

    def __init__(
        self,
        frequencies: tuple[float, ...] | list[float],
        sampling_rate: int,
        n_fbs: int = 5,
        *,
        ensemble: bool = False,
        fusion: bool = False,
    ) -> None:
        if not frequencies or any(float(frequency) <= 0 for frequency in frequencies):
            raise ValueError("bPRCA frequency units must be positive.")
        self.frequencies = tuple(float(frequency) for frequency in frequencies)
        self.sampling_rate = int(sampling_rate)
        self.weights = filterbank_weights(n_fbs)
        self.ensemble = bool(ensemble)
        self.fusion = bool(fusion)
        self.templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None
        self.period_samples: tuple[int, ...] | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "BPRCA":
        x = np.asarray(train_x, dtype=np.float64)
        y = np.asarray(train_y, dtype=np.int64)
        classes = int(np.max(y)) + 1
        _, n_fbs, channels, samples = x.shape
        periods = [min(samples, int(round(self.sampling_rate / frequency))) for frequency in self.frequencies]
        if self.fusion:
            periods.append(samples)
        self.period_samples = tuple(periods)
        self.templates = np.zeros((classes, len(periods), n_fbs, channels, samples), dtype=np.float64)
        self.filters = np.zeros((len(periods), n_fbs, classes, channels), dtype=np.float64)

        for cls in range(classes):
            class_trials = x[y == cls]
            if class_trials.shape[0] < 2:
                raise ValueError("bPRCA requires at least two calibration trials per class.")
            for stream_idx, period_samples in enumerate(periods):
                for fb_idx in range(n_fbs):
                    spatial_filter, prototype = _prca_filter(class_trials[:, fb_idx], period_samples)
                    self.filters[stream_idx, fb_idx, cls] = spatial_filter
                    self.templates[cls, stream_idx, fb_idx] = _repeat_template(prototype, samples)
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.templates is None or self.filters is None:
            raise RuntimeError("bPRCA model is not fitted.")
        trials = np.asarray(x, dtype=np.float64)
        classes, streams, n_fbs, _, samples = self.templates.shape
        if trials.shape[1] != n_fbs or trials.shape[-1] != samples:
            raise ValueError("bPRCA test epochs must match the fitted filter-bank and sample dimensions.")
        scores = np.zeros((trials.shape[0], classes), dtype=np.float64)

        for trial_idx, trial in enumerate(trials):
            for cls in range(classes):
                for stream_idx in range(streams):
                    for fb_idx in range(n_fbs):
                        if self.ensemble:
                            spatial_filters = self.filters[stream_idx, fb_idx]
                            projected_trial = spatial_filters @ trial[fb_idx]
                            projected_template = spatial_filters @ self.templates[cls, stream_idx, fb_idx]
                        else:
                            spatial_filter = self.filters[stream_idx, fb_idx, cls]
                            projected_trial = spatial_filter @ trial[fb_idx]
                            projected_template = spatial_filter @ self.templates[cls, stream_idx, fb_idx]
                        scores[trial_idx, cls] += self.weights[fb_idx] * corr_flat(
                            projected_trial, projected_template
                        )
        return np.argmax(scores, axis=1), scores
