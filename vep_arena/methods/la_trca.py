from __future__ import annotations

import numpy as np
from scipy import signal

from vep_arena.config import BENCHMARK_FREQS
from vep_arena.methods.trca_core import corr_rows, trca_filter


BENCHMARK_9CH_COORDINATES = np.asarray(
    [
        (0.000325, -0.081115, 0.082615),
        (-0.048424, -0.099341, 0.021599),
        (-0.036511, -0.100853, 0.037167),
        (0.000216, -0.102178, 0.050608),
        (0.036782, -0.100849, 0.036397),
        (0.049820, -0.099446, 0.021727),
        (-0.029413, -0.112449, 0.008839),
        (0.000108, -0.114892, 0.014657),
        (0.029843, -0.112156, 0.008800),
    ],
    dtype=np.float64,
)


def align_channels(x: np.ndarray, delays_seconds: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Advance delayed channels using linear interpolation and edge hold."""

    data = np.asarray(x, dtype=np.float64)
    samples = data.shape[-1]
    base = np.arange(samples, dtype=np.float64)
    aligned = np.empty_like(data)
    for channel, delay in enumerate(delays_seconds):
        source = base + float(delay) * sampling_rate
        aligned[..., channel, :] = np.apply_along_axis(
            lambda row: np.interp(source, base, row, left=row[0], right=row[-1]),
            -1,
            data[..., channel, :],
        )
    return aligned


def estimate_wave_delays(
    trials: np.ndarray,
    target_frequency: float,
    sampling_rate: int,
    coordinates: np.ndarray = BENCHMARK_9CH_COORDINATES,
    source_channel: int = 3,
) -> np.ndarray:
    x = np.mean(np.asarray(trials, dtype=np.float64), axis=0)
    sos = signal.butter(4, (9.0, 15.0), btype="bandpass", fs=sampling_rate, output="sos")
    padlen = min(x.shape[-1] - 1, 3 * (2 * sos.shape[0] + 1))
    narrow = signal.sosfiltfilt(sos, x, axis=-1, padlen=padlen)
    phases = np.angle(signal.hilbert(narrow, axis=-1))
    phase_difference = np.angle(
        np.mean(np.exp(1j * (phases[source_channel] - phases)), axis=-1)
    )
    distances = np.linalg.norm(coordinates - coordinates[source_channel], axis=1)
    valid = (distances > 1e-12) & (np.abs(phase_difference) > 1e-4)
    velocity_terms = 2 * np.pi * float(target_frequency) * distances[valid] / np.abs(
        phase_difference[valid]
    )
    velocity = float(np.mean(velocity_terms)) if velocity_terms.size else np.inf
    delays = np.zeros(x.shape[0], dtype=np.float64)
    if np.isfinite(velocity) and velocity > 0:
        delays = distances / velocity
    max_delay = 0.5 / float(target_frequency)
    return np.clip(delays, 0.0, max_delay)


class LATRCA:
    """Latency-aligning TRCA using the paper's wave-propagation model."""

    name = "LATRCA"

    def __init__(
        self,
        n_fbs: int = 5,
        *,
        frequencies: tuple[float, ...] | list[float] = BENCHMARK_FREQS,
        sampling_rate: int = 250,
        coordinates: np.ndarray = BENCHMARK_9CH_COORDINATES,
        source_channel: int = 3,
    ) -> None:
        self.weights = np.asarray(
            [(index + 1) ** (-1.25) + 0.25 for index in range(n_fbs)], dtype=np.float64
        )
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.sampling_rate = int(sampling_rate)
        self.coordinates = np.asarray(coordinates, dtype=np.float64)
        self.source_channel = int(source_channel)
        self.templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None
        self.delays: np.ndarray | None = None
        self._preset_delays: np.ndarray | None = None

    def estimate_delays(self, train_x: np.ndarray, train_y: np.ndarray) -> "LATRCA":
        x = np.asarray(train_x, dtype=np.float64)
        y = np.asarray(train_y, dtype=np.int64)
        classes = int(np.max(y)) + 1
        self._preset_delays = np.zeros((classes, x.shape[2]), dtype=np.float64)
        for cls in range(classes):
            self._preset_delays[cls] = estimate_wave_delays(
                x[y == cls, 0], float(self.frequencies[cls]), self.sampling_rate,
                self.coordinates, self.source_channel,
            )
        return self

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "LATRCA":
        x = np.asarray(train_x, dtype=np.float64)
        y = np.asarray(train_y, dtype=np.int64)
        if x.ndim != 4:
            raise ValueError("train_x must have shape trials x subbands x channels x samples")
        classes = int(np.max(y)) + 1
        _, n_fbs, channels, samples = x.shape
        if self.coordinates.shape != (channels, 3):
            raise ValueError("coordinates must have shape channels x 3")
        if classes > len(self.frequencies):
            raise ValueError("frequencies does not cover every training class")
        self.templates = np.zeros((classes, n_fbs, channels, samples), dtype=np.float64)
        self.filters = np.zeros((n_fbs, classes, channels), dtype=np.float64)
        if self._preset_delays is not None:
            if self._preset_delays.shape != (classes, channels):
                raise ValueError("pre-estimated delays do not match the training data")
            self.delays = self._preset_delays.copy()
        else:
            self.delays = np.zeros((classes, channels), dtype=np.float64)
        for cls in range(classes):
            class_trials = x[y == cls]
            if len(class_trials) < 2:
                raise ValueError("LA-TRCA requires at least two training trials")
            if self._preset_delays is None:
                self.delays[cls] = estimate_wave_delays(
                    class_trials[:, 0], float(self.frequencies[cls]), self.sampling_rate,
                    self.coordinates, self.source_channel,
                )
            aligned = align_channels(class_trials, -self.delays[cls], self.sampling_rate)
            self.templates[cls] = np.mean(aligned, axis=0)
            for fb_idx in range(n_fbs):
                self.filters[fb_idx, cls] = trca_filter(aligned[:, fb_idx])
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.templates is None or self.filters is None or self.delays is None:
            raise RuntimeError("LA-TRCA model is not fitted.")
        epochs = np.asarray(x, dtype=np.float64)
        band_scores = np.zeros(
            (epochs.shape[0], epochs.shape[1], self.templates.shape[0]), dtype=np.float64
        )
        for cls in range(self.templates.shape[0]):
            aligned = align_channels(epochs, -self.delays[cls], self.sampling_rate)
            for fb_idx in range(epochs.shape[1]):
                spatial_filter = self.filters[fb_idx, cls]
                projected_trials = aligned[:, fb_idx].transpose(0, 2, 1) @ spatial_filter
                projected_template = self.templates[cls, fb_idx].T @ spatial_filter
                correlations = corr_rows(projected_trials, projected_template)
                band_scores[:, fb_idx, cls] = correlations**2
        scores = np.einsum("f,tfc->tc", self.weights[: epochs.shape[1]], band_scores)
        return np.argmax(scores, axis=1), scores
