from __future__ import annotations

import numpy as np

from vep_arena.config import BENCHMARK_FREQS
from vep_arena.methods.trca_core import _solve_trca, trca_scores


def _gaussian_bank(
    frequencies: np.ndarray,
    target_frequency: float,
    first_harmonic: int,
    harmonics: int,
    fwhm: float,
) -> np.ndarray:
    sigma = fwhm * (2.0 * np.pi - 1.0) / (4.0 * np.pi)
    response = np.zeros_like(frequencies)
    for harmonic in range(first_harmonic, harmonics + 1):
        center = harmonic * target_frequency
        if center <= frequencies[-1]:
            response += np.exp(-0.5 * ((frequencies - center) / sigma) ** 2)
    return response


def ress_filter(
    trials: np.ndarray,
    target_frequency: float,
    first_harmonic: int,
    *,
    sampling_rate: int = 250,
    harmonics: int = 5,
    signal_fwhm: float = 1.75,
    reference_fwhm: float = 0.25,
    fft_seconds: float = 20.0,
) -> np.ndarray:
    """Fit the Mode-I RESS spatial filter from trials x channels x samples."""

    x = np.asarray(trials, dtype=np.float64)
    x = x - np.mean(x, axis=-1, keepdims=True)
    n_fft = max(x.shape[-1], int(round(fft_seconds * sampling_rate)))
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)
    signal_response = _gaussian_bank(
        frequencies, target_frequency, first_harmonic, harmonics, signal_fwhm
    )
    reference_response = 1.0 - _gaussian_bank(
        frequencies, target_frequency, first_harmonic, harmonics, reference_fwhm
    )
    spectrum = np.fft.rfft(x, n=n_fft, axis=-1)
    signal = np.fft.irfft(spectrum * signal_response, n=n_fft, axis=-1)[..., : x.shape[-1]]
    reference = np.fft.irfft(spectrum * reference_response, n=n_fft, axis=-1)[..., : x.shape[-1]]
    signal = signal - np.mean(signal, axis=-1, keepdims=True)
    reference = reference - np.mean(reference, axis=-1, keepdims=True)
    signal_concat = np.transpose(signal, (1, 0, 2)).reshape(x.shape[1], -1)
    reference_concat = np.transpose(reference, (1, 0, 2)).reshape(x.shape[1], -1)
    return _solve_trca(signal_concat @ signal_concat.T, reference_concat @ reference_concat.T)


class RESS:
    """Rhythmic entrainment source separation for SSVEP classification.

    The defaults reproduce Xu et al.'s Mode-I stimulus-specific RESS configuration.
    Input epochs are trials x subbands x channels x samples.
    """

    name = "RESS"

    def __init__(
        self,
        n_fbs: int = 5,
        *,
        frequencies: tuple[float, ...] | list[float] = BENCHMARK_FREQS,
        sampling_rate: int = 250,
        harmonics: int = 5,
        signal_fwhm: float = 1.75,
        reference_fwhm: float = 0.25,
        ensemble: bool = False,
    ) -> None:
        self.weights = np.asarray(
            [(index + 1) ** (-1.25) + 0.25 for index in range(n_fbs)], dtype=np.float64
        )
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.sampling_rate = int(sampling_rate)
        self.harmonics = int(harmonics)
        self.signal_fwhm = float(signal_fwhm)
        self.reference_fwhm = float(reference_fwhm)
        self.ensemble = bool(ensemble)
        self.templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "RESS":
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
            self.templates[cls] = np.mean(class_trials, axis=0)
            for fb_idx in range(n_fbs):
                self.filters[fb_idx, cls] = ress_filter(
                    class_trials[:, fb_idx],
                    float(self.frequencies[cls]),
                    fb_idx + 1,
                    sampling_rate=self.sampling_rate,
                    harmonics=self.harmonics,
                    signal_fwhm=self.signal_fwhm,
                    reference_fwhm=self.reference_fwhm,
                )
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.templates is None or self.filters is None:
            raise RuntimeError("RESS model is not fitted.")
        scores = trca_scores(
            np.asarray(x, dtype=np.float64),
            self.templates,
            self.filters,
            self.weights,
            ensemble=self.ensemble,
        )
        return np.argmax(scores, axis=1), scores
