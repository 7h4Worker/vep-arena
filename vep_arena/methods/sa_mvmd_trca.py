from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vep_arena.methods.mvmd import mvmd
from vep_arena.methods.trca_core import corr_flat as corrcoef_1d
from vep_arena.methods.trca_core import trca_filter_from_cst as trca_filter


def corrcoef_2d(a: np.ndarray, b: np.ndarray) -> float:
    av = np.ravel(a) - np.mean(a)
    bv = np.ravel(b) - np.mean(b)
    den = np.linalg.norm(av) * np.linalg.norm(bv)
    if den <= 1e-12:
        return 0.0
    return float(np.dot(av, bv) / den)


def paper_component_weights(n_selected: int) -> np.ndarray:
    """Weights for reconstructed signal and selected IMFs.

    Paper Eq. 8 uses a(m) = (m + 1)^-1.25 + 0.25 for m = 0..M.
    """

    return np.asarray([(m + 1) ** (-1.25) + 0.25 for m in range(n_selected + 1)], dtype=np.float64)


def assistance_weights(n_harmonics: int) -> np.ndarray:
    """Sinusoidal amplitudes from the paper: A_i = i^-0.75 + 0.01."""

    idx = np.arange(1, n_harmonics + 1, dtype=np.float64)
    return idx ** (-0.75) + 0.01


def sinusoidal_assistance(
    samples: int,
    fs: int,
    fundamental: float,
    *,
    n_harmonics: int = 7,
    rp: float = 100.0,
    eeg: np.ndarray | None = None,
) -> np.ndarray:
    t = np.arange(samples, dtype=np.float64) / fs
    amps = assistance_weights(n_harmonics)
    signal = np.zeros(samples, dtype=np.float64)
    for harmonic, amp in enumerate(amps, start=1):
        freq = harmonic * fundamental
        if freq < fs / 2:
            signal += amp * np.sin(2 * np.pi * freq * t)
    if eeg is not None:
        eeg_power = float(np.mean(np.asarray(eeg, dtype=np.float64) ** 2))
        sa_power = float(np.mean(signal**2))
        if sa_power > 1e-18 and eeg_power > 1e-18:
            signal *= np.sqrt(rp * eeg_power / sa_power)
    return signal


def sa_mvmd_components(
    trial: np.ndarray,
    fundamental: float,
    *,
    fs: int = 250,
    n_harmonics: int = 7,
    selected_harmonics: int = 5,
    alpha: float = 2000.0,
    rp: float = 100.0,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> np.ndarray:
    """Paper-style SA-MVMD components for one EEG trial.

    Returns components x channels x samples, where component 0 is the
    reconstructed wideband signal and components 1..M are IMFs 2..(M+1).
    """

    trial = np.asarray(trial, dtype=np.float64)
    channels, samples = trial.shape
    sa = sinusoidal_assistance(samples, fs, fundamental, n_harmonics=n_harmonics, rp=rp, eeg=trial)
    augmented = np.concatenate([trial, sa[None, :]], axis=0)
    init_omega = np.zeros(n_harmonics + 1, dtype=np.float64)
    init_omega[1:] = np.arange(1, n_harmonics + 1, dtype=np.float64) * fundamental / fs
    modes, omega = mvmd(
        augmented,
        alpha=alpha,
        tau=0.0,
        n_modes=n_harmonics + 1,
        dc=True,
        init_omega=init_omega,
        tol=tol,
        max_iter=max_iter,
    )
    final_omega = omega[-1]
    order = np.argsort(final_omega)
    modes = modes[order, :channels]
    selected = modes[1 : selected_harmonics + 1]
    reconstructed = np.sum(selected, axis=0)
    return np.concatenate([reconstructed[None, ...], selected], axis=0)


@dataclass
class SAMVMDTRCAModel:
    templates: np.ndarray
    filters: np.ndarray
    component_weights: np.ndarray

    @classmethod
    def fit(cls, components: np.ndarray) -> "SAMVMDTRCAModel":
        """Fit paper-style TRCA filters.

        `components` shape is classes x train_blocks x comps x channels x samples.
        """

        classes, _, comps, channels, samples = components.shape
        templates = np.mean(components, axis=1)
        filters = np.zeros((classes, comps, channels), dtype=np.float64)
        for target in range(classes):
            for comp in range(comps):
                filters[target, comp] = trca_filter(np.transpose(components[target, :, comp], (1, 2, 0)))
        return cls(templates=templates, filters=filters, component_weights=paper_component_weights(comps - 1))

    def score_candidate_trca(self, target: int, candidate_components: np.ndarray) -> float:
        scores = np.zeros(candidate_components.shape[0], dtype=np.float64)
        for comp in range(candidate_components.shape[0]):
            w = self.filters[target, comp]
            test_projection = candidate_components[comp].T @ w
            template_projection = self.templates[target, comp].T @ w
            scores[comp] = corrcoef_1d(test_projection, template_projection)
        return float(np.dot(self.component_weights, scores))

    def score_candidate_etrCA(self, target: int, candidate_components: np.ndarray) -> float:
        scores = np.zeros(candidate_components.shape[0], dtype=np.float64)
        for comp in range(candidate_components.shape[0]):
            w = self.filters[:, comp].T
            test_projection = candidate_components[comp].T @ w
            template_projection = self.templates[target, comp].T @ w
            scores[comp] = corrcoef_2d(test_projection, template_projection)
        return float(np.dot(self.component_weights, scores))

    def predict_from_candidate_components(self, candidate_components: np.ndarray, *, ensemble: bool = False) -> tuple[int, np.ndarray]:
        """Predict from precomputed candidate decompositions.

        `candidate_components` shape is classes x comps x channels x samples.
        """

        scores = np.zeros(candidate_components.shape[0], dtype=np.float64)
        for target in range(candidate_components.shape[0]):
            if ensemble:
                scores[target] = self.score_candidate_etrCA(target, candidate_components[target])
            else:
                scores[target] = self.score_candidate_trca(target, candidate_components[target])
        return int(np.argmax(scores)), scores
