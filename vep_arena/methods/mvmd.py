from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd

from vep_arena.config import BENCHMARK_FREQS, BenchmarkSpec
from vep_arena.methods.cca import _orth, sine_references
from vep_arena.methods.trca_core import corr_flat as corrcoef_1d
from vep_arena.methods.trca_core import trca_filter_from_cst


def mvmd(
    signal: np.ndarray,
    *,
    alpha: float = 2000.0,
    tau: float = 0.0,
    n_modes: int = 5,
    dc: bool = False,
    init: int = 1,
    init_omega: np.ndarray | None = None,
    tol: float = 1e-5,
    max_iter: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Multivariate variational mode decomposition.

    This is a NumPy port of the public MATLAB `MVMD.m` reference by Rehman and
    Aftab. Input and output use EEG-friendly shapes:

    - input: channels x samples
    - modes: modes x channels x samples
    - omega: iteration x modes, normalized cycles/sample on the mirrored signal
    """

    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("MVMD input must be channels x samples.")
    channels, original_samples = x.shape
    if original_samples < 8:
        raise ValueError("MVMD needs at least 8 samples.")
    if original_samples % 2:
        x = np.pad(x, ((0, 0), (0, 1)), mode="edge")
    samples = x.shape[1]
    half = samples // 2

    mirrored = np.concatenate([x[:, half - 1 :: -1], x, x[:, samples - 1 : half - 1 : -1]], axis=1)
    points = mirrored.shape[1]
    freqs = np.arange(points, dtype=np.float64) / points - 0.5

    f_hat = np.fft.fftshift(np.fft.fft(mirrored, axis=1), axes=1)
    f_hat_plus = f_hat.copy()
    f_hat_plus[:, : points // 2] = 0.0
    f_hat_plus_t = f_hat_plus.T

    alpha_vec = np.full(n_modes, float(alpha), dtype=np.float64)
    u_hat = np.zeros((points, channels, n_modes), dtype=np.complex128)
    u_hat_prev = np.zeros_like(u_hat)
    lambda_hat = np.zeros((points, channels), dtype=np.complex128)
    omega = np.zeros((max_iter + 1, n_modes), dtype=np.float64)
    if init_omega is not None:
        init_omega = np.asarray(init_omega, dtype=np.float64)
        if init_omega.shape != (n_modes,):
            raise ValueError(f"init_omega must have shape ({n_modes},).")
        omega[0] = init_omega
    elif init == 1:
        omega[0] = (0.5 / n_modes) * np.arange(n_modes)
    elif init == 2:
        fs_norm = 1.0 / samples
        omega[0] = np.sort(np.exp(np.log(fs_norm) + (np.log(0.5) - np.log(fs_norm)) * np.random.rand(n_modes)))
    if dc:
        omega[0, 0] = 0.0

    u_diff = tol + np.finfo(float).eps
    n = 0
    sum_uk = np.zeros((points, channels), dtype=np.complex128)
    while u_diff > tol and n < max_iter:
        omega[n + 1] = omega[n]
        for k in range(n_modes):
            if k > 0:
                sum_uk = u_hat[:, :, k - 1] + sum_uk - u_hat_prev[:, :, k]
            else:
                sum_uk = u_hat_prev[:, :, n_modes - 1] + sum_uk - u_hat_prev[:, :, k]

            denom = 1.0 + alpha_vec[k] * (freqs - omega[n, k]) ** 2
            residual = f_hat_plus_t - sum_uk - lambda_hat / 2.0
            u_hat[:, :, k] = residual / denom[:, None]

            if (not dc) or k > 0:
                pos = slice(points // 2, points)
                power = np.abs(u_hat[pos, :, k]) ** 2
                den = float(np.sum(power))
                if den > 1e-18:
                    omega[n + 1, k] = float(np.sum(freqs[pos, None] * power) / den)

        lambda_hat = lambda_hat + tau * (np.sum(u_hat, axis=2) - f_hat_plus_t)
        diff = u_hat - u_hat_prev
        u_diff = float(np.finfo(float).eps + np.sum(np.abs(diff) ** 2) / points)
        u_hat_prev = u_hat.copy()
        n += 1

    omega_out = omega[: n + 1]
    full_hat = np.zeros_like(u_hat)
    for c in range(channels):
        full_hat[points // 2 :, c, :] = u_hat[points // 2 :, c, :]
        full_hat[points // 2 : 0 : -1, c, :] = np.conj(u_hat[points // 2 :, c, :])
        full_hat[0, c, :] = np.conj(full_hat[-1, c, :])

    time_modes = np.zeros((n_modes, points, channels), dtype=np.float64)
    for k in range(n_modes):
        for c in range(channels):
            time_modes[k, :, c] = np.real(np.fft.ifft(np.fft.ifftshift(full_hat[:, c, k])))

    start = points // 4
    stop = start + samples
    modes = np.transpose(time_modes[:, start:stop, :], (0, 2, 1))
    return modes[:, :, :original_samples], omega_out


def reference_bank(window: float, harmonics: int = 2, spec: BenchmarkSpec | None = None) -> np.ndarray:
    """All-frequency sine/cosine bank used for efficient sinusoidal assistance."""

    spec = spec or BenchmarkSpec()
    samples = spec.sample_length(window)
    t = (np.arange(1, samples + 1, dtype=np.float64) / spec.sampling_rate)[None, :]
    rows: list[np.ndarray] = []
    for freq in BENCHMARK_FREQS:
        for harmonic in range(1, harmonics + 1):
            rows.append(np.sin(2 * np.pi * harmonic * freq * t))
            rows.append(np.cos(2 * np.pi * harmonic * freq * t))
    return np.concatenate(rows, axis=0)


def assisted_signal(eeg: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Append scaled sinusoidal references to EEG channels."""

    eeg_std = float(np.std(eeg))
    ref_std = float(np.std(refs))
    scale = eeg_std / ref_std if ref_std > 1e-12 else 1.0
    return np.concatenate([eeg, refs * scale], axis=0)


def reconstruct_modes(modes: np.ndarray, drop_first: bool = False) -> np.ndarray:
    start = 1 if drop_first and modes.shape[0] > 1 else 0
    return np.sum(modes[start:], axis=0)


class MVMDCCAClassifier:
    """Calibration-free MVMD-CCA baseline.

    The trial is first decomposed and reconstructed from MVMD modes, then CCA
    is applied to the reconstructed EEG. This follows the MVMD-CCA recognition
    line more closely than treating every mode as an independent filter-bank.
    """

    def __init__(
        self,
        window: float,
        harmonics: int = 5,
        n_modes: int = 5,
        alpha: float = 2000.0,
        max_iter: int = 120,
        tol: float = 1e-5,
        spec: BenchmarkSpec | None = None,
    ) -> None:
        self.window = window
        self.harmonics = harmonics
        self.n_modes = n_modes
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.spec = spec or BenchmarkSpec()
        self.refs_q = [_orth(ref) for ref in sine_references(window, harmonics, self.spec)]
        self.mode_weights = np.asarray([(idx + 1) ** (-1.25) + 0.25 for idx in range(n_modes)], dtype=np.float64)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros((x.shape[0], len(self.refs_q)), dtype=np.float64)
        for trial_idx, trial in enumerate(x):
            modes, _ = mvmd(trial, alpha=self.alpha, n_modes=self.n_modes, init=1, tol=self.tol, max_iter=self.max_iter)
            reconstructed = reconstruct_modes(modes)
            qx = _orth(reconstructed)
            for cls, ref_q in enumerate(self.refs_q):
                vals = svd(qx.T @ ref_q, compute_uv=False, check_finite=False)
                scores[trial_idx, cls] = float(np.clip(vals[0], 0.0, 1.0)) if vals.size else 0.0
        return np.argmax(scores, axis=1), scores


def trca_filter(eeg: np.ndarray) -> np.ndarray:
    """First TRCA spatial filter for channels x samples x trials."""

    return trca_filter_from_cst(eeg)


@dataclass
class SimpleTRCA:
    templates: np.ndarray
    weights: np.ndarray
    ensemble: bool = True

    @classmethod
    def fit(cls, x: np.ndarray, ensemble: bool = True) -> "SimpleTRCA":
        """Fit TRCA on classes x train_blocks x channels x samples."""

        classes, _, channels, samples = x.shape
        templates = np.mean(x, axis=1)
        weights = np.zeros((classes, channels), dtype=np.float64)
        for target in range(classes):
            weights[target] = trca_filter(np.transpose(x[target], (1, 2, 0)))
        return cls(templates=templates, weights=weights, ensemble=ensemble)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros((x.shape[0], self.templates.shape[0]), dtype=np.float64)
        if self.ensemble:
            w = self.weights.T
        for trial_idx, trial in enumerate(x):
            for target in range(self.templates.shape[0]):
                ww = w if self.ensemble else self.weights[target][:, None]
                scores[trial_idx, target] = corrcoef_1d(trial.T @ ww, self.templates[target].T @ ww)
        return np.argmax(scores, axis=1), scores


def mvmd_reconstruct_trial(
    trial: np.ndarray,
    *,
    alpha: float,
    n_modes: int,
    max_iter: int,
    tol: float,
    refs: np.ndarray | None = None,
    drop_first: bool = False,
) -> np.ndarray:
    signal = assisted_signal(trial, refs) if refs is not None else trial
    modes, _ = mvmd(signal, alpha=alpha, n_modes=n_modes, init=1, tol=tol, max_iter=max_iter)
    eeg_modes = modes[:, : trial.shape[0]]
    return reconstruct_modes(eeg_modes, drop_first=drop_first)
