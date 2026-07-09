"""Power spectral density and amplitude spectrum estimation."""
from __future__ import annotations

import numpy as np
from scipy import signal as sig


def compute_psd(
    x: np.ndarray,
    fs: int,
    method: str = "welch",
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate power spectral density.

    Parameters
    ----------
    x : array, shape (..., n_samples)
    fs : sampling rate in Hz
    method : 'welch' or 'fft'
    nfft : FFT length; use values larger than n_samples for zero-padded
        frequency interpolation (e.g. ``nfft=4*n_samples``).

    Returns
    -------
    freqs : (n_freqs,)
    psd : (..., n_freqs)
    """
    x = np.asarray(x, dtype=np.float64)
    n_samples = x.shape[-1]

    if method == "welch":
        if nperseg is None:
            nperseg = min(n_samples, 512)
        freqs, psd = sig.welch(
            x, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, axis=-1,
        )
        return freqs, psd

    if method == "fft":
        if nfft is None:
            nfft = n_samples
        x_c = x - x.mean(axis=-1, keepdims=True)
        fft_vals = np.fft.rfft(x_c, n=nfft, axis=-1)
        psd = (np.abs(fft_vals) ** 2) / (fs * nfft)
        psd[..., 1:-1] *= 2
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        return freqs, psd

    raise ValueError(f"Unknown method: {method!r}")


def amplitude_spectrum(
    x: np.ndarray,
    fs: int,
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum via FFT."""
    x = np.asarray(x, dtype=np.float64)
    if nfft is None:
        nfft = x.shape[-1]
    x_c = x - x.mean(axis=-1, keepdims=True)
    fft_vals = np.fft.rfft(x_c, n=nfft, axis=-1)
    amp = np.abs(fft_vals) * 2.0 / nfft
    amp[..., 0] /= 2
    if nfft % 2 == 0:
        amp[..., -1] /= 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, amp
