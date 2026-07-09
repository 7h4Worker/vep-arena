"""Phase locking value (PLV) and inter-trial phase coherence (ITPC).

Two estimation methods:

* **FFT-based** (``itpc_fft``) — phase at the exact FFT bin closest to
  the target frequency.  More frequency-specific; works well when the
  window contains several full cycles.
* **Filter-based** (``itpc``) — narrow bandpass → Hilbert transform →
  mean PLV over time.  Broader in frequency but captures
  time-resolved phase locking.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig


# ── helpers ─────────────────────────────────────────────────────────

def _extract_phase(
    x: np.ndarray,
    fs: int,
    target_freq: float,
    bandwidth: float = 2.0,
) -> np.ndarray:
    """Instantaneous phase via narrow bandpass + Hilbert.

    Returns (..., n_samples) in radians.
    """
    low = max(target_freq - bandwidth / 2, 0.5)
    high = min(target_freq + bandwidth / 2, fs / 2 - 1)
    if low >= high:
        return np.zeros_like(x)
    sos = sig.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    filtered = sig.sosfiltfilt(sos, x, axis=-1)
    return np.angle(sig.hilbert(filtered, axis=-1))


# ── public API ──────────────────────────────────────────────────────

def itpc(
    x: np.ndarray,
    fs: int,
    target_freq: float,
    bandwidth: float = 2.0,
) -> float:
    """Filter-based inter-trial phase coherence (mean PLV over time).

    Parameters
    ----------
    x : (n_trials, n_samples) — single channel, multiple trials

    Returns
    -------
    ITPC value in [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("itpc expects (n_trials, n_samples)")
    phases = _extract_phase(x, fs, target_freq, bandwidth)
    plv_time = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return float(np.mean(plv_time))


def itpc_fft(
    x: np.ndarray,
    fs: int,
    target_freq: float,
    nfft: int | None = None,
) -> float:
    """FFT-based ITPC at the target frequency bin.

    Parameters
    ----------
    x : (n_trials, n_samples)
    nfft : FFT length; zero-pad for finer bin spacing.

    Returns
    -------
    PLV in [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("itpc_fft expects (n_trials, n_samples)")
    x = x - x.mean(axis=-1, keepdims=True)
    if nfft is None:
        nfft = x.shape[-1]
    fft_vals = np.fft.rfft(x, n=nfft, axis=-1)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    target_bin = int(np.argmin(np.abs(freqs - target_freq)))
    phases = np.angle(fft_vals[:, target_bin])
    return float(np.abs(np.mean(np.exp(1j * phases))))


def plv_profile(
    x: np.ndarray,
    fs: int,
    target_freqs: np.ndarray | tuple[float, ...],
    method: str = "fft",
    bandwidth: float = 2.0,
    nfft: int | None = None,
) -> np.ndarray:
    """ITPC for each target frequency → ``(n_targets,)`` vector.

    Parameters
    ----------
    x : (n_trials, n_samples) for single channel, or
        (n_trials, n_channels, n_samples) — average PLV across channels.
    method : 'fft' or 'filter'
    """
    target_freqs = np.asarray(target_freqs)
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 3:
        per_ch = np.stack(
            [plv_profile(x[:, ch], fs, target_freqs, method, bandwidth, nfft)
             for ch in range(x.shape[1])],
        )
        return np.mean(per_ch, axis=0)

    profile = np.empty(len(target_freqs))
    for i, f0 in enumerate(target_freqs):
        if method == "fft":
            profile[i] = itpc_fft(x, fs, f0, nfft)
        else:
            profile[i] = itpc(x, fs, f0, bandwidth)
    return profile


def rayleigh_test(phases: np.ndarray) -> tuple[float, float]:
    """Rayleigh test for non-uniformity of circular data.

    Returns ``(z_statistic, p_value)``.
    """
    phases = np.asarray(phases).ravel()
    n = len(phases)
    if n < 2:
        return 0.0, 1.0
    r_bar = float(np.abs(np.mean(np.exp(1j * phases))))
    z = n * r_bar ** 2
    p = float(np.exp(-z) * (
        1 + (2 * z - z**2) / (4 * n)
        - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2)
    ))
    return float(z), float(np.clip(p, 0, 1))
