"""Signal-to-noise ratio estimators for SSVEP.

Three levels of SNR definition:

* **narrowband** — power at a single target bin vs. neighbouring bins.
* **harmonic** — cumulative power at fundamental + *N* harmonics vs.
  neighbouring noise at each harmonic.  This is the most common
  "SSVEP SNR" in the literature but papers differ in neighbour count
  and harmonic count.
* **wideband** — total power at all target frequencies (with harmonics)
  vs. remaining power in a frequency band.

All parameters that affect the result (neighbour count, harmonic count,
bandwidth, dB vs. linear) are explicit — there are no hidden defaults
for research-critical choices.
"""
from __future__ import annotations

import numpy as np

from vep_arena.signal.spectrum import compute_psd


def _find_bin(freqs: np.ndarray, freq: float) -> int:
    return int(np.argmin(np.abs(freqs - freq)))


def _neighbor_bins(
    freqs: np.ndarray,
    center_bin: int,
    n_neighbors: int,
    exclude_bins: set[int] | None = None,
) -> np.ndarray:
    exclude = exclude_bins or set()
    neighbors: list[int] = []
    for offset in range(1, len(freqs)):
        for direction in (-1, 1):
            idx = center_bin + direction * offset
            if 0 <= idx < len(freqs) and idx not in exclude and idx != center_bin:
                neighbors.append(idx)
            if len(neighbors) >= n_neighbors:
                return np.array(neighbors, dtype=int)
    return np.array(neighbors, dtype=int)


def _harmonic_bins(freqs: np.ndarray, f0: float, n_harmonics: int) -> set[int]:
    nyquist = freqs[-1]
    bins: set[int] = set()
    for k in range(1, n_harmonics + 1):
        hf = f0 * k
        if hf < nyquist:
            bins.add(_find_bin(freqs, hf))
    return bins


def _collapse_leading(psd: np.ndarray) -> np.ndarray:
    """Average over all axes except the last (frequency axis)."""
    if psd.ndim <= 1:
        return psd
    return np.mean(psd, axis=tuple(range(psd.ndim - 1)))


def _to_db(ratio: float) -> float:
    return float(10 * np.log10(max(ratio, 1e-30)))


# ── public API ──────────────────────────────────────────────────────

def snr_narrowband(
    psd: np.ndarray,
    freqs: np.ndarray,
    target_freq: float,
    n_neighbors: int = 10,
    exclude_harmonics_of: float | None = None,
    n_harmonics_exclude: int = 5,
    db: bool = True,
) -> float:
    """Power at target bin / mean power at neighbouring bins.

    Parameters
    ----------
    psd : (..., n_freqs)  — averaged over leading dims before computation.
    freqs : (n_freqs,)
    target_freq : Hz
    n_neighbors : number of bins for noise floor estimation
    exclude_harmonics_of : exclude bins near harmonics of this frequency
    db : True → 10·log₁₀(ratio)
    """
    psd_1d = _collapse_leading(psd)
    target_bin = _find_bin(freqs, target_freq)

    exclude = {target_bin}
    if exclude_harmonics_of is not None:
        exclude |= _harmonic_bins(freqs, exclude_harmonics_of, n_harmonics_exclude)

    nidx = _neighbor_bins(freqs, target_bin, n_neighbors, exclude)
    if len(nidx) == 0:
        return float("nan")

    noise = float(np.mean(psd_1d[nidx]))
    if noise <= 0 or not np.isfinite(noise):
        return float("nan")

    ratio = float(psd_1d[target_bin]) / noise
    return _to_db(ratio) if db else float(ratio)


def snr_harmonic(
    psd: np.ndarray,
    freqs: np.ndarray,
    target_freq: float,
    n_harmonics: int = 5,
    n_neighbors: int = 10,
    db: bool = True,
    per_harmonic: bool = False,
) -> float | np.ndarray:
    """Cumulative SNR across fundamental + harmonics.

    For each harmonic ``k·f₀`` the signal bin power and mean neighbour
    power are accumulated separately; the final ratio is
    ``Σ signal / Σ noise``.

    Parameters
    ----------
    per_harmonic : if True return ``(n_valid_harmonics,)`` of individual
        SNR values instead of the aggregate.
    """
    psd_1d = _collapse_leading(psd)
    nyquist = freqs[-1]
    all_hbins = _harmonic_bins(freqs, target_freq, n_harmonics)

    signal_powers: list[float] = []
    noise_powers: list[float] = []
    per_h: list[float] = []

    for k in range(1, n_harmonics + 1):
        hf = target_freq * k
        if hf >= nyquist:
            break
        hbin = _find_bin(freqs, hf)
        nidx = _neighbor_bins(freqs, hbin, n_neighbors, all_hbins)
        if len(nidx) == 0:
            continue
        sp = float(psd_1d[hbin])
        np_ = float(np.mean(psd_1d[nidx]))
        signal_powers.append(sp)
        noise_powers.append(np_)
        per_h.append(_to_db(sp / np_) if (db and np_ > 0) else (sp / np_ if np_ > 0 else float("nan")))

    if per_harmonic:
        return np.array(per_h)

    total_s = sum(signal_powers)
    total_n = sum(noise_powers)
    if total_n <= 0:
        return float("nan")
    ratio = total_s / total_n
    return _to_db(ratio) if db else float(ratio)


def snr_wideband(
    psd: np.ndarray,
    freqs: np.ndarray,
    signal_freqs: np.ndarray | list[float],
    signal_bandwidth: float = 0.5,
    noise_band: tuple[float, float] = (1.0, 50.0),
    n_harmonics: int = 5,
    db: bool = True,
) -> float:
    """Total signal power (targets + harmonics) vs. remaining band power.

    Parameters
    ----------
    signal_freqs : fundamental frequencies to treat as "signal"
    signal_bandwidth : Hz window around each signal peak
    noise_band : (low, high) Hz for the total analysis band
    """
    psd_1d = _collapse_leading(psd)
    nyquist = freqs[-1]

    sig_mask = np.zeros(len(freqs), dtype=bool)
    for f0 in signal_freqs:
        for k in range(1, n_harmonics + 1):
            hf = f0 * k
            if hf < nyquist:
                sig_mask |= np.abs(freqs - hf) <= signal_bandwidth

    band_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
    noise_mask = band_mask & ~sig_mask

    if not np.any(sig_mask) or not np.any(noise_mask):
        return float("nan")

    ratio = float(np.sum(psd_1d[sig_mask])) / float(np.sum(psd_1d[noise_mask]))
    return _to_db(ratio) if db else float(ratio)


def ssvep_snr_profile(
    x: np.ndarray,
    fs: int,
    target_freqs: np.ndarray | tuple[float, ...],
    mode: str = "harmonic",
    n_harmonics: int = 5,
    n_neighbors: int = 10,
    db: bool = True,
    psd_method: str = "welch",
    **psd_kwargs,
) -> np.ndarray:
    """Compute SNR for each target frequency → ``(n_targets,)`` vector.

    Parameters
    ----------
    x : (..., n_samples) — e.g. (trials, channels, samples)
    mode : 'narrowband' | 'harmonic' | 'wideband'
    """
    target_freqs = np.asarray(target_freqs)
    freqs, psd = compute_psd(x, fs, method=psd_method, **psd_kwargs)

    profile = np.empty(len(target_freqs))
    for i, f0 in enumerate(target_freqs):
        if mode == "narrowband":
            profile[i] = snr_narrowband(
                psd, freqs, f0, n_neighbors, exclude_harmonics_of=f0, db=db,
            )
        elif mode == "harmonic":
            profile[i] = snr_harmonic(
                psd, freqs, f0, n_harmonics, n_neighbors, db=db,
            )
        elif mode == "wideband":
            profile[i] = snr_wideband(
                psd, freqs, [f0], n_harmonics=n_harmonics, db=db,
            )
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
    return profile
