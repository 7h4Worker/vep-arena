"""Reusable EEG filtering helpers."""
from __future__ import annotations

import numpy as np
from scipy import signal


def remove_dc(data: np.ndarray) -> np.ndarray:
    """Remove channel-wise DC offset."""

    arr = np.asarray(data, dtype=np.float64)
    return arr - np.mean(arr, axis=-1, keepdims=True)


def powerline_comb_filter(
    data: np.ndarray,
    sfreq: float,
    *,
    base_hz: float = 50.0,
    q: float = 35.0,
    remove_dc_offset: bool = True,
) -> np.ndarray:
    """Apply a zero-phase notch comb filter for line-frequency interference."""

    if base_hz <= 0:
        raise ValueError("base_hz must be positive.")
    if q <= 0:
        raise ValueError("q must be positive.")
    sfreq = float(sfreq)
    if base_hz >= sfreq / 2.0:
        raise ValueError(f"base_hz={base_hz} must be below Nyquist for sfreq={sfreq}.")

    out = remove_dc(data) if remove_dc_offset else np.asarray(data, dtype=np.float64)
    b, a = signal.iircomb(w0=float(base_hz), Q=float(q), ftype="notch", fs=sfreq)
    return signal.filtfilt(b, a, out, axis=-1)


def target_band_comb_filter(
    data: np.ndarray,
    sfreq: float,
    centers_hz: tuple[float, ...] | list[float],
    *,
    half_width_hz: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """Keep a union of narrow target-centred passbands with zero phase.

    Adjacent component bands are merged before filtering so a component lying
    in an overlap is not amplified repeatedly. This is suitable for
    target-conditioned SSVEP feature filtering, unlike a power-line notch.
    """

    if half_width_hz <= 0:
        raise ValueError("half_width_hz must be positive.")
    if order <= 0:
        raise ValueError("order must be positive.")
    sfreq = float(sfreq)
    nyquist = sfreq / 2.0
    intervals = sorted(
        (max(0.1, float(center) - half_width_hz), min(nyquist - 0.1, float(center) + half_width_hz))
        for center in centers_hz
        if 0.1 < float(center) < nyquist - 0.1
    )
    if not intervals:
        raise ValueError("No target passbands lie below Nyquist.")
    merged: list[list[float]] = []
    for low, high in intervals:
        if high <= low:
            continue
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    out = np.zeros_like(np.asarray(data, dtype=np.float64))
    for low, high in merged:
        sos = signal.butter(order, (low, high), btype="bandpass", fs=sfreq, output="sos")
        out += signal.sosfiltfilt(sos, data, axis=-1)
    return out
