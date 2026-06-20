from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import cheby1, sosfiltfilt, sosfreqz

from vep_arena.config import BENCHMARK_CHANNELS_9, DATA_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import load_subject_raw, notch_50hz


OUT = Path("D:/ProjData/proj_python/vep_arena/results/benchmark_9ch/preprocessing_compare_2s")


def dnn_sos(fs: int, subbands: int = 3) -> list[np.ndarray]:
    return [cheby1(N=2, rp=1, Wn=[8 * idx, 90], btype="bandpass", fs=fs, output="sos") for idx in range(1, subbands + 1)]


def toolbox_ba(fs: int, n_fbs: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    filters: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n_fbs + 1):
        wp = [(8 * k) / (fs / 2), 90 / (fs / 2)]
        ws = [(8 * k - 2) / (fs / 2), 100 / (fs / 2)]
        gstop = 40
        while gstop >= 20:
            try:
                order, wn = signal.cheb1ord(wp, ws, 3, gstop)
                filters.append(signal.cheby1(order, 0.5, wn, btype="bandpass"))
                break
            except ValueError:
                gstop -= 1
        if gstop < 20:
            raise ValueError(f"Could not create toolbox subband {k}")
    return filters


def apply_toolbox_segment(raw_channel: np.ndarray, spec: BenchmarkSpec, window: float, n_fbs: int = 5) -> np.ndarray:
    fs = spec.sampling_rate
    samples = spec.sample_length(window)
    start = round(spec.cue_seconds * fs)
    latency = round(spec.latency_seconds * fs)
    segment = raw_channel[start : start + latency + samples]
    segment = notch_50hz(segment[None, :], fs)[0]
    filtered = []
    for b, a in toolbox_ba(fs, n_fbs):
        y = signal.filtfilt(b, a, segment, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1))
        filtered.append(y[latency : latency + samples])
    return np.asarray(filtered)


def db(x: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(x), 1e-8))


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=220)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = BenchmarkSpec()
    fs = spec.sampling_rate
    window = 2.0
    subject = 1
    cls = 0
    block = 0
    channel_pos = 4
    channel = BENCHMARK_CHANNELS_9[channel_pos]

    raw = load_subject_raw(DATA_ROOT, subject)
    raw_channel = raw[channel - 1, :, cls, block]
    target = raw_channel[spec.sample_slice(window)]
    samples = spec.sample_length(window)
    t = np.arange(samples) / fs

    dnn_filtered = np.asarray([sosfiltfilt(sos, target) for sos in dnn_sos(fs)])
    arena_filtered = apply_toolbox_segment(raw_channel, spec, window)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8.5), sharex=True)
    axes[0].plot(t, target, color="#333333", linewidth=1.0)
    axes[0].set_title(f"S{subject:02d} class {cls + 1}, block {block + 1}, channel {channel}, 2.0s target window")
    axes[0].set_ylabel("Raw")

    for i in range(3):
        axes[1].plot(t, dnn_filtered[i], linewidth=1.0, label=f"DNN FB{i + 1}: {8 * (i + 1)}-90 Hz")
    axes[1].set_ylabel("DNN")
    axes[1].legend(frameon=False, ncol=3, loc="upper right")

    for i in range(5):
        axes[2].plot(t, arena_filtered[i], linewidth=1.0, label=f"Arena FB{i + 1}: {8 * (i + 1)}-90 Hz")
    axes[2].set_ylabel("Arena")
    axes[2].legend(frameon=False, ncol=3, loc="upper right")

    for i in range(3):
        axes[3].plot(t, dnn_filtered[i] - arena_filtered[i], linewidth=1.0, label=f"FB{i + 1} DNN - Arena")
    axes[3].axhline(0, color="#333333", linewidth=0.8)
    axes[3].set_ylabel("Delta")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(frameon=False, ncol=3, loc="upper right")
    for ax in axes:
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    save(fig, "sample_signal_subband_compare")

    freq = np.linspace(0, fs / 2, 2048)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i, sos in enumerate(dnn_sos(fs)):
        w, h = sosfreqz(sos, worN=freq, fs=fs)
        axes[0].plot(w, db(h), linewidth=1.8, label=f"DNN FB{i + 1}: {8 * (i + 1)}-90 Hz")
    axes[0].set_title("Filter magnitude response")
    axes[0].set_ylabel("DNN bandpass (dB)")
    axes[0].set_ylim(-80, 5)
    axes[0].legend(frameon=False, ncol=3, loc="lower right")

    nb, na = signal.iircomb(50, 35, ftype="notch", fs=fs)
    _, notch_h = signal.freqz(nb, na, worN=freq, fs=fs)
    for i, (b, a) in enumerate(toolbox_ba(fs)):
        w, h = signal.freqz(b, a, worN=freq, fs=fs)
        axes[1].plot(w, db(h), linewidth=1.5, label=f"Arena FB{i + 1}: {8 * (i + 1)}-90 Hz")
    axes[1].plot(freq, db(notch_h), color="#111111", linestyle="--", linewidth=1.4, label="Arena 50 Hz notch")
    axes[1].set_ylabel("Arena filters (dB)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_xlim(0, 110)
    axes[1].set_ylim(-80, 5)
    axes[1].legend(frameon=False, ncol=3, loc="lower right")
    for ax in axes:
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    save(fig, "filter_response_compare")

    corr_rows = []
    for i in range(3):
        corr = float(np.corrcoef(dnn_filtered[i], arena_filtered[i])[0, 1])
        rms_delta = float(np.sqrt(np.mean((dnn_filtered[i] - arena_filtered[i]) ** 2)))
        corr_rows.append(f"| FB{i + 1} | {corr:.4f} | {rms_delta:.4f} |")
    report = [
        "# DNN vs Arena Classical Preprocessing, 2s Sample",
        "",
        f"Sample: subject {subject}, class {cls + 1}, block {block + 1}, channel {channel}, window {window:.1f}s.",
        "",
        "DNN path: crop target window after cue+latency, then 3 Chebyshev-I order-2 bandpass filters `[8*i, 90]` Hz with `sosfiltfilt`; no notch.",
        "",
        "Arena classical path: crop cue-to-latency+target segment, apply 50 Hz notch, apply toolbox-style Chebyshev-I filterbank, then remove latency segment.",
        "",
        "## Same-index subband signal similarity",
        "",
        "| Subband | Pearson r | RMS delta |",
        "| --- | ---: | ---: |",
        *corr_rows,
        "",
        "## Figures",
        "",
        "- `sample_signal_subband_compare.png`",
        "- `filter_response_compare.png`",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
