from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import BENCHMARK_FREQS, PROJECT_ROOT, RUN_ROOT
from vep_arena.data.epochs import CanonicalEpochStore, EpochRequest, epoch_fingerprint
from vep_arena.data.presets import benchmark_9ch_default


def parse_targets(text: str) -> list[int]:
    targets: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            targets.extend(range(int(start), int(end) + 1))
        elif part:
            targets.append(int(part))
    return targets


def channel_name_to_index(channel: str, channels: tuple[int, ...]) -> int:
    names = {
        "PZ": 48,
        "PO3": 54,
        "PO5": 55,
        "PO4": 56,
        "PO6": 57,
        "POZ": 58,
        "O1": 61,
        "OZ": 62,
        "O2": 63,
    }
    value = names.get(channel.upper(), None)
    if value is None:
        value = int(channel)
    return channels.index(value)


def plot_time_grid(
    epochs: np.ndarray,
    targets: list[int],
    channel_idx: int,
    subject: int,
    window: float,
    out: Path,
    fs: int,
) -> None:
    t = np.arange(epochs.shape[-1]) / fs
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, target in zip(axes, targets):
        cls = target - 1
        trials = epochs[cls, :, channel_idx, :]
        trials = trials - trials.mean(axis=1, keepdims=True)
        mean = trials.mean(axis=0)
        for block, row in enumerate(trials, start=1):
            ax.plot(t, row, color="#9aa4b2", alpha=0.45, linewidth=0.8, label="blocks" if block == 1 else None)
        ax.plot(t, mean, color="#174a7c", linewidth=1.8, label="mean")
        ax.set_title(f"T{target:02d}  {BENCHMARK_FREQS[cls]:.1f} Hz", fontsize=10)
        ax.grid(alpha=0.22)
    for ax in axes[len(targets) :]:
        ax.axis("off")
    for ax in axes[-4:]:
        ax.set_xlabel("Time after 0.64s crop (s)")
    for ax in axes[::4]:
        ax.set_ylabel("Oz, demeaned")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(f"S{subject} Oz SSVEP Time-Domain Blocks, {window:.1f}s Canonical Epoch", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.98, 0.96))
    fig.savefig(out / "time_grid_8targets.png", dpi=180)
    plt.close(fig)


def harmonic_power(freqs: np.ndarray, psd: np.ndarray, target_freq: float, band: float = 0.25) -> float:
    keep = np.abs(freqs - target_freq) <= band
    if not np.any(keep):
        return float("nan")
    return float(np.max(psd[keep]))


def plot_psd_grid(
    epochs: np.ndarray,
    targets: list[int],
    channel_idx: int,
    subject: int,
    window: float,
    out: Path,
    fs: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.8), sharex=True, sharey=False)
    axes = axes.ravel()
    for ax, target in zip(axes, targets):
        cls = target - 1
        trials = epochs[cls, :, channel_idx, :]
        trials = trials - trials.mean(axis=1, keepdims=True)
        nperseg = epochs.shape[-1]
        freqs, psd = signal.welch(trials, fs=fs, nperseg=nperseg, axis=-1)
        mean_psd = psd.mean(axis=0)
        f0 = BENCHMARK_FREQS[cls]
        for row in psd:
            ax.semilogy(freqs, row, color="#a8b0ba", alpha=0.35, linewidth=0.75)
        ax.semilogy(freqs, mean_psd, color="#174a7c", linewidth=1.8)
        for harmonic in range(1, 5):
            fh = f0 * harmonic
            if fh <= 60:
                ax.axvline(fh, color="#b45309", alpha=0.68, linestyle="--", linewidth=0.9)
                rows.append(
                    {
                        "target": target,
                        "frequency": f0,
                        "harmonic": harmonic,
                        "harmonic_frequency": fh,
                        "peak_power_near_harmonic": harmonic_power(freqs, mean_psd, fh),
                    }
                )
        ax.set_xlim(4, 60)
        ax.set_title(f"T{target:02d}  {f0:.1f} Hz", fontsize=10)
        ax.grid(alpha=0.22)
    for ax in axes[len(targets) :]:
        ax.axis("off")
    for ax in axes[-4:]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[::4]:
        ax.set_ylabel("PSD")
    fig.suptitle(f"S{subject} Oz PSD, {window:.1f}s Canonical Epoch", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.98, 0.96))
    fig.savefig(out / "psd_grid_8targets.png", dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--targets", default="1-8")
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--channel", default="Oz")
    parser.add_argument("--epoch-cache", type=Path, default=RUN_ROOT / "canonical_epochs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "sample_ssvep_grid_s1")
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    if len(targets) > 8:
        raise ValueError("This grid is designed for up to 8 targets.")

    preset = benchmark_9ch_default()
    store = CanonicalEpochStore(args.epoch_cache)
    request = EpochRequest(preset=preset, subject=args.subject, window=args.window, kind="raw")
    epochs = store.load_or_create(request)
    channel_idx = channel_name_to_index(args.channel, preset.channels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_time_grid(epochs, targets, channel_idx, args.subject, args.window, args.output_dir, preset.spec.sampling_rate)
    peaks = plot_psd_grid(epochs, targets, channel_idx, args.subject, args.window, args.output_dir, preset.spec.sampling_rate)
    peaks.to_csv(args.output_dir / "harmonic_peak_table.csv", index=False)

    manifest = {
        "subject": args.subject,
        "targets": targets,
        "target_frequencies": {str(t): BENCHMARK_FREQS[t - 1] for t in targets},
        "channel": args.channel,
        "window": args.window,
        "epoch_fingerprint": epoch_fingerprint(request),
        "files": ["time_grid_8targets.png", "psd_grid_8targets.png", "harmonic_peak_table.csv"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
