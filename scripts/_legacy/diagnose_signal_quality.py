# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import BENCHMARK_CHANNELS_9, BENCHMARK_FREQS, DATA_ROOT, PROJECT_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import load_subject_filterbank, load_subject_raw, load_subject_toolbox_epochs
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.traditional import CCA, FBCCA, TRCA


def parse_windows(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def oz_channel_index(channels: tuple[int, ...]) -> int:
    return channels.index(62)


def plot_oz_time(raw: np.ndarray, subject: int, target: int, out: Path, spec: BenchmarkSpec) -> None:
    cls = target - 1
    oz = 62 - 1
    fs = spec.sampling_rate
    seconds = raw.shape[1] / fs
    t = np.arange(raw.shape[1]) / fs
    trials = raw[oz, :, cls, :].T
    mean = trials.mean(axis=0)

    plt.figure(figsize=(10, 5.2))
    for block_idx, trial in enumerate(trials, start=1):
        plt.plot(t, trial, color="#9aa4b2", alpha=0.42, linewidth=0.9, label="blocks" if block_idx == 1 else None)
    plt.plot(t, mean, color="#1b4d89", linewidth=2.0, label="mean")
    plt.axvline(spec.cue_seconds, color="#b45309", linestyle="--", linewidth=1.4, label="stimulus onset")
    plt.axvline(spec.cue_seconds + spec.latency_seconds, color="#047857", linestyle="--", linewidth=1.4, label="latency-corrected start")
    plt.xlim(0, min(2.6, seconds))
    plt.xlabel("Time from trial start (s)")
    plt.ylabel("Oz amplitude")
    plt.title(f"S{subject} Target {target} ({BENCHMARK_FREQS[cls]:.1f} Hz) Oz, raw blocks")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.22)
    plt.tight_layout()
    plt.savefig(out / "oz_time_overlay_target01.png", dpi=180)
    plt.close()

    start = round((spec.cue_seconds + spec.latency_seconds) * fs)
    stop = start + round(1.5 * fs)
    tw = (np.arange(stop - start) / fs)
    segs = trials[:, start:stop]
    segs = segs - segs.mean(axis=1, keepdims=True)
    plt.figure(figsize=(10, 5.2))
    for block_idx, trial in enumerate(segs, start=1):
        plt.plot(tw, trial, color="#9aa4b2", alpha=0.46, linewidth=0.9, label="blocks" if block_idx == 1 else None)
    plt.plot(tw, segs.mean(axis=0), color="#1b4d89", linewidth=2.0, label="mean")
    plt.xlabel("Time from latency-corrected stimulus start (s)")
    plt.ylabel("Oz amplitude, demeaned")
    plt.title(f"S{subject} Target {target} Oz, first 1.5 s response window")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.22)
    plt.tight_layout()
    plt.savefig(out / "oz_response_overlay_target01.png", dpi=180)
    plt.close()


def plot_oz_spectrum(raw: np.ndarray, subject: int, target: int, out: Path, spec: BenchmarkSpec) -> None:
    cls = target - 1
    oz = 62 - 1
    fs = spec.sampling_rate
    start = round((spec.cue_seconds + spec.latency_seconds) * fs)
    samples = round(2.0 * fs)
    trials = raw[oz, start : start + samples, cls, :].T
    trials = trials - trials.mean(axis=1, keepdims=True)
    freqs, psd = signal.welch(trials, fs=fs, nperseg=min(samples, 500), axis=-1)
    mean_psd = psd.mean(axis=0)

    plt.figure(figsize=(10, 5.2))
    for row in psd:
        plt.semilogy(freqs, row, color="#9aa4b2", alpha=0.34, linewidth=0.9)
    plt.semilogy(freqs, mean_psd, color="#1b4d89", linewidth=2.0, label="mean PSD")
    f0 = BENCHMARK_FREQS[cls]
    for harmonic in range(1, 6):
        plt.axvline(f0 * harmonic, color="#b45309", alpha=0.68, linestyle="--", linewidth=1.0)
    plt.xlim(1, 60)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(f"S{subject} Target {target} Oz PSD, 2.0 s after latency ({f0:.1f} Hz)")
    plt.grid(alpha=0.22)
    plt.tight_layout()
    plt.savefig(out / "oz_psd_target01.png", dpi=180)
    plt.close()

    f, tt, sxx = signal.spectrogram(trials.mean(axis=0), fs=fs, nperseg=128, noverlap=96)
    plt.figure(figsize=(10, 5.2))
    keep = f <= 60
    plt.pcolormesh(tt, f[keep], 10 * np.log10(sxx[keep] + 1e-12), shading="auto", cmap="magma")
    plt.colorbar(label="Power (dB)")
    for harmonic in range(1, 6):
        plt.axhline(f0 * harmonic, color="cyan", alpha=0.55, linestyle="--", linewidth=0.9)
    plt.xlabel("Time from latency-corrected stimulus start (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"S{subject} Target {target} Oz mean-trial spectrogram")
    plt.tight_layout()
    plt.savefig(out / "oz_spectrogram_target01.png", dpi=180)
    plt.close()


def run_window_trend(subject: int, windows: list[float], out: Path, spec: BenchmarkSpec) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = np.arange(spec.classes, dtype=np.int64)
    for window in windows:
        raw_epochs = None
        filtered_epochs = None
        for method in ("CCA", "FBCCA", "TRCA"):
            if method == "CCA":
                if raw_epochs is None:
                    raw = load_subject_raw(DATA_ROOT, subject)
                    channel_idx = np.asarray(BENCHMARK_CHANNELS_9, dtype=np.int64) - 1
                    start = round((spec.cue_seconds + spec.latency_seconds) * spec.sampling_rate)
                    stop = start + spec.sample_length(window)
                    raw_epochs = np.transpose(raw[channel_idx, start:stop, :, :], (2, 3, 0, 1))[:, :, None, :, :].copy()
                epochs = raw_epochs
                model_factory = lambda: CCA(window=window, harmonics=5, spec=spec)
            else:
                if filtered_epochs is None:
                    filtered_epochs = load_subject_toolbox_epochs(DATA_ROOT, subject, window, n_fbs=5, spec=spec)
                epochs = filtered_epochs
                if method == "FBCCA":
                    model_factory = lambda: FBCCA(window=window, harmonics=5, n_fbs=5, spec=spec)
                else:
                    model_factory = lambda: TRCA(n_fbs=5, ensemble=False)
            for block in range(spec.blocks):
                train_blocks = [b for b in range(spec.blocks) if b != block]
                train_x = epochs[:, train_blocks].reshape(-1, epochs.shape[2], epochs.shape[3], epochs.shape[4])
                train_y = np.repeat(labels, len(train_blocks))
                test_x = epochs[:, block]
                model = model_factory()
                t0 = time.perf_counter()
                model.fit(train_x, train_y)
                pred, _ = model.predict(test_x)
                seconds = time.perf_counter() - t0
                acc = accuracy_score(labels, pred)
                rows.append(
                    {
                        "subject": subject,
                        "method": method,
                        "window": window,
                        "block": block + 1,
                        "accuracy": float(acc),
                        "itr": itr_bits_per_minute(float(acc), spec.classes, window + spec.cue_seconds),
                        "seconds": seconds,
                    }
                )
                print(f"S{subject:02d} {method} w={window:.1f} b={block + 1} acc={acc:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "window_trend_subject.csv", index=False)
    summary = df.groupby(["method", "window"], as_index=False).agg(accuracy=("accuracy", "mean"), itr=("itr", "mean"))
    summary.to_csv(out / "window_trend_summary.csv", index=False)

    plt.figure(figsize=(9.2, 5.2))
    for method, group in summary.groupby("method"):
        plt.plot(group["window"], group["accuracy"], marker="o", label=method)
    plt.xlabel("Window (s)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.title(f"S{subject} CCA/FBCCA/TRCA accuracy vs window")
    plt.tight_layout()
    plt.savefig(out / "window_accuracy_subject.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9.2, 5.2))
    for method, group in summary.groupby("method"):
        plt.plot(group["window"], group["itr"], marker="o", label=method)
    plt.xlabel("Window (s)")
    plt.ylabel("ITR (bits/min)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.title(f"S{subject} CCA/FBCCA/TRCA ITR vs window")
    plt.tight_layout()
    plt.savefig(out / "window_itr_subject.png", dpi=180)
    plt.close()
    return summary


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--windows", default="0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "signal_diagnostics_s1")
    args = parser.parse_args()

    spec = BenchmarkSpec()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_subject_raw(DATA_ROOT, args.subject)
    plot_oz_time(raw, args.subject, args.target, args.output_dir, spec)
    plot_oz_spectrum(raw, args.subject, args.target, args.output_dir, spec)
    summary = run_window_trend(args.subject, parse_windows(args.windows), args.output_dir, spec)

    manifest = {
        "subject": args.subject,
        "target": args.target,
        "target_frequency": BENCHMARK_FREQS[args.target - 1],
        "channel": "Oz",
        "cue_seconds": spec.cue_seconds,
        "latency_seconds": spec.latency_seconds,
        "windows": parse_windows(args.windows),
        "files": [
            "oz_time_overlay_target01.png",
            "oz_response_overlay_target01.png",
            "oz_psd_target01.png",
            "oz_spectrogram_target01.png",
            "window_accuracy_subject.png",
            "window_itr_subject.png",
            "window_trend_subject.csv",
            "window_trend_summary.csv",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Signal Diagnostics",
        "",
        f"Subject: S{args.subject}",
        f"Target: {args.target} ({BENCHMARK_FREQS[args.target - 1]:.1f} Hz)",
        "Channel: Oz",
        "",
        "The time plots overlay the six blocks for one target and mark stimulus onset at 0.5 s plus the latency-corrected analysis start at 0.64 s.",
        "The PSD and spectrogram use the first 2.0 s after the latency-corrected start.",
        "",
        "## Window Trend",
        "",
        markdown_table(summary),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
