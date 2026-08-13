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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vep_arena.config import BENCHMARK_CHANNELS_9, DATA_ROOT, PROJECT_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import benchmark_filterbank, load_subject_raw, notch_50hz
from vep_arena.methods.traditional import FBCCA


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def load_epochs_from_start(
    raw: np.ndarray,
    start_seconds: float,
    window: float,
    n_fbs: int,
    spec: BenchmarkSpec,
) -> np.ndarray:
    """Return classes x blocks x subbands x channels x samples for an absolute trial start."""

    channel_idx = np.asarray(BENCHMARK_CHANNELS_9, dtype=np.int64) - 1
    start = round(start_seconds * spec.sampling_rate)
    samples = spec.sample_length(window)
    stop = start + samples
    out = np.zeros((spec.classes, spec.blocks, n_fbs, len(channel_idx), samples), dtype=np.float64)
    for cls in range(spec.classes):
        for block in range(spec.blocks):
            trial = raw[channel_idx, start:stop, cls, block]
            trial = notch_50hz(trial, spec.sampling_rate)
            out[cls, block] = benchmark_filterbank(trial, spec.sampling_rate, n_fbs)
    return out


def run_offset_sweep(
    subject: int,
    starts: list[float],
    windows: list[float],
    n_fbs: int,
    harmonics: int,
    out: Path,
    spec: BenchmarkSpec,
) -> pd.DataFrame:
    raw = load_subject_raw(DATA_ROOT, subject)
    labels = np.arange(spec.classes, dtype=np.int64)
    rows: list[dict[str, object]] = []
    for window in windows:
        for start in starts:
            epochs = load_epochs_from_start(raw, start, window, n_fbs, spec)
            model = FBCCA(window=window, harmonics=harmonics, n_fbs=n_fbs, spec=spec)
            block_acc = []
            for block in range(spec.blocks):
                train_blocks = [b for b in range(spec.blocks) if b != block]
                train_x = epochs[:, train_blocks].reshape(-1, epochs.shape[2], epochs.shape[3], epochs.shape[4])
                train_y = np.repeat(labels, len(train_blocks))
                test_x = epochs[:, block]
                model.fit(train_x, train_y)
                pred, _ = model.predict(test_x)
                acc = np.mean(pred == labels)
                block_acc.append(float(acc))
                rows.append(
                    {
                        "subject": subject,
                        "method": "FBCCA",
                        "window": window,
                        "start_seconds": start,
                        "block": block + 1,
                        "accuracy": float(acc),
                    }
                )
            print(f"S{subject:02d} FBCCA window={window:.1f} start={start:.2f} acc={np.mean(block_acc):.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "offset_trials.csv", index=False)
    summary = (
        df.groupby(["method", "window", "start_seconds"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"))
        .fillna({"accuracy_sd": 0.0})
    )
    summary.to_csv(out / "offset_summary.csv", index=False)
    return summary


def plot_summary(summary: pd.DataFrame, out: Path, spec: BenchmarkSpec) -> None:
    plt.figure(figsize=(9.2, 5.4))
    for window, group in summary.groupby("window"):
        plt.errorbar(
            group["start_seconds"],
            group["accuracy"],
            yerr=group["accuracy_sd"],
            marker="o",
            capsize=3,
            label=f"{window:.1f}s",
        )
    plt.axvline(spec.cue_seconds, color="#b45309", linestyle="--", linewidth=1.2, label="cue offset 0.50s")
    plt.axvline(
        spec.cue_seconds + spec.latency_seconds,
        color="#047857",
        linestyle="--",
        linewidth=1.2,
        label="cue+latency 0.64s",
    )
    plt.xlabel("Absolute crop start from raw trial start (s)")
    plt.ylabel("Leave-one-block FBCCA accuracy")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.24)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "offset_accuracy_curve.png", dpi=180)
    plt.close()


def markdown_table(df: pd.DataFrame) -> str:
    cols = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            vals.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, out: Path, spec: BenchmarkSpec) -> None:
    wide = summary.pivot(index="window", columns="start_seconds", values="accuracy").reset_index()
    lines = [
        "# Benchmark Offset Check",
        "",
        "This diagnostic treats crop start as an absolute time from raw trial start.",
        f"The standard Benchmark path is cue offset plus visual latency: {spec.cue_seconds:.2f} + {spec.latency_seconds:.2f} = {spec.cue_seconds + spec.latency_seconds:.2f}s.",
        "",
        "## Mean FBCCA Accuracy",
        "",
        markdown_table(wide),
        "",
        "## Interpretation",
        "",
        "- A start near 0.14s tests the mistaken assumption that raw trials begin at visual stimulus onset.",
        "- A start at 0.50s skips cue/pre-stim but keeps the first visual-latency samples.",
        "- A start at 0.64s matches the conventional Benchmark protocol.",
        "- Later starts can be useful diagnostics for steady-state quality, but they are a different evaluation window.",
        "",
        "## Files",
        "",
        "- `offset_trials.csv`",
        "- `offset_summary.csv`",
        "- `offset_accuracy_curve.png`",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--starts", default="0.14,0.50,0.64,1.00,1.14")
    parser.add_argument("--windows", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--n-fbs", type=int, default=5)
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "offset_check_s1")
    args = parser.parse_args()

    spec = BenchmarkSpec()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    starts = parse_floats(args.starts)
    windows = parse_floats(args.windows)
    started = time.perf_counter()
    summary = run_offset_sweep(args.subject, starts, windows, args.n_fbs, args.harmonics, args.output_dir, spec)
    plot_summary(summary, args.output_dir, spec)
    write_report(summary, args.output_dir, spec)
    manifest = {
        "subject": args.subject,
        "dataset": "Tsinghua Benchmark SSVEP",
        "diagnostic": "absolute crop-start sweep",
        "starts": starts,
        "windows": windows,
        "cue_seconds": spec.cue_seconds,
        "latency_seconds": spec.latency_seconds,
        "standard_start_seconds": spec.cue_seconds + spec.latency_seconds,
        "seconds": time.perf_counter() - started,
        "files": ["offset_trials.csv", "offset_summary.csv", "offset_accuracy_curve.png", "report.md"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
