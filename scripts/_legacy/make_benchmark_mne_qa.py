# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-20
# Last updated: 2026-06-20
# Description: Generate low-intrusion MNE QA figures for Benchmark epochs.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import BENCHMARK_CHANNELS_9, DATA_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import load_subject_trials
from vep_arena.neuroviz.mne_bridge import EpochMetadata, epochs_array_from_class_block_trials


BENCHMARK_9CH_NAMES = ("Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2")


def save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--class-id", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmark_9ch/mne_qa_smoke"))
    parser.add_argument("--topomap", action="store_true", help="Also try MNE topomap plotting.")
    args = parser.parse_args()

    spec = BenchmarkSpec()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trials = load_subject_trials(
        DATA_ROOT,
        args.subject,
        args.window,
        channels=BENCHMARK_CHANNELS_9,
        spec=spec,
    )
    epochs = epochs_array_from_class_block_trials(
        trials,
        channel_names=BENCHMARK_9CH_NAMES,
        sampling_rate=spec.sampling_rate,
        metadata=EpochMetadata(
            subject=args.subject,
            block=args.block,
            window=args.window,
            preprocess="raw_epoch",
        ),
    )

    class_key = f"target/{args.class_id}"
    evoked = epochs[class_key].average()

    fig = epochs[class_key].plot(show=False, scalings="auto", n_epochs=1, n_channels=len(BENCHMARK_9CH_NAMES))
    save_figure(fig, args.output_dir / "epoch_trace.png")

    fig = evoked.plot(spatial_colors=True, show=False)
    save_figure(fig, args.output_dir / "evoked_trace.png")

    figures = ["epoch_trace.png", "evoked_trace.png"]
    if args.topomap:
        try:
            fig = evoked.plot_topomap(times=np.linspace(0.1, args.window - 0.1, 4), show=False)
            save_figure(fig, args.output_dir / "evoked_topomap.png")
            figures.append("evoked_topomap.png")
        except Exception as exc:
            (args.output_dir / "topomap_error.txt").write_text(str(exc), encoding="utf-8")
    else:
        (args.output_dir / "topomap_skipped.txt").write_text(
            "Topomap is skipped by default. Run with --topomap to try it in a configured MNE plotting environment.\n",
            encoding="utf-8",
        )

    data = epochs.get_data(copy=True)
    freqs, psd = welch(data, fs=spec.sampling_rate, nperseg=min(data.shape[-1], 256), axis=-1)
    mask = (freqs >= 1.0) & (freqs <= 100.0)
    mean_psd = psd.mean(axis=(0, 1))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(freqs[mask], 10 * np.log10(np.maximum(mean_psd[mask], 1e-20)), color="#315aa3", linewidth=1.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title("Mean Welch PSD across exported MNE epochs/channels")
    ax.grid(True, alpha=0.25)
    save_figure(fig, args.output_dir / "psd.png")
    figures.append("psd.png")

    manifest = {
        "dataset": "Tsinghua Benchmark SSVEP",
        "subject": args.subject,
        "block": args.block,
        "window": args.window,
        "class_id": args.class_id,
        "channels": list(BENCHMARK_9CH_NAMES),
        "channel_indices": list(BENCHMARK_CHANNELS_9),
        "sampling_rate": spec.sampling_rate,
        "preprocess": "raw_epoch",
        "figures": figures,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
