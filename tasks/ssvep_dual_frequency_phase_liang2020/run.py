from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.dual_frequency import (  # noqa: E402
    LIANG2020_CHANNELS,
    LIANG2020_SAMPLING_RATE,
    iter_liang_mat_files,
    liang2020_root,
    load_liang_mat,
)


def parse_subjects(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(item) for item in part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    return values


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def select_records(records, max_files_per_experiment: int):
    buckets: dict[str, list[object]] = defaultdict(list)
    for record in records:
        if len(buckets[record.experiment]) < max_files_per_experiment:
            buckets[record.experiment].append(record)
    selected = []
    for experiment in sorted(buckets):
        selected.extend(buckets[experiment])
    return selected


def plot_time_frequency(epoch, output: Path, title: str) -> None:
    fs = LIANG2020_SAMPLING_RATE
    x = epoch.x
    first_trial = x[:, :, 0]
    time = np.arange(first_trial.shape[1]) / fs
    channel_indices = [0, len(LIANG2020_CHANNELS) - 2]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for idx in channel_indices:
        axes[0].plot(time, first_trial[idx], lw=0.8, label=LIANG2020_CHANNELS[idx])
    axes[0].set_title(f"{title} time-domain check")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")

    for idx in channel_indices:
        freqs, psd = signal.welch(first_trial[idx], fs=fs, nperseg=min(first_trial.shape[1], fs * 2))
        keep = (freqs >= 1.0) & (freqs <= 60.0)
        axes[1].plot(freqs[keep], psd[keep], lw=0.9, label=LIANG2020_CHANNELS[idx])
    axes[1].set_title("Welch PSD check")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power")
    axes[1].legend(loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke inventory for Liang2020 dual-frequency phase SSVEP data.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "smoke_inventory")
    parser.add_argument("--subjects", default="1")
    parser.add_argument("--experiments", default="exp1,exp2")
    parser.add_argument("--max-files-per-experiment", type=int, default=4)
    args = parser.parse_args()

    root = liang2020_root(args.root)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    experiments = parse_csv(args.experiments)

    records = iter_liang_mat_files(root=root, experiments=experiments, subjects=subjects)
    selected = select_records(records, max_files_per_experiment=args.max_files_per_experiment)
    if not selected:
        raise SystemExit("No Liang2020 .mat files matched the requested smoke scope.")

    file_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    first_epoch = None
    first_title = ""
    for record in selected:
        epoch = load_liang_mat(record.path)
        if first_epoch is None:
            first_epoch = epoch
            first_title = f"{record.experiment} sub-{record.subject:02d} condition-{record.condition} run-{record.run}"
        rel_path = record.path.relative_to(root)
        duration = epoch.n_samples / LIANG2020_SAMPLING_RATE
        file_rows.append(
            {
                "dataset": "liang2020_dual_frequency_phase",
                "experiment": record.experiment,
                "batch": record.batch,
                "subject": record.subject,
                "condition": record.condition,
                "run": record.run,
                "file": str(rel_path),
                "loader": epoch.engine,
                "channels": epoch.n_channels,
                "samples": epoch.n_samples,
                "trials": epoch.n_trials,
                "duration_seconds": duration,
                "label_min": int(epoch.labels.min()),
                "label_max": int(epoch.labels.max()),
                "unique_labels": int(np.unique(epoch.labels).size),
            }
        )
        for trial_index, label in enumerate(epoch.labels):
            trial_rows.append(
                {
                    "dataset": "liang2020_dual_frequency_phase",
                    "experiment": record.experiment,
                    "subject": record.subject,
                    "condition": record.condition,
                    "run": record.run,
                    "trial_index": int(trial_index),
                    "target_id": int(label),
                    "samples": epoch.n_samples,
                    "duration_seconds": duration,
                }
            )

    files_df = pd.DataFrame(file_rows)
    trials_df = pd.DataFrame(trial_rows)
    summary_df = (
        files_df.groupby(["experiment", "subject", "condition"], as_index=False)
        .agg(
            files=("file", "count"),
            trials=("trials", "sum"),
            samples_min=("samples", "min"),
            samples_max=("samples", "max"),
            label_min=("label_min", "min"),
            label_max=("label_max", "max"),
            unique_labels_max=("unique_labels", "max"),
        )
        .sort_values(["experiment", "subject", "condition"])
    )

    files_df.to_csv(output / "files.csv", index=False)
    trials_df.to_csv(output / "trials.csv", index=False)
    summary_df.to_csv(output / "summary.csv", index=False)
    plot_time_frequency(first_epoch, output / "liang2020_smoke_time_frequency.png", first_title)

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "subjects": subjects,
        "experiments": experiments,
        "max_files_per_experiment": args.max_files_per_experiment,
        "sampling_rate": LIANG2020_SAMPLING_RATE,
        "channels": list(LIANG2020_CHANNELS),
        "files": len(files_df),
        "trials": len(trials_df),
        "outputs": {
            "files": "files.csv",
            "trials": "trials.csv",
            "summary": "summary.csv",
            "qa_figure": "liang2020_smoke_time_frequency.png",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
