from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.dual_frequency import (  # noqa: E402
    SUN2024_OCCIPITAL9,
    SUN2024_SPLITS,
    iter_sun_cnt_files,
    sun_annotation_rows,
    sun_target_repeat_trials,
    sun2024_root,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_subjects(text: str, available: list[int]) -> list[int]:
    if text.strip().lower() == "all":
        return available
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(value) for value in part.split("-", 1))
            values.extend(range(start, stop + 1))
        else:
            values.append(int(part))
    return sorted(dict.fromkeys(values))


def protocol_name(repetitions: Counter[int]) -> str:
    counts = set(repetitions.values())
    if len(repetitions) != 40 or len(counts) != 1:
        return "offline_irregular_target_repetitions"
    count = next(iter(counts))
    if count == 5:
        return "offline_paper_5_repetitions"
    if count == 8:
        return "offline_extended_8_repetitions"
    return f"offline_{count}_repetitions_per_target"


def plot_time_frequency(record, trial, output: Path) -> None:
    import mne

    raw = mne.io.read_raw_cnt(record.path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    samples = min(int(round(2.0 * sfreq)), raw.n_times - trial.start_sample)
    picks = [channel for channel in SUN2024_OCCIPITAL9 if channel in raw.ch_names]
    data = raw.get_data(picks=picks[:3], start=trial.start_sample, stop=trial.start_sample + samples)
    time = np.arange(samples) / sfreq
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5), constrained_layout=True)
    for index, channel in enumerate(picks[:3]):
        axes[0].plot(time, data[index] * 1e6, linewidth=0.75, label=channel)
    axes[0].set_title(f"Sun2024 offline S{record.subject:02d} target {trial.target_id}, repetition {trial.repetition}")
    axes[0].set_xlabel("Stimulus time (s)")
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].legend(loc="upper right")
    for index, channel in enumerate(picks[:3]):
        frequency, power = signal.welch(data[index], fs=sfreq, nperseg=samples)
        keep = (frequency >= 1.0) & (frequency <= 60.0)
        axes[1].plot(frequency[keep], power[keep], linewidth=0.9, label=channel)
    axes[1].set_title("Raw-event spectral sanity check")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power")
    axes[1].legend(loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol smoke for Sun2024 public offline CNT data.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "protocol_smoke")
    args = parser.parse_args()

    root = sun2024_root(args.root)
    records = iter_sun_cnt_files(root=root, splits=["forty_targets"])
    subjects = parse_subjects(args.subjects, sorted(record.subject for record in records))
    records = [record for record in records if record.subject in set(subjects)]
    if not records:
        raise ValueError("No forty-target offline records matched the requested subjects.")
    args.output.mkdir(parents=True, exist_ok=True)

    file_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    repetition_rows: list[dict[str, object]] = []
    figure_record = next((record for record in records if record.subject == 10), records[0])
    figure_trial = None
    for record in records:
        import mne

        raw = mne.io.read_raw_cnt(record.path, preload=False, verbose="ERROR")
        trials = sun_target_repeat_trials(sun_annotation_rows(record.path), n_targets=40)
        repetitions = Counter(trial.target_id for trial in trials)
        name = protocol_name(repetitions)
        if figure_trial is None and record.subject == figure_record.subject:
            figure_trial = trials[0]
        file_rows.append(
            {
                "split": "forty_targets_offline",
                "subject": record.subject,
                "file": str(record.path.relative_to(root)),
                "sampling_rate": float(raw.info["sfreq"]),
                "channels": len(raw.ch_names),
                "stimulus_trials": len(trials),
                "min_repetitions_per_target": min(repetitions.values()),
                "max_repetitions_per_target": max(repetitions.values()),
                "protocol": name,
            }
        )
        for target_id in range(1, 41):
            repetition_rows.append(
                {
                    "subject": record.subject,
                    "target_id": target_id,
                    "repetitions": repetitions[target_id],
                    "protocol": name,
                }
            )
        for trial in trials:
            trial_rows.append(
                {
                    "split": "forty_targets_offline",
                    "subject": record.subject,
                    "trial_index": trial.trial_index,
                    "target_id": trial.target_id,
                    "repetition": trial.repetition,
                    "start_sample": trial.start_sample,
                    "end_sample": trial.end_sample,
                    "start_seconds": trial.start_seconds,
                    "end_seconds": trial.end_seconds,
                    "duration_seconds": trial.duration_seconds,
                    "protocol": name,
                }
            )

    files = pd.DataFrame(file_rows)
    trials = pd.DataFrame(trial_rows)
    repetitions = pd.DataFrame(repetition_rows)
    protocol = (
        files.groupby("protocol", as_index=False)
        .agg(subjects=("subject", "nunique"), trials=("stimulus_trials", "sum"), channels=("channels", "min"))
        .sort_values("protocol")
    )
    files.to_csv(args.output / "files.csv", index=False)
    trials.to_csv(args.output / "trial_ledger.csv", index=False)
    repetitions.to_csv(args.output / "target_repetitions.csv", index=False)
    protocol.to_csv(args.output / "protocol_summary.csv", index=False)
    if figure_trial is not None:
        plot_time_frequency(figure_record, figure_trial, args.output / "sun2024_protocol_smoke_time_frequency.png")
    manifest = {
        "task": "ssvep_efficient_dual_frequency_sun2024",
        "available_local_splits": sorted(SUN2024_SPLITS),
        "online_data_available": False,
        "subjects": subjects,
        "split": "forty_targets_offline",
        "split_unit": "target_repetition",
        "outputs": ["files.csv", "trial_ledger.csv", "target_repetitions.csv", "protocol_summary.csv", "sun2024_protocol_smoke_time_frequency.png"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
