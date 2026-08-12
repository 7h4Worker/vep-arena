from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.dual_frequency import (  # noqa: E402
    iter_sun_cnt_files,
    sun_annotation_rows,
    sun_target_repeat_trials,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FS = 250.0
ONSET_SHIFT_SECONDS = 0.14


def parse_subjects(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def mean_psd(epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs, psd = signal.welch(epochs, fs=FS, nperseg=epochs.shape[-1], axis=-1, detrend="constant")
    return freqs, np.mean(psd, axis=(0, 1))


def narrow_peaks(freqs: np.ndarray, psd: np.ndarray, state: str, subject: int) -> list[dict[str, float | int | str]]:
    db = 10.0 * np.log10(np.maximum(psd, 1e-30))
    rows: list[dict[str, float | int | str]] = []
    for index in range(3, len(freqs) - 3):
        if not 1.0 <= freqs[index] <= 110.0:
            continue
        local = np.median(np.concatenate((db[index - 3 : index - 1], db[index + 2 : index + 4])))
        rows.append(
            {
                "subject": subject,
                "state": state,
                "frequency_hz": float(freqs[index]),
                "power_db": float(db[index]),
                "local_baseline_db": float(local),
                "narrow_peak_db": float(db[index] - local),
            }
        )
    return sorted(rows, key=lambda row: float(row["narrow_peak_db"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect evidence for an author comb filter in Sun2024 CNT recordings.")
    parser.add_argument("--subjects", default="11,13")
    parser.add_argument("--repetitions", type=int, default=5, choices=[5, 8])
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "comb_filter_psd",
    )
    args = parser.parse_args()

    import mne

    args.output.mkdir(parents=True, exist_ok=True)
    records = {record.subject: record for record in iter_sun_cnt_files(splits=["forty_targets"])}
    subjects = parse_subjects(args.subjects)
    missing = sorted(set(subjects) - set(records))
    if missing:
        raise ValueError(f"Missing subjects: {missing}")
    samples = int(round(args.window * FS))
    psd_rows: list[dict[str, float | int | str]] = []
    peak_rows: list[dict[str, float | int | str]] = []
    plotted: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}

    for subject in subjects:
        record = records[subject]
        trials = [
            trial
            for trial in sun_target_repeat_trials(sun_annotation_rows(record.path), n_targets=40)
            if trial.repetition <= args.repetitions
        ]
        raw = mne.io.read_raw_cnt(record.path, preload=False, verbose="ERROR")
        raw.pick(mne.pick_types(raw.info, eeg=True, meg=False, stim=False, exclude=[]))
        raw.load_data(verbose="ERROR")
        raw.resample(FS, npad="auto", verbose="ERROR")
        data = raw.get_data().astype(np.float64, copy=False)
        stimulation: list[np.ndarray] = []
        cue: list[np.ndarray] = []
        for trial in trials:
            stim_start = int(round((trial.start_seconds + ONSET_SHIFT_SECONDS) * FS))
            cue_start = int(round((trial.start_seconds - args.window) * FS))
            if cue_start < 0 or stim_start + samples > data.shape[-1]:
                continue
            stimulation.append(data[:, stim_start : stim_start + samples])
            cue.append(data[:, cue_start : cue_start + samples])
        if len(stimulation) != len(trials):
            raise ValueError(f"Subject {subject}: an epoch crossed a recording boundary.")
        for state, epochs in (("cue_pre_stimulus", np.stack(cue)), ("stimulation", np.stack(stimulation))):
            freqs, psd = mean_psd(epochs)
            plotted[(subject, state)] = (freqs, psd)
            for freq, power in zip(freqs, psd):
                psd_rows.append(
                    {
                        "subject": subject,
                        "state": state,
                        "frequency_hz": float(freq),
                        "power_db": float(10.0 * np.log10(max(power, 1e-30))),
                    }
                )
            peak_rows.extend(narrow_peaks(freqs, psd, state, subject))

    with (args.output / "psd.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("subject", "state", "frequency_hz", "power_db"))
        writer.writeheader()
        writer.writerows(psd_rows)
    with (args.output / "narrow_peaks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("subject", "state", "frequency_hz", "power_db", "local_baseline_db", "narrow_peak_db"),
        )
        writer.writeheader()
        writer.writerows(peak_rows)

    fig, axes = plt.subplots(len(subjects), 2, figsize=(12, 3.4 * len(subjects)), dpi=160, sharex="col")
    axes = np.atleast_2d(axes)
    for row, subject in enumerate(subjects):
        for state, color in (("cue_pre_stimulus", "#5e6b7a"), ("stimulation", "#167d65")):
            freqs, psd = plotted[(subject, state)]
            power = 10.0 * np.log10(np.maximum(psd, 1e-30))
            axes[row, 0].plot(freqs, power, label=state, color=color, linewidth=1.2)
            axes[row, 1].plot(freqs, power, label=state, color=color, linewidth=1.2)
        for axis, limits in ((axes[row, 0], (1, 110)), (axes[row, 1], (42, 58))):
            axis.axvline(50, color="#b63a3a", linestyle="--", linewidth=0.9, label="50 Hz")
            axis.set_xlim(*limits)
            axis.set_ylabel(f"S{subject:02d} PSD (dB)")
            axis.grid(alpha=0.25)
        axes[row, 0].legend(frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel("Frequency (Hz)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_title("Cue-period and stimulation PSD")
    axes[0, 1].set_title("50 Hz neighbourhood")
    fig.tight_layout()
    fig.savefig(args.output / "comb_filter_psd.png")
    plt.close(fig)

    at_50 = [row for row in psd_rows if abs(float(row["frequency_hz"]) - 50.0) < 1e-9]
    summary = {
        "task": "ssvep_efficient_dual_frequency_sun2024",
        "subjects": subjects,
        "repetitions": args.repetitions,
        "window_seconds": args.window,
        "fixed_onset_shift_seconds": ONSET_SHIFT_SECONDS,
        "analysis": "Welch PSD averaged across target-repetition epochs and EEG channels.",
        "author_comb_status": "paper states comb filtering but publishes neither code nor parameters",
        "power_at_50hz_db": at_50,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
