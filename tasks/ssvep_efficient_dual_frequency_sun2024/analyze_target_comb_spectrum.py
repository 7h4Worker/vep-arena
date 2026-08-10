from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.dual_frequency import (  # noqa: E402
    iter_sun_cnt_files,
    sun_annotation_rows,
    sun_stimulus_codebook,
    sun_target_repeat_trials,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FS = 250.0
ONSET_SHIFT_SECONDS = 0.14


def parse_subjects(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def target_features(left: float, right: float) -> list[tuple[str, float]]:
    features = [("f_left", left), ("f_right", right)]
    for harmonic in range(2, 5):
        for eye, frequency in (("left", left), ("right", right)):
            value = harmonic * frequency
            if value < 90.0:
                features.append((f"h{harmonic}_{eye}", value))
    features.extend((("sum", left + right), ("abs_diff", abs(left - right))))
    return [(name, value) for name, value in features if 1.0 <= value < 90.0]


def build_swap_map(codebook: list[dict[str, object]]) -> dict[int, int]:
    lookup = {
        (round(float(row["freq_for_left_eye"]), 6), round(float(row["freq_for_right_eye"]), 6)): int(row["trigger_num"])
        for row in codebook
    }
    pairs: dict[int, int] = {}
    for row in codebook:
        left = round(float(row["freq_for_left_eye"]), 6)
        right = round(float(row["freq_for_right_eye"]), 6)
        target = int(row["trigger_num"])
        partner = lookup.get((right, left))
        if partner is None:
            raise ValueError(f"Target {target} has no swapped-frequency partner.")
        pairs[target] = partner
    return pairs


def load_target_psd(record, repetitions: int, window: float) -> tuple[np.ndarray, np.ndarray]:
    import mne

    samples = int(round(window * FS))
    trials = [
        trial
        for trial in sun_target_repeat_trials(sun_annotation_rows(record.path), n_targets=40)
        if trial.repetition <= repetitions
    ]
    if len(trials) != 40 * repetitions:
        raise ValueError(f"Subject {record.subject} has {len(trials)} selected trials, expected {40 * repetitions}.")
    raw = mne.io.read_raw_cnt(record.path, preload=False, verbose="ERROR")
    raw.pick(mne.pick_types(raw.info, eeg=True, meg=False, stim=False, exclude=[]))
    raw.load_data(verbose="ERROR")
    raw.resample(FS, npad="auto", verbose="ERROR")
    data = raw.get_data().astype(np.float64, copy=False)
    per_target: list[list[np.ndarray]] = [[] for _ in range(40)]
    for trial in trials:
        start = int(round((trial.start_seconds + ONSET_SHIFT_SECONDS) * FS))
        epoch = data[:, start : start + samples]
        if epoch.shape[-1] != samples:
            raise ValueError(f"Subject {record.subject}: target {trial.target_id} crosses recording boundary.")
        per_target[trial.target_id - 1].append(epoch)
    if any(len(epochs) != repetitions for epochs in per_target):
        raise ValueError(f"Subject {record.subject}: target repetition coverage is incomplete.")
    nfft = 4 * samples
    freqs, psd = signal.welch(
        np.asarray([np.stack(epochs) for epochs in per_target]),
        fs=FS,
        nperseg=samples,
        nfft=nfft,
        axis=-1,
        detrend="constant",
    )
    return freqs, np.mean(psd, axis=(1, 2))


def nearest_power(freqs: np.ndarray, power_db: np.ndarray, frequency: float) -> tuple[float, float]:
    index = int(np.argmin(np.abs(freqs - frequency)))
    return float(freqs[index]), float(power_db[index])


def main() -> None:
    parser = argparse.ArgumentParser(description="Map target-conditioned spectra before designing Sun2024 comb passbands.")
    parser.add_argument("--subjects", default="11,13")
    parser.add_argument("--repetitions", type=int, choices=[5, 8], default=5)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "target_comb_spectrum")
    args = parser.parse_args()

    codebook = sun_stimulus_codebook(kind="forty_targets")
    if [int(row["trigger_num"]) for row in codebook] != list(range(1, 41)):
        raise ValueError("Expected target triggers 1 through 40 in the Sun2024 codebook.")
    swap_map = build_swap_map(codebook)
    records = {record.subject: record for record in iter_sun_cnt_files(splits=["forty_targets"])}
    subjects = parse_subjects(args.subjects)
    missing = sorted(set(subjects) - set(records))
    if missing:
        raise ValueError(f"Missing subjects: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)

    contrast_rows: list[dict[str, float | int | str]] = []
    heatmaps: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for subject in subjects:
        freqs, psd = load_target_psd(records[subject], args.repetitions, args.window)
        power_db = 10.0 * np.log10(np.maximum(psd, 1e-30))
        heatmaps[subject] = (freqs, power_db)
        for row in codebook:
            target = int(row["trigger_num"])
            target_index = target - 1
            partner = swap_map[target]
            nonpair = [index for index in range(40) if index not in {target_index, partner - 1}]
            for feature, frequency in target_features(float(row["freq_for_left_eye"]), float(row["freq_for_right_eye"])):
                bin_frequency, own_power = nearest_power(freqs, power_db[target_index], frequency)
                _, partner_power = nearest_power(freqs, power_db[partner - 1], frequency)
                _, nonpair_power = nearest_power(freqs, np.median(power_db[nonpair], axis=0), frequency)
                contrast_rows.append(
                    {
                        "subject": subject,
                        "target_id": target,
                        "group": "O" if target <= 20 else "S",
                        "swap_target_id": partner,
                        "feature": feature,
                        "nominal_frequency_hz": frequency,
                        "fft_bin_hz": bin_frequency,
                        "target_power_db": own_power,
                        "swap_power_db": partner_power,
                        "nonpair_median_power_db": nonpair_power,
                        "target_minus_nonpair_db": own_power - nonpair_power,
                        "target_minus_swap_db": own_power - partner_power,
                    }
                )

    with (args.output / "target_feature_contrast.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(contrast_rows[0]))
        writer.writeheader()
        writer.writerows(contrast_rows)

    feature_rows: list[dict[str, float | int | str]] = []
    for subject in subjects:
        for feature in sorted({str(row["feature"]) for row in contrast_rows}):
            values = [float(row["target_minus_nonpair_db"]) for row in contrast_rows if row["subject"] == subject and row["feature"] == feature]
            feature_rows.append(
                {
                    "subject": subject,
                    "feature": feature,
                    "median_target_minus_nonpair_db": float(np.median(values)),
                    "mean_target_minus_nonpair_db": float(np.mean(values)),
                    "fraction_positive": float(np.mean(np.asarray(values) > 0.0)),
                }
            )
    with (args.output / "feature_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)

    fig, axes = plt.subplots(len(subjects), 2, figsize=(13, 4.4 * len(subjects)), dpi=160, gridspec_kw={"width_ratios": (2.1, 1)})
    axes = np.atleast_2d(axes)
    for row, subject in enumerate(subjects):
        freqs, power_db = heatmaps[subject]
        keep = (freqs >= 6.0) & (freqs <= 70.0)
        centered = power_db[:, keep] - np.median(power_db[:, keep], axis=1, keepdims=True)
        image = axes[row, 0].imshow(
            centered,
            aspect="auto",
            origin="lower",
            extent=(float(freqs[keep][0]), float(freqs[keep][-1]), 1, 40),
            cmap="magma",
            vmin=-4,
            vmax=8,
        )
        axes[row, 0].set_title(f"S{subject:02d}: target-conditioned PSD above target median")
        axes[row, 0].set_xlabel("Frequency (Hz)")
        axes[row, 0].set_ylabel("Target ID")
        fig.colorbar(image, ax=axes[row, 0], label="dB")
        rows = [item for item in feature_rows if item["subject"] == subject]
        rows.sort(key=lambda item: float(item["median_target_minus_nonpair_db"]), reverse=True)
        axes[row, 1].barh(
            [str(item["feature"]) for item in rows],
            [float(item["median_target_minus_nonpair_db"]) for item in rows],
            color="#167d65",
        )
        axes[row, 1].axvline(0, color="#333333", linewidth=0.8)
        axes[row, 1].invert_yaxis()
        axes[row, 1].set_title("Median target contrast")
        axes[row, 1].set_xlabel("dB vs non-pair targets")
    fig.tight_layout()
    fig.savefig(args.output / "target_comb_spectrum.png")
    plt.close(fig)

    summary = {
        "task": "ssvep_efficient_dual_frequency_sun2024",
        "subjects": subjects,
        "repetitions": args.repetitions,
        "window_seconds": args.window,
        "fixed_onset_shift_seconds": ONSET_SHIFT_SECONDS,
        "fft_resolution_hz": FS / (4 * int(round(args.window * FS))),
        "physical_epoch_resolution_hz": 1.0 / args.window,
        "group_assignment": "targets 1-20 are O and targets 21-40 are S, per paper Eq. (10)",
        "swap_pairs": swap_map,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
