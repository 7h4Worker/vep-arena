from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.dual_frequency import (  # noqa: E402
    iter_sun_cnt_files,
    sun_annotation_rows,
    sun_stimulus_codebook,
    sun_target_repeat_trials,
)
from vep_arena.methods.trca_core import corr_rows, trca_filter  # noqa: E402
from vep_arena.signal.filters import target_band_comb_filter  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FS = 250.0
ONSET_SHIFT_SECONDS = 0.14


def parse_subjects(text: str, available: list[int]) -> list[int]:
    if text.strip().lower() == "all":
        return available
    return sorted({int(item.strip()) for item in text.split(",") if item.strip()})


def parse_windows(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_pair_centers(codebook: list[dict[str, object]]) -> dict[int, tuple[float, ...]]:
    by_frequency = {
        (round(float(row["freq_for_left_eye"]), 6), round(float(row["freq_for_right_eye"]), 6)): int(row["trigger_num"])
        for row in codebook
    }
    centers: dict[int, tuple[float, ...]] = {}
    for row in codebook:
        left = float(row["freq_for_left_eye"])
        right = float(row["freq_for_right_eye"])
        target = int(row["trigger_num"])
        partner = by_frequency.get((round(right, 6), round(left, 6)))
        if partner is None:
            raise ValueError(f"Target {target} has no swapped-frequency partner.")
        pair_id = min(target, partner)
        values = (left, right, 2.0 * left, 2.0 * right, left + right)
        centers[pair_id] = tuple(sorted({round(value, 6) for value in values if 1.0 <= value < 90.0}))
    if len(centers) != 20:
        raise ValueError(f"Expected 20 swapped pairs, got {len(centers)}.")
    return centers


def itr_bits_per_min(accuracy: float, window: float) -> float:
    p = min(max(float(accuracy), 1e-12), 1.0 - 1e-12)
    bits = math.log2(40) + p * math.log2(p) + (1.0 - p) * math.log2((1.0 - p) / 39.0)
    return bits * 60.0 / (window + 0.5)


def write_confusion(path: Path, true: np.ndarray, predicted: np.ndarray, title: str) -> None:
    matrix = confusion_matrix(true, predicted, labels=list(range(40)))
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    image = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted target")
    ax.set_ylabel("True target")
    ax.set_xticks(np.arange(0, 40, 5))
    ax.set_yticks(np.arange(0, 40, 5))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_subject(record, repetitions: int):
    import mne

    trials = [
        trial
        for trial in sun_target_repeat_trials(sun_annotation_rows(record.path), n_targets=40)
        if trial.repetition <= repetitions
    ]
    labels = np.asarray([trial.target_id - 1 for trial in trials], dtype=np.int64)
    folds = np.asarray([trial.repetition for trial in trials], dtype=np.int64)
    if Counter(labels.tolist()) != Counter({target: repetitions for target in range(40)}):
        raise ValueError(f"Subject {record.subject}: target repetitions are incomplete.")
    raw = mne.io.read_raw_cnt(record.path, preload=False, verbose="ERROR")
    raw.pick(mne.pick_types(raw.info, eeg=True, meg=False, stim=False, exclude=[]))
    raw.load_data(verbose="ERROR")
    raw.resample(FS, npad="auto", verbose="ERROR")
    starts = np.asarray([int(round((trial.start_seconds + ONSET_SHIFT_SECONDS) * FS)) for trial in trials], dtype=np.int64)
    return raw.get_data().astype(np.float64, copy=False), starts, labels, folds


def class_score(
    epochs: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    class_id: int,
    repetitions: int,
) -> np.ndarray:
    scores = np.empty(labels.size, dtype=np.float64)
    for fold in range(1, repetitions + 1):
        train = (labels == class_id) & (folds != fold)
        test = folds == fold
        spatial = trca_filter(epochs[train])
        template = np.mean(epochs[train], axis=0)
        projected_test = epochs[test].transpose(0, 2, 1) @ spatial
        projected_template = template.T @ spatial
        scores[test] = corr_rows(projected_test, projected_template)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-conditioned passband-comb TRCA smoke for Sun2024.")
    parser.add_argument("--subjects", default="11,13")
    parser.add_argument("--repetitions", type=int, choices=[5, 8], default=5)
    parser.add_argument("--windows", default="0.6,1.0")
    parser.add_argument("--half-width", type=float, default=0.5)
    parser.add_argument("--comb-mode", choices=["pair_specific", "global_union"], default="pair_specific")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "target_comb_trca_smoke",
    )
    args = parser.parse_args()

    codebook = sun_stimulus_codebook(kind="forty_targets")
    pair_centers = build_pair_centers(codebook)
    global_centers = tuple(sorted({center for centers in pair_centers.values() for center in centers}))
    target_pair = {}
    for target, row in enumerate(codebook, start=1):
        left = round(float(row["freq_for_left_eye"]), 6)
        right = round(float(row["freq_for_right_eye"]), 6)
        partner = next(
            int(candidate["trigger_num"])
            for candidate in codebook
            if round(float(candidate["freq_for_left_eye"]), 6) == right
            and round(float(candidate["freq_for_right_eye"]), 6) == left
        )
        target_pair[target - 1] = min(target, partner)
    records = {record.subject: record for record in iter_sun_cnt_files(splits=["forty_targets"])}
    subjects = parse_subjects(args.subjects, sorted(records))
    missing = sorted(set(subjects) - set(records))
    if missing:
        raise ValueError(f"Missing subjects: {missing}")
    windows = parse_windows(args.windows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []

    for subject in subjects:
        data, starts, labels, folds = load_subject(records[subject], args.repetitions)
        for window in windows:
            samples = int(round(window * FS))
            if np.max(starts + samples) > data.shape[-1]:
                raise ValueError(f"Subject {subject}: window exceeds recording boundary.")
            scores = np.empty((labels.size, 40), dtype=np.float64)
            if args.comb_mode == "global_union":
                print(f"[Sun2024 comb] S{subject:02d} {window:.1f}s global union", flush=True)
                filtered = target_band_comb_filter(data, FS, global_centers, half_width_hz=args.half_width)
                epochs = np.stack([filtered[:, start : start + samples] for start in starts])
                for class_id in range(40):
                    scores[:, class_id] = class_score(epochs, labels, folds, class_id, args.repetitions)
                del filtered, epochs
            else:
                for pair_id, centers in pair_centers.items():
                    print(f"[Sun2024 comb] S{subject:02d} {window:.1f}s pair={pair_id:02d}", flush=True)
                    filtered = target_band_comb_filter(data, FS, centers, half_width_hz=args.half_width)
                    epochs = np.stack([filtered[:, start : start + samples] for start in starts])
                    for class_id, class_pair in target_pair.items():
                        if class_pair == pair_id:
                            scores[:, class_id] = class_score(epochs, labels, folds, class_id, args.repetitions)
                    del filtered, epochs
            predicted = np.argmax(scores, axis=1)
            accuracy = float(np.mean(predicted == labels))
            summary_rows.append(
                {
                    "subject": subject,
                    "window": window,
                    "method": f"{args.comb_mode}_passband_comb_trca",
                    "repetitions": args.repetitions,
                    "folds": args.repetitions,
                    "trials": labels.size,
                    "channels": "all",
                    "components": "f_left,f_right,h2_left,h2_right,sum",
                    "half_width_hz": args.half_width,
                    "accuracy": accuracy,
                    "itr_bits_per_min": itr_bits_per_min(accuracy, window),
                }
            )
            for index, trial in enumerate(range(labels.size)):
                trial_rows.append(
                    {
                        "subject": subject,
                        "window": window,
                        "fold": int(folds[index]),
                        "target_id": int(labels[index] + 1),
                        "predicted_id": int(predicted[index] + 1),
                        "correct": int(predicted[index] == labels[index]),
                    }
                )
            write_confusion(
                args.output / "figures" / f"confusion_target_comb_trca_s{subject:02d}_{window:.1f}s.png",
                labels,
                predicted,
                f"Sun2024 {args.comb_mode} passband comb TRCA S{subject:02d}, {window:.1f}s",
            )
            print(f"[Sun2024 comb] S{subject:02d} {window:.1f}s accuracy={accuracy:.4f}", flush=True)

    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)
    manifest = {
        "task": "ssvep_efficient_dual_frequency_sun2024",
        "protocol": "target_repetition_loo",
        "subjects": subjects,
        "windows": windows,
        "repetitions": args.repetitions,
        "fixed_onset_shift_seconds": ONSET_SHIFT_SECONDS,
        "comb_type": args.comb_mode,
        "components": ["f_left", "f_right", "h2_left", "h2_right", "sum"],
        "half_width_hz": args.half_width,
        "pair_centers_hz": pair_centers,
        "global_union_centers_hz": global_centers,
        "status": "empirical passband-comb sensitivity smoke, not an author-code reproduction",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
