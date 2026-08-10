from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy import signal
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.dual_frequency import (  # noqa: E402
    SUN2024_OCCIPITAL9,
    iter_sun_cnt_files,
    sun_annotation_rows,
    sun_target_repeat_trials,
)
from vep_arena.methods.traditional import TRCA  # noqa: E402
from vep_arena.signal.filters import powerline_comb_filter, remove_dc  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_BANDS = ((8.0, 90.0), (16.0, 90.0), (24.0, 90.0), (32.0, 90.0), (40.0, 90.0))
ONSET_SHIFT_SECONDS = 0.14
TRIAL_FIELDS = ("subject", "window", "method", "fold", "target_id", "repetition", "predicted_id", "correct")
SUMMARY_FIELDS = ("subject", "window", "method", "repetitions", "folds", "trials", "channels", "preprocess", "accuracy", "itr_bits_per_min")


def parse_subjects(text: str, available: list[int]) -> list[int]:
    if text.strip().lower() == "all":
        return available
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, stop = (int(value) for value in item.split("-", 1))
            values.extend(range(start, stop + 1))
        else:
            values.append(int(item))
    return sorted(dict.fromkeys(values))


def parse_windows(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_methods(text: str) -> list[str]:
    methods = [item.strip().upper() for item in text.split(",") if item.strip()]
    unknown = sorted(set(methods) - {"TRCA", "ETRCA"})
    if unknown:
        raise ValueError(f"This protocol smoke supports only TRCA/ETRCA, not {unknown}.")
    return methods


def select_channels(raw, selection: str) -> list[str]:
    if selection == "all":
        import mne

        return [raw.ch_names[index] for index in mne.pick_types(raw.info, eeg=True, meg=False, stim=False, exclude=[])]
    available = {name.upper(): name for name in raw.ch_names}
    return [available[name.upper()] for name in SUN2024_OCCIPITAL9 if name.upper() in available]


def filterbank_epochs(
    raw_path: Path,
    trials,
    *,
    window: float,
    channels: str,
    onset_shift: float,
    n_bands: int,
    preprocess: str,
    comb_f0: float,
    comb_q: float,
) -> np.ndarray:
    import mne

    raw = mne.io.read_raw_cnt(raw_path, preload=False, verbose="ERROR")
    selected = select_channels(raw, channels)
    if not selected:
        raise ValueError(f"No EEG channels selected from {raw_path}")
    raw.pick(selected)
    first = min(trial.start_seconds + onset_shift for trial in trials)
    last = max(trial.start_seconds + onset_shift + window for trial in trials)
    crop_tmin = max(0.0, first - 3.0)
    raw.crop(tmin=crop_tmin, tmax=min(float(raw.times[-1]), last + 3.0))
    raw.load_data(verbose="ERROR")
    if float(raw.info["sfreq"]) != 250.0:
        raw.resample(250.0, npad="auto", verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data().astype(np.float64, copy=False)
    if preprocess == "comb_filterbank":
        data = powerline_comb_filter(data, sfreq, base_hz=comb_f0, q=comb_q, remove_dc_offset=True)
    elif preprocess == "filterbank":
        data = remove_dc(data)
    else:
        raise ValueError(f"Unknown preprocess mode: {preprocess}")
    samples = int(round(window * sfreq))
    starts = [int(round((trial.start_seconds + onset_shift - crop_tmin) * sfreq)) for trial in trials]
    if max(start + samples for start in starts) > data.shape[-1]:
        raise ValueError(f"Epoch exceeds cropped data in {raw_path}")
    epochs = np.empty((len(trials), n_bands, data.shape[0], samples), dtype=np.float64)
    for band_index, (low, high) in enumerate(DEFAULT_BANDS[:n_bands]):
        sos = signal.butter(4, (low, min(high, sfreq / 2.0 - 1.0)), btype="bandpass", fs=sfreq, output="sos")
        filtered = signal.sosfiltfilt(sos, data, axis=-1)
        for trial_index, start in enumerate(starts):
            epochs[trial_index, band_index] = filtered[:, start : start + samples]
    return epochs


def itr_bits_per_min(accuracy: float, window: float) -> float:
    p = min(max(float(accuracy), 1e-12), 1.0 - 1e-12)
    bits = math.log2(40) + p * math.log2(p) + (1.0 - p) * math.log2((1.0 - p) / 39.0)
    return bits * 60.0 / (window + 0.5)


def write_confusion(path: Path, true: list[int], predicted: list[int], title: str) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-repetition LOO smoke for Sun2024 offline 40-target data.")
    parser.add_argument("--subjects", default="10")
    parser.add_argument("--repetitions", type=int, choices=[5, 8], default=5)
    parser.add_argument("--repetition-start", type=int, default=1, help="First per-target repetition included in the target-wise folds.")
    parser.add_argument("--windows", default="0.6,1.0")
    parser.add_argument("--methods", default="TRCA,ETRCA")
    parser.add_argument("--channels", choices=["all", "occipital9"], default="all")
    parser.add_argument("--n-bands", type=int, default=5, choices=range(1, 6))
    parser.add_argument(
        "--preprocess",
        choices=["comb_filterbank", "filterbank"],
        default="filterbank",
        help="Use comb_filterbank only for an explicitly labelled, unverified author-comb assumption.",
    )
    parser.add_argument("--comb-f0", type=float, default=50.0)
    parser.add_argument("--comb-q", type=float, default=35.0)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "offline_target_repetition_smoke")
    args = parser.parse_args()

    records = iter_sun_cnt_files(splits=["forty_targets"])
    subjects = parse_subjects(args.subjects, sorted(record.subject for record in records))
    wanted = {record.subject: record for record in records if record.subject in set(subjects)}
    if sorted(wanted) != subjects:
        raise ValueError(f"Missing requested subjects: {sorted(set(subjects) - set(wanted))}")
    windows = parse_windows(args.windows)
    methods = parse_methods(args.methods)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)

    trial_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    aggregate: dict[tuple[str, float], tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for subject in subjects:
        record = wanted[subject]
        all_trials = sun_target_repeat_trials(sun_annotation_rows(record.path), n_targets=40)
        counts = Counter(trial.target_id for trial in all_trials)
        last_repetition = args.repetition_start + args.repetitions - 1
        if min(counts.values()) < last_repetition:
            raise ValueError(f"Subject {subject} has only {min(counts.values())} repetitions for at least one target; need {last_repetition}.")
        trials = [trial for trial in all_trials if args.repetition_start <= trial.repetition <= last_repetition]
        repetitions = np.asarray([trial.repetition for trial in trials], dtype=np.int64)
        labels = np.asarray([trial.target_id - 1 for trial in trials], dtype=np.int64)
        for fold in range(1, args.repetitions + 1):
            test = labels[repetitions == fold]
            train = labels[repetitions != fold]
            if sorted(test.tolist()) != list(range(40)) or Counter(train.tolist()) != Counter({target: args.repetitions - 1 for target in range(40)}):
                raise ValueError(f"Subject {subject}, repetition fold {fold} does not have target-wise LOO coverage.")
        for window in windows:
            print(f"[Sun2024] subject={subject:02d} window={window:.1f}s target-repetition LOO", flush=True)
            x = filterbank_epochs(
                record.path,
                trials,
                window=window,
                channels=args.channels,
                onset_shift=ONSET_SHIFT_SECONDS,
                n_bands=args.n_bands,
                preprocess=args.preprocess,
                comb_f0=args.comb_f0,
                comb_q=args.comb_q,
            )
            for method in methods:
                true_all: list[int] = []
                predicted_all: list[int] = []
                for fold in range(1, args.repetitions + 1):
                    train_mask = repetitions != fold
                    test_mask = repetitions == fold
                    model = TRCA(n_fbs=args.n_bands, ensemble=method == "ETRCA")
                    model.fit(x[train_mask], labels[train_mask])
                    predicted, _ = model.predict(x[test_mask])
                    true = labels[test_mask]
                    for index, trial_index in enumerate(np.flatnonzero(test_mask)):
                        trial = trials[trial_index]
                        trial_rows.append({"subject": subject, "window": window, "method": method, "fold": fold, "target_id": int(true[index] + 1), "repetition": trial.repetition, "predicted_id": int(predicted[index] + 1), "correct": int(predicted[index] == true[index])})
                    true_all.extend(true.tolist())
                    predicted_all.extend(predicted.tolist())
                accuracy = float(np.mean(np.asarray(true_all) == np.asarray(predicted_all)))
                preprocess_label = args.preprocess if args.preprocess == "filterbank" else f"comb_filterbank(f0={args.comb_f0:g},q={args.comb_q:g})"
                summary_rows.append({"subject": subject, "window": window, "method": method, "repetitions": args.repetitions, "folds": args.repetitions, "trials": len(true_all), "channels": args.channels, "preprocess": preprocess_label, "accuracy": accuracy, "itr_bits_per_min": itr_bits_per_min(accuracy, window)})
                aggregate[(method, window)][0].extend(true_all)
                aggregate[(method, window)][1].extend(predicted_all)
                print(f"[Sun2024] subject={subject:02d} window={window:.1f}s method={method} acc={accuracy:.4f}", flush=True)

    with (args.output / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        writer.writerows(trial_rows)
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    for (method, window), (true, predicted) in aggregate.items():
        write_confusion(args.output / "figures" / f"confusion_{method.lower()}_{window:.1f}s.png", true, predicted, f"Sun2024 target-repetition LOO {method}, {window:.1f}s")
    manifest = {"task": "ssvep_efficient_dual_frequency_sun2024", "protocol": "offline_target_repetition_loo", "subjects": subjects, "repetitions": args.repetitions, "repetition_range": [args.repetition_start, args.repetition_start + args.repetitions - 1], "windows": windows, "methods": methods, "onset_shift": ONSET_SHIFT_SECONDS, "preprocess": args.preprocess, "comb_filter_assumption": {"base_hz": args.comb_f0, "q": args.comb_q, "paper_exact_parameters_available": False}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
