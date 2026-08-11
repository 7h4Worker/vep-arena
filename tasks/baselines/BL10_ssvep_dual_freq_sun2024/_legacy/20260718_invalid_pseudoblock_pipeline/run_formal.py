from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
from scipy import signal
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.dual_frequency import (
    SUN2024_OCCIPITAL9,
    iter_sun_cnt_files,
    pair_sun_trial_events,
    sun_annotation_rows,
    sun_stimulus_codebook,
)
from vep_arena.methods.btrca import BTRCA
from vep_arena.methods.traditional import TRCA
from vep_arena.signal.filters import powerline_comb_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_BANDS = (
    (8.0, 90.0),
    (16.0, 90.0),
    (24.0, 90.0),
    (32.0, 90.0),
    (40.0, 90.0),
)


def parse_ints(text: str | None, default: list[int]) -> list[int]:
    if text is None or text.strip().lower() == "all":
        return default
    out: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start, stop = item.split("-", 1)
            out.extend(range(int(start), int(stop) + 1))
        else:
            out.append(int(item))
    return sorted(dict.fromkeys(out))


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_methods(text: str) -> list[str]:
    methods = [item.strip().upper() for item in text.split(",") if item.strip()]
    valid = {"TRCA", "ETRCA", "BTRCA", "EBTRCA"}
    unknown = sorted(set(methods) - valid)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return methods


def select_channels(raw, mode: str) -> list[str]:
    if mode == "all":
        import mne

        picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, exclude=[])
        return [raw.ch_names[idx] for idx in picks]
    wanted = {name.upper(): name for name in SUN2024_OCCIPITAL9}
    available = {name.upper(): name for name in raw.ch_names}
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise ValueError(f"Missing occipital channels in CNT: {missing}")
    return [available[name.upper()] for name in SUN2024_OCCIPITAL9]


def build_swap_pairs(codebook: list[dict[str, object]]) -> list[tuple[int, int]]:
    groups: dict[tuple[float, float], list[int]] = defaultdict(list)
    for row in codebook:
        trigger = int(row["trigger_num"])
        left = float(row["freq_for_left_eye"])
        right = float(row["freq_for_right_eye"])
        key = tuple(sorted((left, right)))
        groups[key].append(trigger - 1)
    pairs: list[tuple[int, int]] = []
    for key, labels in sorted(groups.items()):
        if len(labels) != 2:
            raise ValueError(f"Expected exactly two swapped targets for {key}, got {labels}")
        a, b = sorted(labels)
        pairs.append((a, b))
    if len(pairs) != 20:
        raise ValueError(f"Expected 20 swapped pairs for Sun2024 40-target codebook, got {len(pairs)}")
    return pairs


def apply_notch(data: np.ndarray, sfreq: float, notch_hz: float | None) -> np.ndarray:
    if notch_hz is None or notch_hz <= 0:
        return data
    out = np.asarray(data, dtype=np.float64)
    nyquist = sfreq / 2.0
    freq = float(notch_hz)
    while freq < nyquist - 1.0:
        b, a = signal.iirnotch(w0=freq, Q=30.0, fs=sfreq)
        out = signal.filtfilt(b, a, out, axis=-1)
        freq += float(notch_hz)
    return out


def filterband(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    high = min(high, sfreq / 2.0 - 1.0)
    if low >= high:
        raise ValueError(f"Invalid band {low}-{high} for sampling rate {sfreq}")
    sos = signal.butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, data, axis=-1)


def preprocess_continuous(
    data: np.ndarray,
    sfreq: float,
    *,
    preprocess: str,
    notch_hz: float | None,
    comb_f0: float,
    comb_q: float,
) -> np.ndarray:
    if preprocess == "filterbank":
        return apply_notch(data, sfreq, notch_hz)
    if preprocess == "comb_filterbank":
        return powerline_comb_filter(data, sfreq, base_hz=comb_f0, q=comb_q, remove_dc_offset=True)
    raise ValueError(f"Unknown preprocess mode: {preprocess}")


def load_subject_epochs(
    cnt_path: Path,
    trials,
    *,
    window: float,
    channels: str,
    n_bands: int,
    onset_shift: float,
    resample_hz: float,
    notch_hz: float | None,
    preprocess: str,
    comb_f0: float,
    comb_q: float,
    crop_padding: float,
) -> np.ndarray:
    import mne

    raw = mne.io.read_raw_cnt(cnt_path, preload=False, verbose="ERROR")
    selected = select_channels(raw, channels)
    raw.pick(selected)
    crop_tmin = 0.0
    if crop_padding >= 0:
        first_start = min(trial.start_seconds + onset_shift for trial in trials)
        last_stop = max(trial.start_seconds + onset_shift + window for trial in trials)
        crop_tmin = max(0.0, first_start - float(crop_padding))
        crop_tmax = min(float(raw.times[-1]), last_stop + float(crop_padding))
        raw.crop(tmin=crop_tmin, tmax=crop_tmax)
    raw.load_data(verbose="ERROR")
    if resample_hz > 0 and abs(float(raw.info["sfreq"]) - resample_hz) > 1e-6:
        raw.resample(resample_hz, npad="auto", verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data().astype(np.float64, copy=False)
    data = preprocess_continuous(
        data,
        sfreq,
        preprocess=preprocess,
        notch_hz=notch_hz,
        comb_f0=comb_f0,
        comb_q=comb_q,
    )

    samples = int(round(window * sfreq))
    starts = [int(round((trial.start_seconds + onset_shift - crop_tmin) * sfreq)) for trial in trials]
    stops = [start + samples for start in starts]
    max_stop = max(stops)
    if max_stop > data.shape[1]:
        raise ValueError(f"Requested epoch stop {max_stop} exceeds data length {data.shape[1]} in {cnt_path}")

    bands = DEFAULT_BANDS[:n_bands]
    epochs = np.zeros((len(trials), n_bands, data.shape[0], samples), dtype=np.float64)
    for band_idx, (low, high) in enumerate(bands):
        filtered = filterband(data, sfreq, low, high)
        for trial_idx, (start, stop) in enumerate(zip(starts, stops, strict=True)):
            epochs[trial_idx, band_idx] = filtered[:, start:stop]
    return epochs


def make_model(method: str, n_bands: int, pairs: list[tuple[int, int]]):
    if method == "TRCA":
        return TRCA(n_fbs=n_bands, ensemble=False)
    if method == "ETRCA":
        return TRCA(n_fbs=n_bands, ensemble=True)
    if method == "BTRCA":
        return BTRCA(pairs, n_fbs=n_bands, ensemble=False)
    if method == "EBTRCA":
        return BTRCA(pairs, n_fbs=n_bands, ensemble=True)
    raise ValueError(method)


def itr_bits_per_min(classes: int, accuracy: float, selection_time: float) -> float:
    p = min(max(float(accuracy), 1e-12), 1.0 - 1e-12)
    n = float(classes)
    bits = math.log2(n) + p * math.log2(p) + (1.0 - p) * math.log2((1.0 - p) / (n - 1.0))
    return bits * 60.0 / selection_time


def write_confusion(path: Path, y_true: list[int], y_pred: list[int], title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(40)))
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    image = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(0, 40, 5))
    ax.set_yticks(np.arange(0, 40, 5))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


TRIAL_FIELDS = (
    "subject", "window", "method", "fold", "trial_index", "block", "trial_in_block", "target_id",
    "predicted_id", "correct", "score_true", "score_pred",
)
SUMMARY_FIELDS = (
    "subject", "window", "method", "channels", "preprocess", "n_bands", "folds", "skipped_folds",
    "trials", "accuracy", "itr_bits_per_min",
)
SKIPPED_FOLD_FIELDS = ("subject", "fold", "reason", "missing_training_classes", "test_trials")
FAILURE_FIELDS = ("subject", "error")


def evaluate_subject(payload: dict[str, object]) -> dict[str, object]:
    """Run one subject independently so the parent can checkpoint it safely."""

    subject = int(payload["subject"])
    cnt_path = Path(str(payload["cnt_path"]))
    windows = [float(value) for value in payload["windows"]]
    methods = [str(value) for value in payload["methods"]]
    pairs = [tuple(pair) for pair in payload["pairs"]]
    max_blocks = payload["max_blocks"]
    max_folds = payload["max_folds"]
    trials = pair_sun_trial_events(sun_annotation_rows(cnt_path), n_targets=40)
    if max_blocks is not None:
        trials = [trial for trial in trials if trial.block <= int(max_blocks)]
    if not trials:
        raise ValueError(f"No trials left for subject {subject} after max_blocks={max_blocks}.")

    y = np.asarray([trial.target_id - 1 for trial in trials], dtype=np.int64)
    blocks = np.asarray([trial.block for trial in trials], dtype=np.int64)
    fold_values = sorted(np.unique(blocks).tolist())
    if max_folds is not None:
        fold_values = fold_values[: int(max_folds)]

    valid_folds: list[int] = []
    skipped_folds: list[dict[str, object]] = []
    expected_classes = set(range(40))
    for fold in fold_values:
        train_mask = blocks != fold
        missing_classes = sorted(expected_classes - set(y[train_mask].tolist()))
        if missing_classes:
            skipped_folds.append(
                {
                    "subject": subject,
                    "fold": int(fold),
                    "reason": "training_class_coverage_incomplete",
                    "missing_training_classes": ",".join(str(cls + 1) for cls in missing_classes),
                    "test_trials": int(np.sum(blocks == fold)),
                }
            )
            continue
        valid_folds.append(int(fold))
    if not valid_folds:
        raise ValueError(f"Subject {subject} has no leave-block-out folds with all 40 training classes.")

    trial_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for window in windows:
        print(f"[Sun2024] subject={subject:02d} window={window:.2f}s load/filter pid={os.getpid()}", flush=True)
        x = load_subject_epochs(
            cnt_path,
            trials,
            window=window,
            channels=str(payload["channels"]),
            n_bands=int(payload["n_bands"]),
            onset_shift=float(payload["onset_shift"]),
            resample_hz=float(payload["resample_hz"]),
            notch_hz=float(payload["notch_hz"]),
            preprocess=str(payload["preprocess"]),
            comb_f0=float(payload["comb_f0"]),
            comb_q=float(payload["comb_q"]),
            crop_padding=float(payload["crop_padding"]),
        )
        for method in methods:
            y_true_subject: list[int] = []
            y_pred_subject: list[int] = []
            for fold in valid_folds:
                train_mask = blocks != fold
                test_mask = blocks == fold
                model = make_model(method, int(payload["n_bands"]), pairs)
                model.fit(x[train_mask], y[train_mask])
                pred, scores = model.predict(x[test_mask])
                true = y[test_mask]
                test_indices = np.flatnonzero(test_mask)
                for local_idx, trial_index in enumerate(test_indices):
                    trial_rows.append(
                        {
                            "subject": subject,
                            "window": window,
                            "method": method,
                            "fold": int(fold),
                            "trial_index": int(trials[trial_index].trial_index),
                            "block": int(trials[trial_index].block),
                            "trial_in_block": int(trials[trial_index].trial_in_block),
                            "target_id": int(true[local_idx] + 1),
                            "predicted_id": int(pred[local_idx] + 1),
                            "correct": int(pred[local_idx] == true[local_idx]),
                            "score_true": float(scores[local_idx, true[local_idx]]),
                            "score_pred": float(scores[local_idx, pred[local_idx]]),
                        }
                    )
                y_true_subject.extend(true.tolist())
                y_pred_subject.extend(pred.tolist())
            accuracy = float(np.mean(np.asarray(y_true_subject) == np.asarray(y_pred_subject)))
            summary_rows.append(
                {
                    "subject": subject,
                    "window": window,
                    "method": method,
                    "channels": str(payload["channels"]),
                    "preprocess": str(payload["preprocess"]),
                    "n_bands": int(payload["n_bands"]),
                    "folds": len(valid_folds),
                    "skipped_folds": len(skipped_folds),
                    "trials": len(y_true_subject),
                    "accuracy": accuracy,
                    "itr_bits_per_min": itr_bits_per_min(40, accuracy, window + float(payload["search_time"])),
                }
            )
            print(
                f"[Sun2024] subject={subject:02d} window={window:.2f}s method={method} "
                f"acc={accuracy:.4f} folds={len(valid_folds)}",
                flush=True,
            )
    return {"subject": subject, "trial_rows": trial_rows, "summary_rows": summary_rows, "skipped_folds": skipped_folds}


def read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint(
    output: Path,
    manifest: dict[str, object],
    trial_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    skipped_folds: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> None:
    write_rows(output / "trials.csv", TRIAL_FIELDS, trial_rows)
    write_rows(output / "summary.csv", SUMMARY_FIELDS, summary_rows)
    write_rows(output / "skipped_folds.csv", SKIPPED_FOLD_FIELDS, skipped_folds)
    write_rows(output / "failures.csv", FAILURE_FIELDS, failures)
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_subjects"] = sorted({int(row["subject"]) for row in summary_rows})
    manifest["failed_subjects"] = sorted({int(row["subject"]) for row in failures})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    records = iter_sun_cnt_files(splits=["forty_targets"])
    available_subjects = sorted(record.subject for record in records)
    subjects = parse_ints(args.subjects, available_subjects)
    wanted = {record.subject: record for record in records if record.subject in set(subjects)}
    missing = sorted(set(subjects) - set(wanted))
    if missing:
        raise ValueError(f"Missing Sun2024 forty-target CNT files for subjects: {missing}")
    windows = parse_floats(args.windows)
    methods = parse_methods(args.methods)
    codebook = sun_stimulus_codebook(kind="forty_targets")
    pairs = build_swap_pairs(codebook)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    trial_rows = read_rows(output / "trials.csv") if args.resume else []
    summary_rows = read_rows(output / "summary.csv") if args.resume else []
    skipped_folds = read_rows(output / "skipped_folds.csv") if args.resume else []
    failures = read_rows(output / "failures.csv") if args.resume else []
    expected_rows_per_subject = len(windows) * len(methods)
    completed_subjects = {
        int(subject)
        for subject in subjects
        if sum(int(row["subject"]) == subject for row in summary_rows) == expected_rows_per_subject
    }
    pending_subjects = [subject for subject in subjects if subject not in completed_subjects]

    manifest: dict[str, object] = {
        "task": "ssvep_efficient_dual_frequency_sun2024",
        "split": "forty_targets",
        "subjects": subjects,
        "windows": windows,
        "methods": methods,
        "channels": args.channels,
        "n_bands": args.n_bands,
        "bands": DEFAULT_BANDS[: args.n_bands],
        "preprocess": args.preprocess,
        "onset_shift": args.onset_shift,
        "resample_hz": args.resample_hz,
        "notch_hz": args.notch_hz,
        "comb_f0": args.comb_f0,
        "comb_q": args.comb_q,
        "crop_padding": args.crop_padding,
        "max_blocks": args.max_blocks,
        "max_folds": args.max_folds,
        "codebook_rows": len(codebook),
        "swap_pairs": pairs,
        "workers": args.workers,
        "parallel_unit": "subject",
        "status": "partial",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payloads = [
        {
            "subject": subject,
            "cnt_path": str(wanted[subject].path),
            "windows": windows,
            "methods": methods,
            "pairs": pairs,
            "channels": args.channels,
            "n_bands": args.n_bands,
            "preprocess": args.preprocess,
            "onset_shift": args.onset_shift,
            "resample_hz": args.resample_hz,
            "notch_hz": args.notch_hz,
            "comb_f0": args.comb_f0,
            "comb_q": args.comb_q,
            "crop_padding": args.crop_padding,
            "search_time": args.search_time,
            "max_blocks": args.max_blocks,
            "max_folds": args.max_folds,
        }
        for subject in pending_subjects
    ]

    def accept_result(result: dict[str, object]) -> None:
        trial_rows.extend(result["trial_rows"])
        summary_rows.extend(result["summary_rows"])
        skipped_folds.extend(result["skipped_folds"])
        checkpoint(output, manifest, trial_rows, summary_rows, skipped_folds, failures)
        print(f"[Sun2024] checkpoint subject={int(result['subject']):02d} pending={len(payloads)}", flush=True)

    if args.workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_map = {pool.submit(evaluate_subject, payload): int(payload["subject"]) for payload in payloads}
            for future in as_completed(future_map):
                subject = future_map[future]
                try:
                    accept_result(future.result())
                except Exception as error:
                    failures.append({"subject": subject, "error": repr(error)})
                    checkpoint(output, manifest, trial_rows, summary_rows, skipped_folds, failures)
                    print(f"[Sun2024] failed subject={subject:02d} error={error!r}", flush=True)
    else:
        for payload in payloads:
            subject = int(payload["subject"])
            try:
                accept_result(evaluate_subject(payload))
            except Exception as error:
                failures.append({"subject": subject, "error": repr(error)})
                checkpoint(output, manifest, trial_rows, summary_rows, skipped_folds, failures)
                print(f"[Sun2024] failed subject={subject:02d} error={error!r}", flush=True)

    aggregate_predictions: dict[tuple[str, float], tuple[list[int], list[int]]] = {}
    for row in trial_rows:
        key = (str(row["method"]), float(row["window"]))
        if key not in aggregate_predictions:
            aggregate_predictions[key] = ([], [])
        aggregate_predictions[key][0].append(int(row["target_id"]) - 1)
        aggregate_predictions[key][1].append(int(row["predicted_id"]) - 1)
    for (method, window), (y_true, y_pred) in aggregate_predictions.items():
        write_confusion(
            figures_dir / f"confusion_{method.lower()}_{window:.2f}s.png",
            y_true,
            y_pred,
            f"Sun2024 {method} {window:.2f}s",
        )
    manifest["status"] = "complete" if len(completed_subjects | {int(row['subject']) for row in summary_rows}) == len(subjects) and not failures else "partial"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint(output, manifest, trial_rows, summary_rows, skipped_folds, failures)
    if failures:
        raise RuntimeError(f"Sun2024 run completed with {len(failures)} failed subject(s); see {output / 'failures.csv'}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Formal Sun2024 dual-frequency 40-target TRCA/bTRCA runner.")
    parser.add_argument("--subjects", default="all", help="Subjects, e.g. all, 1, 1-3, or 1,2,5.")
    parser.add_argument("--windows", default="0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0")
    parser.add_argument("--methods", default="TRCA,ETRCA,BTRCA,EBTRCA")
    parser.add_argument("--channels", choices=["occipital9", "all"], default="occipital9")
    parser.add_argument("--n-bands", type=int, default=5, choices=range(1, len(DEFAULT_BANDS) + 1))
    parser.add_argument(
        "--preprocess",
        choices=["filterbank", "comb_filterbank"],
        default="filterbank",
        help="Continuous preprocessing before the TRCA-style filter-bank bands.",
    )
    parser.add_argument("--onset-shift", type=float, default=0.14)
    parser.add_argument("--resample-hz", type=float, default=250.0)
    parser.add_argument("--notch-hz", type=float, default=50.0)
    parser.add_argument("--comb-f0", type=float, default=50.0)
    parser.add_argument("--comb-q", type=float, default=35.0)
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=5.0,
        help="Seconds retained before/after requested trials before loading CNT data; negative disables cropping.",
    )
    parser.add_argument("--search-time", type=float, default=0.5)
    parser.add_argument("--max-blocks", type=int, default=None, help="Keep only blocks <= this value before CV.")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent subject workers; parent owns result writes.")
    parser.add_argument("--resume", action="store_true", help="Skip subjects with a complete subject/window/method grid in summary.csv.")
    parser.add_argument("--output", default="tasks/baselines/BL10_ssvep_dual_freq_sun2024/results/formal_run")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
