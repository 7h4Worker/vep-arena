from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from run import (
    BASE_FREQS,
    CHANNEL66,
    DEFAULT_DATASET,
    FB_NUM,
    FS,
    LATENCY,
    LAG,
    RAW_REL,
    fit_tdca_single,
    itr_bits_per_minute,
    predict_tdca,
    projection_matrices,
)
from run_online_highest import filterbank_online, load_online


PAPER_TABLE2 = {
    "S1": {"targets": 160, "window_ms": 250, "accuracy_percent": 96.88, "itr_bpm": 551.42},
    "S3": {"targets": 80, "window_ms": 200, "accuracy_percent": 93.75, "itr_bpm": 479.20},
    "S4": {"targets": 80, "window_ms": 250, "accuracy_percent": 91.00, "itr_bpm": 425.45},
    "S5": {"targets": 120, "window_ms": 200, "accuracy_percent": 88.67, "itr_bpm": 481.36},
    "S7": {"targets": 80, "window_ms": 200, "accuracy_percent": 95.25, "itr_bpm": 492.58},
    "S10": {"targets": 120, "window_ms": 200, "accuracy_percent": 82.67, "itr_bpm": 432.58},
    "S11": {"targets": 80, "window_ms": 200, "accuracy_percent": 87.50, "itr_bpm": 427.75},
    "S12": {"targets": 80, "window_ms": 200, "accuracy_percent": 91.75, "itr_bpm": 462.08},
    "S14": {"targets": 80, "window_ms": 300, "accuracy_percent": 94.75, "itr_bpm": 427.05},
    "S15": {"targets": 160, "window_ms": 250, "accuracy_percent": 96.50, "itr_bpm": 547.77},
}


def half_up_samples(window_ms: int) -> int:
    return int(math.floor(window_ms / 1000 * FS + 0.5))


def subject_freqs(targets: int) -> np.ndarray:
    if targets % len(BASE_FREQS) != 0:
        raise ValueError(f"Cannot map {targets} targets onto {len(BASE_FREQS)} base frequencies")
    return np.repeat(BASE_FREQS, targets // len(BASE_FREQS))


def load_or_filter_online(
    dataset_root: Path,
    subject: str,
    split: str,
    expected_targets: int,
    force_filter: bool,
) -> np.ndarray:
    cache_dir = dataset_root / "derivatives" / "tdca_sample" / "online_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"filtered_online_{split}_{subject}_{expected_targets}target_66ch_float32.npy"
    if cache.exists() and not force_filter:
        arr = np.load(cache, mmap_mode="r")
        print(f"{subject} {split}: loaded cache {arr.shape}", flush=True)
        return arr

    raw_path = dataset_root / RAW_REL / "data" / "online" / split / f"{subject}.mat"
    raw = load_online(raw_path)[CHANNEL66]
    if raw.shape[2] != expected_targets:
        raise ValueError(f"{subject} {split} has {raw.shape[2]} targets, expected {expected_targets}")
    print(f"{subject} {split}: raw {raw.shape}", flush=True)
    filtered = filterbank_online(raw)
    np.save(cache, filtered)
    print(f"{subject} {split}: saved {cache}", flush=True)
    return np.load(cache, mmap_mode="r")


def run_subject(dataset_root: Path, subject: str, force_filter: bool) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    started = time.perf_counter()
    cfg = PAPER_TABLE2[subject]
    targets = int(cfg["targets"])
    window_ms = int(cfg["window_ms"])
    time_samples = half_up_samples(window_ms)
    needed = time_samples + LATENCY + LAG

    train = load_or_filter_online(dataset_root, subject, "training", targets, force_filter)
    test = load_or_filter_online(dataset_root, subject, "testing", targets, force_filter)
    if needed > train.shape[1] or needed > test.shape[1]:
        raise ValueError(f"{subject}: {window_ms} ms needs {needed}, train/test samples={train.shape[1]}/{test.shape[1]}")

    freqs = subject_freqs(targets)
    p_cond = projection_matrices(time_samples, freqs)
    model = fit_tdca_single(np.asarray(train, dtype=np.float32), time_samples, p_cond)

    correct = np.zeros(FB_NUM, dtype=np.int64)
    confusion = np.zeros((targets, targets), dtype=np.int64)
    pred_rows = []
    total = 0
    for block in range(test.shape[3]):
        for cond in range(targets):
            epoch = np.asarray(test[:, :, cond, block, :], dtype=np.float32)
            pred = predict_tdca(epoch, model, time_samples, p_cond)
            correct += pred == cond
            final_pred = int(pred[-1])
            confusion[cond, final_pred] += 1
            total += 1
            pred_rows.append(
                {
                    "subject": subject,
                    "block": block + 1,
                    "trial_index": total,
                    "true": cond + 1,
                    "pred": final_pred + 1,
                    "correct": int(final_pred == cond),
                    "targets": targets,
                    "window_ms": window_ms,
                    "time_samples": time_samples,
                    "method": "TDCA",
                }
            )
        print(f"{subject}: test_block={block + 1}/{test.shape[3]} fb5_acc={correct[-1] / total:.6f}", flush=True)

    accuracy = float(correct[-1] / total)
    itr = itr_bits_per_minute(targets, accuracy, window_ms / 1000 + 0.5)
    paper_acc = float(cfg["accuracy_percent"]) / 100
    paper_itr = float(cfg["itr_bpm"])
    row = {
        "subject": subject,
        "targets": targets,
        "channels": 66,
        "filter_banks": FB_NUM,
        "train_blocks": int(train.shape[3]),
        "test_blocks": int(test.shape[3]),
        "window_ms": window_ms,
        "time_samples": time_samples,
        "total_trials": total,
        "correct_trials": int(correct[-1]),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "itr_bpm": itr,
        "paper_accuracy_percent": float(cfg["accuracy_percent"]),
        "paper_itr_bpm": paper_itr,
        "delta_accuracy_pp": accuracy * 100 - float(cfg["accuracy_percent"]),
        "delta_correct_trials": int(correct[-1]) - round(paper_acc * total),
        "delta_itr_bpm": itr - paper_itr,
        "seconds": time.perf_counter() - started,
    }
    return row, pred_rows, confusion


def write_rows(
    out_dir: Path,
    rows: list[dict[str, object]],
    pred_rows: list[dict[str, object]],
    confusions: dict[str, np.ndarray],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "online_table2_reproduction.csv"
    fields = [
        "subject",
        "targets",
        "channels",
        "filter_banks",
        "train_blocks",
        "test_blocks",
        "window_ms",
        "time_samples",
        "total_trials",
        "correct_trials",
        "accuracy",
        "accuracy_percent",
        "itr_bpm",
        "paper_accuracy_percent",
        "paper_itr_bpm",
        "delta_accuracy_pp",
        "delta_correct_trials",
        "delta_itr_bpm",
        "seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "online_table2_reproduction.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pred_fields = [
        "subject",
        "block",
        "trial_index",
        "true",
        "pred",
        "correct",
        "targets",
        "window_ms",
        "time_samples",
        "method",
    ]
    pred_path = out_dir / "online_table2_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(pred_rows)
    confusion_dir = out_dir / "confusions"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    for key, matrix in confusions.items():
        np.save(confusion_dir / f"confusion_{key}.npy", matrix)
    md = [
        "# Online Table 2 Reproduction",
        "",
        "| subject | targets | window_ms | local acc % | paper acc % | d correct | local ITR | paper ITR | d ITR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['subject']} | {row['targets']} | {row['window_ms']} | "
            f"{float(row['accuracy_percent']):.3f} | {float(row['paper_accuracy_percent']):.3f} | "
            f"{int(row['delta_correct_trials'])} | {float(row['itr_bpm']):.3f} | "
            f"{float(row['paper_itr_bpm']):.3f} | {float(row['delta_itr_bpm']):.3f} |"
        )
    (out_dir / "online_table2_reproduction.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    manifest = {
        "runner": str(Path(__file__).resolve()),
        "method": "TDCA",
        "dataset": "ssvep_hd_200target online release",
        "paper_table": "Table 2 online BCI experiment",
        "summary_csv": str(csv_path),
        "predictions_csv": str(pred_path),
        "confusion_dir": str(confusion_dir),
        "subjects": [row["subject"] for row in rows],
        "prediction_rows": len(pred_rows),
        "status": "complete" if len(rows) == len(PAPER_TABLE2) else "partial",
    }
    (out_dir / "online_table2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {pred_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--subjects", default=",".join(PAPER_TABLE2.keys()))
    parser.add_argument("--force-filter", action="store_true")
    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    out_dir = Path(__file__).resolve().parent / "results"
    rows = []
    pred_rows = []
    confusions = {}
    for subject in subjects:
        row, subject_preds, subject_confusion = run_subject(args.dataset_root, subject, args.force_filter)
        rows.append(row)
        pred_rows.extend(subject_preds)
        confusions[f"{subject}_{row['targets']}target"] = subject_confusion
        write_rows(out_dir, rows, pred_rows, confusions)
        print(row, flush=True)


if __name__ == "__main__":
    main()
