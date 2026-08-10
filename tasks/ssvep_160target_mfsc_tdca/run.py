from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.mfsc160 import (  # noqa: E402
    MFSC160_CHANNELS,
    MFSC160_CLASSES,
    MFSC160_DEFAULT_FILTER_BANKS,
    MFSC160_SAMPLING_RATE,
    load_or_filter_mfsc160_subject,
    mfsc160_epoch_tensor,
    mfsc160_root,
    mfsc160_sequence_references,
    parse_subjects,
    read_codebook,
)
from vep_arena.metrics import itr_bits_per_minute  # noqa: E402
from vep_arena.methods.tdca import TDCA  # noqa: E402
from vep_arena.methods.traditional import filterbank_weights  # noqa: E402


TASK_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SubjectTask:
    root: str
    split: str
    subject: int
    segment_windows: tuple[float, ...]
    n_bands: int
    harmonics: int
    n_components: int
    n_delay: int
    force_filter: bool
    done_windows: tuple[float, ...]


def parse_windows(text: str) -> list[float]:
    if ":" in text:
        start, step, stop = (float(item) for item in text.split(":", 2))
        values: list[float] = []
        current = start
        while current <= stop + 1e-9:
            values.append(round(current, 10))
            current += step
        return values
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def existing_keys(path: Path) -> set[tuple[str, int, float, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int, float, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "complete":
                keys.add((row["split"], int(row["subject"]), float(row["segment_window"]), int(row["n_delay"])))
    return keys


def append_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_subject(task: SubjectTask) -> dict[str, object]:
    started = time.perf_counter()
    root = Path(task.root)
    codebook = read_codebook(root)
    filtered = load_or_filter_mfsc160_subject(
        root=root,
        split=task.split,
        subject=task.subject,
        n_bands=task.n_bands,
        force_filter=task.force_filter,
    )
    targets, blocks, n_bands, channels, _ = filtered.shape
    labels = np.arange(targets, dtype=np.int64)
    done = set(float(item) for item in task.done_windows)
    trial_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []

    for segment_window in task.segment_windows:
        if float(segment_window) in done:
            continue
        window_started = time.perf_counter()
        x = mfsc160_epoch_tensor(filtered, segment_seconds=segment_window)
        segment_samples = int(round(segment_window * MFSC160_SAMPLING_RATE))
        refs = mfsc160_sequence_references(codebook, segment_samples=segment_samples, harmonics=task.harmonics)
        fb_weights = filterbank_weights(task.n_bands)
        block_acc: list[float] = []
        confusion = np.zeros((targets, targets), dtype=np.int64)
        for block_idx in range(blocks):
            train_blocks = [idx for idx in range(blocks) if idx != block_idx]
            train_x = x[:, train_blocks].transpose(0, 1, 2, 3, 4).reshape(-1, n_bands, channels, x.shape[-1])
            train_y = np.repeat(labels, len(train_blocks))
            test_x = x[:, block_idx]
            model = TDCA(n_components=task.n_components, n_delay=task.n_delay, fb_weights=fb_weights)
            fit_started = time.perf_counter()
            model.fit(np.asarray(train_x, dtype=np.float64), train_y, refs)
            fit_done = time.perf_counter()
            pred, scores_by_band = model.predict(np.asarray(test_x, dtype=np.float64))
            pred_done = time.perf_counter()
            scores = np.einsum("f,tfc->tc", model.fb_weights, scores_by_band)
            accuracy = float(np.mean(pred == labels))
            block_acc.append(accuracy)
            for target_idx, pred_idx in enumerate(pred):
                confusion[target_idx, int(pred_idx)] += 1
                pred_rows.append(
                    {
                        "method": "TDCA",
                        "split": task.split,
                        "subject": task.subject,
                        "segment_window": segment_window,
                        "total_window": segment_window * 4,
                        "n_delay": task.n_delay,
                        "block": block_idx + 1,
                        "true": target_idx + 1,
                        "pred": int(pred_idx) + 1,
                        "correct": int(pred_idx == target_idx),
                        "score_true": float(scores[target_idx, target_idx]),
                        "score_pred": float(scores[target_idx, int(pred_idx)]),
                    }
                )
            runtime_rows.extend(
                [
                    {
                        "method": "TDCA",
                        "split": task.split,
                        "subject": task.subject,
                        "segment_window": segment_window,
                        "n_delay": task.n_delay,
                        "block": block_idx + 1,
                        "stage": "fit",
                        "seconds": fit_done - fit_started,
                        "worker_pid": os.getpid(),
                    },
                    {
                        "method": "TDCA",
                        "split": task.split,
                        "subject": task.subject,
                        "segment_window": segment_window,
                        "n_delay": task.n_delay,
                        "block": block_idx + 1,
                        "stage": "predict",
                        "seconds": pred_done - fit_done,
                        "worker_pid": os.getpid(),
                    },
                ]
            )

        mean_acc = float(np.mean(block_acc))
        trial_rows.append(
            {
                "method": "TDCA",
                "split": task.split,
                "subject": task.subject,
                "segment_window": segment_window,
                "total_window": segment_window * 4,
                "targets": targets,
                "channels": channels,
                "blocks": blocks,
                "n_bands": n_bands,
                "harmonics": task.harmonics,
                "n_components": task.n_components,
                "n_delay": task.n_delay,
                "accuracy": mean_acc,
                "itr_bpm": float(itr_bits_per_minute(mean_acc, targets, segment_window * 4 + 0.5)),
                "seconds": time.perf_counter() - window_started,
                "status": "complete",
                "reason": "",
            }
        )
        runtime_rows.append(
            {
                "method": "TDCA",
                "split": task.split,
                "subject": task.subject,
                "segment_window": segment_window,
                "n_delay": task.n_delay,
                "block": "__all__",
                "stage": "subject_window_total",
                "seconds": time.perf_counter() - window_started,
                "worker_pid": os.getpid(),
            }
        )

    return {
        "subject": task.subject,
        "seconds": time.perf_counter() - started,
        "trial_rows": trial_rows,
        "pred_rows": pred_rows,
        "runtime_rows": runtime_rows,
    }


def rebuild_summary(out_dir: Path) -> None:
    trials_path = out_dir / "trials.csv"
    if not trials_path.exists():
        return
    df = pd.read_csv(trials_path)
    if df.empty:
        return
    complete = df[df["status"] == "complete"].copy()
    if complete.empty:
        return
    group_cols = ["method", "split", "segment_window", "total_window", "n_delay"]
    summary = (
        complete.groupby(group_cols, as_index=False)
        .agg(
            subjects=("subject", "nunique"),
            accuracy=("accuracy", "mean"),
            accuracy_sem=("accuracy", lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            itr_bpm=("itr_bpm", "mean"),
            itr_sem=("itr_bpm", lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            seconds=("seconds", "sum"),
        )
        .sort_values(["split", "segment_window", "n_delay"])
    )
    summary.to_csv(out_dir / "summary.csv", index=False)


def write_outputs(out_dir: Path, result: dict[str, object], fields: dict[str, list[str]]) -> None:
    append_csv(out_dir / "trials.csv", list(result["trial_rows"]), fields["trials"])
    append_csv(out_dir / "predictions.csv", list(result["pred_rows"]), fields["predictions"])
    append_csv(out_dir / "runtime.csv", list(result["runtime_rows"]), fields["runtime"])
    rebuild_summary(out_dir)


def plot_smoke_confusion(out_dir: Path) -> None:
    pred_path = out_dir / "predictions.csv"
    if not pred_path.exists():
        return
    preds = pd.read_csv(pred_path)
    if preds.empty:
        return
    best_window = (
        preds.groupby("segment_window")
        .agg(accuracy=("correct", "mean"))
        .sort_values(["accuracy"], ascending=False)
        .index[0]
    )
    subset = preds[preds["segment_window"] == best_window]
    confusion = np.zeros((MFSC160_CLASSES, MFSC160_CLASSES), dtype=np.int64)
    for row in subset.itertuples(index=False):
        confusion[int(row.true) - 1, int(row.pred) - 1] += 1
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(confusion, cmap="Blues", interpolation="nearest")
    ax.set_title(f"MFSC160 TDCA smoke confusion, segment={float(best_window):g}s")
    ax.set_xlabel("Predicted target")
    ax.set_ylabel("True target")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_dir / "mfsc160_tdca_smoke_confusion.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="TDCA runner for the 160-target MFSC SSVEP dataset.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--split", choices=["offline", "online"], default="offline")
    parser.add_argument("--subjects", default="1")
    parser.add_argument("--segment-windows", default="0.4,1.0")
    parser.add_argument("--output", type=Path, default=TASK_DIR / "results" / "tdca_smoke")
    parser.add_argument("--n-bands", type=int, default=MFSC160_DEFAULT_FILTER_BANKS)
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-delay", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force-filter", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-smoke", action="store_true")
    args = parser.parse_args()

    if args.workers > 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    root = mfsc160_root(args.root)
    out_dir = args.output if args.output.is_absolute() else TASK_DIR / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects, split=args.split)
    windows = tuple(parse_windows(args.segment_windows))
    done = existing_keys(out_dir / "trials.csv") if args.resume else set()

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "runner": str(Path(__file__).resolve()),
        "dataset": "Chen2021 160-target MFSC SSVEP",
        "dataset_root": str(root),
        "split": args.split,
        "subjects": subjects,
        "segment_windows": list(windows),
        "total_windows": [window * 4 for window in windows],
        "sampling_rate": MFSC160_SAMPLING_RATE,
        "channels": list(MFSC160_CHANNELS),
        "targets": MFSC160_CLASSES,
        "n_bands": args.n_bands,
        "harmonics": args.harmonics,
        "n_components": args.n_components,
        "n_delay": args.n_delay,
        "workers": args.workers,
        "status": "partial",
        "outputs": {
            "trials": "trials.csv",
            "predictions": "predictions.csv",
            "runtime": "runtime.csv",
            "summary": "summary.csv",
            "smoke_confusion": "mfsc160_tdca_smoke_confusion.png" if args.plot_smoke else None,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    tasks = []
    for subject in subjects:
        done_windows = tuple(window for split, subj, window, delay in done if split == args.split and subj == subject and delay == args.n_delay)
        pending = tuple(window for window in windows if window not in set(done_windows))
        if pending:
            tasks.append(
                SubjectTask(
                    root=str(root),
                    split=args.split,
                    subject=subject,
                    segment_windows=pending,
                    n_bands=args.n_bands,
                    harmonics=args.harmonics,
                    n_components=args.n_components,
                    n_delay=args.n_delay,
                    force_filter=args.force_filter,
                    done_windows=done_windows,
                )
            )

    fields = {
        "trials": [
            "method",
            "split",
            "subject",
            "segment_window",
            "total_window",
            "targets",
            "channels",
            "blocks",
            "n_bands",
            "harmonics",
            "n_components",
            "n_delay",
            "accuracy",
            "itr_bpm",
            "seconds",
            "status",
            "reason",
        ],
        "predictions": [
            "method",
            "split",
            "subject",
            "segment_window",
            "total_window",
            "n_delay",
            "block",
            "true",
            "pred",
            "correct",
            "score_true",
            "score_pred",
        ],
        "runtime": ["method", "split", "subject", "segment_window", "n_delay", "block", "stage", "seconds", "worker_pid"],
    }

    if args.workers <= 1:
        for task in tasks:
            result = run_subject(task)
            write_outputs(out_dir, result, fields)
            print(f"done {args.split} S{task.subject} seconds={result['seconds']:.1f}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_map = {pool.submit(run_subject, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                result = future.result()
                write_outputs(out_dir, result, fields)
                print(f"done {args.split} S{task.subject} seconds={result['seconds']:.1f}", flush=True)

    rebuild_summary(out_dir)
    if args.plot_smoke:
        plot_smoke_confusion(out_dir)
    expected = {(args.split, subject, window, args.n_delay) for subject in subjects for window in windows}
    complete = existing_keys(out_dir / "trials.csv")
    manifest["status"] = "complete" if expected.issubset(complete) else "partial"
    manifest["completed_configs"] = len(expected & complete)
    manifest["expected_configs"] = len(expected)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
