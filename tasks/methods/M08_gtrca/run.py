from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[2]
sys.path.insert(0, str(PROJECT))

from vep_arena.config import CACHE_ROOT, DATA_ROOT, BenchmarkSpec
from vep_arena.data.epochs import CanonicalEpochStore, EpochRequest
from vep_arena.data.presets import benchmark_9ch_default
from vep_arena.evaluation import parse_range, parse_windows
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.gtrca import gTRCA
from vep_arena.methods.traditional import TRCA


def _load_subject(
    subject: int, window: float, cache_window: float, n_fbs: int,
    root: Path, epoch_cache: Path,
) -> np.ndarray:
    preset = benchmark_9ch_default(root)
    return CanonicalEpochStore(epoch_cache).load_or_create(
        EpochRequest(
            preset=preset, subject=subject, window=window, kind="filterbank",
            n_fbs=n_fbs, cache_window=cache_window,
        )
    )


def _write(rows: list[dict], result_dir: Path, manifest: dict) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["method", "window", "subject", "block"])
    frame.to_csv(result_dir / "trials.csv", index=False)
    subject = frame.groupby(["method", "window", "subject"], as_index=False).agg(
        accuracy=("accuracy", "mean"), itr=("itr", "mean"), seconds=("seconds", "sum")
    )
    summary = subject.groupby(["method", "window"], as_index=False).agg(
        accuracy=("accuracy", "mean"), accuracy_sem=("accuracy", "sem"),
        itr=("itr", "mean"), itr_sem=("itr", "sem"), subjects=("subject", "nunique"),
        seconds=("seconds", "sum"),
    )
    subject.to_csv(result_dir / "subject.csv", index=False)
    summary.to_csv(result_dir / "summary.csv", index=False)
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(7.5, 5.0))
    plt.bar(summary["method"], summary["accuracy"], yerr=summary["accuracy_sem"], capsize=4)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.02)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "accuracy_compare.png", dpi=180)
    plt.close()
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--train-subjects", default="1-30")
    parser.add_argument("--test-subjects", default="31-35")
    parser.add_argument("--blocks", default="1-6")
    parser.add_argument("--windows", default="1.0")
    parser.add_argument("--cache-window", type=float)
    parser.add_argument("--methods", default="GTRCA,TRCA")
    parser.add_argument("--n-fbs", type=int, default=1)
    parser.add_argument("--task-name", default="benchmark_gtrca")
    parser.add_argument("--epoch-cache", type=Path, default=CACHE_ROOT / "canonical_epochs")
    args = parser.parse_args()
    spec = BenchmarkSpec()
    train_subjects = parse_range(args.train_subjects)
    test_subjects = parse_range(args.test_subjects)
    if set(train_subjects) & set(test_subjects):
        raise ValueError("train and test subjects must be disjoint")
    blocks = parse_range(args.blocks)
    block_indices = [block - 1 for block in blocks]
    windows = parse_windows(args.windows)
    cache_window = args.cache_window or max(windows)
    methods = [value.strip().upper() for value in args.methods.split(",") if value.strip()]
    rows: list[dict] = []
    labels = np.arange(spec.classes, dtype=np.int64)
    for window in windows:
        group_epochs = []
        for subject in train_subjects:
            epochs = _load_subject(
                subject, window, cache_window, args.n_fbs, args.root, args.epoch_cache
            )
            group_epochs.append(epochs[:, block_indices].reshape(-1, *epochs.shape[2:]))
        group_x = np.asarray(group_epochs)
        group_y = np.repeat(labels, len(blocks))
        group_model = gTRCA(n_fbs=args.n_fbs)
        fit_started = time.perf_counter()
        group_model.fit(group_x, group_y)
        group_fit_seconds = time.perf_counter() - fit_started
        for subject in test_subjects:
            epochs = _load_subject(
                subject, window, cache_window, args.n_fbs, args.root, args.epoch_cache
            )
            epochs = epochs[:, block_indices]
            for test_idx, block in enumerate(blocks):
                for method in methods:
                    started = time.perf_counter()
                    if method == "GTRCA":
                        predictions, _ = group_model.predict(epochs[:, test_idx])
                        seconds = time.perf_counter() - started + group_fit_seconds / (
                            len(test_subjects) * len(blocks)
                        )
                    elif method == "TRCA":
                        train_indices = [index for index in range(len(blocks)) if index != test_idx]
                        train_x = epochs[:, train_indices].reshape(-1, *epochs.shape[2:])
                        train_y = np.repeat(labels, len(train_indices))
                        model = TRCA(n_fbs=args.n_fbs, ensemble=False).fit(train_x, train_y)
                        predictions, _ = model.predict(epochs[:, test_idx])
                        seconds = time.perf_counter() - started
                    else:
                        raise ValueError(f"unknown method: {method}")
                    accuracy = float(np.mean(predictions == labels))
                    rows.append(
                        {
                            "method": method, "window": window, "subject": subject,
                            "block": block, "accuracy": accuracy,
                            "itr": itr_bits_per_minute(
                                accuracy, spec.classes, window + spec.cue_seconds
                            ),
                            "samples": spec.classes, "seconds": seconds,
                        }
                    )
                    print(f"{method} w={window} s={subject} b={block} acc={accuracy:.3f}", flush=True)
    result_dir = TASK / "results" / args.task_name
    _write(
        rows, result_dir,
        {
            "task": "M08_gtrca", "paper": "10.1038/s41598-019-56962-2",
            "dataset": "Tsinghua Benchmark SSVEP", "root": str(args.root),
            "train_subjects": train_subjects, "test_subjects": test_subjects,
            "blocks": blocks, "windows": windows, "methods": methods,
            "group_protocol": "disjoint-subject group fit plus Eq.21 zero-training prediction",
            "trca_protocol": "test-subject leave-one-block-out with five calibration blocks",
        },
    )
    print(result_dir)


if __name__ == "__main__":
    main()
