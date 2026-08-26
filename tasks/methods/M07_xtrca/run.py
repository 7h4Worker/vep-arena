from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from vep_arena.methods.traditional import TRCA
from vep_arena.methods.xtrca import xTRCA


def _run_unit(task: dict) -> list[dict]:
    spec = BenchmarkSpec()
    window = float(task["window"])
    blocks = [int(value) for value in task["blocks"]]
    preset = benchmark_9ch_default(Path(str(task["root"])))
    epochs = CanonicalEpochStore(Path(str(task["epoch_cache"]))).load_or_create(
        EpochRequest(
            preset=preset, subject=int(task["subject"]), window=window, kind="filterbank",
            n_fbs=int(task["n_fbs"]), cache_window=float(task["cache_window"]),
        )
    )
    labels = np.arange(spec.classes, dtype=np.int64)
    rows: list[dict] = []
    for test_idx, block in enumerate(blocks):
        train_indices = [index for index in range(len(blocks)) if index != test_idx]
        train_x = epochs[:, train_indices].reshape(-1, *epochs.shape[2:])
        train_y = np.repeat(labels, len(train_indices))
        for method in task["methods"]:
            if method == "XTRCA":
                model = xTRCA(
                    n_fbs=int(task["n_fbs"]), search_samples=int(task["search_samples"]),
                    max_cycles=int(task["max_cycles"]),
                )
            elif method == "TRCA":
                model = TRCA(n_fbs=int(task["n_fbs"]), ensemble=False)
            else:
                raise ValueError(f"unknown method: {method}")
            started = time.perf_counter()
            model.fit(train_x, train_y)
            predictions, _ = model.predict(epochs[:, test_idx])
            seconds = time.perf_counter() - started
            accuracy = float(np.mean(predictions == labels))
            rows.append(
                {
                    "method": method, "window": window, "subject": int(task["subject"]),
                    "block": block, "accuracy": accuracy,
                    "itr": itr_bits_per_minute(accuracy, spec.classes, window + spec.cue_seconds),
                    "samples": spec.classes, "seconds": seconds,
                }
            )
    return rows


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
    plt.figure(figsize=(8.5, 5.2))
    for method, group in summary.groupby("method"):
        plt.errorbar(group["window"], group["accuracy"], yerr=group["accuracy_sem"], marker="o", capsize=3, label=method)
    plt.xlabel("Window (s)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "accuracy_curve.png", dpi=180)
    plt.close()
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--blocks", default="1-6")
    parser.add_argument("--windows", default="0.2:0.2:1.0")
    parser.add_argument("--cache-window", type=float)
    parser.add_argument("--methods", default="XTRCA,TRCA")
    parser.add_argument("--n-fbs", type=int, default=1)
    parser.add_argument("--search-ms", type=float, default=20.0)
    parser.add_argument("--max-cycles", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task-name", default="benchmark_xtrca")
    parser.add_argument("--epoch-cache", type=Path, default=CACHE_ROOT / "canonical_epochs")
    args = parser.parse_args()
    subjects = parse_range(args.subjects)
    blocks = parse_range(args.blocks)
    windows = parse_windows(args.windows)
    cache_window = args.cache_window or max(windows)
    methods = [value.strip().upper() for value in args.methods.split(",") if value.strip()]
    search_samples = int(round(args.search_ms * BenchmarkSpec().sampling_rate / 1000.0))
    tasks = [
        {
            "root": str(args.root), "epoch_cache": str(args.epoch_cache), "subject": subject,
            "window": window, "cache_window": cache_window, "blocks": blocks, "methods": methods,
            "n_fbs": args.n_fbs, "search_samples": search_samples, "max_cycles": args.max_cycles,
        }
        for window in windows for subject in subjects
    ]
    rows: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_unit, task) for task in tasks]
            for future in as_completed(futures):
                rows.extend(future.result())
                print(f"completed {len(rows)} rows", flush=True)
    else:
        for task in tasks:
            rows.extend(_run_unit(task))
            print(f"completed w={task['window']} s={task['subject']}", flush=True)
    result_dir = TASK / "results" / args.task_name
    _write(
        rows, result_dir,
        {
            "task": "M07_xtrca", "paper": "10.1016/j.neuroimage.2019.04.049",
            "dataset": "Tsinghua Benchmark SSVEP", "root": str(args.root),
            "subjects": subjects, "blocks": blocks, "windows": windows, "methods": methods,
            "search_ms": args.search_ms, "max_cycles": args.max_cycles,
            "protocol": "subject-specific leave-one-block-out (five training blocks)",
            "ssvep_adaptation": "periodic circular timing shifts within each epoch",
        },
    )
    print(result_dir)


if __name__ == "__main__":
    main()
