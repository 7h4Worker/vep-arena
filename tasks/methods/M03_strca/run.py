from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[2]
sys.path.insert(0, str(PROJECT))

from vep_arena.config import CACHE_ROOT, DATA_ROOT, BENCHMARK_FREQS, BenchmarkSpec
from vep_arena.data.epochs import CanonicalEpochStore, EpochRequest
from vep_arena.data.presets import benchmark_9ch_default
from vep_arena.data.toolbox_adapter import toolbox_beta, toolbox_dataset_info
from vep_arena.evaluation import parse_range, parse_windows
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.strca import sTRCA
from vep_arena.methods.traditional import TRCA


def _load(task: dict) -> tuple[np.ndarray, tuple[float, ...], BenchmarkSpec, float]:
    dataset = str(task["dataset"])
    subject = int(task["subject"])
    window = float(task["window"])
    n_fbs = int(task["n_fbs"])
    if dataset == "benchmark":
        spec = BenchmarkSpec()
        preset = benchmark_9ch_default(Path(str(task["root"])))
        epochs = CanonicalEpochStore(Path(str(task["epoch_cache"]))).load_or_create(
            EpochRequest(
                preset=preset, subject=subject, window=window, kind="filterbank",
                n_fbs=n_fbs, cache_window=float(task["cache_window"]),
            )
        )
        return epochs, BENCHMARK_FREQS, spec, spec.cue_seconds
    info = toolbox_dataset_info("beta", root=Path(str(task["root"])))
    adapter = toolbox_beta(root=Path(str(task["root"])), download=False)
    blocks = [int(value) for value in task["blocks"]]
    targets = list(info.targets)
    batch = adapter.get_trials(
        subject, blocks, targets, "occipital_9ch", window,
        preprocess="toolbox_fb", n_bands=n_fbs, filter_window=float(task["cache_window"]),
    )
    epochs = batch.x.reshape(len(blocks), len(targets), *batch.x.shape[1:]).transpose(1, 0, 2, 3, 4)
    spec = BenchmarkSpec(
        subjects=len(info.subjects), blocks=len(blocks), classes=len(targets),
        sampling_rate=info.sampling_rate, cue_seconds=info.break_seconds,
        latency_seconds=info.latency_seconds or 0.0,
    )
    return epochs, info.frequencies, spec, info.break_seconds


def _run_unit(task: dict) -> list[dict]:
    epochs, frequencies, spec, break_seconds = _load(task)
    blocks = [int(value) for value in task["blocks"]]
    labels = np.arange(spec.classes, dtype=np.int64)
    rows: list[dict] = []
    for calibration_blocks in [int(value) for value in task["calibration_blocks"]]:
        for fold, train_tuple in enumerate(combinations(range(len(blocks)), calibration_blocks), start=1):
            train_indices = list(train_tuple)
            test_indices = [index for index in range(len(blocks)) if index not in train_indices]
            train_x = epochs[:, train_indices].reshape(-1, *epochs.shape[2:])
            train_y = np.repeat(labels, calibration_blocks)
            for method in task["methods"]:
                if method == "STRCA":
                    model = sTRCA(
                        n_fbs=int(task["n_fbs"]), frequencies=frequencies,
                        sampling_rate=spec.sampling_rate, harmonics=int(task["harmonics"]),
                        local_range=float(task["local_range"]),
                    )
                elif method == "TRCA":
                    model = TRCA(n_fbs=int(task["n_fbs"]), ensemble=False)
                else:
                    raise ValueError(f"unknown method: {method}")
                started = time.perf_counter()
                model.fit(train_x, train_y)
                fit_seconds = time.perf_counter() - started
                for test_idx in test_indices:
                    predict_started = time.perf_counter()
                    predictions, _ = model.predict(epochs[:, test_idx])
                    predict_seconds = time.perf_counter() - predict_started
                    accuracy = float(np.mean(predictions == labels))
                    rows.append(
                        {
                            "dataset": task["dataset"], "method": method,
                            "window": float(task["window"]), "calibration_blocks": calibration_blocks,
                            "subject": int(task["subject"]), "fold": fold, "block": blocks[test_idx],
                            "accuracy": accuracy,
                            "itr": itr_bits_per_minute(
                                accuracy, spec.classes, float(task["window"]) + break_seconds
                            ),
                            "samples": spec.classes,
                            "seconds": fit_seconds / len(test_indices) + predict_seconds,
                        }
                    )
    return rows


def _write(rows: list[dict], result_dir: Path, manifest: dict) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "method", "calibration_blocks", "window", "subject", "fold", "block"]
    )
    frame.to_csv(result_dir / "trials.csv", index=False)
    keys = ["dataset", "method", "calibration_blocks", "window"]
    subject = frame.groupby([*keys, "subject"], as_index=False).agg(
        accuracy=("accuracy", "mean"), itr=("itr", "mean"), seconds=("seconds", "sum")
    )
    summary = subject.groupby(keys, as_index=False).agg(
        accuracy=("accuracy", "mean"), accuracy_sem=("accuracy", "sem"),
        itr=("itr", "mean"), itr_sem=("itr", "sem"), subjects=("subject", "nunique"),
        seconds=("seconds", "sum"),
    )
    subject.to_csv(result_dir / "subject.csv", index=False)
    summary.to_csv(result_dir / "summary.csv", index=False)
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    for calibration_blocks, group in summary.groupby("calibration_blocks"):
        plt.figure(figsize=(8.5, 5.2))
        for method, method_rows in group.groupby("method"):
            plt.errorbar(
                method_rows["window"], method_rows["itr"], yerr=method_rows["itr_sem"],
                marker="o", capsize=3, label=method,
            )
        plt.xlabel("Window (s)")
        plt.ylabel("ITR (bits/min)")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / f"itr_cal{calibration_blocks}.png", dpi=180)
        plt.close()
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["benchmark", "beta"], default="benchmark")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--subjects")
    parser.add_argument("--blocks")
    parser.add_argument("--windows", default="0.5:0.1:1.0")
    parser.add_argument("--calibration-blocks", default="2-4")
    parser.add_argument("--methods", default="STRCA,TRCA")
    parser.add_argument("--n-fbs", type=int, default=5)
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--local-range", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task-name")
    parser.add_argument("--epoch-cache", type=Path, default=CACHE_ROOT / "canonical_epochs")
    args = parser.parse_args()
    if args.dataset == "benchmark":
        root = args.root or DATA_ROOT
        subjects = parse_range(args.subjects or "1-35")
        blocks = parse_range(args.blocks or "1-6")
    else:
        info = toolbox_dataset_info("beta", root=args.root)
        root = info.root
        subjects = parse_range(args.subjects) if args.subjects else list(info.subjects)
        blocks = parse_range(args.blocks) if args.blocks else list(info.blocks)
    windows = parse_windows(args.windows)
    calibration_blocks = parse_range(args.calibration_blocks)
    if min(calibration_blocks) < 2 or max(calibration_blocks) >= len(blocks):
        raise ValueError("sTRCA calibration blocks must be between 2 and total_blocks - 1")
    methods = [value.strip().upper() for value in args.methods.split(",") if value.strip()]
    result_dir = TASK / "results" / (args.task_name or f"{args.dataset}_strca")
    tasks = [
        {
            "dataset": args.dataset, "root": str(root), "epoch_cache": str(args.epoch_cache),
            "subject": subject, "window": window, "cache_window": max(windows),
            "blocks": blocks, "calibration_blocks": calibration_blocks, "methods": methods,
            "n_fbs": args.n_fbs, "harmonics": args.harmonics, "local_range": args.local_range,
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
            print(f"completed {task['dataset']} w={task['window']} s={task['subject']}", flush=True)
    _write(
        rows, result_dir,
        {
            "task": "M03_strca", "paper": "10.1016/j.measurement.2024.114959",
            "dataset": args.dataset, "root": str(root), "subjects": subjects, "blocks": blocks,
            "windows": windows, "calibration_blocks": calibration_blocks, "methods": methods,
            "harmonics": args.harmonics, "local_range": args.local_range,
            "protocol": "exhaustive training-block combinations; every remaining block tested",
            "preprocessing": "BL01/BL02 9ch canonical filterbank and dataset-specific latency",
        },
    )
    print(result_dir)


if __name__ == "__main__":
    main()
