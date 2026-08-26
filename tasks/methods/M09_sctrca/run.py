from __future__ import annotations

import argparse
import itertools
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
from scipy import signal

TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[2]
sys.path.insert(0, str(PROJECT))

from vep_arena.config import CACHE_ROOT, DATA_ROOT, BENCHMARK_FREQS, BenchmarkSpec
from vep_arena.data.epochs import CanonicalEpochStore, EpochRequest
from vep_arena.data.presets import benchmark_9ch_default
from vep_arena.data.toolbox_adapter import toolbox_beta
from vep_arena.evaluation import parse_range, parse_windows
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.sctrca import scTRCA
from vep_arena.methods.traditional import TRCA


def _normalize(x: np.ndarray) -> np.ndarray:
    centered = x - np.mean(x, axis=-1, keepdims=True)
    return centered / np.maximum(np.std(centered, axis=-1, keepdims=True), 1e-12)


def _paper_preprocess(dataset: object, x: np.ndarray) -> np.ndarray:
    sampling_rate = int(dataset.srate)
    passband = [6 / (sampling_rate / 2), 90 / (sampling_rate / 2)]
    stopband = [4 / (sampling_rate / 2), 100 / (sampling_rate / 2)]
    order, cutoff = signal.cheb1ord(passband, stopband, 3, 40)
    numerator, denominator = signal.cheby1(order, 0.5, cutoff, btype="bandpass")
    return signal.filtfilt(numerator, denominator, x, axis=-1, padtype="odd")


def _load(task: dict) -> tuple[np.ndarray, tuple[float, ...], float]:
    blocks = [int(value) for value in task["blocks"]]
    window = float(task["window"])
    n_fbs = int(task["n_fbs"])
    if task["dataset"] == "benchmark":
        preset = benchmark_9ch_default(Path(str(task["root"])))
        epochs = CanonicalEpochStore(Path(str(task["epoch_cache"]))).load_or_create(
            EpochRequest(
                preset=preset, subject=int(task["subject"]), window=window,
                kind="filterbank", n_fbs=n_fbs, cache_window=float(task["cache_window"]),
            )
        )
        block_indices = [block - 1 for block in blocks]
        return epochs[:, block_indices].transpose(1, 0, 2, 3, 4), BENCHMARK_FREQS, 0.5
    dataset = toolbox_beta(root=Path(str(task["root"])), download=False)
    if bool(task["paper_bandpass"]):
        dataset.preprocess_fun = _paper_preprocess
    targets = list(dataset.info.targets)
    batch = dataset.get_trials(
        int(task["subject"]), blocks, targets, "occipital_9ch", window,
        preprocess="toolbox_preprocess" if bool(task["paper_bandpass"]) else "toolbox_fb",
        n_bands=n_fbs, filter_window=float(task["cache_window"]),
    )
    beta_x = batch.x[:, None] if batch.x.ndim == 3 else batch.x
    epochs = beta_x.reshape(len(blocks), len(targets), *beta_x.shape[1:])
    return epochs, tuple(dataset.info.frequencies), float(dataset.info.break_seconds)


def _run_unit(task: dict) -> list[dict]:
    epochs, frequencies, shift_seconds = _load(task)
    blocks = [int(value) for value in task["blocks"]]
    training_trials = int(task["training_trials"])
    labels = np.arange(epochs.shape[1], dtype=np.int64)
    rows: list[dict] = []
    for test_idx, test_block in enumerate(blocks):
        candidates = [index for index in range(len(blocks)) if index != test_idx]
        for train_indices in itertools.combinations(candidates, training_trials):
            train_x = epochs[list(train_indices)].reshape(-1, *epochs.shape[2:])
            train_y = np.tile(labels, training_trials)
            test_x = epochs[test_idx]
            fold = "+".join(str(blocks[index]) for index in train_indices)
            for method in task["methods"]:
                if method in {"SCTRCA", "ESCTRCA"}:
                    model = scTRCA(
                        n_fbs=int(task["n_fbs"]), frequencies=frequencies,
                        sampling_rate=250, harmonics=int(task["harmonics"]),
                        ensemble=method == "ESCTRCA",
                    )
                    model_x = train_x
                    model_test = test_x
                elif method in {"TRCA", "ETRCA"}:
                    model = TRCA(
                        n_fbs=int(task["n_fbs"]), ensemble=method == "ETRCA"
                    )
                    model_x = _normalize(train_x)
                    model_test = _normalize(test_x)
                else:
                    raise ValueError(f"unknown method: {method}")
                started = time.perf_counter()
                model.fit(model_x, train_y)
                predictions, _ = model.predict(model_test)
                seconds = time.perf_counter() - started
                accuracy = float(np.mean(predictions == labels))
                rows.append(
                    {
                        "dataset": task["dataset"], "method": method,
                        "training_trials": training_trials, "window": float(task["window"]),
                        "subject": int(task["subject"]), "test_block": test_block,
                        "train_blocks": fold, "accuracy": accuracy,
                        "itr": itr_bits_per_minute(
                            accuracy, len(labels), float(task["window"]) + shift_seconds
                        ),
                        "samples": len(labels), "seconds": seconds,
                    }
                )
    return rows


def _write(rows: list[dict], result_dir: Path, manifest: dict) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "method", "training_trials", "window", "subject", "test_block", "train_blocks"]
    )
    frame.to_csv(result_dir / "trials.csv", index=False)
    subject = frame.groupby(
        ["dataset", "method", "training_trials", "window", "subject"], as_index=False
    ).agg(accuracy=("accuracy", "mean"), itr=("itr", "mean"), seconds=("seconds", "sum"))
    summary = subject.groupby(
        ["dataset", "method", "training_trials", "window"], as_index=False
    ).agg(
        accuracy=("accuracy", "mean"), accuracy_sem=("accuracy", "sem"),
        itr=("itr", "mean"), itr_sem=("itr", "sem"), subjects=("subject", "nunique"),
        seconds=("seconds", "sum"),
    )
    subject.to_csv(result_dir / "subject.csv", index=False)
    summary.to_csv(result_dir / "summary.csv", index=False)
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(8.5, 5.2))
    for (dataset, method), group in summary.groupby(["dataset", "method"]):
        plt.plot(group["window"], group["itr"], marker="o", label=f"{dataset}:{method}")
    plt.xlabel("Window (s)")
    plt.ylabel("ITR (bits/min)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "itr_curve.png", dpi=180)
    plt.close()
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["benchmark", "beta"], default="benchmark")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--subjects")
    parser.add_argument("--blocks")
    parser.add_argument("--training-trials", type=int, default=2)
    parser.add_argument("--windows", default="1.0")
    parser.add_argument("--cache-window", type=float)
    parser.add_argument("--methods", default="SCTRCA,TRCA")
    parser.add_argument("--n-fbs", type=int, default=1)
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--toolbox-filterbank", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task-name", default="sctrca_reproduction")
    parser.add_argument("--epoch-cache", type=Path, default=CACHE_ROOT / "canonical_epochs")
    args = parser.parse_args()
    if args.dataset == "benchmark":
        root = args.root or DATA_ROOT
        subjects = parse_range(args.subjects or "1-35")
        blocks = parse_range(args.blocks or "1-6")
    else:
        root = args.root or Path("D:/ProjData/datasets/ssvep_beta")
        subjects = parse_range(args.subjects or "1-70")
        blocks = parse_range(args.blocks or "1-4")
    if not 2 <= args.training_trials < len(blocks):
        raise ValueError("training-trials must be at least 2 and smaller than the block count")
    windows = parse_windows(args.windows)
    cache_window = args.cache_window or max(windows)
    methods = [value.strip().upper() for value in args.methods.split(",") if value.strip()]
    tasks = [
        {
            "dataset": args.dataset, "root": str(root), "epoch_cache": str(args.epoch_cache),
            "subject": subject, "blocks": blocks, "training_trials": args.training_trials,
            "window": window, "cache_window": cache_window, "methods": methods,
            "n_fbs": args.n_fbs, "harmonics": args.harmonics,
            "paper_bandpass": not args.toolbox_filterbank,
        }
        for window in windows for subject in subjects
    ]
    rows: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_unit, task) for task in tasks]
            for future in as_completed(futures):
                rows.extend(future.result())
                print(f"completed {len(rows)} folds", flush=True)
    else:
        for task in tasks:
            rows.extend(_run_unit(task))
            print(f"completed {task['dataset']} w={task['window']} s={task['subject']}", flush=True)
    result_dir = TASK / "results" / args.task_name
    _write(
        rows, result_dir,
        {
            "task": "M09_sctrca", "paper": "10.1088/1741-2552/abfdfa",
            "dataset": args.dataset, "root": str(root), "subjects": subjects,
            "blocks": blocks, "training_trials": args.training_trials, "windows": windows,
            "methods": methods, "harmonics": args.harmonics,
            "paper_bandpass": not args.toolbox_filterbank,
            "protocol": "all training-block combinations within leave-one-block-out",
            "preprocessing": "paper 9ch protocol; BETA uses 6-90 Hz Chebyshev I; channel z-score",
            "task_table_note": "TASK.md copied msTRCA row; paper scTRCA targets are recorded separately",
        },
    )
    print(result_dir)


if __name__ == "__main__":
    main()
