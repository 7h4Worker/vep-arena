from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[2]
sys.path.insert(0, str(PROJECT))

from vep_arena.config import DATA_ROOT, RUN_ROOT, BenchmarkSpec
from vep_arena.data.epochs import CanonicalEpochStore, EpochRequest, epoch_fingerprint
from vep_arena.data.presets import DatasetPreset, benchmark_9ch_default, benchmark_64ch_default
from vep_arena.evaluation import (
    completed_windows,
    load_existing_rows,
    parse_range,
    parse_windows,
    write_outputs,
)
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.traditional import CCA, ECCA, FBCCA, SSCOR, TRCA


def resolve_benchmark_preset(name: str, data_root: Path) -> DatasetPreset:
    if name == "benchmark_9ch":
        return benchmark_9ch_default(data_root)
    if name == "benchmark_64ch":
        return benchmark_64ch_default(data_root)
    raise ValueError(f"Unknown Benchmark channel preset: {name}")


def make_model(name: str, window: float, args: argparse.Namespace, spec: BenchmarkSpec):
    if name == "CCA":
        return CCA(window=window, harmonics=args.harmonics, spec=spec)
    if name == "FBCCA":
        return FBCCA(window=window, harmonics=args.harmonics, n_fbs=args.n_fbs, spec=spec)
    if name == "ECCA":
        return ECCA(window=window, harmonics=args.harmonics, n_fbs=args.n_fbs, spec=spec)
    if name == "TRCA":
        return TRCA(n_fbs=args.n_fbs, ensemble=False)
    if name == "ETRCA":
        return TRCA(n_fbs=args.n_fbs, ensemble=True)
    if name == "SSCOR":
        return SSCOR(n_fbs=args.n_fbs, ensemble=False)
    if name == "ESSCOR":
        return SSCOR(n_fbs=args.n_fbs, ensemble=True)
    raise ValueError(f"Unknown method: {name}")


def save_score_matrix(
    out_dir: Path, method: str, window: float, subject: int, block: int, scores: np.ndarray,
) -> None:
    method_dir = out_dir / "score_matrices" / method.lower()
    method_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        method_dir / f"w{window:g}_s{subject:02d}_b{block}.npz",
        scores=np.asarray(scores, dtype=np.float32),
    )


def save_trca_artifact(
    out_dir: Path, method: str, window: float, subject: int, block: int, model: object,
) -> None:
    filters = getattr(model, "filters", None)
    if filters is None:
        return
    method_dir = out_dir / "model_artifacts" / method.lower()
    method_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        method_dir / f"w{window:g}_s{subject:02d}_b{block}.npz",
        filters=np.asarray(filters, dtype=np.float32),
    )


def run_subject_window_task(task: dict) -> dict:
    spec = BenchmarkSpec()
    data_root = Path(str(task["data_root"]))
    epoch_cache = Path(str(task["epoch_cache"]))
    result_dir = Path(str(task["result_dir"]))
    preset = resolve_benchmark_preset(str(task.get("channel_preset", "benchmark_9ch")), data_root)
    store = CanonicalEpochStore(epoch_cache)
    subject = int(task["subject"])
    window = float(task["window"])
    cache_window = float(task["cache_window"])
    methods = [str(m) for m in task["methods"]]
    blocks = [int(b) for b in task["blocks"]]
    harmonics = int(task["harmonics"])
    n_fbs = int(task["n_fbs"])
    force_epochs = bool(task["force_epochs"])
    save_scores = bool(task["save_score_matrices"])
    save_filters = bool(task["save_trca_filters"])
    args = argparse.Namespace(harmonics=harmonics, n_fbs=n_fbs)

    load_started = time.perf_counter()
    data_by_method: dict[str, np.ndarray] = {}
    if any(m in methods for m in ("CCA",)):
        req = EpochRequest(preset=preset, subject=subject, window=window, kind="raw", cache_window=cache_window)
        raw = store.load_or_create(req, force=force_epochs)
        for m in ("CCA",):
            if m in methods:
                data_by_method[m] = raw[:, :, None, :, :]
    if any(m in methods for m in ("FBCCA", "ECCA", "TRCA", "ETRCA", "SSCOR", "ESSCOR")):
        req = EpochRequest(
            preset=preset, subject=subject, window=window,
            kind="filterbank", n_fbs=n_fbs, cache_window=cache_window,
        )
        filtered_epochs = store.load_or_create(req, force=force_epochs)
        for m in ("FBCCA", "ECCA", "TRCA", "ETRCA", "SSCOR", "ESSCOR"):
            if m in methods:
                data_by_method[m] = filtered_epochs

    labels = np.arange(spec.classes, dtype=np.int64)
    trial_rows: list[dict] = []
    pred_rows: list[dict] = []
    runtime_rows: list[dict] = []
    logs: list[str] = []
    for block in blocks:
        train_blocks = [b - 1 for b in blocks if b != block]
        test_block = block - 1
        for method in methods:
            epochs = data_by_method[method]
            train_x = epochs[:, train_blocks].reshape(-1, epochs.shape[2], epochs.shape[3], epochs.shape[4])
            train_y = np.repeat(labels, len(train_blocks))
            test_x = epochs[:, test_block]
            model = make_model(method, window, args, spec)
            t0 = time.perf_counter()
            model.fit(train_x, train_y)
            fit_done = time.perf_counter()
            pred, scores = model.predict(test_x)
            predict_done = time.perf_counter()
            if save_scores:
                save_score_matrix(result_dir, method, window, subject, block, scores)
            if save_filters and method in ("TRCA", "ETRCA", "SSCOR", "ESSCOR"):
                save_trca_artifact(result_dir, method, window, subject, block, model)
            artifact_done = time.perf_counter()
            seconds = predict_done - t0
            acc = np.mean(pred == labels)
            itr = itr_bits_per_minute(float(acc), spec.classes, window + spec.cue_seconds)
            base_runtime = {
                "method": method, "window": window, "subject": subject, "block": block,
                "worker_pid": os.getpid(), "parallel_unit": "subject_window",
            }
            runtime_rows.extend([
                {**base_runtime, "stage": "fit", "seconds": fit_done - t0},
                {**base_runtime, "stage": "predict", "seconds": predict_done - fit_done},
                {**base_runtime, "stage": "artifact_write", "seconds": artifact_done - predict_done},
                {**base_runtime, "stage": "fit_predict", "seconds": seconds},
            ])
            trial_rows.append({
                "method": method, "window": window, "subject": subject, "block": block,
                "accuracy": float(acc), "itr": float(itr), "samples": spec.classes, "seconds": seconds,
            })
            for true_label, pred_label in zip(labels, pred):
                pred_rows.append({
                    "method": method, "window": window, "subject": subject, "block": block,
                    "true": int(true_label), "pred": int(pred_label),
                    "score_true": float(scores[int(true_label), int(true_label)]),
                    "score_pred": float(scores[int(true_label), int(pred_label)]),
                })
            logs.append(f"{method} w={window:.1f} s={subject:02d} b={block} acc={acc:.3f} sec={seconds:.2f}")

    return {
        "trial_rows": trial_rows,
        "pred_rows": pred_rows,
        "runtime_rows": [
            {
                "method": "__all__", "window": window, "subject": subject, "block": "__all__",
                "stage": "load_epochs_and_unit_total", "seconds": time.perf_counter() - load_started,
                "worker_pid": os.getpid(), "parallel_unit": "subject_window",
            },
            *runtime_rows,
        ],
        "logs": logs,
        "epoch_fingerprint": epoch_fingerprint(
            EpochRequest(
                preset=preset, subject=subject, window=window,
                kind="filterbank", n_fbs=n_fbs, cache_window=cache_window,
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--task-name", default="BL01_ssvep_benchmark")
    parser.add_argument("--channel-preset", choices=["benchmark_9ch", "benchmark_64ch"], default="benchmark_9ch")
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--blocks", default="1-6")
    parser.add_argument("--windows", default="default")
    parser.add_argument("--methods", default="CCA,FBCCA,ECCA,TRCA,ETRCA,SSCOR,ESSCOR")
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--n-fbs", type=int, default=5)
    parser.add_argument("--epoch-cache", type=Path, default=RUN_ROOT / "canonical_epochs")
    parser.add_argument("--force-epochs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-score-matrices", action="store_true")
    parser.add_argument("--save-trca-filters", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    spec = BenchmarkSpec()
    preset = resolve_benchmark_preset(args.channel_preset, args.data_root)
    subjects = parse_range(args.subjects)
    blocks = parse_range(args.blocks)
    windows = parse_windows(args.windows)
    cache_window = max(windows)
    methods = [x.strip().upper() for x in args.methods.split(",") if x.strip()]
    result_dir = TASK / "results" / args.task_name
    run_dir = RUN_ROOT / args.task_name
    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest: dict = {
        "task_name": args.task_name,
        "dataset": "Tsinghua Benchmark SSVEP",
        "channel_preset": args.channel_preset,
        "channels": list(preset.channels),
        "n_channels": len(preset.channels),
        "protocol": "subject-specific leave-one-block-out",
        "subjects": subjects,
        "blocks": blocks,
        "windows": windows,
        "methods": methods,
        "execution": {
            "workers": args.workers,
            "parallel_unit": "subject_window" if args.workers > 1 else "serial",
            "timing_ledger": "runtime.csv",
            "timing_stages": ["fit", "predict", "artifact_write", "fit_predict", "load_epochs_and_unit_total"],
        },
        "started_at_utc": started_at,
        "epoch_preset": preset.manifest(),
        "epoch_cache": str(args.epoch_cache),
        "preprocessing": {
            "cue_seconds": spec.cue_seconds,
            "visual_latency_seconds": spec.latency_seconds,
            "crop_start_seconds": preset.crop_start_seconds,
            "notch": "50 Hz iircomb Q=35",
            "filterbank": "SSVEP-Analysis-Toolbox Benchmark filterbank, 5 subbands by default",
            "epoch_cache_policy": "subject_max_window_slice",
            "epoch_cache_window": cache_window,
            "itr_trial_seconds": "window + 0.5 s gaze shift",
        },
    }

    if args.resume:
        trial_rows, pred_rows, runtime_rows = load_existing_rows(result_dir)
        done_windows = completed_windows(trial_rows, methods, subjects, blocks)
        if done_windows:
            completed = []
            for window, done_methods in sorted(done_windows.items()):
                completed.append(f"{window:g}: {'/'.join(sorted(done_methods))}")
            print("resume: completed " + "; ".join(completed), flush=True)
    else:
        trial_rows: list[dict] = []
        pred_rows: list[dict] = []
        runtime_rows: list[dict] = []
        done_windows: dict = {}

    started = time.perf_counter()
    for window in windows:
        pending_methods = [m for m in methods if m not in done_windows.get(float(window), set())]
        for m in methods:
            if m not in pending_methods:
                print(f"resume skip {m} window={window:.1f}", flush=True)
        if not pending_methods:
            manifest["seconds"] = time.perf_counter() - started
            write_outputs(result_dir, run_dir, trial_rows, pred_rows, runtime_rows, manifest, spec, complete=False)
            print(f"checkpoint window={window:.1f} -> {result_dir}", flush=True)
            continue

        tasks = [
            {
                "data_root": str(args.data_root),
                "channel_preset": args.channel_preset,
                "epoch_cache": str(args.epoch_cache),
                "result_dir": str(result_dir),
                "subject": subject,
                "window": window,
                "cache_window": cache_window,
                "methods": pending_methods,
                "blocks": blocks,
                "harmonics": args.harmonics,
                "n_fbs": args.n_fbs,
                "force_epochs": args.force_epochs,
                "save_score_matrices": args.save_score_matrices,
                "save_trca_filters": args.save_trca_filters,
            }
            for subject in subjects
        ]
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_map = {executor.submit(run_subject_window_task, task): task for task in tasks}
                for future in as_completed(future_map):
                    result = future.result()
                    trial_rows.extend(result["trial_rows"])
                    pred_rows.extend(result["pred_rows"])
                    runtime_rows.extend(result["runtime_rows"])
                    manifest["last_epoch_fingerprint"] = result["epoch_fingerprint"]
                    for line in result["logs"]:
                        print(line, flush=True)
        else:
            for task in tasks:
                result = run_subject_window_task(task)
                trial_rows.extend(result["trial_rows"])
                pred_rows.extend(result["pred_rows"])
                runtime_rows.extend(result["runtime_rows"])
                manifest["last_epoch_fingerprint"] = result["epoch_fingerprint"]
                for line in result["logs"]:
                    print(line, flush=True)
        manifest["seconds"] = time.perf_counter() - started
        write_outputs(result_dir, run_dir, trial_rows, pred_rows, runtime_rows, manifest, spec, complete=False)
        print(f"checkpoint window={window:.1f} -> {result_dir}", flush=True)

    manifest["seconds"] = time.perf_counter() - started
    manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_outputs(result_dir, run_dir, trial_rows, pred_rows, runtime_rows, manifest, spec, complete=True)
    print(result_dir)


if __name__ == "__main__":
    main()
