from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[2]
sys.path.insert(0, str(PROJECT))

from vep_arena.config import CACHE_ROOT, DATA_ROOT, BenchmarkSpec
from vep_arena.data.epochs import CanonicalEpochStore, EpochRequest, epoch_fingerprint
from vep_arena.data.presets import benchmark_9ch_default
from vep_arena.evaluation import (
    completed_windows,
    load_existing_rows,
    parse_range,
    parse_windows,
    write_outputs,
)
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.traditional import CCA, ECCA, FBCCA, SSCOR, TRCA

INPUT = TASK / "results" / "input"


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


def run_subject_window_task(task: dict) -> dict:
    spec = BenchmarkSpec()
    data_root = Path(str(task["data_root"]))
    epoch_cache = Path(str(task["epoch_cache"]))
    result_dir = Path(str(task["result_dir"]))
    preset = benchmark_9ch_default(data_root)
    store = CanonicalEpochStore(epoch_cache)
    subject = int(task["subject"])
    window = float(task["window"])
    cache_window = float(task["cache_window"])
    methods = [str(m) for m in task["methods"]]
    blocks = [int(b) for b in task["blocks"]]
    harmonics = int(task["harmonics"])
    n_fbs = int(task["n_fbs"])
    force_epochs = bool(task["force_epochs"])
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


def copy_runner_outputs(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("predictions.csv", "trials.csv", "summary.csv", "subject.csv", "block.csv", "runtime.csv", "manifest.json"):
        src = source / name
        if src.exists():
            shutil.copy2(src, dest / name)
    fig_src = source / "figures"
    if fig_src.exists():
        fig_dest = dest / "runner_figures"
        if fig_dest.exists():
            shutil.rmtree(fig_dest)
        shutil.copytree(fig_src, fig_dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--source-result", type=Path, default=None)
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--blocks", default="1-6")
    parser.add_argument("--windows", default=None)
    parser.add_argument("--methods", default="CCA,FBCCA,ECCA,TRCA,ETRCA")
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--n-fbs", type=int, default=5)
    parser.add_argument("--epoch-cache", type=Path, default=CACHE_ROOT / "canonical_epochs")
    parser.add_argument("--force-epochs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task-name", default=None)
    args = parser.parse_args()

    subjects_text = args.subjects or ("1-2" if args.smoke else "1-35")
    windows_text = args.windows or ("0.2,0.5,1.0" if args.smoke else "default")
    task_name = args.task_name or ("benchmark_decision_channel_capacity_smoke" if args.smoke else "benchmark_decision_channel_capacity_full")

    if args.reuse_existing:
        source = args.source_result or (TASK / "results" / task_name)
        if not (source / "predictions.csv").exists():
            raise FileNotFoundError(f"Missing predictions.csv under {source}")
        copy_runner_outputs(source, INPUT)
        print(INPUT)
        return

    spec = BenchmarkSpec()
    preset = benchmark_9ch_default(args.data_root)
    subjects = parse_range(subjects_text)
    blocks = parse_range(args.blocks)
    windows = parse_windows(windows_text)
    cache_window = max(windows)
    methods = [x.strip().upper() for x in args.methods.split(",") if x.strip()]
    result_dir = TASK / "results" / task_name
    run_dir = CACHE_ROOT / task_name
    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "task_name": task_name,
        "dataset": "Tsinghua Benchmark SSVEP",
        "channel_preset": "benchmark_9ch",
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
        },
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
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
        if not pending_methods:
            print(f"resume skip window={window:.1f}", flush=True)
            continue
        tasks = [
            {
                "data_root": str(args.data_root),
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
    copy_runner_outputs(result_dir, INPUT)
    print(result_dir)


if __name__ == "__main__":
    main()
