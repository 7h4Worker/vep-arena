# Author: Liu Haifeng <liuhf@728@gmail.com>
# Description: Benchmark runner; shared bookkeeping lives in vep_arena.evaluation.
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vep_arena.config import DATA_ROOT, PROJECT_ROOT, RUN_ROOT, WINDOWS, BenchmarkSpec
from vep_arena.data.epochs import CanonicalEpochStore, EpochRequest, epoch_fingerprint
from vep_arena.data.presets import DatasetPreset, benchmark_9ch_default, benchmark_64ch_default
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.traditional import CCA, ECCA, FBCCA, SSCOR, TRCA
# Re-export names used by legacy callers, without duplicate implementations.
from vep_arena.evaluation import (
    confusion_counts, parse_range, parse_windows, summarize, plot_outputs,
    write_report, write_outputs, load_existing_rows, completed_windows,
)
from vep_arena.run_contract import prepare_run


def save_score_matrix(out_dir: Path, method: str, window: float, subject: int, block: int, scores: np.ndarray) -> None:
    directory = out_dir / "score_matrices" / method.lower()
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"w{window:g}_s{subject:02d}_b{block}.npz", scores=np.asarray(scores, dtype=np.float32))


def save_trca_artifact(out_dir: Path, method: str, window: float, subject: int, block: int, model: object) -> None:
    filters = getattr(model, "filters", None)
    if filters is None:
        return
    directory = out_dir / "model_artifacts" / method.lower()
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"w{window:g}_s{subject:02d}_b{block}.npz", filters=np.asarray(filters, dtype=np.float32))


def method_names(text: str) -> list[str]:
    names = [x.strip().upper() for x in text.split(",") if x.strip()]
    allowed = {"CCA", "FBCCA", "ECCA", "TRCA", "ETRCA", "SSCOR", "ESSCOR"}
    if not names or len(names) != len(set(names)) or not set(names) <= allowed:
        raise ValueError(f"Methods must be unique entries from {sorted(allowed)}")
    return names


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


def resolve_benchmark_preset(name: str, data_root: Path) -> DatasetPreset:
    if name == "benchmark_9ch":
        return benchmark_9ch_default(data_root)
    if name == "benchmark_64ch":
        return benchmark_64ch_default(data_root)
    raise ValueError(f"Unknown Benchmark channel preset: {name}")


def load_epochs(name: str, subject: int, window: float, args: argparse.Namespace, store: CanonicalEpochStore) -> np.ndarray:
    preset = resolve_benchmark_preset(getattr(args, "channel_preset", "benchmark_9ch"), args.data_root)
    cache_window = getattr(args, "cache_window", window)
    if name == "CCA":
        raw = store.load_or_create(EpochRequest(preset=preset, subject=subject, window=window, kind="raw", cache_window=cache_window))
        return raw[:, :, None, :, :]
    return store.load_or_create(EpochRequest(preset=preset, subject=subject, window=window,
        kind="filterbank", n_fbs=args.n_fbs, cache_window=cache_window))


def run_subject_window_task(task: dict) -> dict:
    spec = BenchmarkSpec()
    data_root, epoch_cache, result_dir = Path(task["data_root"]), Path(task["epoch_cache"]), Path(task["result_dir"])
    preset = resolve_benchmark_preset(str(task.get("channel_preset", "benchmark_9ch")), data_root)
    store = CanonicalEpochStore(epoch_cache)
    subject, window, cache_window = int(task["subject"]), float(task["window"]), float(task["cache_window"])
    methods, blocks = list(task["methods"]), [int(b) for b in task["blocks"]]
    harmonics, n_fbs = int(task["harmonics"]), int(task["n_fbs"])
    force_epochs = bool(task["force_epochs"])
    save_scores, save_filters = bool(task["save_score_matrices"]), bool(task["save_trca_filters"])
    args = argparse.Namespace(harmonics=harmonics, n_fbs=n_fbs)
    load_started = time.perf_counter()
    data_by_method = {}
    if "CCA" in methods:
        req = EpochRequest(preset=preset, subject=subject, window=window, kind="raw", cache_window=cache_window)
        data_by_method["CCA"] = store.load_or_create(req, force=force_epochs)[:, :, None, :, :]
    fb_methods = {"FBCCA", "ECCA", "TRCA", "ETRCA", "SSCOR", "ESSCOR"}
    if any(method in fb_methods for method in methods):
        req = EpochRequest(preset=preset, subject=subject, window=window, kind="filterbank", n_fbs=n_fbs, cache_window=cache_window)
        filtered = store.load_or_create(req, force=force_epochs)
        data_by_method.update({m: filtered for m in methods if m in fb_methods})
    labels = np.arange(spec.classes, dtype=np.int64)
    trial_rows, pred_rows, runtime_rows, logs = [], [], [], []
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
            base = dict(method=method, window=window, subject=subject, block=block)
            timing = dict(**base, worker_pid=os.getpid(), parallel_unit="subject_window")
            runtime_rows.extend([dict(**timing, stage="fit", seconds=fit_done - t0),
                dict(**timing, stage="predict", seconds=predict_done - fit_done),
                dict(**timing, stage="artifact_write", seconds=artifact_done - predict_done),
                dict(**timing, stage="fit_predict", seconds=seconds)])
            trial_rows.append(dict(**base, accuracy=float(acc), itr=float(itr), samples=spec.classes, seconds=seconds))
            for true_label, pred_label in zip(labels, pred):
                pred_rows.append(dict(**base, true=int(true_label), pred=int(pred_label),
                    score_true=float(scores[int(true_label), int(true_label)]),
                    score_pred=float(scores[int(true_label), int(pred_label)])))
            logs.append(f"{method} w={window:.1f} s={subject:02d} b={block} acc={acc:.3f} sec={seconds:.2f}")
    return dict(trial_rows=trial_rows, pred_rows=pred_rows, runtime_rows=[
        dict(method="__all__", window=window, subject=subject, block="__all__", stage="load_epochs_and_unit_total",
             seconds=time.perf_counter() - load_started, worker_pid=os.getpid(), parallel_unit="subject_window"), *runtime_rows],
        logs=logs, epoch_fingerprint=epoch_fingerprint(EpochRequest(preset=preset, subject=subject,
            window=window, kind="filterbank", n_fbs=n_fbs, cache_window=cache_window)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--task-name", default="traditional_9ch")
    parser.add_argument("--channel-preset", choices=["benchmark_9ch", "benchmark_64ch"], default="benchmark_9ch")
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--blocks", default="1-6")
    parser.add_argument("--windows", default="default")
    parser.add_argument("--methods", default="CCA,FBCCA,TRCA")
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--n-fbs", type=int, default=5)
    parser.add_argument("--epoch-cache", type=Path, default=RUN_ROOT / "canonical_epochs")
    parser.add_argument("--force-epochs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-score-matrices", action="store_true")
    parser.add_argument("--save-trca-filters", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--source-manifest", type=Path, help="Explicit source hash inventory; required for resume")
    args = parser.parse_args()
    spec = BenchmarkSpec()
    preset = resolve_benchmark_preset(args.channel_preset, args.data_root)
    subjects, blocks = parse_range(args.subjects), parse_range(args.blocks)
    windows, methods = parse_windows(args.windows), method_names(args.methods)
    if not set(subjects) <= set(range(1, spec.subjects + 1)) or not set(blocks) <= set(range(1, spec.blocks + 1)):
        raise ValueError("Subjects/blocks outside Benchmark scope")
    if len(blocks) < 2:
        raise ValueError("Leave-one-block-out requires at least two blocks")
    result_dir, run_dir = PROJECT_ROOT / "results" / args.task_name, RUN_ROOT / args.task_name
    cache_window = max(windows)
    manifest = dict(task_name=args.task_name, dataset="Tsinghua Benchmark SSVEP", channel_preset=args.channel_preset,
        channels=list(preset.channels), n_channels=len(preset.channels), protocol="subject-specific leave-one-block-out",
        subjects=subjects, blocks=blocks, windows=windows, methods=methods,
        started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        execution=dict(workers=args.workers, parallel_unit="subject_window" if args.workers > 1 else "serial",
            timing_ledger="runtime.csv", timing_stages=["fit", "predict", "artifact_write", "fit_predict", "load_epochs_and_unit_total"]),
        epoch_preset=preset.manifest(), epoch_cache=str(args.epoch_cache),
        preprocessing=dict(cue_seconds=spec.cue_seconds, visual_latency_seconds=spec.latency_seconds,
            crop_start_seconds=preset.crop_start_seconds, notch="50 Hz iircomb Q=35",
            filterbank="SSVEP-Analysis-Toolbox Benchmark filterbank", n_fbs=args.n_fbs, harmonics=args.harmonics,
            epoch_cache_policy="subject_max_window_slice", epoch_cache_window=cache_window,
            itr_trial_seconds="window + 0.5 s gaze shift"))
    prepare_run(manifest, args, PROJECT_ROOT, result_dir, classes=spec.classes)
    trial_rows, pred_rows, runtime_rows = load_existing_rows(result_dir, manifest) if args.resume else ([], [], [])
    done = completed_windows(trial_rows, methods, subjects, blocks)
    started = time.perf_counter()
    for window in windows:
        pending = [m for m in methods if m not in done.get(float(window), set())]
        if not pending:
            print(f"resume skip window={window:g}", flush=True)
            continue
        tasks = [dict(data_root=str(args.data_root), channel_preset=args.channel_preset, epoch_cache=str(args.epoch_cache),
            result_dir=str(result_dir), subject=s, window=window, cache_window=cache_window, methods=pending, blocks=blocks,
            harmonics=args.harmonics, n_fbs=args.n_fbs, force_epochs=args.force_epochs,
            save_score_matrices=args.save_score_matrices, save_trca_filters=args.save_trca_filters) for s in subjects]
        def consume(result):
            trial_rows.extend(result["trial_rows"])
            pred_rows.extend(result["pred_rows"])
            runtime_rows.extend(result["runtime_rows"])
            manifest["last_epoch_fingerprint"] = result["epoch_fingerprint"]
            for line in result["logs"]:
                print(line, flush=True)
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for future in as_completed([executor.submit(run_subject_window_task, task) for task in tasks]):
                    consume(future.result())
        else:
            for task in tasks:
                consume(run_subject_window_task(task))
        manifest["seconds"] = time.perf_counter() - started
        write_outputs(result_dir, run_dir, trial_rows, pred_rows, runtime_rows, manifest, spec, False)
    manifest["seconds"] = time.perf_counter() - started
    manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_outputs(result_dir, run_dir, trial_rows, pred_rows, runtime_rows, manifest, spec, True)
    print(result_dir)


if __name__ == "__main__":
    main()
