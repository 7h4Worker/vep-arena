# Author: Liu Haifeng <liuhf@728@gmail.com>
# Description: TDCA runner retaining its CLI and method computation.
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
from vep_arena.config import DATA_ROOT, PROJECT_ROOT, RUN_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import reference_signals
from vep_arena.data.epochs import CanonicalEpochStore, EpochRequest, epoch_fingerprint
from vep_arena.data.presets import benchmark_9ch_default
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.tdca import TDCA
from vep_arena.evaluation import (
    confusion_counts, plot_outputs, summarize, write_report, parse_range, parse_windows,
    completed_units, write_outputs, load_existing_rows,
)
from vep_arena.run_contract import atomic_json, prepare_run

# Historical import name; safe resume now also requires expected_manifest.
load_existing = load_existing_rows


def save_score_matrix(out_dir: Path, window: float, subject: int, block: int, scores: np.ndarray) -> None:
    directory = out_dir / "score_matrices" / "tdca"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"w{window:g}_s{subject:02d}_b{block}.npz", scores=np.asarray(scores, dtype=np.float32))


def save_tdca_artifact(out_dir: Path, window: float, subject: int, block: int, model: TDCA) -> None:
    directory = out_dir / "model_artifacts" / "tdca"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"w{window:g}_s{subject:02d}_b{block}.npz",
        filters=np.asarray(model.filters, dtype=np.float32), fb_weights=np.asarray(model.fb_weights, dtype=np.float32),
        n_components=np.asarray([model.n_components], dtype=np.int64), n_delay=np.asarray([model.padding_len], dtype=np.int64))


def run_subject_window_task(task: dict) -> dict:
    spec = BenchmarkSpec()
    data_root, epoch_cache, result_dir = Path(task["data_root"]), Path(task["epoch_cache"]), Path(task["result_dir"])
    preset = benchmark_9ch_default(data_root)
    store = CanonicalEpochStore(epoch_cache)
    subject, window = int(task["subject"]), float(task["window"])
    blocks = [int(b) for b in task["blocks"]]
    load_started = time.perf_counter()
    req = EpochRequest(preset=preset, subject=subject, window=window, kind="filterbank",
        n_fbs=int(task["n_fbs"]), extra_samples=int(task["n_delay"]), cache_window=float(task["cache_window"]))
    epochs = store.load_or_create(req, force=bool(task["force_epochs"]))
    refs = reference_signals(window, int(task["harmonics"]), spec)
    labels = np.arange(spec.classes, dtype=np.int64)
    trial_rows, pred_rows, runtime_rows, logs = [], [], [], []
    for block in blocks:
        train_blocks = [b - 1 for b in blocks if b != block]
        train_x = epochs[:, train_blocks].reshape(-1, epochs.shape[2], epochs.shape[3], epochs.shape[4])
        train_y = np.repeat(labels, len(train_blocks))
        test_x = epochs[:, block - 1]
        model = TDCA(n_components=int(task["n_components"]), n_delay=int(task["n_delay"]))
        t0 = time.perf_counter()
        model.fit(train_x, train_y, refs)
        fit_done = time.perf_counter()
        pred, band_scores = model.predict(test_x)
        predict_done = time.perf_counter()
        scores = np.einsum("f,tfc->tc", model.fb_weights, band_scores)
        if task["save_score_matrices"]:
            save_score_matrix(result_dir, window, subject, block, scores)
        if task["save_model_artifacts"]:
            save_tdca_artifact(result_dir, window, subject, block, model)
        artifact_done = time.perf_counter()
        acc, seconds = float(np.mean(pred == labels)), predict_done - t0
        itr = itr_bits_per_minute(acc, spec.classes, window + spec.cue_seconds)
        base = dict(method="TDCA", window=window, subject=subject, block=block)
        timing = dict(**base, worker_pid=os.getpid(), parallel_unit="subject_window")
        runtime_rows.extend([dict(**timing, stage="fit", seconds=fit_done - t0),
            dict(**timing, stage="predict", seconds=predict_done - fit_done),
            dict(**timing, stage="artifact_write", seconds=artifact_done - predict_done),
            dict(**timing, stage="fit_predict", seconds=seconds)])
        trial_rows.append(dict(**base, accuracy=acc, itr=float(itr), samples=spec.classes, seconds=seconds))
        for true_label, pred_label in zip(labels, pred):
            pred_rows.append(dict(**base, true=int(true_label), pred=int(pred_label),
                score_true=float(scores[int(true_label), int(true_label)]),
                score_pred=float(scores[int(true_label), int(pred_label)])))
        logs.append(f"TDCA w={window:.1f} s={subject:02d} b={block} acc={acc:.3f} sec={seconds:.2f}")
    return dict(trial_rows=trial_rows, pred_rows=pred_rows, runtime_rows=[
        dict(method="__all__", window=window, subject=subject, block="__all__", stage="load_epochs_and_unit_total",
             seconds=time.perf_counter() - load_started, worker_pid=os.getpid(), parallel_unit="subject_window"), *runtime_rows],
        logs=logs, epoch_fingerprint=epoch_fingerprint(req))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--task-name", default="tdca_9ch_w02_2s")
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--blocks", default="1-6")
    parser.add_argument("--windows", default="0.2:0.2:2.0")
    parser.add_argument("--n-fbs", type=int, default=5)
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-delay", type=int, default=5)
    parser.add_argument("--epoch-cache", type=Path, default=RUN_ROOT / "canonical_epochs")
    parser.add_argument("--force-epochs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-score-matrices", action="store_true")
    parser.add_argument("--save-model-artifacts", action="store_true")
    parser.add_argument("--prebuild-epochs", action="store_true")
    parser.add_argument("--prebuild-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--source-manifest", type=Path, help="Explicit source hash inventory; required for resume")
    args = parser.parse_args()
    if args.prebuild_only and not args.prebuild_epochs:
        raise ValueError("--prebuild-only requires --prebuild-epochs")
    spec, preset = BenchmarkSpec(), benchmark_9ch_default(args.data_root)
    subjects, blocks, windows = parse_range(args.subjects), parse_range(args.blocks), parse_windows(args.windows)
    if not set(subjects) <= set(range(1, spec.subjects + 1)) or not set(blocks) <= set(range(1, spec.blocks + 1)):
        raise ValueError("Subjects/blocks outside Benchmark scope")
    if len(blocks) < 2:
        raise ValueError("Leave-one-block-out requires at least two blocks")
    cache_window = max(windows)
    result_dir, run_dir = PROJECT_ROOT / "results" / args.task_name, RUN_ROOT / args.task_name
    manifest = dict(task_name=args.task_name, method="TDCA", dataset="Tsinghua Benchmark SSVEP", channels=list(preset.channels),
        protocol="subject-specific leave-one-block-out", subjects=subjects, blocks=blocks, windows=windows,
        n_fbs=args.n_fbs, harmonics=args.harmonics, n_components=args.n_components, n_delay=args.n_delay,
        execution=dict(workers=args.workers, parallel_unit="subject_window" if args.workers > 1 else "serial",
            timing_ledger="runtime.csv", timing_stages=["fit", "predict", "artifact_write", "fit_predict", "load_epochs_and_unit_total"]),
        started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), epoch_preset=preset.manifest(), epoch_cache=str(args.epoch_cache),
        preprocessing=dict(cue_seconds=spec.cue_seconds, visual_latency_seconds=spec.latency_seconds,
            crop_start_seconds=preset.crop_start_seconds, notch="50 Hz iircomb Q=35",
            filterbank="SSVEP-Analysis-Toolbox Benchmark filterbank", n_fbs=args.n_fbs, tdca_extra_samples=args.n_delay,
            epoch_cache_policy="subject_max_window_slice", epoch_cache_window=cache_window, itr_trial_seconds="window + 0.5 s gaze shift"))
    prepare_run(manifest, args, PROJECT_ROOT, result_dir, classes=spec.classes)
    trial_rows, pred_rows, runtime_rows = load_existing(result_dir, manifest) if args.resume else ([], [], [])
    done = completed_units(trial_rows, subjects, blocks)
    started = time.perf_counter()
    if args.prebuild_epochs:
        store = CanonicalEpochStore(args.epoch_cache)
        for window in windows:
            for subject in subjects:
                req = EpochRequest(preset=preset, subject=subject, window=window, kind="filterbank", n_fbs=args.n_fbs,
                    extra_samples=args.n_delay, cache_window=cache_window)
                store.load_or_create(req, force=args.force_epochs)
                print(f"epoch w={window:g} s={subject:02d}", flush=True)
        if args.prebuild_only:
            manifest.update(status="prebuilt", seconds=time.perf_counter() - started)
            atomic_json(result_dir / "manifest.json", manifest)
            print(result_dir)
            return
    for window in windows:
        tasks = [dict(data_root=str(args.data_root), epoch_cache=str(args.epoch_cache), result_dir=str(result_dir),
            subject=s, window=window, cache_window=cache_window, blocks=blocks, n_fbs=args.n_fbs, harmonics=args.harmonics,
            n_components=args.n_components, n_delay=args.n_delay, force_epochs=args.force_epochs,
            save_score_matrices=args.save_score_matrices, save_model_artifacts=args.save_model_artifacts)
            for s in subjects if (float(window), int(s)) not in done]
        if not tasks:
            print(f"resume skip window={window:g}", flush=True)
            continue
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
