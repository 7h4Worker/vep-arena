# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_traditional_benchmark import (
    confusion_counts,
    plot_outputs,
    summarize,
    write_report,
)
from vep_arena.config import DATA_ROOT, PROJECT_ROOT, RUN_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import reference_signals
from vep_arena.data.epochs import CanonicalEpochStore, EpochRequest, epoch_fingerprint
from vep_arena.data.presets import benchmark_9ch_default
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.tdca import TDCA


def parse_range(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        elif part:
            values.append(int(part))
    return values


def parse_windows(text: str) -> list[float]:
    if ":" in text:
        start, step, stop = [float(x) for x in text.split(":", 2)]
        values = []
        current = start
        while current <= stop + 1e-9:
            values.append(round(current, 10))
            current += step
        return values
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def save_score_matrix(out_dir: Path, window: float, subject: int, block: int, scores: np.ndarray) -> None:
    method_dir = out_dir / "score_matrices" / "tdca"
    method_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        method_dir / f"w{window:g}_s{subject:02d}_b{block}.npz",
        scores=np.asarray(scores, dtype=np.float32),
    )


def save_tdca_artifact(out_dir: Path, window: float, subject: int, block: int, model: TDCA) -> None:
    method_dir = out_dir / "model_artifacts" / "tdca"
    method_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        method_dir / f"w{window:g}_s{subject:02d}_b{block}.npz",
        filters=np.asarray(model.filters, dtype=np.float32),
        fb_weights=np.asarray(model.fb_weights, dtype=np.float32),
        n_components=np.asarray([model.n_components], dtype=np.int64),
        n_delay=np.asarray([model.padding_len], dtype=np.int64),
    )


def load_existing(result_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    preds: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    if (result_dir / "trials.csv").exists():
        rows = pd.read_csv(result_dir / "trials.csv").to_dict("records")
    if (result_dir / "predictions.csv").exists():
        preds = pd.read_csv(result_dir / "predictions.csv").to_dict("records")
    if (result_dir / "runtime.csv").exists():
        runtime = pd.read_csv(result_dir / "runtime.csv").to_dict("records")
    return rows, preds, runtime


def completed_units(rows: list[dict[str, object]], subjects: list[int], blocks: list[int]) -> set[tuple[float, int]]:
    if not rows:
        return set()
    df = pd.DataFrame(rows)
    expected = len(blocks)
    done = set()
    for (window, subject), group in df.groupby(["window", "subject"]):
        if int(subject) in subjects and len(group) >= expected:
            done.add((float(window), int(subject)))
    return done


def write_outputs(
    result_dir: Path,
    run_dir: Path,
    trial_rows: list[dict[str, object]],
    pred_rows: list[dict[str, object]],
    runtime_rows: list[dict[str, object]],
    manifest: dict[str, object],
    spec: BenchmarkSpec,
    *,
    complete: bool,
) -> None:
    trials = pd.DataFrame(trial_rows)
    preds = pd.DataFrame(pred_rows)
    if trials.empty:
        return
    summary, subject, block = summarize(trials, spec)
    trials.to_csv(result_dir / "trials.csv", index=False)
    preds.to_csv(result_dir / "predictions.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(result_dir / "runtime.csv", index=False)
    summary.to_csv(result_dir / "summary.csv", index=False)
    subject.to_csv(result_dir / "subject.csv", index=False)
    block.to_csv(result_dir / "block.csv", index=False)
    np.save(run_dir / "confusion_tdca.npy", confusion_counts(preds["true"], preds["pred"], spec.classes))
    manifest["status"] = "complete" if complete else "partial"
    manifest["rows_written"] = int(len(trials))
    manifest["windows_written"] = sorted(float(x) for x in trials["window"].unique())
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_outputs(summary, subject, block, result_dir)
    write_report(summary, result_dir)


def run_subject_window_task(task: dict[str, object]) -> dict[str, object]:
    spec = BenchmarkSpec()
    data_root = Path(str(task["data_root"]))
    epoch_cache = Path(str(task["epoch_cache"]))
    result_dir = Path(str(task["result_dir"]))
    preset = benchmark_9ch_default(data_root)
    store = CanonicalEpochStore(epoch_cache)
    subject = int(task["subject"])
    window = float(task["window"])
    cache_window = float(task["cache_window"])
    blocks = [int(block) for block in task["blocks"]]
    n_fbs = int(task["n_fbs"])
    harmonics = int(task["harmonics"])
    n_components = int(task["n_components"])
    n_delay = int(task["n_delay"])
    force_epochs = bool(task["force_epochs"])
    save_scores = bool(task["save_score_matrices"])
    save_model = bool(task["save_model_artifacts"])

    load_started = time.perf_counter()
    req = EpochRequest(
        preset=preset,
        subject=subject,
        window=window,
        kind="filterbank",
        n_fbs=n_fbs,
        extra_samples=n_delay,
        cache_window=cache_window,
    )
    epochs = store.load_or_create(req, force=force_epochs)
    refs = reference_signals(window, harmonics, spec)
    labels = np.arange(spec.classes, dtype=np.int64)

    trial_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    logs: list[str] = []
    for block in blocks:
        train_blocks = [b - 1 for b in blocks if b != block]
        test_block = block - 1
        train_x = epochs[:, train_blocks].reshape(-1, epochs.shape[2], epochs.shape[3], epochs.shape[4])
        train_y = np.repeat(labels, len(train_blocks))
        test_x = epochs[:, test_block]
        model = TDCA(n_components=n_components, n_delay=n_delay)
        t0 = time.perf_counter()
        model.fit(train_x, train_y, refs)
        fit_done = time.perf_counter()
        pred, band_scores = model.predict(test_x)
        predict_done = time.perf_counter()
        scores = np.einsum("f,tfc->tc", model.fb_weights, band_scores)
        if save_scores:
            save_score_matrix(result_dir, window, subject, block, scores)
        if save_model:
            save_tdca_artifact(result_dir, window, subject, block, model)
        artifact_done = time.perf_counter()
        acc = float(np.mean(pred == labels))
        fit_predict_seconds = predict_done - t0
        itr = itr_bits_per_minute(acc, spec.classes, window + spec.cue_seconds)
        base_runtime = {
            "method": "TDCA",
            "window": window,
            "subject": subject,
            "block": block,
            "worker_pid": os.getpid(),
            "parallel_unit": "subject_window",
        }
        runtime_rows.extend(
            [
                {**base_runtime, "stage": "fit", "seconds": fit_done - t0},
                {**base_runtime, "stage": "predict", "seconds": predict_done - fit_done},
                {**base_runtime, "stage": "artifact_write", "seconds": artifact_done - predict_done},
                {**base_runtime, "stage": "fit_predict", "seconds": fit_predict_seconds},
            ]
        )
        trial_rows.append(
            {
                "method": "TDCA",
                "window": window,
                "subject": subject,
                "block": block,
                "accuracy": acc,
                "itr": float(itr),
                "samples": spec.classes,
                "seconds": fit_predict_seconds,
            }
        )
        for true_label, pred_label in zip(labels, pred):
            pred_rows.append(
                {
                    "method": "TDCA",
                    "window": window,
                    "subject": subject,
                    "block": block,
                    "true": int(true_label),
                    "pred": int(pred_label),
                    "score_true": float(scores[int(true_label), int(true_label)]),
                    "score_pred": float(scores[int(true_label), int(pred_label)]),
                }
            )
        logs.append(f"TDCA w={window:.1f} s={subject:02d} b={block} acc={acc:.3f} sec={fit_predict_seconds:.2f}")

    return {
        "trial_rows": trial_rows,
        "pred_rows": pred_rows,
        "runtime_rows": [
            {
                "method": "__all__",
                "window": window,
                "subject": subject,
                "block": "__all__",
                "stage": "load_epochs_and_unit_total",
                "seconds": time.perf_counter() - load_started,
                "worker_pid": os.getpid(),
                "parallel_unit": "subject_window",
            },
            *runtime_rows,
        ],
        "logs": logs,
        "epoch_fingerprint": epoch_fingerprint(req),
    }


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
    args = parser.parse_args()

    spec = BenchmarkSpec()
    preset = benchmark_9ch_default(args.data_root)
    subjects = parse_range(args.subjects)
    blocks = parse_range(args.blocks)
    windows = parse_windows(args.windows)
    cache_window = max(windows)
    result_dir = PROJECT_ROOT / "results" / args.task_name
    run_dir = RUN_ROOT / args.task_name
    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    store = CanonicalEpochStore(args.epoch_cache)

    manifest: dict[str, object] = {
        "task_name": args.task_name,
        "method": "TDCA",
        "dataset": "Tsinghua Benchmark SSVEP",
        "channels": "Benchmark 9ch: Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, O2",
        "protocol": "subject-specific leave-one-block-out",
        "subjects": subjects,
        "blocks": blocks,
        "windows": windows,
        "n_fbs": args.n_fbs,
        "harmonics": args.harmonics,
        "n_components": args.n_components,
        "n_delay": args.n_delay,
        "execution": {
            "workers": args.workers,
            "parallel_unit": "subject_window" if args.workers > 1 else "serial",
            "timing_ledger": "runtime.csv",
            "timing_stages": ["fit", "predict", "artifact_write", "fit_predict", "load_epochs_and_unit_total"],
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
            "tdca_extra_samples": args.n_delay,
            "epoch_cache_policy": "subject_max_window_slice",
            "epoch_cache_window": cache_window,
            "itr_trial_seconds": "window + 0.5 s gaze shift",
        },
    }

    if args.resume:
        trial_rows, pred_rows, runtime_rows = load_existing(result_dir)
        done = completed_units(trial_rows, subjects, blocks)
        if trial_rows:
            existing_windows = sorted(float(x) for x in pd.DataFrame(trial_rows)["window"].unique())
            windows_for_manifest = sorted(set(existing_windows + windows))
            manifest["windows"] = windows_for_manifest
        print(f"resume: {len(done)} subject-window units complete", flush=True)
    else:
        trial_rows = []
        pred_rows = []
        runtime_rows = []
        done = set()

    started = time.perf_counter()
    if args.prebuild_epochs:
        total = len(subjects) * len(windows)
        built = 0
        for window in windows:
            for subject in subjects:
                req = EpochRequest(
                    preset=preset,
                    subject=subject,
                    window=window,
                    kind="filterbank",
                    n_fbs=args.n_fbs,
                    extra_samples=args.n_delay,
                    cache_window=cache_window,
                )
                store.load_or_create(req, force=args.force_epochs)
                built += 1
                print(f"epoch {built}/{total} w={window:.1f} s={subject:02d}", flush=True)
        if args.prebuild_only:
            manifest["seconds"] = time.perf_counter() - started
            manifest["status"] = "prebuilt"
            manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(result_dir)
            return

    for window in windows:
        tasks = [
            {
                "data_root": str(args.data_root),
                "epoch_cache": str(args.epoch_cache),
                "result_dir": str(result_dir),
                "subject": subject,
                "window": window,
                "cache_window": cache_window,
                "blocks": blocks,
                "n_fbs": args.n_fbs,
                "harmonics": args.harmonics,
                "n_components": args.n_components,
                "n_delay": args.n_delay,
                "force_epochs": args.force_epochs,
                "save_score_matrices": args.save_score_matrices,
                "save_model_artifacts": args.save_model_artifacts,
            }
            for subject in subjects
            if (float(window), int(subject)) not in done
        ]
        if not tasks:
            print(f"resume skip window={window:.1f}", flush=True)
            continue
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
