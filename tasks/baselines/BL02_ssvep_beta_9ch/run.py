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

from vep_arena.config import CACHE_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import reference_signals
from vep_arena.data.toolbox_adapter import (
    ToolboxDatasetAdapter,
    toolbox_beta,
    toolbox_dataset_info,
    toolbox_wearable_dry,
    toolbox_wearable_wet,
)
from vep_arena.evaluation import (
    completed_windows,
    load_existing_rows,
    parse_range,
    parse_windows,
    write_outputs,
)
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.tdca import TDCA
from vep_arena.methods.traditional import CCA, ECCA, FBCCA, TRCA

WINDOW_PRESETS = {
    "w02_10_step01": "0.2:0.1:1.0",
    "w02_10_step02": "0.2:0.2:1.0",
    "w02_20_step01": "0.2:0.1:2.0",
    "w02_20_step02": "0.2:0.2:2.0",
}


def open_dataset(name: str, root: Path, download: bool = False) -> ToolboxDatasetAdapter:
    if name == "beta":
        return toolbox_beta(root=root, download=download)
    if name == "wearable_wet":
        return toolbox_wearable_wet(root=root, download=download)
    if name == "wearable_dry":
        return toolbox_wearable_dry(root=root, download=download)
    raise ValueError(f"Unknown dataset: {name}")


def spec_from_info(info) -> BenchmarkSpec:
    return BenchmarkSpec(
        subjects=len(info.subjects),
        blocks=len(info.blocks),
        classes=len(info.targets),
        sampling_rate=info.sampling_rate,
        cue_seconds=info.break_seconds,
        latency_seconds=info.latency_seconds or 0.0,
    )


def make_model(method: str, window: float, spec: BenchmarkSpec, info, args: argparse.Namespace):
    if method == "CCA":
        return CCA(window=window, harmonics=args.harmonics, spec=spec, frequencies=info.frequencies, phases_pi=info.phases)
    if method == "FBCCA":
        return FBCCA(
            window=window, harmonics=args.harmonics, n_fbs=args.n_bands, spec=spec,
            frequencies=info.frequencies, phases_pi=info.phases,
        )
    if method == "ECCA":
        return ECCA(
            window=window, harmonics=args.harmonics, n_fbs=args.n_bands, spec=spec,
            frequencies=info.frequencies, phases_pi=info.phases,
        )
    if method == "TRCA":
        return TRCA(n_fbs=args.n_bands, ensemble=False)
    if method == "ETRCA":
        return TRCA(n_fbs=args.n_bands, ensemble=True)
    if method == "TDCA":
        return TDCA(n_components=args.n_components, n_delay=args.n_delay)
    raise ValueError(f"Unknown method: {method}")


def run_subject_window_task(task: dict) -> dict:
    dataset_name = str(task["dataset"])
    root = Path(str(task["root"]))
    subject = int(task["subject"])
    window = float(task["window"])
    filter_window = float(task["filter_window"])
    methods = [str(m) for m in task["methods"]]
    blocks = [int(b) for b in task["blocks"]]
    targets = [int(t) for t in task["targets"]]
    channels = str(task["channels"])
    args = argparse.Namespace(
        harmonics=int(task["harmonics"]),
        n_bands=int(task["n_bands"]),
        n_components=int(task["n_components"]),
        n_delay=int(task["n_delay"]),
    )

    dataset = open_dataset(dataset_name, root=root, download=False)
    info = dataset.info
    spec = spec_from_info(info)
    labels = np.asarray(targets, dtype=np.int64)
    load_started = time.perf_counter()

    data_by_method: dict[str, np.ndarray] = {}
    if "CCA" in methods:
        data_by_method["CCA"] = dataset.get_trials(
            subject, blocks, targets, channels, window, preprocess="raw", n_bands=1, filter_window=filter_window,
        ).x
    if any(m in methods for m in ("FBCCA", "ECCA", "TRCA", "ETRCA")):
        fb = dataset.get_trials(
            subject, blocks, targets, channels, window, preprocess="toolbox_fb", n_bands=args.n_bands, filter_window=filter_window,
        ).x
        for m in ("FBCCA", "ECCA", "TRCA", "ETRCA"):
            if m in methods:
                data_by_method[m] = fb
    if "TDCA" in methods:
        data_by_method["TDCA"] = dataset.get_trials(
            subject, blocks, targets, channels, window, preprocess="toolbox_fb",
            n_bands=args.n_bands, filter_window=filter_window, extra_samples=args.n_delay,
        ).x

    refs = reference_signals(window, args.harmonics, spec, frequencies=info.frequencies, phases_pi=info.phases)
    trial_rows: list[dict] = []
    pred_rows: list[dict] = []
    runtime_rows: list[dict] = []
    logs: list[str] = []

    for block_idx, block in enumerate(blocks):
        train_blocks = [idx for idx, item in enumerate(blocks) if item != block]
        test_block = block_idx
        for method in methods:
            epochs = data_by_method[method].reshape(len(blocks), len(targets), *data_by_method[method].shape[1:])
            train_x = epochs[train_blocks].reshape(-1, epochs.shape[2], epochs.shape[3], epochs.shape[4])
            train_y = np.tile(labels, len(train_blocks))
            test_x = epochs[test_block]
            model = make_model(method, window, spec, info, args)
            t0 = time.perf_counter()
            if method == "TDCA":
                model.fit(train_x, train_y, refs)
            else:
                model.fit(train_x, train_y)
            fit_done = time.perf_counter()
            pred, scores = model.predict(test_x)
            predict_done = time.perf_counter()
            if method == "TDCA":
                scores = np.einsum("f,tfc->tc", model.fb_weights, scores)
            fit_predict_seconds = predict_done - t0
            acc = float(np.mean(pred == labels))
            itr = itr_bits_per_minute(acc, spec.classes, window + info.break_seconds)
            base_runtime = {
                "method": method, "window": window, "subject": subject, "block": block,
                "worker_pid": os.getpid(), "parallel_unit": "subject_window",
            }
            runtime_rows.extend([
                {**base_runtime, "stage": "fit", "seconds": fit_done - t0},
                {**base_runtime, "stage": "predict", "seconds": predict_done - fit_done},
                {**base_runtime, "stage": "fit_predict", "seconds": fit_predict_seconds},
            ])
            trial_rows.append({
                "method": method, "window": window, "subject": subject, "block": block,
                "accuracy": acc, "itr": float(itr), "samples": len(targets), "seconds": fit_predict_seconds,
            })
            for true_label, pred_label in zip(labels, pred):
                pred_rows.append({
                    "method": method, "window": window, "subject": subject, "block": block,
                    "true": int(true_label), "pred": int(pred_label),
                    "score_true": float(scores[int(np.where(labels == true_label)[0][0]), int(true_label)]),
                    "score_pred": float(scores[int(np.where(labels == true_label)[0][0]), int(pred_label)]),
                })
            logs.append(f"{method} {dataset_name} w={window:.1f} s={subject:03d} b={block} acc={acc:.3f} sec={fit_predict_seconds:.2f}")

    return {
        "trial_rows": trial_rows,
        "pred_rows": pred_rows,
        "runtime_rows": [
            {
                "method": "__all__", "window": window, "subject": subject, "block": "__all__",
                "stage": "load_dataset_and_unit_total", "seconds": time.perf_counter() - load_started,
                "worker_pid": os.getpid(), "parallel_unit": "subject_window",
            },
            *runtime_rows,
        ],
        "logs": logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["beta", "wearable_wet", "wearable_dry"], required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--task-name")
    parser.add_argument("--subjects", default="1")
    parser.add_argument("--blocks")
    parser.add_argument("--targets")
    parser.add_argument("--channels", default="occipital_9ch")
    parser.add_argument("--windows", default="0.2:0.2:2.0")
    parser.add_argument("--window-preset", choices=sorted(WINDOW_PRESETS))
    parser.add_argument("--methods", default="CCA,FBCCA,ECCA,TRCA,TDCA")
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--n-bands", type=int, default=5)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-delay", type=int, default=5)
    parser.add_argument("--ensure-data", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    info = toolbox_dataset_info(args.dataset, root=args.root)
    root = info.root
    subjects = parse_range(args.subjects)
    blocks = parse_range(args.blocks) if args.blocks else list(info.blocks)
    targets = parse_range(args.targets) if args.targets else list(info.targets)
    window_text = WINDOW_PRESETS[args.window_preset] if args.window_preset else args.windows
    windows = parse_windows(window_text)
    methods = [x.strip().upper() for x in args.methods.split(",") if x.strip()]
    filter_window = max(windows)
    spec = spec_from_info(info)
    task_name = args.task_name or f"{args.dataset}_traditional_w{windows[0]:g}_{windows[-1]:g}s"
    result_dir = TASK / "results" / task_name
    run_dir = CACHE_ROOT / task_name
    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.ensure_data:
        print(f"ensure_data: materializing {args.dataset} at {root}", flush=True)
        open_dataset(args.dataset, root=root, download=True)
        print("ensure_data: complete", flush=True)

    manifest: dict = {
        "task_name": task_name,
        "dataset": info.name,
        "dataset_id": info.id,
        "root": str(root),
        "channels": args.channels,
        "protocol": "subject-specific leave-one-block-out",
        "subjects": subjects,
        "blocks": blocks,
        "targets": targets,
        "windows": windows,
        "filter_window": filter_window,
        "methods": methods,
        "n_bands": args.n_bands,
        "harmonics": args.harmonics,
        "execution": {
            "workers": args.workers,
            "parallel_unit": "subject_window" if args.workers > 1 else "serial",
            "timing_ledger": "runtime.csv",
        },
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preprocessing": {
            "source": "SSVEP-Analysis-Toolbox preprocess/filterbank",
            "filter_window_policy": "filter at max requested window, then crop to each window",
            "latency_seconds": info.latency_seconds,
            "itr_trial_seconds": "window + dataset break_seconds",
        },
    }

    if args.resume:
        trial_rows, pred_rows, runtime_rows = load_existing_rows(result_dir)
        done_windows = completed_windows(trial_rows, methods, subjects, blocks)
        if done_windows:
            print(
                "resume: completed "
                + "; ".join(f"{w:g}: {'/'.join(sorted(d))}" for w, d in sorted(done_windows.items())),
                flush=True,
            )
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
                "dataset": args.dataset,
                "root": str(root),
                "subject": subject,
                "window": window,
                "filter_window": filter_window,
                "methods": pending_methods,
                "blocks": blocks,
                "targets": targets,
                "channels": args.channels,
                "harmonics": args.harmonics,
                "n_bands": args.n_bands,
                "n_components": args.n_components,
                "n_delay": args.n_delay,
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
                    for line in result["logs"]:
                        print(line, flush=True)
        else:
            for task in tasks:
                result = run_subject_window_task(task)
                trial_rows.extend(result["trial_rows"])
                pred_rows.extend(result["pred_rows"])
                runtime_rows.extend(result["runtime_rows"])
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
