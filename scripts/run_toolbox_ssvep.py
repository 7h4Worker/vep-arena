"""BETA/Wearable runner with shared, validated result bookkeeping."""
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
from vep_arena.config import PROJECT_ROOT, RUN_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import reference_signals
from vep_arena.data.toolbox_adapter import (
    ToolboxDatasetAdapter, toolbox_beta, toolbox_dataset_info, toolbox_wearable_dry, toolbox_wearable_wet,
)
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.tdca import TDCA
from vep_arena.methods.traditional import CCA, ECCA, FBCCA, TRCA
from vep_arena.evaluation import (
    confusion_counts, plot_outputs, summarize, write_report, parse_range, parse_windows,
    completed_windows, write_outputs, load_existing_rows,
)
from vep_arena.run_contract import prepare_run

load_existing = load_existing_rows
WINDOW_PRESETS = {"w02_10_step01": "0.2:0.1:1.0", "w02_10_step02": "0.2:0.2:1.0",
                  "w02_20_step01": "0.2:0.1:2.0", "w02_20_step02": "0.2:0.2:2.0"}


def method_names(text: str) -> list[str]:
    methods = [x.strip().upper() for x in text.split(",") if x.strip()]
    allowed = {"CCA", "FBCCA", "ECCA", "TRCA", "ETRCA", "TDCA"}
    if not methods or len(methods) != len(set(methods)) or not set(methods) <= allowed:
        raise ValueError(f"Methods must be unique entries from {sorted(allowed)}")
    return methods


def open_dataset(name: str, root: Path, download: bool = False) -> ToolboxDatasetAdapter:
    if name == "beta":
        return toolbox_beta(root=root, download=download)
    if name == "wearable_wet":
        return toolbox_wearable_wet(root=root, download=download)
    if name == "wearable_dry":
        return toolbox_wearable_dry(root=root, download=download)
    raise ValueError(f"Unknown dataset: {name}")


def spec_from_info(info) -> BenchmarkSpec:
    return BenchmarkSpec(subjects=len(info.subjects), blocks=len(info.blocks), classes=len(info.targets),
        sampling_rate=info.sampling_rate, cue_seconds=info.break_seconds, latency_seconds=info.latency_seconds or 0.0)


def make_model(method: str, window: float, spec: BenchmarkSpec, info, args: argparse.Namespace):
    if method == "CCA":
        return CCA(window=window, harmonics=args.harmonics, spec=spec, frequencies=info.frequencies, phases_pi=info.phases)
    if method == "FBCCA":
        return FBCCA(window=window, harmonics=args.harmonics, n_fbs=args.n_bands, spec=spec,
                     frequencies=info.frequencies, phases_pi=info.phases)
    if method == "ECCA":
        return ECCA(window=window, harmonics=args.harmonics, n_fbs=args.n_bands, spec=spec,
                    frequencies=info.frequencies, phases_pi=info.phases)
    if method == "TRCA":
        return TRCA(n_fbs=args.n_bands, ensemble=False)
    if method == "ETRCA":
        return TRCA(n_fbs=args.n_bands, ensemble=True)
    if method == "TDCA":
        return TDCA(n_components=args.n_components, n_delay=args.n_delay)
    raise ValueError(f"Unknown method: {method}")


def run_subject_window_task(task: dict) -> dict:
    dataset_name, root = str(task["dataset"]), Path(task["root"])
    subject, window, filter_window = int(task["subject"]), float(task["window"]), float(task["filter_window"])
    methods, blocks, targets = list(task["methods"]), list(task["blocks"]), list(task["targets"])
    channels = str(task["channels"])
    args = argparse.Namespace(harmonics=int(task["harmonics"]), n_bands=int(task["n_bands"]),
                              n_components=int(task["n_components"]), n_delay=int(task["n_delay"]))
    dataset = open_dataset(dataset_name, root=root, download=False)
    info, spec = dataset.info, spec_from_info(dataset.info)
    labels = np.asarray(targets, dtype=np.int64)
    load_started, data_by_method = time.perf_counter(), {}
    if "CCA" in methods:
        data_by_method["CCA"] = dataset.get_trials(subject, blocks, targets, channels, window,
            preprocess="raw", n_bands=1, filter_window=filter_window).x
    fb_methods = {"FBCCA", "ECCA", "TRCA", "ETRCA"}
    if any(m in fb_methods for m in methods):
        fb = dataset.get_trials(subject, blocks, targets, channels, window, preprocess="toolbox_fb",
                                n_bands=args.n_bands, filter_window=filter_window).x
        data_by_method.update({m: fb for m in methods if m in fb_methods})
    if "TDCA" in methods:
        data_by_method["TDCA"] = dataset.get_trials(subject, blocks, targets, channels, window,
            preprocess="toolbox_fb", n_bands=args.n_bands, filter_window=filter_window, extra_samples=args.n_delay).x
    refs = reference_signals(window, args.harmonics, spec, frequencies=info.frequencies, phases_pi=info.phases)
    trial_rows, pred_rows, runtime_rows, logs = [], [], [], []
    for block_idx, block in enumerate(blocks):
        train_blocks = [idx for idx, item in enumerate(blocks) if item != block]
        for method in methods:
            epochs = data_by_method[method].reshape(len(blocks), len(targets), *data_by_method[method].shape[1:])
            train_x = epochs[train_blocks].reshape(-1, epochs.shape[2], epochs.shape[3], epochs.shape[4])
            train_y, test_x = np.tile(labels, len(train_blocks)), epochs[block_idx]
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
            seconds, acc = predict_done - t0, float(np.mean(pred == labels))
            itr = itr_bits_per_minute(acc, spec.classes, window + info.break_seconds)
            base = dict(method=method, window=window, subject=subject, block=block)
            timing = dict(**base, worker_pid=os.getpid(), parallel_unit="subject_window")
            runtime_rows.extend([dict(**timing, stage="fit", seconds=fit_done - t0),
                dict(**timing, stage="predict", seconds=predict_done - fit_done), dict(**timing, stage="fit_predict", seconds=seconds)])
            trial_rows.append(dict(**base, accuracy=acc, itr=float(itr), samples=len(targets), seconds=seconds))
            for row_index, (true_label, pred_label) in enumerate(zip(labels, pred)):
                pred_rows.append(dict(**base, true=int(true_label), pred=int(pred_label),
                    score_true=float(scores[row_index, int(true_label)]), score_pred=float(scores[row_index, int(pred_label)])))
            logs.append(f"{method} {dataset_name} w={window:.1f} s={subject:03d} b={block} acc={acc:.3f} sec={seconds:.2f}")
    return dict(trial_rows=trial_rows, pred_rows=pred_rows, runtime_rows=[
        dict(method="__all__", window=window, subject=subject, block="__all__", stage="load_dataset_and_unit_total",
             seconds=time.perf_counter() - load_started, worker_pid=os.getpid(), parallel_unit="subject_window"), *runtime_rows], logs=logs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["beta", "wearable_wet", "wearable_dry"], required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--task-name")
    parser.add_argument("--output-dir", type=Path)
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
    parser.add_argument("--source-manifest", type=Path, help="Explicit source hash inventory; required for resume")
    args = parser.parse_args()
    info = toolbox_dataset_info(args.dataset, root=args.root)
    root = info.root
    subjects, blocks = parse_range(args.subjects), parse_range(args.blocks) if args.blocks else list(info.blocks)
    targets = parse_range(args.targets) if args.targets else list(info.targets)
    windows = parse_windows(WINDOW_PRESETS[args.window_preset] if args.window_preset else args.windows)
    methods, spec = method_names(args.methods), spec_from_info(info)
    if not set(subjects) <= set(info.subjects) or not set(blocks) <= set(info.blocks):
        raise ValueError("Subjects/blocks outside declared dataset scope")
    if len(blocks) < 2:
        raise ValueError("Leave-one-block-out requires at least two blocks")
    if targets != list(info.targets):
        raise ValueError("Target subsets require model label/score remapping; this runner currently accepts the full target set")
    task_name = args.task_name or f"{args.dataset}_traditional_w{windows[0]:g}_{windows[-1]:g}s"
    result_dir, run_dir = args.output_dir or (PROJECT_ROOT / "results" / task_name), RUN_ROOT / task_name
    manifest = dict(task_name=task_name, dataset=info.name, dataset_id=info.id, root=str(root), channels=args.channels,
        protocol="subject-specific leave-one-block-out", subjects=subjects, blocks=blocks, targets=targets,
        windows=windows, filter_window=max(windows), methods=methods, n_bands=args.n_bands, harmonics=args.harmonics,
        execution=dict(workers=args.workers, parallel_unit="subject_window" if args.workers > 1 else "serial", timing_ledger="runtime.csv"),
        started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        preprocessing=dict(source="SSVEP-Analysis-Toolbox preprocess/filterbank",
            filter_window_policy="filter at max requested window, then crop to each window",
            latency_seconds=info.latency_seconds, itr_trial_seconds="window + dataset break_seconds", break_seconds=info.break_seconds))
    if args.ensure_data and (args.source_manifest is not None or args.resume):
        raise ValueError("Materialize data first, then build its source inventory; do not replace data while resuming")
    prepare_run(manifest, args, PROJECT_ROOT, result_dir, classes=spec.classes)
    if args.ensure_data:
        open_dataset(args.dataset, root=root, download=True)
    trial_rows, pred_rows, runtime_rows = load_existing(result_dir, manifest) if args.resume else ([], [], [])
    done = completed_windows(trial_rows, methods, subjects, blocks)
    started = time.perf_counter()
    for window in windows:
        pending = [m for m in methods if m not in done.get(float(window), set())]
        if not pending:
            print(f"resume skip window={window:g}", flush=True)
            continue
        tasks = [dict(dataset=args.dataset, root=str(root), subject=s, window=window, filter_window=max(windows),
            methods=pending, blocks=blocks, targets=targets, channels=args.channels, harmonics=args.harmonics,
            n_bands=args.n_bands, n_components=args.n_components, n_delay=args.n_delay) for s in subjects]
        def consume(result):
            trial_rows.extend(result["trial_rows"])
            pred_rows.extend(result["pred_rows"])
            runtime_rows.extend(result["runtime_rows"])
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
