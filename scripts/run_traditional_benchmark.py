from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import DATA_ROOT, PROJECT_ROOT, RUN_ROOT, WINDOWS, BenchmarkSpec
from vep_arena.data.epochs import CanonicalEpochStore, EpochRequest, epoch_fingerprint
from vep_arena.data.presets import benchmark_9ch_default
from vep_arena.metrics import itr_bits_per_minute, sem
from vep_arena.methods.traditional import CCA, FBCCA, TRCA


def confusion_counts(true: pd.Series, pred: pd.Series, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for t, p in zip(true.to_numpy(dtype=np.int64), pred.to_numpy(dtype=np.int64)):
        if 0 <= t < classes and 0 <= p < classes:
            matrix[t, p] += 1
    return matrix


def save_score_matrix(
    out_dir: Path,
    method: str,
    window: float,
    subject: int,
    block: int,
    scores: np.ndarray,
) -> None:
    method_dir = out_dir / "score_matrices" / method.lower()
    method_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        method_dir / f"w{window:g}_s{subject:02d}_b{block}.npz",
        scores=np.asarray(scores, dtype=np.float32),
    )


def save_trca_artifact(
    out_dir: Path,
    method: str,
    window: float,
    subject: int,
    block: int,
    model: object,
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
    if text == "default":
        return list(WINDOWS)
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def method_names(text: str) -> list[str]:
    return [x.strip().upper() for x in text.split(",") if x.strip()]


def make_model(name: str, window: float, args: argparse.Namespace, spec: BenchmarkSpec):
    if name == "CCA":
        return CCA(window=window, harmonics=args.harmonics, spec=spec)
    if name == "FBCCA":
        return FBCCA(window=window, harmonics=args.harmonics, n_fbs=args.n_fbs, spec=spec)
    if name == "TRCA":
        return TRCA(n_fbs=args.n_fbs, ensemble=False)
    if name == "ETRCA":
        return TRCA(n_fbs=args.n_fbs, ensemble=True)
    raise ValueError(f"Unknown method: {name}")


def load_epochs(
    name: str,
    subject: int,
    window: float,
    args: argparse.Namespace,
    store: CanonicalEpochStore,
) -> np.ndarray:
    preset = benchmark_9ch_default(args.data_root)
    if name == "CCA":
        raw = store.load_or_create(EpochRequest(preset=preset, subject=subject, window=window, kind="raw"))
        return raw[:, :, None, :, :]
    return store.load_or_create(EpochRequest(preset=preset, subject=subject, window=window, kind="filterbank", n_fbs=args.n_fbs))


def write_outputs(
    result_dir: Path,
    run_dir: Path,
    trial_rows: list[dict[str, object]],
    pred_rows: list[dict[str, object]],
    runtime_rows: list[dict[str, object]],
    manifest: dict[str, object],
    spec: BenchmarkSpec,
    complete: bool,
) -> None:
    trials = pd.DataFrame(trial_rows)
    preds = pd.DataFrame(pred_rows)
    if trials.empty:
        return
    summary, subject_df, block_df = summarize(trials, spec)
    trials.to_csv(result_dir / "trials.csv", index=False)
    preds.to_csv(result_dir / "predictions.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(result_dir / "runtime.csv", index=False)
    summary.to_csv(result_dir / "summary.csv", index=False)
    subject_df.to_csv(result_dir / "subject.csv", index=False)
    block_df.to_csv(result_dir / "block.csv", index=False)
    labels = np.arange(spec.classes, dtype=np.int64)
    for method in sorted(trials["method"].unique()):
        rows = preds[preds["method"] == method]
        np.save(run_dir / f"confusion_{method.lower()}.npy", confusion_counts(rows["true"], rows["pred"], spec.classes))
    manifest["status"] = "complete" if complete else "partial"
    manifest["rows_written"] = int(len(trials))
    manifest["windows_written"] = sorted(float(x) for x in trials["window"].unique())
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_outputs(summary, subject_df, block_df, result_dir)
    write_report(summary, result_dir)


def summarize(trials: pd.DataFrame, spec: BenchmarkSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject = (
        trials.groupby(["method", "window", "subject"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), itr=("itr", "mean"), block_sd=("accuracy", "std"), seconds=("seconds", "sum"))
        .fillna({"block_sd": 0.0})
    )
    block = (
        trials.groupby(["method", "window", "block"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), itr=("itr", "mean"), seconds=("seconds", "sum"))
    )
    rows = []
    for (method, window), group in subject.groupby(["method", "window"]):
        rows.append(
            {
                "method": method,
                "window": float(window),
                "accuracy": float(group["accuracy"].mean()),
                "accuracy_sem": sem(group["accuracy"].to_numpy()),
                "itr": float(group["itr"].mean()),
                "itr_sem": sem(group["itr"].to_numpy()),
                "block_sd": float(group["block_sd"].mean()),
                "subjects": int(group["subject"].nunique()),
                "samples": int(spec.classes * spec.blocks * group["subject"].nunique()),
                "seconds": float(group["seconds"].sum()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["method", "window"])
    return summary, subject, block


def plot_outputs(summary: pd.DataFrame, subject: pd.DataFrame, block: pd.DataFrame, out: Path) -> None:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    methods = list(summary["method"].drop_duplicates())

    plt.figure(figsize=(8.5, 5.2))
    for method in methods:
        rows = summary[summary["method"] == method]
        plt.errorbar(rows["window"], rows["accuracy"], yerr=rows["accuracy_sem"], marker="o", capsize=3, label=method)
    plt.xlabel("Window (s)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "accuracy_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    for method in methods:
        rows = summary[summary["method"] == method]
        plt.errorbar(rows["window"], rows["itr"], yerr=rows["itr_sem"], marker="o", capsize=3, label=method)
    plt.xlabel("Window (s)")
    plt.ylabel("ITR (bits/min)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "itr_curve.png", dpi=180)
    plt.close()

    piv = summary.pivot(index="method", columns="window", values="accuracy").reindex(methods)
    plt.figure(figsize=(9.5, max(2.8, 0.45 * len(piv) + 1.5)))
    plt.imshow(piv.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(label="Accuracy")
    plt.xticks(range(len(piv.columns)), [f"{x:.1f}" for x in piv.columns])
    plt.yticks(range(len(piv.index)), piv.index)
    plt.xlabel("Window (s)")
    plt.tight_layout()
    plt.savefig(figdir / "accuracy_heatmap.png", dpi=180)
    plt.close()

    best_window = summary.sort_values("accuracy").groupby("method").tail(1)[["method", "window"]]
    box_rows = []
    for _, row in best_window.iterrows():
        box_rows.append(subject[(subject["method"] == row["method"]) & (subject["window"] == row["window"])])
    if box_rows:
        box_df = pd.concat(box_rows, ignore_index=True)
        labels = list(best_window["method"])
        data = [box_df[box_df["method"] == method]["accuracy"].to_numpy() for method in labels]
        plt.figure(figsize=(8.5, 4.8))
        plt.boxplot(data, tick_labels=labels, showmeans=True)
        plt.ylabel("Subject Accuracy at Best Window")
        plt.ylim(0, 1.02)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figdir / "subject_box_best.png", dpi=180)
        plt.close()

    for method in methods:
        rows = subject[subject["method"] == method]
        piv = rows.pivot(index="subject", columns="window", values="accuracy").sort_index()
        plt.figure(figsize=(10.5, 7.2))
        plt.imshow(piv.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(label="Accuracy")
        plt.xticks(range(len(piv.columns)), [f"{x:.1f}" for x in piv.columns], rotation=45)
        plt.yticks(range(len(piv.index)), [str(x) for x in piv.index])
        plt.xlabel("Window (s)")
        plt.ylabel("Subject")
        plt.title(f"{method} Subject x Window Accuracy")
        plt.tight_layout()
        plt.savefig(figdir / f"subject_window_heatmap_{method.lower()}.png", dpi=180)
        plt.close()

    plt.figure(figsize=(9.2, 5.2))
    for method in methods:
        rows = block[block["method"] == method]
        block_summary = rows.groupby("window", as_index=False).agg(
            accuracy=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
        )
        plt.errorbar(
            block_summary["window"],
            block_summary["accuracy"],
            yerr=block_summary["accuracy_sd"],
            marker="o",
            capsize=3,
            label=method,
        )
    plt.xlabel("Window (s)")
    plt.ylabel("Block-CV Accuracy")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "block_cv_accuracy_curve.png", dpi=180)
    plt.close()


def write_report(summary: pd.DataFrame, out: Path) -> None:
    ranking = summary.groupby("method", as_index=False)["accuracy"].mean().sort_values("accuracy", ascending=False)
    best = summary.sort_values("accuracy").groupby("window").tail(1).sort_values("window")
    lines = [
        "# Traditional Benchmark 9ch Evaluation",
        "",
        "Protocol: Tsinghua Benchmark 9-channel, subject-specific leave-one-block-out.",
        "Preprocessing: 0.5 s cue skipped, 0.14 s visual latency handled, 50 Hz notch, toolbox-style filterbank for FBCCA/TRCA.",
        "",
        "## Mean Ranking",
        "",
        "| Rank | Method | Mean accuracy |",
        "| ---: | --- | ---: |",
    ]
    for rank, row in enumerate(ranking.itertuples(index=False), start=1):
        lines.append(f"| {rank} | {row.method} | {row.accuracy:.4f} |")
    lines.extend(["", "## Best By Window", "", "| Window | Best method | Accuracy |", "| ---: | --- | ---: |"])
    for row in best.itertuples(index=False):
        lines.append(f"| {row.window:.1f}s | {row.method} | {row.accuracy:.4f} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `trials.csv`: method/window/subject/block rows.",
            "- `summary.csv`: method/window aggregates with SEM.",
            "- `subject.csv`: subject-level aggregates.",
            "- `block.csv`: block-level aggregates.",
            "- `predictions.csv`: trial-level predictions and true-label scores.",
            "- `runtime.csv`: standardized stage timing ledger.",
            "- `figures/accuracy_curve.png`",
            "- `figures/itr_curve.png`",
            "- `figures/accuracy_heatmap.png`",
            "- `figures/subject_box_best.png`",
            "- `figures/subject_window_heatmap_<method>.png`",
            "- `figures/block_cv_accuracy_curve.png`",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_rows(result_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    trials_path = result_dir / "trials.csv"
    preds_path = result_dir / "predictions.csv"
    runtime_path = result_dir / "runtime.csv"
    trial_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    if trials_path.exists():
        trial_rows = pd.read_csv(trials_path).to_dict("records")
    if preds_path.exists():
        pred_rows = pd.read_csv(preds_path).to_dict("records")
    if runtime_path.exists():
        runtime_rows = pd.read_csv(runtime_path).to_dict("records")
    return trial_rows, pred_rows, runtime_rows


def completed_windows(
    trial_rows: list[dict[str, object]],
    methods: list[str],
    subjects: list[int],
    blocks: list[int],
) -> dict[float, set[str]]:
    if not trial_rows:
        return {}
    trials = pd.DataFrame(trial_rows)
    expected = len(subjects) * len(blocks)
    done: dict[float, set[str]] = {}
    for (window, method), group in trials.groupby(["window", "method"]):
        if method in methods and len(group) >= expected:
            done.setdefault(float(window), set()).add(str(method))
    return done


def run_subject_window_task(task: dict[str, object]) -> dict[str, object]:
    spec = BenchmarkSpec()
    data_root = Path(str(task["data_root"]))
    epoch_cache = Path(str(task["epoch_cache"]))
    result_dir = Path(str(task["result_dir"]))
    preset = benchmark_9ch_default(data_root)
    store = CanonicalEpochStore(epoch_cache)
    subject = int(task["subject"])
    window = float(task["window"])
    methods = [str(method) for method in task["methods"]]
    blocks = [int(block) for block in task["blocks"]]
    harmonics = int(task["harmonics"])
    n_fbs = int(task["n_fbs"])
    force_epochs = bool(task["force_epochs"])
    save_scores = bool(task["save_score_matrices"])
    save_filters = bool(task["save_trca_filters"])
    args = argparse.Namespace(harmonics=harmonics, n_fbs=n_fbs)

    load_started = time.perf_counter()
    data_by_method: dict[str, np.ndarray] = {}
    if "CCA" in methods:
        req = EpochRequest(preset=preset, subject=subject, window=window, kind="raw")
        raw = store.load_or_create(req, force=force_epochs)
        data_by_method["CCA"] = raw[:, :, None, :, :]
    if any(method in methods for method in ("FBCCA", "TRCA", "ETRCA")):
        req = EpochRequest(preset=preset, subject=subject, window=window, kind="filterbank", n_fbs=n_fbs)
        filtered_epochs = store.load_or_create(req, force=force_epochs)
        for method in ("FBCCA", "TRCA", "ETRCA"):
            if method in methods:
                data_by_method[method] = filtered_epochs

    labels = np.arange(spec.classes, dtype=np.int64)
    trial_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    logs: list[str] = []
    runtime_rows: list[dict[str, object]] = []
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
            if save_filters and method in ("TRCA", "ETRCA"):
                save_trca_artifact(result_dir, method, window, subject, block, model)
            artifact_done = time.perf_counter()
            seconds = predict_done - t0
            acc = np.mean(pred == labels)
            itr = itr_bits_per_minute(float(acc), spec.classes, window + spec.cue_seconds)
            base_runtime = {
                "method": method,
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
                    {**base_runtime, "stage": "fit_predict", "seconds": seconds},
                ]
            )
            trial_rows.append(
                {
                    "method": method,
                    "window": window,
                    "subject": subject,
                    "block": block,
                    "accuracy": float(acc),
                    "itr": float(itr),
                    "samples": spec.classes,
                    "seconds": seconds,
                }
            )
            for true_label, pred_label in zip(labels, pred):
                pred_rows.append(
                    {
                        "method": method,
                        "window": window,
                        "subject": subject,
                        "block": block,
                        "true": int(true_label),
                        "pred": int(pred_label),
                        "score_true": float(scores[int(true_label), int(true_label)]),
                        "score_pred": float(scores[int(true_label), int(pred_label)]),
                    }
                )
            logs.append(f"{method} w={window:.1f} s={subject:02d} b={block} acc={acc:.3f} sec={seconds:.2f}")

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
        "epoch_fingerprint": epoch_fingerprint(
            EpochRequest(preset=preset, subject=subject, window=window, kind="filterbank", n_fbs=n_fbs)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--task-name", default="traditional_9ch")
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
    args = parser.parse_args()

    spec = BenchmarkSpec()
    preset = benchmark_9ch_default(args.data_root)
    store = CanonicalEpochStore(args.epoch_cache)
    subjects = parse_range(args.subjects)
    blocks = parse_range(args.blocks)
    windows = parse_windows(args.windows)
    methods = method_names(args.methods)
    result_dir = PROJECT_ROOT / "results" / args.task_name
    run_dir = RUN_ROOT / args.task_name
    result_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = {
        "task_name": args.task_name,
        "dataset": "Tsinghua Benchmark SSVEP",
        "channels": "Benchmark 9ch: Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, O2",
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
        trial_rows = []
        pred_rows = []
        runtime_rows = []
        done_windows = {}
    started = time.perf_counter()
    for window in windows:
        pending_methods = [method for method in methods if method not in done_windows.get(float(window), set())]
        for method in methods:
            if method not in pending_methods:
                print(f"resume skip {method} window={window:.1f}", flush=True)
        if not pending_methods:
            manifest["seconds"] = time.perf_counter() - started
            write_outputs(result_dir, run_dir, trial_rows, pred_rows, runtime_rows, manifest, spec, complete=False)
            print(f"checkpoint window={window:.1f} -> {result_dir}", flush=True)
            continue

        tasks = [
            {
                "data_root": str(args.data_root),
                "epoch_cache": str(args.epoch_cache),
                "result_dir": str(result_dir),
                "subject": subject,
                "window": window,
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
