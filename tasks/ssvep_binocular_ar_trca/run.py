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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.binocular import (  # noqa: E402
    AR_CLASSES,
    AR_GAZE_SHIFT_SECONDS,
    AR_OCCIPITAL,
    AR_SESSIONS,
    BINOCULAR_AR_ROOT,
    ar_apply_filterbank,
    ar_task_n_fbs,
    available_ar_subjects,
    load_ar_task_epochs,
)
from vep_arena.metrics import itr_bits_per_minute, sem  # noqa: E402
from vep_arena.methods.traditional import CCA, FBCCA, TRCA, multi_frequency_reference_signals  # noqa: E402


def parse_range(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    return values


def parse_csv(text: str, upper: bool = True) -> list[str]:
    items = [item.strip() for item in text.split(",") if item.strip()]
    return [item.upper() for item in items] if upper else items


def parse_windows(text: str) -> list[float]:
    if ":" in text:
        lo, step, hi = (float(x) for x in text.split(":", 2))
        count = int(round((hi - lo) / step)) + 1
        return [round(lo + idx * step, 10) for idx in range(count)]
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def make_model(method: str, n_fbs: int, window: float, references: list[np.ndarray], harmonics: int) -> CCA | FBCCA | TRCA:
    if method == "CCA":
        return CCA(window=window, harmonics=harmonics, references=references)
    if method == "FBCCA":
        return FBCCA(window=window, harmonics=harmonics, n_fbs=n_fbs, references=references)
    if method == "TRCA":
        return TRCA(n_fbs=n_fbs, ensemble=False)
    if method == "ETRCA":
        return TRCA(n_fbs=n_fbs, ensemble=True)
    raise ValueError(f"Unsupported method for this task: {method}")


def run_unit(task: dict[str, object]) -> dict[str, object]:
    root = Path(str(task["root"]))
    subject = int(task["subject"])
    paradigm = str(task["task"])
    window = float(task["window"])
    methods = [str(method) for method in task["methods"]]
    sessions = [str(session) for session in task["sessions"]]
    n_fbs = int(task["n_fbs"])
    harmonics = int(task.get("harmonics", 5))
    started = time.perf_counter()

    try:
        data = load_ar_task_epochs(
            root=root,
            subject=subject,
            task=paradigm,
            window_seconds=window,
            channels=AR_OCCIPITAL,
            sessions=sessions,
            n_fbs=n_fbs,
            filterbank=False,
        )
    except FileNotFoundError as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
            "trial_rows": [],
            "pred_rows": [],
            "runtime_rows": [],
            "manifest_rows": [
                {
                    "subject": subject,
                    "task": paradigm,
                    "window": window,
                    "methods": json.dumps(methods),
                    "status": "skipped",
                    "reason": str(exc),
                }
            ],
        }

    raw_epochs = data.x
    fb_epochs = ar_apply_filterbank(raw_epochs[:, :, 0], task=paradigm, n_fbs=n_fbs, fs=data.sampling_rate)
    references = multi_frequency_reference_signals(
        target_frequencies=data.target_freqs,
        samples=raw_epochs.shape[-1],
        fs=data.sampling_rate,
        harmonics=harmonics,
    )
    labels = np.arange(AR_CLASSES, dtype=np.int64)
    blocks = list(range(raw_epochs.shape[1]))
    trial_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    logs: list[str] = []

    for method in methods:
        for block_idx in blocks:
            train_blocks = [idx for idx in blocks if idx != block_idx]
            method_epochs = raw_epochs if method == "CCA" else fb_epochs
            train_x = method_epochs[:, train_blocks].reshape(
                -1,
                method_epochs.shape[2],
                method_epochs.shape[3],
                method_epochs.shape[4],
            )
            train_y = np.repeat(labels, len(train_blocks))
            test_x = method_epochs[:, block_idx]
            model = make_model(method, n_fbs=n_fbs, window=window, references=references, harmonics=harmonics)
            t0 = time.perf_counter()
            model.fit(train_x, train_y)
            fit_done = time.perf_counter()
            pred, scores = model.predict(test_x)
            predict_done = time.perf_counter()
            accuracy = float(np.mean(pred == labels))
            itr = itr_bits_per_minute(accuracy, AR_CLASSES, window + AR_GAZE_SHIFT_SECONDS)
            block_key = data.block_keys[block_idx]
            trial_rows.append(
                {
                    "method": method,
                    "task": paradigm,
                    "window": window,
                    "subject": subject,
                    "block": block_idx + 1,
                    "block_key": block_key,
                    "accuracy": accuracy,
                    "itr": float(itr),
                    "samples": AR_CLASSES,
                    "seconds": predict_done - t0,
                }
            )
            for true_label, pred_label in zip(labels, pred):
                pred_rows.append(
                    {
                        "method": method,
                        "task": paradigm,
                        "window": window,
                        "subject": subject,
                        "block": block_idx + 1,
                        "block_key": block_key,
                        "true": int(true_label),
                        "pred": int(pred_label),
                        "score_true": float(scores[int(true_label), int(true_label)]),
                        "score_pred": float(scores[int(true_label), int(pred_label)]),
                    }
                )
            runtime_base = {
                "method": method,
                "task": paradigm,
                "window": window,
                "subject": subject,
                "block": block_idx + 1,
                "worker_pid": os.getpid(),
            }
            runtime_rows.extend(
                [
                    {**runtime_base, "stage": "fit", "seconds": fit_done - t0},
                    {**runtime_base, "stage": "predict", "seconds": predict_done - fit_done},
                    {**runtime_base, "stage": "fit_predict", "seconds": predict_done - t0},
                ]
            )
            logs.append(f"{method} {paradigm} sub-{subject:03d} w={window:g} {block_key} acc={accuracy:.3f}")

    runtime_rows.append(
        {
            "method": "__all__",
            "task": paradigm,
            "window": window,
            "subject": subject,
            "block": "__all__",
            "stage": "unit_total",
            "seconds": time.perf_counter() - started,
            "worker_pid": os.getpid(),
        }
    )
    return {
        "status": "complete",
        "trial_rows": trial_rows,
        "pred_rows": pred_rows,
        "runtime_rows": runtime_rows,
        "logs": logs,
        "manifest_rows": [
            {
                "subject": subject,
                "task": paradigm,
                "window": window,
                "methods": json.dumps(methods),
                "status": "complete",
                "blocks": len(blocks),
                "channels": len(data.channels),
                "n_fbs": n_fbs,
                "harmonics": harmonics,
                "samples": int(raw_epochs.shape[-1]),
                "target_freqs": json.dumps(data.target_freqs),
            }
        ],
    }


def load_existing(result_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    def read(name: str) -> list[dict[str, object]]:
        path = result_dir / name
        return pd.read_csv(path).to_dict("records") if path.exists() and path.stat().st_size > 0 else []

    return read("trials.csv"), read("predictions.csv"), read("runtime.csv"), read("unit_manifest.csv")


def completed_methods_by_unit(unit_rows: list[dict[str, object]]) -> dict[tuple[int, str, float], set[str]]:
    rows = [row for row in unit_rows if row.get("status") == "complete"]
    if not rows:
        return {}
    done: dict[tuple[int, str, float], set[str]] = {}
    df = pd.DataFrame(rows)
    for (subject, task, window), group in df.groupby(["subject", "task", "window"]):
        recorded_methods: set[str] = set()
        if "methods" in group.columns:
            for raw in group["methods"].dropna():
                try:
                    recorded_methods.update(str(item) for item in json.loads(str(raw)))
                except json.JSONDecodeError:
                    recorded_methods.update(item.strip() for item in str(raw).split(",") if item.strip())
        done[(int(subject), str(task), float(window))] = recorded_methods
    return done


def summarize(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trials.empty:
        return pd.DataFrame(), pd.DataFrame()
    subject = (
        trials.groupby(["method", "task", "window", "subject"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), itr=("itr", "mean"), blocks=("block", "nunique"), seconds=("seconds", "sum"))
        .sort_values(["task", "method", "window", "subject"])
    )
    rows = []
    for (method, task, window), group in subject.groupby(["method", "task", "window"]):
        rows.append(
            {
                "method": method,
                "task": task,
                "window": float(window),
                "accuracy": float(group["accuracy"].mean()),
                "accuracy_sem": sem(group["accuracy"].to_numpy()),
                "itr": float(group["itr"].mean()),
                "itr_sem": sem(group["itr"].to_numpy()),
                "subjects": int(group["subject"].nunique()),
                "blocks": int(group["blocks"].sum()),
                "seconds": float(group["seconds"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "method", "window"]), subject


def confusion_counts(preds: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((AR_CLASSES, AR_CLASSES), dtype=np.int64)
    for row in preds.itertuples(index=False):
        matrix[int(row.true), int(row.pred)] += 1
    return matrix


def plot_outputs(summary: pd.DataFrame, subject: pd.DataFrame, result_dir: Path) -> None:
    if summary.empty:
        return
    figdir = result_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, fname in (
        ("accuracy", "Accuracy", "accuracy_curve.png"),
        ("itr", "ITR (bits/min)", "itr_curve.png"),
    ):
        tasks = list(summary["task"].drop_duplicates())
        fig, axes = plt.subplots(len(tasks), 1, figsize=(8.8, max(3.2, 2.7 * len(tasks))), sharex=True)
        if len(tasks) == 1:
            axes = [axes]
        for ax, task_name in zip(axes, tasks):
            rows_task = summary[summary["task"] == task_name]
            for method in rows_task["method"].drop_duplicates():
                rows = rows_task[rows_task["method"] == method]
                ax.errorbar(
                    rows["window"],
                    rows[metric],
                    yerr=rows[f"{metric}_sem"],
                    marker="o",
                    capsize=3,
                    label=method,
                )
            ax.set_title(task_name)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            if metric == "accuracy":
                ax.set_ylim(0, 1.02)
        axes[-1].set_xlabel("Window (s)")
        fig.tight_layout()
        fig.savefig(figdir / fname, dpi=180)
        plt.close(fig)

    piv = summary.pivot_table(index=["task", "method"], columns="window", values="accuracy")
    fig, ax = plt.subplots(figsize=(9.5, max(3.0, 0.42 * len(piv) + 1.4)))
    im = ax.imshow(piv.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Accuracy")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{x:g}" for x in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{idx[0]} / {idx[1]}" for idx in piv.index])
    ax.set_xlabel("Window (s)")
    fig.tight_layout()
    fig.savefig(figdir / "accuracy_heatmap.png", dpi=180)
    plt.close(fig)

    if not subject.empty:
        best = summary.sort_values("accuracy").groupby(["task", "method"], as_index=False).tail(1)
        rows = []
        labels = []
        for row in best.itertuples(index=False):
            values = subject[
                (subject["task"] == row.task)
                & (subject["method"] == row.method)
                & (subject["window"] == row.window)
            ]["accuracy"].to_numpy()
            if values.size:
                rows.append(values)
                labels.append(f"{row.task}\n{row.method}")
        if rows:
            fig, ax = plt.subplots(figsize=(max(7.5, 1.2 * len(rows)), 4.8))
            ax.boxplot(rows, tick_labels=labels, showmeans=True)
            ax.set_ylim(0, 1.02)
            ax.set_ylabel("Subject Accuracy at Best Window")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figdir / "subject_box_best.png", dpi=180)
            plt.close(fig)


def write_report(summary: pd.DataFrame, result_dir: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Binocular AR Baseline Task Report",
        "",
        "Protocol: Ke et al. 2025 binocular AR epoch data, paper 10-channel preset, subject-specific leave-one-block-out.",
        "Crop: event onset + 0.14 s visual latency; ITR selection time uses window + 1 s gaze shift, matching the official MATLAB scripts.",
        "Preprocessing: Arena SciPy approximation of paper preprocessing, then official-code-style Chebyshev filterbank.",
        "",
        "## Run",
        "",
        f"- Subjects: `{manifest['subjects']}`",
        f"- Tasks: `{manifest['tasks']}`",
        f"- Windows: `{manifest['windows']}`",
        f"- Methods: `{manifest['methods']}`",
        "",
    ]
    if not summary.empty:
        lines.extend(["## Best Accuracy By Task", "", "| Task | Method | Window | Accuracy | ITR | Subjects |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        best = summary.sort_values("accuracy").groupby("task", as_index=False).tail(1).sort_values("task")
        for row in best.itertuples(index=False):
            lines.append(f"| {row.task} | {row.method} | {row.window:g}s | {row.accuracy:.4f} | {row.itr:.2f} | {row.subjects} |")
        lines.extend(["", "## Output Files", "", "- `trials.csv`", "- `predictions.csv`", "- `summary.csv`", "- `subject.csv`", "- `runtime.csv`", "- `unit_manifest.csv`", "- `figures/accuracy_curve.png`", "- `figures/itr_curve.png`", "- `figures/accuracy_heatmap.png`"])
    (result_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    result_dir: Path,
    trial_rows: list[dict[str, object]],
    pred_rows: list[dict[str, object]],
    runtime_rows: list[dict[str, object]],
    unit_rows: list[dict[str, object]],
    manifest: dict[str, object],
    complete: bool,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    trials = pd.DataFrame(trial_rows)
    preds = pd.DataFrame(pred_rows)
    runtime = pd.DataFrame(runtime_rows)
    units = pd.DataFrame(unit_rows)
    trials.to_csv(result_dir / "trials.csv", index=False)
    preds.to_csv(result_dir / "predictions.csv", index=False)
    runtime.to_csv(result_dir / "runtime.csv", index=False)
    units.to_csv(result_dir / "unit_manifest.csv", index=False)
    summary, subject = summarize(trials)
    summary.to_csv(result_dir / "summary.csv", index=False)
    subject.to_csv(result_dir / "subject.csv", index=False)
    if not preds.empty:
        conf_dir = result_dir / "confusions"
        conf_dir.mkdir(exist_ok=True)
        for (task_name, method, window), group in preds.groupby(["task", "method", "window"]):
            np.save(conf_dir / f"{task_name}_{method}_w{float(window):g}.npy", confusion_counts(group))
    manifest["status"] = "complete" if complete else "partial"
    manifest["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["rows_written"] = int(len(trials))
    (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_outputs(summary, subject, result_dir)
    write_report(summary, result_dir, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCA/FBCCA/TRCA/ETRCA on Ke 2025 binocular AR SSVEP epoch data.")
    parser.add_argument("--root", type=Path, default=BINOCULAR_AR_ROOT)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "latest")
    parser.add_argument("--subjects", default="1")
    parser.add_argument("--tasks", default="LF,DFDP,DFDP1")
    parser.add_argument("--windows", default="0.5,1.0")
    parser.add_argument("--methods", default="TRCA,ETRCA")
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--sessions", default=",".join(AR_SESSIONS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    subjects = parse_range(args.subjects)
    available = set(available_ar_subjects(args.root))
    missing = [subject for subject in subjects if subject not in available]
    if missing:
        raise ValueError(f"Missing complete subject zips: {missing}")
    tasks = parse_csv(args.tasks)
    windows = parse_windows(args.windows)
    methods = parse_csv(args.methods)
    sessions = parse_csv(args.sessions, upper=False)
    result_dir = args.out

    if args.resume:
        trial_rows, pred_rows, runtime_rows, unit_rows = load_existing(result_dir)
        done_methods = completed_methods_by_unit(unit_rows)
    else:
        trial_rows, pred_rows, runtime_rows, unit_rows, done_methods = [], [], [], [], {}

    manifest = {
        "dataset": "Ke2025 binocular AR SSVEP",
        "paper_doi": "10.1038/s41597-025-05696-0",
        "data_source": "Figshare article 26768287",
        "method_source": "public MATLAB classification code distributed with the dataset",
        "implementation_identity": (
            "Arena CCA/FBCCA/TRCA/ETRCA integration using the public protocol; "
            "SciPy preprocessing is an explicit numerical adaptation"
        ),
        "evidence_role": "paper-referenced Arena reproduction; not bitwise official-code parity",
        "root": str(args.root),
        "subjects": subjects,
        "tasks": tasks,
        "windows": windows,
        "methods": methods,
        "harmonics": args.harmonics,
        "sessions": sessions,
        "channels": list(AR_OCCIPITAL),
        "classes": AR_CLASSES,
        "protocol": "subject-specific leave-one-block-out over sessions x blocks",
        "crop": "event onset + 0.14 s visual latency",
        "itr_trial_seconds": "window + 1.0 s gaze shift",
        "workers": args.workers,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    work = []
    for subject in subjects:
        for task_name in tasks:
            for window in windows:
                key = (subject, task_name, float(window))
                missing_methods = [method for method in methods if method not in done_methods.get(key, set())]
                if not missing_methods:
                    if not args.quiet:
                        print(f"resume skip sub-{subject:03d} {task_name} w={window:g}", flush=True)
                    continue
                work.append(
                    {
                        "root": str(args.root),
                        "subject": subject,
                        "task": task_name,
                        "window": window,
                        "methods": missing_methods,
                        "sessions": sessions,
                        "n_fbs": ar_task_n_fbs(task_name),
                        "harmonics": args.harmonics,
                    }
                )

    result_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    if args.workers > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_unit, item) for item in work]
            for future in as_completed(futures):
                result = future.result()
                trial_rows.extend(result["trial_rows"])
                pred_rows.extend(result["pred_rows"])
                runtime_rows.extend(result["runtime_rows"])
                unit_rows.extend(result["manifest_rows"])
                if not args.quiet:
                    for line in result.get("logs", []):
                        print(line, flush=True)
                completed += 1
                if completed % max(1, args.checkpoint_every) == 0:
                    write_outputs(result_dir, trial_rows, pred_rows, runtime_rows, unit_rows, manifest, complete=False)
                    print(f"checkpoint {completed}/{len(work)} -> {result_dir}", flush=True)
    else:
        for item in work:
            result = run_unit(item)
            trial_rows.extend(result["trial_rows"])
            pred_rows.extend(result["pred_rows"])
            runtime_rows.extend(result["runtime_rows"])
            unit_rows.extend(result["manifest_rows"])
            if not args.quiet:
                for line in result.get("logs", []):
                    print(line, flush=True)
            completed += 1
            if completed % max(1, args.checkpoint_every) == 0:
                write_outputs(result_dir, trial_rows, pred_rows, runtime_rows, unit_rows, manifest, complete=False)
                print(f"checkpoint {completed}/{len(work)} -> {result_dir}", flush=True)

    manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_outputs(result_dir, trial_rows, pred_rows, runtime_rows, unit_rows, manifest, complete=True)
    print(result_dir)


if __name__ == "__main__":
    main()
