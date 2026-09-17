"""Shared block-evaluation utilities; scientific method code stays in methods/."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import uuid
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vep_arena.channel.confusion import confusion_counts
from vep_arena.config import BenchmarkSpec, WINDOWS
from vep_arena.metrics import sem
from vep_arena.run_contract import (
    UNIT_KEY, atomic_json, fingerprint, require_same_run, resume_config,
    file_sha256, unique_frame, validate_predictions, validate_trials,
    validate_coverage, verify_files,
)


def parse_range(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            raise ValueError(f"Invalid non-negative integer range: {part!r}")
        bounds = [int(value) for value in part.split("-")]
        start, stop = bounds[0], bounds[-1]
        if stop < start or stop - start > 100000:
            raise ValueError("Range must be ascending and bounded")
        values.extend(range(start, stop + 1))
    if len(values) != len(set(values)):
        raise ValueError("Duplicate selections are not allowed")
    return values


def parse_windows(text: str) -> list[float]:
    text = text.strip()
    if text == "default":
        return list(WINDOWS)
    if ":" in text:
        start, step, stop = map(float, text.split(":"))
        if not all(math.isfinite(x) for x in (start, step, stop)):
            raise ValueError("Windows must be finite")
        if start <= 0 or step <= 0 or stop < start:
            raise ValueError("Expected positive start:step:stop with stop >= start")
        span = (stop - start) / step
        if not math.isfinite(span) or span > 100000:
            raise ValueError("Too many windows")
        count = math.floor(span + 1e-9) + 1
        values = [round(start + i * step, 10) for i in range(count)]
    else:
        values = [float(value.strip()) for value in text.split(",")]
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Windows must be finite and positive")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate windows are not allowed")
    return values


def summarize(trials: pd.DataFrame, spec: BenchmarkSpec | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep historical equal-block/equal-subject means; count actual samples.

    This helper accepts one method/window/subject/block row. Repeated training
    combinations must retain a split dimension and use task-specific aggregation.
    The spec parameter remains for caller compatibility, not sample estimation.
    """
    validate_trials(trials)
    subject = trials.groupby(["method", "window", "subject"], as_index=False).agg(
        accuracy=("accuracy", "mean"), itr=("itr", "mean"),
        block_sd=("accuracy", "std"), seconds=("seconds", "sum"), samples=("samples", "sum"),
    ).fillna({"block_sd": 0.0})
    block = trials.groupby(["method", "window", "block"], as_index=False).agg(
        accuracy=("accuracy", "mean"), itr=("itr", "mean"),
        seconds=("seconds", "sum"), samples=("samples", "sum"),
    )
    rows = []
    for (method, window), group in subject.groupby(["method", "window"]):
        rows.append({"method": method, "window": float(window),
            "accuracy": float(group["accuracy"].mean()), "accuracy_sem": sem(group["accuracy"].to_numpy()),
            "itr": float(group["itr"].mean()), "itr_sem": sem(group["itr"].to_numpy()),
            "block_sd": float(group["block_sd"].mean()), "subjects": int(group["subject"].nunique()),
            "samples": int(group["samples"].sum()), "seconds": float(group["seconds"].sum())})
    return pd.DataFrame(rows).sort_values(["method", "window"]), subject, block


def plot_outputs(summary: pd.DataFrame, subject: pd.DataFrame, block: pd.DataFrame, out: Path) -> None:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    methods = list(summary["method"].drop_duplicates())
    for metric, label in [("accuracy", "Accuracy"), ("itr", "ITR (bits/min)")]:
        plt.figure(figsize=(8.5, 5.2))
        for method in methods:
            rows = summary[summary["method"] == method]
            plt.errorbar(rows["window"], rows[metric], yerr=rows[f"{metric}_sem"], marker="o", capsize=3, label=method)
        plt.xlabel("Window (s)")
        plt.ylabel(label)
        if metric == "accuracy":
            plt.ylim(0, 1.02)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / f"{metric}_curve.png", dpi=180)
        plt.close()
    piv = summary.pivot(index="method", columns="window", values="accuracy").reindex(methods)
    plt.figure(figsize=(9.5, max(2.8, 0.45 * len(piv) + 1.5)))
    plt.imshow(piv.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(label="Accuracy")
    plt.xticks(range(len(piv.columns)), [f"{x:g}" for x in piv.columns])
    plt.yticks(range(len(piv.index)), piv.index)
    plt.xlabel("Window (s)")
    plt.tight_layout()
    plt.savefig(figdir / "accuracy_heatmap.png", dpi=180)
    plt.close()
    best = summary.sort_values("accuracy").groupby("method").tail(1)
    data = [subject[(subject.method == r.method) & (subject.window == r.window)].accuracy.to_numpy()
            for r in best.itertuples()]
    plt.figure(figsize=(8.5, 4.8))
    plt.boxplot(data, tick_labels=best["method"].tolist(), showmeans=True)
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
        plt.xticks(range(len(piv.columns)), [f"{x:g}" for x in piv.columns], rotation=45)
        plt.yticks(range(len(piv.index)), [str(x) for x in piv.index])
        plt.xlabel("Window (s)")
        plt.ylabel("Subject")
        plt.title(f"{method} Subject x Window Accuracy")
        plt.tight_layout()
        plt.savefig(figdir / f"subject_window_heatmap_{method.lower()}.png", dpi=180)
        plt.close()
    plt.figure(figsize=(9.2, 5.2))
    for method in methods:
        rows = block[block["method"] == method].groupby("window", as_index=False).agg(
            accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        ).fillna({"accuracy_sd": 0.0})
        plt.errorbar(rows.window, rows.accuracy, yerr=rows.accuracy_sd, marker="o", capsize=3, label=method)
    plt.xlabel("Window (s)")
    plt.ylabel("Block-CV Accuracy")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "block_cv_accuracy_curve.png", dpi=180)
    plt.close()


def write_report(summary: pd.DataFrame, out: Path, manifest: dict | None = None) -> None:
    """Render declared facts, never substitute Benchmark settings for missing metadata."""
    manifest = manifest or {}
    lines = ["# Evaluation Report", ""]
    for name, value in [("Dataset", manifest.get("dataset", "unspecified")),
        ("Protocol", manifest.get("protocol", "unspecified")),
        ("Preprocessing", manifest.get("preprocess", manifest.get("preprocessing", "unspecified"))),
        ("Method inputs", manifest.get("method_inputs", "unspecified")),
        ("Status", manifest.get("status", "unspecified"))]:
        lines.extend([f"{name}: {json.dumps(value, ensure_ascii=False)}", ""])
    lines.extend(["Aggregation: equal block means within subjects, then equal subject means.",
        "Samples are actual evaluated records, not estimates from full dataset size.",
        "These are descriptive results; compatibility/acceptance is not implied.", "",
        "| Method | Window (s) | Accuracy | ITR | Samples |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.method} | {row.window:g} | {row.accuracy:.4f} | {row.itr:.4f} | {row.samples} |")
    lines.extend(["", "See manifest.json for the artifact ledger and declared limitations."])
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def completed_windows(trial_rows: list[dict], methods: list[str], subjects: list[int], blocks: list[int]) -> dict[float, set[str]]:
    if any(not values or len(values) != len(set(values)) for values in (methods, subjects, blocks)):
        raise ValueError("Completion scope must be non-empty and unique")
    if not trial_rows:
        return {}
    trials = pd.DataFrame(trial_rows)
    unique_frame(trials, UNIT_KEY, "trials")
    expected, done = set(product(subjects, blocks)), {}
    for (window, method), group in trials.groupby(["window", "method"]):
        if method in methods and set(zip(group.subject, group.block)) == expected:
            done.setdefault(float(window), set()).add(str(method))
    return done


def completed_units(rows: list[dict], subjects: list[int], blocks: list[int]) -> set[tuple[float, int]]:
    if any(not values or len(values) != len(set(values)) for values in (subjects, blocks)):
        raise ValueError("Completion scope must be non-empty and unique")
    if not rows:
        return set()
    frame = pd.DataFrame(rows)
    unique_frame(frame, UNIT_KEY, "trials")
    if frame.method.nunique() != 1:
        raise ValueError("completed_units requires one method; use completed_windows for multiple methods")
    return {(float(window), int(subject)) for (window, subject), group in frame.groupby(["window", "subject"])
            if subject in subjects and set(group.block) == set(blocks)}


def load_existing_rows(result_dir: Path, expected_manifest: dict | None = None):
    paths = [result_dir / name for name in ["trials.csv", "predictions.csv", "runtime.csv"]]
    manifest_path = result_dir / "manifest.json"
    if not manifest_path.exists() and not any(p.exists() for p in paths):
        if result_dir.exists() and any(result_dir.iterdir()):
            raise ValueError("Non-empty output without a committed checkpoint; use a new directory")
        return [], [], []
    if expected_manifest is None:
        raise ValueError("Safe resume requires expected_manifest; old artifacts remain readable without resume")
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_same_run(saved, expected_manifest)
    verify_files(result_dir, saved)
    frames = [pd.read_csv(path) for path in paths]
    validate_predictions(frames[0], frames[1], saved.get("classes"))
    validate_coverage(frames[0], expected_manifest, complete=False)
    # Legacy runners checkpoint entire windows. A partly written window is not
    # automatically appended or deduplicated; it needs an explicit recovery run.
    done = completed_windows(frames[0].to_dict("records"), expected_manifest["methods"],
                             expected_manifest["subjects"], expected_manifest["blocks"])
    for window, method in frames[0][["window", "method"]].drop_duplicates().itertuples(index=False):
        if method not in done.get(float(window), set()):
            raise ValueError("Incomplete window checkpoint: restart it in a new output directory")
    expected_manifest["run_id"] = saved.get("run_id")
    expected_manifest["execution_history"] = [*saved.get("execution_history", []),
        {"command": saved.get("code", {}).get("command"), "started_at_utc": saved.get("started_at_utc"),
         "finished_at_utc": saved.get("finished_at_utc"), "seconds": saved.get("seconds")}]
    return tuple(frame.to_dict("records") for frame in frames)


def write_outputs(result_dir: Path, run_dir: Path | None, trial_rows: list[dict], pred_rows: list[dict],
                  runtime_rows: list[dict], manifest: dict, spec: BenchmarkSpec, complete: bool) -> None:
    """Validate first; commit the manifest last to make torn checkpoints detectable.

    Filenames for tables/figures stay. Confusions are keyed by method AND window.
    An unqualified compatibility mirror is emitted only for single-window runs.
    There is no multi-file rollback or concurrent-writer support.
    """
    trials, preds = pd.DataFrame(trial_rows), pd.DataFrame(pred_rows)
    if trials.empty:
        if complete:
            raise ValueError("An empty run cannot be complete")
        return
    identity = validate_predictions(trials, preds, spec.classes)
    methods = manifest.get("methods", [manifest.get("method")])
    if any(not isinstance(m, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", m) for m in methods):
        raise ValueError("Method identifiers must be filename-safe")
    if len({m.lower() for m in methods}) != len(methods):
        raise ValueError("Method identifiers collide in output filenames")
    manifest["methods"] = methods
    validate_coverage(trials, manifest, complete=complete)
    manifest.setdefault("run_id", uuid.uuid4().hex)
    manifest.update({"schema_version": "1.0", "artifact_profile": "classification", "classes": spec.classes,
        "label_base": 0, "status": "complete" if complete else "partial", "rows_written": len(trials),
        "windows_written": sorted(float(w) for w in trials.window.unique())})
    limitations = list(manifest.get("limitations", []))
    if identity == "true":
        limitations.append("legacy_one_trial_per_target_block; raw trial provenance not verified")
    manifest["limitations"] = sorted(set(limitations))
    manifest["structural_validation"] = {"status": "pass", "real_data_acceptance": "not_run"}
    summary, subject, block = summarize(trials, spec)
    result_dir.mkdir(parents=True, exist_ok=True)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    runtime = pd.DataFrame(runtime_rows)
    if runtime.empty:
        runtime = pd.DataFrame(columns=[*UNIT_KEY, "stage", "seconds"])
    manifest["confusions"] = []
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=result_dir) as directory:
        stage = Path(directory)
        for name, frame in [("trials", trials), ("predictions", preds), ("runtime", runtime),
                            ("summary", summary), ("subject", subject), ("block", block)]:
            frame.to_csv(stage / f"{name}.csv", index=False)
        (stage / "confusions").mkdir()
        for (method, window), rows in preds.groupby(["method", "window"]):
            relative = f"confusions/confusion_{method.lower()}_w{float(window)!r}.npy"
            matrix = confusion_counts(rows.true, rows.pred, spec.classes)
            np.save(stage / relative, matrix)
            manifest["confusions"].append({"path": relative, "method": method, "window": float(window)})
            if run_dir is not None:
                # Compatibility mirrors are not authoritative; use the result ledger.
                np.save(run_dir / Path(relative).name, matrix)
                if len(manifest["windows"]) == 1:
                    np.save(run_dir / f"confusion_{method.lower()}.npy", matrix)
        plot_outputs(summary, subject, block, stage)
        write_report(summary, stage, manifest)
        manifest["artifacts"] = [{"path": p.relative_to(stage).as_posix(), "sha256": file_sha256(p)}
            for p in sorted(stage.rglob("*")) if p.is_file()]
        for directory_name in ("score_matrices", "model_artifacts"):
            for path in sorted((result_dir / directory_name).rglob("*")):
                if path.is_file():
                    manifest["artifacts"].append({"path": path.relative_to(result_dir).as_posix(),
                                                  "sha256": file_sha256(path)})
        manifest["config_fingerprint"] = fingerprint(resume_config(manifest))
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                destination = result_dir / path.relative_to(stage)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination)
        atomic_json(result_dir / "manifest.json", manifest)
