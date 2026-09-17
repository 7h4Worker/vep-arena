from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vep_arena.channel.confusion import confusion_counts
from vep_arena.config import BenchmarkSpec, WINDOWS
from vep_arena.metrics import sem


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
    if ":" in text:
        start, step, stop = [float(x) for x in text.split(":", 2)]
        values: list[float] = []
        current = start
        while current <= stop + 1e-9:
            values.append(round(current, 10))
            current += step
        return values
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def summarize(
    trials: pd.DataFrame, spec: BenchmarkSpec
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject = (
        trials.groupby(["method", "window", "subject"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            itr=("itr", "mean"),
            block_sd=("accuracy", "std"),
            seconds=("seconds", "sum"),
        )
        .fillna({"block_sd": 0.0})
    )
    block = trials.groupby(["method", "window", "block"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        itr=("itr", "mean"),
        seconds=("seconds", "sum"),
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


def plot_outputs(
    summary: pd.DataFrame,
    subject: pd.DataFrame,
    block: pd.DataFrame,
    out: Path,
) -> None:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    methods = list(summary["method"].drop_duplicates())

    plt.figure(figsize=(8.5, 5.2))
    for method in methods:
        rows = summary[summary["method"] == method]
        plt.errorbar(
            rows["window"], rows["accuracy"],
            yerr=rows["accuracy_sem"], marker="o", capsize=3, label=method,
        )
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
        plt.errorbar(
            rows["window"], rows["itr"],
            yerr=rows["itr_sem"], marker="o", capsize=3, label=method,
        )
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
        box_rows.append(
            subject[(subject["method"] == row["method"]) & (subject["window"] == row["window"])]
        )
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
            block_summary["window"], block_summary["accuracy"],
            yerr=block_summary["accuracy_sd"], marker="o", capsize=3, label=method,
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
    ranking = (
        summary.groupby("method", as_index=False)["accuracy"]
        .mean()
        .sort_values("accuracy", ascending=False)
    )
    best = summary.sort_values("accuracy").groupby("window").tail(1).sort_values("window")
    lines = [
        "# Traditional Benchmark Evaluation",
        "",
        "Protocol: Tsinghua Benchmark selected-channel preset, subject-specific leave-one-block-out.",
        "Preprocessing: 0.5 s cue skipped, 0.14 s visual latency handled, 50 Hz notch, toolbox-style filterbank for FBCCA/TRCA.",
        "",
        "## Mean Ranking",
        "",
        "| Rank | Method | Mean accuracy |",
        "| ---: | --- | ---: |",
    ]
    for rank, row in enumerate(ranking.itertuples(index=False), start=1):
        lines.append(f"| {rank} | {row.method} | {row.accuracy:.4f} |")
    lines.extend(
        ["", "## Best By Window", "", "| Window | Best method | Accuracy |", "| ---: | --- | ---: |"]
    )
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


def write_outputs(
    result_dir: Path,
    run_dir: Path | None,
    trial_rows: list[dict],
    pred_rows: list[dict],
    runtime_rows: list[dict],
    manifest: dict,
    spec: BenchmarkSpec,
    *,
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
    if run_dir is not None:
        for method in sorted(trials["method"].unique()):
            rows = preds[preds["method"] == method]
            np.save(
                run_dir / f"confusion_{method.lower()}.npy",
                confusion_counts(rows["true"], rows["pred"], spec.classes),
            )
    manifest["status"] = "complete" if complete else "partial"
    manifest["rows_written"] = int(len(trials))
    manifest["windows_written"] = sorted(float(x) for x in trials["window"].unique())
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    plot_outputs(summary, subject_df, block_df, result_dir)
    write_report(summary, result_dir)


def load_existing_rows(
    result_dir: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    trial_rows: list[dict] = []
    pred_rows: list[dict] = []
    runtime_rows: list[dict] = []
    if (result_dir / "trials.csv").exists():
        trial_rows = pd.read_csv(result_dir / "trials.csv").to_dict("records")
    if (result_dir / "predictions.csv").exists():
        pred_rows = pd.read_csv(result_dir / "predictions.csv").to_dict("records")
    if (result_dir / "runtime.csv").exists():
        runtime_rows = pd.read_csv(result_dir / "runtime.csv").to_dict("records")
    return trial_rows, pred_rows, runtime_rows


def completed_windows(
    trial_rows: list[dict],
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


def completed_units(
    rows: list[dict],
    subjects: list[int],
    blocks: list[int],
) -> set[tuple[float, int]]:
    if not rows:
        return set()
    df = pd.DataFrame(rows)
    expected = len(blocks)
    done: set[tuple[float, int]] = set()
    for (window, subject), group in df.groupby(["window", "subject"]):
        if int(subject) in subjects and len(group) >= expected:
            done.add((float(window), int(subject)))
    return done
