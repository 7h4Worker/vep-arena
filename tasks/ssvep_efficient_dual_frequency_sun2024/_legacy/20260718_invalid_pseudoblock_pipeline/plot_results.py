from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TASK_ROOT = Path(__file__).resolve().parent
RESULTS = TASK_ROOT / "results"
METHOD_ORDER = ("TRCA", "ETRCA", "BTRCA", "EBTRCA")


def sem(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def normalize_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "itr" not in out.columns and "itr_bits_per_min" in out.columns:
        out["itr"] = out["itr_bits_per_min"]
    if "itr" not in out.columns and "itr_bits_min" in out.columns:
        out["itr"] = out["itr_bits_min"]
    if "itr" not in out.columns:
        raise ValueError("Expected an ITR column named itr, itr_bits_per_min, or itr_bits_min.")
    return out


def summarize_subject_rows(subject: pd.DataFrame) -> pd.DataFrame:
    summary = (
        subject.groupby(["method", "window"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_sem=("accuracy", sem),
            itr=("itr", "mean"),
            itr_sem=("itr", sem),
            subjects=("subject", "nunique"),
            folds=("folds", "mean"),
            trials=("trials", "mean"),
        )
        .sort_values(["method", "window"])
    )
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    return summary.sort_values(["method", "window"]).reset_index(drop=True)


def plot_outputs(summary: pd.DataFrame, subject: pd.DataFrame, result_dir: Path, title: str) -> list[Path]:
    figdir = result_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for metric, ylabel, fname in (
        ("accuracy", "Accuracy", "accuracy_curve.png"),
        ("itr", "ITR (bits/min)", "itr_curve.png"),
    ):
        fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=180)
        for method in METHOD_ORDER:
            rows = summary[summary["method"] == method]
            if rows.empty:
                continue
            ax.errorbar(
                rows["window"],
                rows[metric],
                yerr=rows[f"{metric}_sem"],
                marker="o",
                capsize=3,
                linewidth=2,
                label=method,
            )
        ax.set_title(title)
        ax.set_xlabel("Window (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        if metric == "accuracy":
            ax.set_ylim(0, 1.02)
        fig.tight_layout()
        path = figdir / fname
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    piv = summary.pivot_table(index="method", columns="window", values="accuracy")
    fig, ax = plt.subplots(figsize=(9.0, 3.6), dpi=180)
    im = ax.imshow(piv.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Accuracy")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{float(x):g}" for x in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([str(x) for x in piv.index])
    ax.set_xlabel("Window (s)")
    ax.set_title(title)
    fig.tight_layout()
    path = figdir / "accuracy_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    best = summary.sort_values("accuracy").groupby("method", as_index=False).tail(1)
    rows = []
    labels = []
    for row in best.itertuples(index=False):
        values = subject[(subject["method"] == row.method) & (subject["window"] == row.window)]["accuracy"].to_numpy()
        if values.size:
            rows.append(values)
            labels.append(f"{row.method}\n{float(row.window):g}s")
    if rows:
        fig, ax = plt.subplots(figsize=(max(7.0, 1.25 * len(rows)), 4.6), dpi=180)
        ax.boxplot(rows, tick_labels=labels, showmeans=True)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Subject Accuracy at Best Window")
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(title)
        fig.tight_layout()
        path = figdir / "subject_box_best.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    return paths


def write_report(summary: pd.DataFrame, result_dir: Path, run_name: str, source_rows: int) -> Path:
    best = summary.sort_values("accuracy").groupby("method", as_index=False).tail(1).sort_values("method")
    lines = [
        "# Sun2024 Formal Result Report",
        "",
        f"Run: `{run_name}`",
        "",
        "This report is generated from the run's subject-level `summary.csv`.",
        "It does not include smoke results or paper reference lines in the figures.",
        "",
        "## Source",
        "",
        f"- Subject-level rows: `{source_rows}`",
        "- Aggregated file: `summary_aggregate.csv`",
        "- Subject-level copy: `subject.csv`",
        "",
        "## Best Accuracy By Method",
        "",
        "| Method | Window | Accuracy | Accuracy SEM | ITR | ITR SEM | Subjects |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"| {row.method} | {float(row.window):g}s | {row.accuracy:.4f} | "
            f"{row.accuracy_sem:.4f} | {row.itr:.2f} | {row.itr_sem:.2f} | {int(row.subjects)} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `summary.csv` original subject-level runner output",
            "- `summary_aggregate.csv` method/window mean and SEM",
            "- `subject.csv` normalized subject-level table",
            "- `figures/accuracy_curve.png`",
            "- `figures/itr_curve.png`",
            "- `figures/accuracy_heatmap.png`",
            "- `figures/subject_box_best.png`",
            "",
            "## Interpretation Note",
            "",
            "The completed occipital9 run is still a diagnostic baseline, not a paper-level reproduction. "
            "The queued all-channel comb-filter run should become the current formal comparison once it finishes.",
        ]
    )
    path = result_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def plot_formal_run(run_name: str) -> list[Path]:
    result_dir = RESULTS / run_name
    source = result_dir / "summary.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    subject = normalize_summary_columns(pd.read_csv(source))
    summary = summarize_subject_rows(subject)
    subject.to_csv(result_dir / "subject.csv", index=False)
    summary.to_csv(result_dir / "summary_aggregate.csv", index=False)
    paths = plot_outputs(summary, subject, result_dir, f"Sun2024 {run_name}")
    paths.append(write_report(summary, result_dir, run_name, len(subject)))
    return paths


def plot_smoke_diagnostics(run_names: list[str]) -> Path:
    frames = []
    for run_name in run_names:
        path = RESULTS / run_name / "summary.csv"
        if path.exists():
            df = normalize_summary_columns(pd.read_csv(path))
            df["run"] = run_name
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No smoke summary.csv files found.")
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["run"].map(
        {
            "formal_smoke_20260717": "occipital9 fb, 2 folds",
            "formal_smoke_all64_20260717": "all64 fb, 2 folds",
            "formal_smoke_all64_no_latency_20260717": "all64 fb, no latency",
            "formal_smoke_all64_comb_20260717": "all64 comb, 2 folds",
            "formal_smoke_all64_comb_first5_20260717": "all64 comb first5, 2 folds",
            "formal_smoke_all64_comb_first5_fullfold_s01_20260717": "all64 comb first5, 5 folds",
        }
    ).fillna(df["run"])

    out_dir = RESULTS / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(dict.fromkeys(df["label"].tolist()))
    x = np.arange(len(labels))
    width = 0.18
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), dpi=180, sharex=True)
    for idx, method in enumerate(METHOD_ORDER):
        rows = df[df["method"] == method].set_index("label").reindex(labels)
        offset = (idx - 1.5) * width
        axes[0].bar(x + offset, rows["accuracy"], width=width, label=method)
        axes[1].bar(x + offset, rows["itr"], width=width, label=method)
    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("ITR (bits/min)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set_ylim(0, 1.02)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best", ncols=4, fontsize=8)
    fig.suptitle("Sun2024 Diagnostic Smoke Only: not comparable as a formal curve")
    fig.tight_layout()
    path = out_dir / "smoke_acc_itr_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Sun2024 formal results.")
    parser.add_argument("--run", default="formal_full_20260717_occipital9")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()

    paths = plot_formal_run(args.run)
    if args.include_smoke:
        paths.append(
            plot_smoke_diagnostics(
                [
                    "formal_smoke_20260717",
                    "formal_smoke_all64_20260717",
                    "formal_smoke_all64_no_latency_20260717",
                    "formal_smoke_all64_comb_20260717",
                    "formal_smoke_all64_comb_first5_20260717",
                    "formal_smoke_all64_comb_first5_fullfold_s01_20260717",
                ]
            )
        )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
