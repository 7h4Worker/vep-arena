# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-18
# Last updated: 2026-06-18
# Description: Plot Arena-side DNN global vs fine-tuned comparison.
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import RESULT_ROOT


METHOD_LABELS = {
    "DNN-global-pt": "Global checkpoint",
    "DNN-finetuned-pt": "Checkpoint + subject fine-tune",
    "DNN-global-trained": "Arena-trained global",
    "DNN-finetuned-trained": "Arena-trained + subject fine-tune",
}


def load_summary(root: Path) -> pd.DataFrame:
    if (root / "summary.csv").exists():
        return pd.read_csv(root / "summary.csv")
    frames = [pd.read_csv(path) for path in sorted(root.glob("window_*s/summary.csv"))]
    if not frames:
        raise FileNotFoundError(f"No summary.csv files found under {root}")
    summary = pd.concat(frames, ignore_index=True).sort_values(["method", "window"])
    summary.to_csv(root / "summary.csv", index=False)
    return summary


def plot_metric(summary: pd.DataFrame, out_path: Path, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
    for method, group in summary.groupby("method"):
        group = group.sort_values("window")
        ax.plot(
            group["window"],
            group[metric],
            marker="o",
            linewidth=2,
            markersize=4,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("Window (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(summary["window"].min() - 0.03, summary["window"].max() + 0.03)
    if metric == "accuracy":
        ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_lift(summary: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    pivot = summary.pivot_table(index="window", columns="method", values="accuracy", aggfunc="mean")
    lift_rows = []
    if {"DNN-global-pt", "DNN-finetuned-pt"}.issubset(pivot.columns):
        lift_rows.append(("Checkpoint fine-tune lift", pivot["DNN-finetuned-pt"] - pivot["DNN-global-pt"]))
    if {"DNN-global-trained", "DNN-finetuned-trained"}.issubset(pivot.columns):
        lift_rows.append(("Arena-trained fine-tune lift", pivot["DNN-finetuned-trained"] - pivot["DNN-global-trained"]))
    if not lift_rows:
        return pd.DataFrame()
    lift = pd.DataFrame({"window": pivot.index})
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=150)
    for label, series in lift_rows:
        lift[label] = series.to_numpy()
        ax.plot(pivot.index, series, marker="o", linewidth=2, markersize=4, label=label)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("Accuracy lift")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return lift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=RESULT_ROOT / "dnn_w02_2s_arena")
    args = parser.parse_args()

    args.input_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.input_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    summary = load_summary(args.input_dir)
    summary["label"] = summary["method"].map(METHOD_LABELS).fillna(summary["method"])
    summary.to_csv(args.input_dir / "summary.csv", index=False)
    plot_metric(summary, figure_dir / "dnn_accuracy_comparison.png", "accuracy", "Accuracy")
    plot_metric(summary, figure_dir / "dnn_itr_comparison.png", "itr", "ITR (bits/min)")
    lift = plot_lift(summary, figure_dir / "dnn_finetune_lift.png")
    if not lift.empty:
        lift.to_csv(args.input_dir / "finetune_lift.csv", index=False)
    best = summary.sort_values("accuracy", ascending=False).iloc[0]
    lines = [
        "# DNN Arena Comparison",
        "",
        f"Input directory: `{args.input_dir}`",
        "",
        "| Method | Windows | Best accuracy | Best window |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, group in summary.groupby("method"):
        best_group = group.sort_values("accuracy", ascending=False).iloc[0]
        lines.append(
            f"| {METHOD_LABELS.get(method, method)} | {group['window'].nunique()} | "
            f"{best_group['accuracy']:.4f} | {best_group['window']:.1f} |"
        )
    lines.extend(
        [
            "",
            f"Best observed row: {METHOD_LABELS.get(best['method'], best['method'])} at {best['window']:.1f}s, accuracy {best['accuracy']:.4f}.",
            "",
            "## Figures",
            "",
            "![DNN accuracy comparison](figures/dnn_accuracy_comparison.png)",
            "",
            "![DNN ITR comparison](figures/dnn_itr_comparison.png)",
            "",
            "![DNN fine-tune lift](figures/dnn_finetune_lift.png)",
        ]
    )
    (args.input_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.input_dir / "report.md")


if __name__ == "__main__":
    main()
