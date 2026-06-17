# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT = Path("D:/ProjData/proj_python/vep_arena")
BASE = PROJECT / "results" / "benchmark_9ch"
OUT = BASE / "final_compare_plus_sa_mvmd"

BASE_SUMMARY = BASE / "final_compare" / "summary.csv"
SA_SUMMARY = BASE / "sa_mvmd_paper" / "summary.csv"

METHOD_ORDER = [
    "CCA",
    "FBCCA",
    "FBTRCA",
    "TDCA",
    "DNN",
    "TRCA-Net",
    "SSVEPFormer",
    "SA-MVMD-TRCA",
    "SA-MVMD-eTRCA",
]

COLORS = {
    "CCA": "#6b7280",
    "FBCCA": "#0891b2",
    "FBTRCA": "#2f6f5f",
    "TDCA": "#b15f18",
    "DNN": "#315aa3",
    "TRCA-Net": "#bf3f55",
    "SSVEPFormer": "#7b59b6",
    "SA-MVMD-TRCA": "#d97941",
    "SA-MVMD-eTRCA": "#315aa3",
}

SHARED_WINDOWS = [0.4, 0.6, 0.8, 1.0]


def apply_style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    if "subject_sem" in df.columns and "accuracy_sem" not in df.columns:
        rename["subject_sem"] = "accuracy_sem"
    if "mean_block_std" in df.columns and "block_sd" not in df.columns:
        rename["mean_block_std"] = "block_sd"
    if "block_std_sem" in df.columns and "block_sd_sem" not in df.columns:
        rename["block_std_sem"] = "block_sd_sem"
    if "itr_sd" in df.columns and "itr_sem" not in df.columns:
        rename["itr_sd"] = "itr_sem"
    if rename:
        df = df.rename(columns=rename)
    if "accuracy_sem" not in df.columns:
        df["accuracy_sem"] = np.nan
    if "itr_sem" not in df.columns:
        df["itr_sem"] = np.nan
    if "block_sd" not in df.columns:
        df["block_sd"] = np.nan
    if "block_sd_sem" not in df.columns:
        df["block_sd_sem"] = np.nan
    if "subjects" not in df.columns:
        df["subjects"] = np.nan
    cols = ["method", "window", "accuracy", "accuracy_sem", "itr", "itr_sem", "block_sd", "block_sd_sem", "subjects"]
    return df[cols].copy()


def load_summary() -> pd.DataFrame:
    base = normalize_summary(pd.read_csv(BASE_SUMMARY))
    sa = normalize_summary(pd.read_csv(SA_SUMMARY))
    summary = pd.concat([base, sa], ignore_index=True)
    summary["window"] = summary["window"].astype(float).round(1)
    summary = summary[summary["method"].isin(METHOD_ORDER)].copy()
    return summary


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=220)
    fig.savefig(OUT / "figures" / f"{name}.svg")
    plt.close(fig)


def plot_curve(summary: pd.DataFrame, metric: str, err: str, ylabel: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("window")
        if sub.empty:
            continue
        ax.errorbar(
            sub["window"],
            sub[metric],
            yerr=sub[err],
            marker="o",
            linewidth=1.8,
            capsize=3,
            markersize=4,
            color=COLORS[method],
            label=method,
        )
    ax.set_xlabel("Signal window (s)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(summary["window"].unique()))
    if metric == "accuracy":
        ax.set_ylim(0.0, 1.0)
    apply_style(ax)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    save(fig, name)


def plot_overview(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))
    for ax, metric, err, ylabel, title in [
        (axes[0], "accuracy", "accuracy_sem", "Accuracy", "Accuracy by window"),
        (axes[1], "itr", "itr_sem", "ITR (bits/min)", "ITR by window"),
    ]:
        for method in METHOD_ORDER:
            sub = summary[summary["method"] == method].sort_values("window")
            if sub.empty:
                continue
            ax.errorbar(
                sub["window"],
                sub[metric],
                yerr=sub[err],
                marker="o",
                linewidth=1.8,
                capsize=3,
                markersize=4,
                color=COLORS[method],
                label=method,
            )
        ax.set_xlabel("Signal window (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(sorted(summary["window"].unique()))
        if metric == "accuracy":
            ax.set_ylim(0.0, 1.0)
        apply_style(ax)
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="lower right")
    save(fig, "overview")


def make_report(summary: pd.DataFrame) -> None:
    shared = summary[summary["window"].isin(SHARED_WINDOWS)].copy()
    ranking = (
        shared.groupby("method", as_index=False)["accuracy"]
        .mean()
        .sort_values("accuracy", ascending=False)
    )
    best = shared.loc[shared.groupby("window")["accuracy"].idxmax()].sort_values("window")
    peak_itr = summary.loc[summary.groupby("method")["itr"].idxmax(), ["method", "window", "itr"]].sort_values("itr", ascending=False)

    lines = [
        "# Benchmark 9ch Final Comparison + SA-MVMD",
        "",
        "This view adds the paper-faithful SA-MVMD-TRCA and SA-MVMD-eTRCA results to the existing 9ch benchmark comparison.",
        "Shared-window ranking is computed on 0.4-1.0 s so every method is compared on the same windows.",
        "",
        "## Shared-Window Ranking",
        "",
        "| Rank | Method | Mean accuracy on 0.4-1.0 s |",
        "| ---: | --- | ---: |",
    ]
    for i, row in enumerate(ranking.itertuples(index=False), start=1):
        lines.append(f"| {i} | {row.method} | {row.accuracy:.4f} |")

    lines += [
        "",
        "## Best Method By Shared Window",
        "",
        "| Window | Best method | Accuracy |",
        "| ---: | --- | ---: |",
    ]
    for row in best.itertuples(index=False):
        lines.append(f"| {row.window:.1f}s | {row.method} | {row.accuracy:.4f} |")

    lines += [
        "",
        "## Peak ITR",
        "",
        "| Method | Peak window | Peak ITR |",
        "| --- | ---: | ---: |",
    ]
    for row in peak_itr.itertuples(index=False):
        lines.append(f"| {row.method} | {row.window:.1f}s | {row.itr:.2f} |")

    lines += [
        "",
        "## Notes",
        "",
        "- SA-MVMD is paper-faithful and uses its own 0.4-2.0 s window sweep.",
        "- The combined line plots show those extra long windows, but the shared-window ranking stays on 0.4-1.0 s for fairness.",
        "- DNN, TRCA-Net, TDCA, FBTRCA, CCA, FBCCA, and SSVEPFormer are kept from the existing benchmark comparison.",
        "",
        "## Files",
        "",
        "- `figures/overview.png`",
        "- `figures/accuracy_curve.png`",
        "- `figures/itr_curve.png`",
        "- `summary.csv`",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    summary.to_csv(OUT / "summary.csv", index=False)
    plot_curve(summary, "accuracy", "accuracy_sem", "Accuracy", "accuracy_curve")
    plot_curve(summary, "itr", "itr_sem", "ITR (bits/min)", "itr_curve")
    plot_overview(summary)
    make_report(summary)
    print(OUT)


if __name__ == "__main__":
    main()
