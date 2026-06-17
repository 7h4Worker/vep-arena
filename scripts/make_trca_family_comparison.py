# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Build TRCA-family comparison figures from completed Arena outputs.
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path("D:/ProjData/proj_python/vep_arena")
OUT = PROJECT / "results" / "benchmark_9ch" / "trca_family_comparison"
FIG = OUT / "figures"

TRADITIONAL = PROJECT / "results" / "traditional_9ch_core" / "summary.csv"
WONG = PROJECT / "results" / "benchmark_9ch" / "wong_multistimulus_04_10" / "summary.csv"
SA_MVMD = PROJECT / "results" / "benchmark_9ch" / "sa_mvmd_paper" / "summary.csv"

METHOD_ORDER = [
    "TRCA",
    "MSETRCA",
    "MSCCA+MSETRCA",
    "SA-MVMD-TRCA",
    "SA-MVMD-eTRCA",
]

COLORS = {
    "TRCA": "#455a64",
    "MSETRCA": "#2f6f5f",
    "MSCCA+MSETRCA": "#126f9a",
    "SA-MVMD-TRCA": "#d97941",
    "SA-MVMD-eTRCA": "#315aa3",
}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "subject_sem" in df.columns and "accuracy_sem" not in df.columns:
        df = df.rename(columns={"subject_sem": "accuracy_sem"})
    if "accuracy_sem" not in df.columns:
        df["accuracy_sem"] = np.nan
    if "itr_sem" not in df.columns:
        df["itr_sem"] = np.nan
    if "block_sd" not in df.columns:
        df["block_sd"] = np.nan
    if "subjects" not in df.columns:
        df["subjects"] = np.nan
    return df[["method", "window", "accuracy", "accuracy_sem", "itr", "itr_sem", "block_sd", "subjects"]]


def load_summary() -> pd.DataFrame:
    parts = []
    trad = normalize(pd.read_csv(TRADITIONAL))
    parts.append(trad[trad["method"].eq("TRCA")])
    wong = normalize(pd.read_csv(WONG))
    parts.append(wong[wong["method"].isin(["MSETRCA", "MSCCA+MSETRCA"])])
    sa = normalize(pd.read_csv(SA_MVMD))
    parts.append(sa[sa["method"].isin(["SA-MVMD-TRCA", "SA-MVMD-eTRCA"])])
    summary = pd.concat(parts, ignore_index=True)
    summary["window"] = summary["window"].astype(float).round(1)
    return summary[summary["method"].isin(METHOD_ORDER)].copy()


def apply_style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=220)
    fig.savefig(FIG / f"{name}.svg")
    plt.close(fig)


def plot_accuracy_curve(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for method in METHOD_ORDER:
        rows = summary[summary["method"].eq(method)].sort_values("window")
        if rows.empty:
            continue
        ax.errorbar(
            rows["window"],
            rows["accuracy"],
            yerr=rows["accuracy_sem"],
            marker="o",
            linewidth=1.9,
            capsize=3,
            markersize=4.5,
            color=COLORS[method],
            label=method,
        )
    ax.set_xlabel("Signal window (s)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.45, 1.01)
    ax.set_xticks(sorted(summary["window"].unique()))
    apply_style(ax)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    save(fig, "trca_family_accuracy_curve")


def plot_itr_curve(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for method in METHOD_ORDER:
        rows = summary[summary["method"].eq(method)].sort_values("window")
        if rows.empty:
            continue
        ax.errorbar(
            rows["window"],
            rows["itr"],
            yerr=rows["itr_sem"],
            marker="o",
            linewidth=1.9,
            capsize=3,
            markersize=4.5,
            color=COLORS[method],
            label=method,
        )
    ax.set_xlabel("Signal window (s)")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_xticks(sorted(summary["window"].unique()))
    apply_style(ax)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    save(fig, "trca_family_itr_curve")


def plot_one_second_bar(summary: pd.DataFrame) -> None:
    rows = summary[summary["window"].eq(1.0)].set_index("method").reindex(METHOD_ORDER).dropna(subset=["accuracy"])
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(rows))
    ax.bar(x, rows["accuracy"], yerr=rows["accuracy_sem"], capsize=4, color=[COLORS[m] for m in rows.index])
    ax.set_xticks(x, rows.index, rotation=20, ha="right")
    ax.set_ylabel("Accuracy at 1.0 s")
    ax.set_ylim(0.86, 1.0)
    for i, value in enumerate(rows["accuracy"]):
        ax.text(i, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    apply_style(ax)
    save(fig, "trca_family_1s_accuracy_bar")


def write_report(summary: pd.DataFrame) -> None:
    one_s = summary[summary["window"].eq(1.0)].set_index("method").reindex(METHOD_ORDER).dropna(subset=["accuracy"])
    ranking = one_s.sort_values("accuracy", ascending=False)
    lines = [
        "# TRCA Family Comparison",
        "",
        "This report compares completed TRCA-family outputs already present in VEP Arena.",
        "",
        "## 1.0 s Accuracy",
        "",
        "| Method | Accuracy | SEM | ITR |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, row in ranking.iterrows():
        lines.append(f"| {method} | {row.accuracy:.4f} | {row.accuracy_sem:.4f} | {row.itr:.2f} |")
    lines += [
        "",
        "## Files",
        "",
        "- `summary.csv`",
        "- `figures/trca_family_accuracy_curve.png`",
        "- `figures/trca_family_itr_curve.png`",
        "- `figures/trca_family_1s_accuracy_bar.png`",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    summary.to_csv(OUT / "summary.csv", index=False)
    plot_accuracy_curve(summary)
    plot_itr_curve(summary)
    plot_one_second_bar(summary)
    write_report(summary)
    print(OUT)


if __name__ == "__main__":
    main()
