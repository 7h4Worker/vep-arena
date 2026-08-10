"""Unified accuracy and ITR curves for the JBHI35 five-subject historical run."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results" / "final_five_execution_20260730"
OUTPUT = RESULTS / "figures"
OUTPUT.mkdir(exist_ok=True)

TDCA_SUBJECT = pd.read_csv(RESULTS / "tdca_full" / "subject.csv")
PERIODIC_SUBJECT = pd.read_csv(RESULTS / "periodic_receivers_full" / "subject.csv")
ETRCA_SUBJECT = pd.read_csv(RESULTS / "etrca_2s" / "subject.csv")

def _group_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["method", "window_seconds"], as_index=False).agg(
        acc_mean=("accuracy", "mean"),
        acc_sem=("accuracy", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        itr_mean=("itr_bpm", "mean"),
        itr_sem=("itr_bpm", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
    )
    return grouped.sort_values(["method", "window_seconds"])


tdca_stats = _group_stats(TDCA_SUBJECT)
periodic_stats = _group_stats(PERIODIC_SUBJECT)

etrca_signed = ETRCA_SUBJECT[ETRCA_SUBJECT["variant"] == "arena_independent_signed"]
etrca_acc = etrca_signed["accuracy"].mean()
etrca_acc_sem = etrca_signed["accuracy"].std(ddof=1) / np.sqrt(len(etrca_signed))
etrca_itr = etrca_signed["itr_bpm"].mean()
etrca_itr_sem = etrca_signed["itr_bpm"].std(ddof=1) / np.sqrt(len(etrca_signed))

METHOD_ORDER = [
    "SPECTRAL_TDCA", "EFUSIONCA", "FUSIONCA", "EEG_TDCA",
    "EBPRCA", "BPRCA",
]
METHOD_LABELS = {
    "SPECTRAL_TDCA": "Spectral-TDCA",
    "EEG_TDCA": "EEG-TDCA",
    "BPRCA": "bPRCA",
    "EBPRCA": "E-bPRCA",
    "FUSIONCA": "FusionCA",
    "EFUSIONCA": "E-FusionCA",
}
COLORS = {
    "SPECTRAL_TDCA": "#D62728",
    "EEG_TDCA": "#FF7F0E",
    "EFUSIONCA": "#2CA02C",
    "FUSIONCA": "#1F77B4",
    "EBPRCA": "#9467BD",
    "BPRCA": "#8C564B",
}
MARKERS = {
    "SPECTRAL_TDCA": "s",
    "EEG_TDCA": "D",
    "EFUSIONCA": "^",
    "FUSIONCA": "o",
    "EBPRCA": "v",
    "BPRCA": "x",
}

all_stats = pd.concat([tdca_stats, periodic_stats], ignore_index=True)


def _plot_metric(metric_mean: str, metric_sem: str, ylabel: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in METHOD_ORDER:
        subset = all_stats[all_stats["method"] == method]
        if subset.empty:
            continue
        ax.errorbar(
            subset["window_seconds"],
            subset[metric_mean],
            yerr=subset[metric_sem],
            marker=MARKERS[method],
            color=COLORS[method],
            label=METHOD_LABELS[method],
            markersize=6,
            capsize=3,
            linewidth=1.5,
        )
    if metric_mean == "acc_mean":
        ax.errorbar(
            2.0, etrca_acc, yerr=etrca_acc_sem,
            marker="*", color="#17BECF", markersize=12, capsize=4,
            linewidth=0, label="eTRCA (2s only)",
        )
        ax.axhline(1/35, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Chance (1/35)")
    else:
        ax.errorbar(
            2.0, etrca_itr, yerr=etrca_itr_sem,
            marker="*", color="#17BECF", markersize=12, capsize=4,
            linewidth=0, label="eTRCA (2s only)",
        )
    ax.set_xlabel("Window length (s)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(0.2, 2.1, 0.2))
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(OUTPUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT / filename}")


_plot_metric("acc_mean", "acc_sem", "Accuracy", "JBHI35 Five-Subject: Accuracy vs. Window Length", "unified_accuracy.png")
_plot_metric("itr_mean", "itr_sem", "ITR (bits/min)", "JBHI35 Five-Subject: ITR vs. Window Length", "unified_itr.png")

print("\nSummary at key time points:")
print("=" * 80)
for window in [0.4, 1.0, 2.0]:
    print(f"\n--- Window = {window} s ---")
    subset = all_stats[all_stats["window_seconds"] == window]
    for method in METHOD_ORDER:
        row = subset[subset["method"] == method]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"  {METHOD_LABELS[method]:14s}  Acc={r['acc_mean']:.1%} ± {r['acc_sem']:.1%}  ITR={r['itr_mean']:.1f} ± {r['itr_sem']:.1f} bpm")
    if window == 2.0:
        print(f"  {'eTRCA':14s}  Acc={etrca_acc:.1%} ± {etrca_acc_sem:.1%}  ITR={etrca_itr:.1f} ± {etrca_itr_sem:.1f} bpm")
