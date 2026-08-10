"""Complete figure set for JBHI35 five-subject historical run.

Outputs:
1. unified_accuracy.png / unified_itr.png — all methods, mean ± SEM
2. per_subject_accuracy.png / per_subject_itr.png — one subplot per subject, all methods
3. per_method_accuracy.png / per_method_itr.png — one subplot per method, all subjects
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "final_five_execution_20260730"
OUTPUT = RESULTS / "figures"
OUTPUT.mkdir(exist_ok=True)

TDCA_SUBJECT = pd.read_csv(RESULTS / "tdca_full" / "subject.csv")
PERIODIC_SUBJECT = pd.read_csv(RESULTS / "periodic_receivers_full" / "subject.csv")
ETRCA_SUBJECT = pd.read_csv(RESULTS / "etrca_2s" / "subject.csv")

etrca_signed = ETRCA_SUBJECT[ETRCA_SUBJECT["variant"] == "arena_independent_signed"].copy()
etrca_signed["method"] = "eTRCA"
etrca_signed["window_seconds"] = 2.0
etrca_signed = etrca_signed[["subject", "method", "window_seconds", "accuracy", "itr_bpm"]]

all_subject = pd.concat([TDCA_SUBJECT, PERIODIC_SUBJECT, etrca_signed], ignore_index=True)

SUBJECTS = ["S01", "S03", "S04", "S05", "S06"]
METHOD_ORDER = ["SPECTRAL_TDCA", "EFUSIONCA", "EEG_TDCA", "FUSIONCA", "EBPRCA", "BPRCA", "eTRCA"]
METHOD_LABELS = {
    "SPECTRAL_TDCA": "Spectral-TDCA",
    "EEG_TDCA": "EEG-TDCA",
    "BPRCA": "bPRCA",
    "EBPRCA": "E-bPRCA",
    "FUSIONCA": "FusionCA",
    "EFUSIONCA": "E-FusionCA",
    "eTRCA": "eTRCA",
}
COLORS = {
    "SPECTRAL_TDCA": "#D62728",
    "EEG_TDCA": "#FF7F0E",
    "EFUSIONCA": "#2CA02C",
    "FUSIONCA": "#1F77B4",
    "EBPRCA": "#9467BD",
    "BPRCA": "#8C564B",
    "eTRCA": "#17BECF",
}
MARKERS = {
    "SPECTRAL_TDCA": "s",
    "EEG_TDCA": "D",
    "EFUSIONCA": "^",
    "FUSIONCA": "o",
    "EBPRCA": "v",
    "BPRCA": "x",
    "eTRCA": "*",
}

windows = np.arange(0.2, 2.01, 0.2).round(1)


def _group_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["method", "window_seconds"], as_index=False).agg(
        acc_mean=("accuracy", "mean"),
        acc_sem=("accuracy", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        itr_mean=("itr_bpm", "mean"),
        itr_sem=("itr_bpm", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
    )
    return grouped.sort_values(["method", "window_seconds"])


# === Figure 1 & 2: Unified mean ± SEM ===
all_no_etrca = all_subject[all_subject["method"] != "eTRCA"]
stats = _group_stats(all_no_etrca)
etrca_stats = _group_stats(all_subject[all_subject["method"] == "eTRCA"])

for metric, sem_col, ylabel, title_suffix, fname in [
    ("acc_mean", "acc_sem", "Accuracy", "Accuracy", "unified_accuracy.png"),
    ("itr_mean", "itr_sem", "ITR (bits/min)", "ITR", "unified_itr.png"),
]:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for method in METHOD_ORDER:
        if method == "eTRCA":
            row = etrca_stats[etrca_stats["method"] == "eTRCA"]
            if not row.empty:
                r = row.iloc[0]
                ax.errorbar(r["window_seconds"], r[metric], yerr=r[sem_col],
                            marker="*", color=COLORS[method], markersize=14,
                            capsize=4, linewidth=0, label=METHOD_LABELS[method] + " (2s)")
            continue
        subset = stats[stats["method"] == method]
        if subset.empty:
            continue
        ax.errorbar(subset["window_seconds"], subset[metric], yerr=subset[sem_col],
                    marker=MARKERS[method], color=COLORS[method],
                    label=METHOD_LABELS[method], markersize=6, capsize=3, linewidth=1.6)
    if metric == "acc_mean":
        ax.axhline(1/35, color="gray", ls="--", lw=0.8, alpha=0.5, label="Chance")
    ax.set_xlabel("Window length (s)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"JBHI35 Five-Subject Historical Set: {title_suffix} vs. Window Length", fontsize=12)
    ax.set_xticks(windows)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.savefig(OUTPUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT / fname}")


# === Figure 3 & 4: Per-subject (subplots), all methods ===
for metric, ylabel, title_suffix, fname in [
    ("accuracy", "Accuracy", "Accuracy", "per_subject_accuracy.png"),
    ("itr_bpm", "ITR (bits/min)", "ITR", "per_subject_itr.png"),
]:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True, sharey=True)
    for idx, (subj, ax) in enumerate(zip(SUBJECTS, axes)):
        subj_data = all_subject[all_subject["subject"] == subj]
        for method in METHOD_ORDER:
            mdata = subj_data[subj_data["method"] == method].sort_values("window_seconds")
            if mdata.empty:
                continue
            if method == "eTRCA":
                ax.plot(mdata["window_seconds"], mdata[metric],
                        marker="*", color=COLORS[method], markersize=12,
                        linewidth=0, label=METHOD_LABELS[method])
            else:
                ax.plot(mdata["window_seconds"], mdata[metric],
                        marker=MARKERS[method], color=COLORS[method],
                        label=METHOD_LABELS[method], markersize=5, linewidth=1.4)
        if metric == "accuracy":
            ax.axhline(1/35, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(subj, fontsize=11)
        ax.set_xlabel("Window (s)", fontsize=9)
        ax.set_xticks([0.4, 0.8, 1.2, 1.6, 2.0])
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=10)
    axes[-1].legend(loc="upper left", fontsize=7, bbox_to_anchor=(1.02, 1))
    fig.suptitle(f"JBHI35 Per-Subject {title_suffix}", fontsize=13)
    fig.savefig(OUTPUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT / fname}")


# === Figure 5 & 6: Per-method (subplots), all subjects + mean ===
methods_with_curve = [m for m in METHOD_ORDER if m != "eTRCA"]
for metric, ylabel, title_suffix, fname in [
    ("accuracy", "Accuracy", "Accuracy", "per_method_accuracy.png"),
    ("itr_bpm", "ITR (bits/min)", "ITR", "per_method_itr.png"),
]:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True, sharey=True)
    axes_flat = axes.flatten()
    subj_colors = {"S01": "#1f77b4", "S03": "#ff7f0e", "S04": "#2ca02c", "S05": "#d62728", "S06": "#9467bd"}
    for idx, method in enumerate(methods_with_curve):
        ax = axes_flat[idx]
        mdata = all_subject[all_subject["method"] == method]
        for subj in SUBJECTS:
            sdata = mdata[mdata["subject"] == subj].sort_values("window_seconds")
            ax.plot(sdata["window_seconds"], sdata[metric],
                    marker="o", color=subj_colors[subj], markersize=4,
                    linewidth=1.0, alpha=0.6, label=subj)
        mean_data = mdata.groupby("window_seconds", as_index=False).agg(
            mean=(metric, "mean"),
            sem=(metric, lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        )
        ax.errorbar(mean_data["window_seconds"], mean_data["mean"],
                    yerr=mean_data["sem"], color="black", linewidth=2.2,
                    marker="s", markersize=5, capsize=3, label="Mean ± SEM")
        if metric == "accuracy":
            ax.axhline(1/35, color="gray", ls="--", lw=0.7, alpha=0.5)
        ax.set_title(METHOD_LABELS[method], fontsize=11)
        ax.set_xlabel("Window (s)", fontsize=9)
        ax.set_xticks([0.4, 0.8, 1.2, 1.6, 2.0])
        ax.grid(alpha=0.2)
        if idx % 3 == 0:
            ax.set_ylabel(ylabel, fontsize=10)
    axes_flat[-1].legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.02, 1))
    fig.suptitle(f"JBHI35 Per-Method {title_suffix}: Individual Subjects + Mean", fontsize=13)
    fig.savefig(OUTPUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT / fname}")

print("\nAll figures generated.")
