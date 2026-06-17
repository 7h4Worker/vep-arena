# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


PROJECT = Path("D:/ProjData/proj_python/vep_arena")
ARENA_RESULTS = PROJECT / "results" / "benchmark_9ch"
OUT = ARENA_RESULTS / "final_compare"
TRCANET_OFFICIAL = (
    Path("D:/ProjData/proj_python/trcanet_benchmark_pytorch")
    / "outputs"
    / "sweeps"
    / "official_preprocess_20260522"
    / "analysis"
)
CCA_FBCCA_RUN = PROJECT / "runs" / "cca_fbcca" / "subject_block.csv"

METHOD_ORDER = ["CCA", "FBCCA", "FBTRCA", "TDCA", "DNN", "TRCA-Net", "SSVEPFormer"]
COLORS = {
    "CCA": "#6b7280",
    "FBCCA": "#0891b2",
    "FBTRCA": "#2f6f5f",
    "TDCA": "#b15f18",
    "DNN": "#315aa3",
    "TRCA-Net": "#bf3f55",
    "SSVEPFormer": "#7b59b6",
}
WINDOWS = [round(x, 1) for x in np.arange(0.2, 1.01, 0.1)]


def itr(classes: int, accuracy: float, seconds: float) -> float:
    if accuracy < 1 / classes:
        return 0.0
    if accuracy >= 1.0:
        return math.log2(classes) * 60 / seconds
    return (
        math.log2(classes)
        + accuracy * math.log2(accuracy)
        + (1 - accuracy) * math.log2((1 - accuracy) / (classes - 1))
    ) * 60 / seconds


def sem(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


def load_trials() -> pd.DataFrame:
    trials = pd.read_csv(ARENA_RESULTS / "trials.csv")
    trials = trials[trials["method"] != "TRCA-Net"].copy()

    trca = pd.read_csv(TRCANET_OFFICIAL / "subject_block_results.csv")
    trca = trca.rename(columns={"signal_length": "window"})
    trca["method"] = "TRCA-Net"
    trca["samples"] = 40
    trca = trca[["method", "window", "subject", "block", "accuracy", "samples", "seconds"]]

    cca_fbcca = pd.read_csv(CCA_FBCCA_RUN)
    cca_fbcca = cca_fbcca[["method", "window", "subject", "block", "accuracy", "samples", "seconds"]]

    combined = pd.concat([trials, trca, cca_fbcca], ignore_index=True)
    combined["window"] = combined["window"].round(1)
    return combined[combined["method"].isin(METHOD_ORDER)].copy()


def aggregate(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject = (
        trials.groupby(["method", "window", "subject"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            block_std=("accuracy", lambda x: float(np.std(x, ddof=1))),
            blocks=("block", "nunique"),
            samples=("samples", "sum"),
        )
    )
    subject["itr"] = subject.apply(lambda r: itr(40, float(r.accuracy), float(r.window) + 0.5), axis=1)

    block = (
        trials.groupby(["method", "window", "block"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            subject_std=("accuracy", lambda x: float(np.std(x, ddof=1))),
            subjects=("subject", "nunique"),
        )
    )

    summary = (
        subject.groupby(["method", "window"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            subject_sd=("accuracy", lambda x: float(np.std(x, ddof=1))),
            subject_sem=("accuracy", sem),
            itr=("itr", "mean"),
            itr_sd=("itr", lambda x: float(np.std(x, ddof=1))),
            itr_sem=("itr", sem),
            block_sd=("block_std", "mean"),
            block_sd_sem=("block_std", sem),
            subjects=("subject", "nunique"),
        )
    )
    return summary, subject, block


def paired_stats(subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        wide = subject[subject["window"] == window].pivot(index="subject", columns="method", values="accuracy")
        for a, b in itertools.combinations(METHOD_ORDER, 2):
            if a not in wide.columns or b not in wide.columns:
                continue
            pair = wide[[a, b]].dropna()
            if len(pair) < 2:
                continue
            diff = pair[a] - pair[b]
            stat = ttest_rel(pair[a], pair[b])
            dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else float("nan")
            rows.append(
                {
                    "window": window,
                    "method_a": a,
                    "method_b": b,
                    "mean_a": pair[a].mean(),
                    "mean_b": pair[b].mean(),
                    "delta_a_minus_b": diff.mean(),
                    "t": stat.statistic,
                    "p": stat.pvalue,
                    "cohen_dz": dz,
                    "subjects": len(pair),
                }
            )
    return pd.DataFrame(rows)


def apply_style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(OUT / "figures" / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def plot_curves(summary: pd.DataFrame) -> None:
    for metric, err, ylabel, name in [
        ("accuracy", "subject_sem", "Accuracy", "accuracy_curve"),
        ("itr", "itr_sem", "ITR (bits/min)", "itr_curve"),
        ("block_sd", "block_sd_sem", "Within-subject block SD", "block_stability"),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for method in METHOD_ORDER:
            sub = summary[summary["method"] == method].sort_values("window")
            ax.errorbar(
                sub["window"],
                sub[metric],
                yerr=sub[err],
                marker="o",
                markersize=4.5,
                linewidth=1.9,
                capsize=3,
                color=COLORS[method],
                label=method,
            )
        ax.set_xlabel("Signal window (s)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(WINDOWS)
        if metric == "accuracy":
            ax.set_ylim(0.2, 1.0)
        apply_style(ax)
        ax.legend(frameon=False, ncol=4, loc="lower right" if metric == "accuracy" else "best")
        save(fig, name)


def plot_bars(summary: pd.DataFrame) -> None:
    windows = [0.5, 1.0]
    width = 0.105
    x = np.arange(len(windows))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for i, method in enumerate(METHOD_ORDER):
        rows = summary[(summary["method"] == method) & (summary["window"].isin(windows))].set_index("window")
        vals = [rows.loc[w, "accuracy"] for w in windows]
        errs = [rows.loc[w, "subject_sem"] for w in windows]
        ax.bar(x + (i - 3) * width, vals, width, yerr=errs, capsize=3, color=COLORS[method], label=method)
    ax.set_xticks(x, [f"{w:.1f}s" for w in windows])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.0)
    apply_style(ax)
    ax.legend(frameon=False, ncol=4, loc="lower right")
    save(fig, "accuracy_bars_05_10")


def plot_delta(summary: pd.DataFrame) -> None:
    wide = summary.pivot(index="window", columns="method", values="accuracy")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for method in METHOD_ORDER:
        if method == "DNN":
            continue
        delta = wide[method] - wide["DNN"]
        ax.plot(delta.index, delta.values, marker="o", linewidth=1.9, color=COLORS[method], label=method)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Signal window (s)")
    ax.set_ylabel("Accuracy delta vs DNN")
    ax.set_xticks(WINDOWS)
    apply_style(ax)
    ax.legend(frameon=False, ncol=2)
    save(fig, "delta_vs_dnn")


def plot_subject_box(subject: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, window in zip(axes, [0.5, 1.0]):
        data = [
            subject[(subject["method"] == method) & (subject["window"] == window)]["accuracy"].to_numpy()
            for method in METHOD_ORDER
        ]
        bp = ax.boxplot(data, tick_labels=METHOD_ORDER, patch_artist=True, showfliers=False)
        for patch, method in zip(bp["boxes"], METHOD_ORDER):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.68)
            patch.set_edgecolor("#333333")
        ax.set_title(f"{window:.1f}s")
        ax.tick_params(axis="x", labelrotation=25)
        apply_style(ax)
    axes[0].set_ylabel("Subject mean accuracy")
    axes[0].set_ylim(0.2, 1.0)
    save(fig, "subject_box_05_10")


def plot_overview(summary: pd.DataFrame, subject: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    ax = axes[0, 0]
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("window")
        ax.plot(sub["window"], sub["accuracy"], marker="o", linewidth=1.9, color=COLORS[method], label=method)
    ax.set_title("Accuracy by window")
    ax.set_xlabel("Signal window (s)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.2, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    apply_style(ax)

    ax = axes[0, 1]
    rows = summary[summary["window"].isin([0.5, 1.0])]
    bars = rows.pivot(index="method", columns="window", values="accuracy").loc[METHOD_ORDER]
    x = np.arange(len(METHOD_ORDER))
    ax.bar(x - 0.18, bars[0.5], 0.36, color=[COLORS[m] for m in METHOD_ORDER], alpha=0.65, label="0.5s")
    ax.bar(x + 0.18, bars[1.0], 0.36, color=[COLORS[m] for m in METHOD_ORDER], alpha=0.95, label="1.0s")
    ax.set_xticks(x, METHOD_ORDER, rotation=25)
    ax.set_title("Typical windows")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.legend(
        handles=[
            Patch(facecolor="#555555", alpha=0.45, label="0.5s"),
            Patch(facecolor="#555555", alpha=0.85, label="1.0s"),
        ],
        frameon=False,
    )
    apply_style(ax)

    ax = axes[1, 0]
    wide = summary.pivot(index="window", columns="method", values="accuracy")
    for method in METHOD_ORDER:
        if method == "DNN":
            continue
        ax.plot(wide.index, wide[method] - wide["DNN"], marker="o", linewidth=1.7, color=COLORS[method], label=method)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Delta vs DNN")
    ax.set_xlabel("Signal window (s)")
    ax.set_ylabel("Accuracy delta")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    apply_style(ax)

    ax = axes[1, 1]
    data = [
        subject[(subject["method"] == method) & (subject["window"] == 1.0)]["accuracy"].to_numpy()
        for method in METHOD_ORDER
    ]
    bp = ax.boxplot(data, tick_labels=METHOD_ORDER, patch_artist=True, showfliers=False)
    for patch, method in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.68)
    ax.set_title("Subject distribution at 1.0s")
    ax.set_ylabel("Subject mean accuracy")
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_ylim(0.2, 1.0)
    apply_style(ax)
    save(fig, "overview")


def make_report(summary: pd.DataFrame, subject: pd.DataFrame, stats: pd.DataFrame) -> None:
    acc = summary.pivot(index="window", columns="method", values="accuracy")
    sems = summary.pivot(index="window", columns="method", values="subject_sem")
    avg = summary.groupby("method", as_index=False)["accuracy"].mean().sort_values("accuracy", ascending=False)
    best = summary.loc[summary.groupby("window")["accuracy"].idxmax()].sort_values("window")
    peak_itr = summary.loc[summary.groupby("method")["itr"].idxmax(), ["method", "window", "itr"]].sort_values("itr", ascending=False)

    lines = [
        "# Benchmark 9ch Final Comparison",
        "",
        "Protocol: subject-specific leave-one-block-out on the Tsinghua Benchmark 9-channel subset.",
        "TRCA-Net uses the later `official_preprocess_20260522` run as the main entry.",
        "",
        "## Overall Ranking",
        "",
        "| Rank | Method | Mean accuracy across windows |",
        "| ---: | --- | ---: |",
    ]
    for i, row in enumerate(avg.itertuples(index=False), start=1):
        lines.append(f"| {i} | {row.method} | {row.accuracy:.4f} |")

    lines += [
        "",
        "## Best By Window",
        "",
        "| Window | Best method | Accuracy |",
        "| ---: | --- | ---: |",
    ]
    for row in best.itertuples(index=False):
        lines.append(f"| {row.window:.1f}s | {row.method} | {row.accuracy:.4f} |")

    lines += [
        "",
        "## Accuracy",
        "",
        "| Window | " + " | ".join(METHOD_ORDER) + " |",
        "| ---: | " + " | ".join(["---:" for _ in METHOD_ORDER]) + " |",
    ]
    for window in WINDOWS:
        cells = []
        for method in METHOD_ORDER:
            cells.append(f"{acc.loc[window, method]:.4f} +/- {sems.loc[window, method]:.4f}")
        lines.append(f"| {window:.1f}s | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Peak ITR",
        "",
        "| Method | Peak window | Peak ITR |",
        "| --- | ---: | ---: |",
    ]
    for row in peak_itr.itertuples(index=False):
        lines.append(f"| {row.method} | {row.window:.1f}s | {row.itr:.2f} |")

    dnn_tests = stats[(stats["method_b"] == "DNN") | (stats["method_a"] == "DNN")].copy()
    dnn_tests = dnn_tests[dnn_tests["window"].isin([0.5, 1.0])]
    lines += [
        "",
        "## Notes",
        "",
        "- CCA and FBCCA are calibration-free reference methods here; they use the same Arena trial crop and output schema, but they do not train with this subject's remaining blocks.",
        "- FBCCA follows the public TRCA-SSVEP implementation of Chen/Nakanishi style FBCCA: 5 filter banks, 5 harmonics, and `(idx + 1)^(-1.25) + 0.25` subband weights.",
        "- DNN is the strongest across most short-to-mid windows in this local set.",
        "- TDCA catches up at 1.0s and is the best entry there by a small margin.",
        "- TRCA-Net official-preprocess is close to DNN around 0.5-0.6s but does not exceed DNN at 1.0s in this run.",
        "- SSVEPFormer is consistently below the other entries here; this likely reflects the current local reproduction/configuration rather than a final claim about the architecture family.",
        "- Error bars in figures are SEM across subjects; boxplots show subject mean accuracies across blocks.",
        "",
        "## Files",
        "",
        "- `figures/overview.png`: compact four-panel summary",
        "- `figures/accuracy_curve.png`: accuracy curve with SEM",
        "- `figures/itr_curve.png`: ITR curve with SEM",
        "- `figures/accuracy_bars_05_10.png`: grouped bars at 0.5s and 1.0s",
        "- `figures/delta_vs_dnn.png`: accuracy delta relative to DNN",
        "- `figures/subject_box_05_10.png`: subject distribution at 0.5s and 1.0s",
        "- `figures/block_stability.png`: within-subject block SD",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    trials = load_trials()
    summary, subject, block = aggregate(trials)
    stats = paired_stats(subject)

    trials.to_csv(OUT / "trials.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    subject.to_csv(OUT / "subject.csv", index=False)
    block.to_csv(OUT / "block.csv", index=False)
    stats.to_csv(OUT / "paired_stats.csv", index=False)

    source_rows = [
        {
            "method": "TRCA-Net",
            "source": str(TRCANET_OFFICIAL / "subject_block_results.csv"),
            "note": "official-preprocess run, replaces first-pass TRCA-Net in final comparison",
        },
        {
            "method": "DNN/FBTRCA/TDCA/SSVEPFormer",
            "source": str(ARENA_RESULTS / "trials.csv"),
            "note": "previous VEP Arena consolidated source rows",
        },
        {
            "method": "CCA/FBCCA",
            "source": str(CCA_FBCCA_RUN),
            "note": "Arena-native run using TRCA-SSVEP FBCCA reference settings",
        },
    ]
    pd.DataFrame(source_rows).to_csv(OUT / "source_ledger.csv", index=False)

    plot_curves(summary)
    plot_bars(summary)
    plot_delta(summary)
    plot_subject_box(subject)
    plot_overview(summary, subject)
    make_report(summary, subject, stats)
    print(OUT)


if __name__ == "__main__":
    main()
