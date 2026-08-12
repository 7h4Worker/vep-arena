# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


COLORS = {
    "MVMD-CCA": "#116A7B",
    "SA-MVMD-TRCA": "#D97941",
}


def sem(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


def apply_style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("D:/ProjData/proj_python/vep_arena/runs/mvmd_methods"))
    parser.add_argument("--out-dir", type=Path, default=Path("D:/ProjData/proj_python/vep_arena/results/benchmark_9ch/mvmd_methods"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    subject_block = pd.read_csv(args.run_dir / "subject_block.csv")
    predictions = pd.read_csv(args.run_dir / "predictions.csv")
    subject = (
        subject_block.groupby(["method", "window", "subject"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), itr=("itr", "mean"), block_sd=("accuracy", lambda x: float(np.std(x, ddof=1))))
    )
    summary = (
        subject.groupby(["method", "window"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            subject_sem=("accuracy", sem),
            itr=("itr", "mean"),
            itr_sem=("itr", sem),
            block_sd=("block_sd", "mean"),
            block_sd_sem=("block_sd", sem),
            subjects=("subject", "nunique"),
        )
    )
    subject.to_csv(args.out_dir / "subject.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)

    windows = sorted(summary["window"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for metric, err, ylabel, ax in [
        ("accuracy", "subject_sem", "Accuracy", axes[0]),
        ("itr", "itr_sem", "ITR (bits/min)", axes[1]),
        ("block_sd", "block_sd_sem", "Within-subject block SD", axes[2]),
    ]:
        for method in [m for m in COLORS if m in summary["method"].unique()]:
            sub = summary[summary["method"] == method].sort_values("window")
            ax.errorbar(
                sub["window"],
                sub[metric],
                yerr=sub[err],
                marker="o",
                capsize=3,
                linewidth=2,
                color=COLORS[method],
                label=method,
            )
        ax.set_xticks(windows)
        ax.set_xlabel("Signal window (s)")
        ax.set_ylabel(ylabel)
        apply_style(ax)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "mvmd_summary.png", dpi=220)
    fig.savefig(fig_dir / "mvmd_summary.svg")
    plt.close(fig)

    methods = [m for m in COLORS if m in predictions["method"].unique()]
    confusion_files = []
    for window in windows:
        fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 4.8), squeeze=False)
        for ax, method in zip(axes[0], methods):
            rows = predictions[(predictions["method"] == method) & (predictions["window"] == window)]
            cm = confusion_matrix(rows["true"], rows["pred"], labels=np.arange(40), normalize="true")
            im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{method} confusion ({window:.1f}s)")
            ax.set_xlabel("Predicted target")
            ax.set_ylabel("True target")
            ax.set_xticks(range(0, 40, 5))
            ax.set_yticks(range(0, 40, 5))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            np.save(args.out_dir / f"confusion_{method.lower().replace('-', '_')}_{window:.1f}s.npy", cm)
        fig.tight_layout()
        png = fig_dir / f"confusion_{window:.1f}s.png"
        fig.savefig(png, dpi=220)
        fig.savefig(fig_dir / f"confusion_{window:.1f}s.svg")
        confusion_files.append(png.name)
        plt.close(fig)

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    lines = [
        "# MVMD Method Benchmark",
        "",
        "This report contains exploratory Arena adapter outputs for MVMD-CCA and SA-MVMD-TRCA style decoding.",
        "",
        "> Status: exploratory only. This is not a faithful paper reproduction and should not be included in the main leaderboard.",
        "",
        "## Protocol",
        "",
        f"- Dataset: Benchmark SSVEP 9ch",
        f"- Subjects: {manifest.get('subjects')}",
        f"- Blocks: {manifest.get('blocks')}",
        f"- Windows: {manifest.get('windows')}",
        f"- MVMD parameters: K={manifest.get('n_modes')}, alpha={manifest.get('alpha')}, max_iter={manifest.get('max_iter')}, tol={manifest.get('tol')}",
        f"- Assist harmonics: {manifest.get('assist_harmonics')}; drop first mode: {manifest.get('drop_first_mode')}",
        "- SA-MVMD-TRCA note: efficient local variant using all-frequency sinusoidal assistance, not official author code and not the paper algorithm.",
        "- Known mismatch: the paper uses target-frequency sinusoidal assistance and fuses TRCA information from IMFs/reconstructed signals; this adapter uses a simplified reconstruction path.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Figures",
        "",
        "- `figures/mvmd_summary.png`",
        *[f"- `figures/{name}`" for name in confusion_files],
    ]
    (args.out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
