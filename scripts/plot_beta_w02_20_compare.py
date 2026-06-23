from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
OUT = RESULTS / "beta_toolbox_9ch_w02_20s_compare"
SOURCES = [
    RESULTS / "beta_toolbox_9ch_w02_20s_cca_fbcca_trca_tdca" / "summary.csv",
    RESULTS / "beta_toolbox_9ch_w02_20s_etrca" / "summary.csv",
]
METHOD_ORDER = ["CCA", "FBCCA", "TRCA", "ETRCA", "TDCA"]
COLORS = {
    "CCA": "#5B6C8F",
    "FBCCA": "#2F9C95",
    "TRCA": "#D8893A",
    "ETRCA": "#C84E5A",
    "TDCA": "#6C5BA7",
}


def load_summary() -> pd.DataFrame:
    frames = []
    for path in SOURCES:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    summary = pd.concat(frames, ignore_index=True)
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    summary = summary.sort_values(["method", "window"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    return summary


def plot_metric(ax: plt.Axes, summary: pd.DataFrame, metric: str, sem: str, ylabel: str) -> None:
    for method in METHOD_ORDER:
        rows = summary[summary["method"] == method]
        if rows.empty:
            continue
        ax.errorbar(
            rows["window"],
            rows[metric],
            yerr=rows[sem],
            marker="o",
            linewidth=2.2,
            markersize=4.8,
            capsize=2.8,
            color=COLORS[method],
            label=method,
        )
    ax.set_xlabel("Window length (s)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.grid(True, axis="y", color="#D8DEE9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    summary = load_summary()
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), constrained_layout=True)
    plot_metric(axes[0], summary, "accuracy", "accuracy_sem", "Accuracy")
    plot_metric(axes[1], summary, "itr", "itr_sem", "ITR (bits/min)")
    axes[0].set_ylim(0, 0.95)
    axes[1].set_ylim(0, 180)
    axes[0].set_title("BETA 9ch Accuracy (0.2-2.0s)")
    axes[1].set_title("BETA 9ch ITR (0.2-2.0s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("BETA SSVEP: Arena Toolbox-Preprocessed Baselines", y=1.03, fontsize=14)
    fig.text(
        0.5,
        -0.025,
        "Protocol: subject-specific leave-one-block-out; 70 subjects x 4 blocks; ITR uses current Arena denominator: window + break.",
        ha="center",
        va="top",
        fontsize=9,
        color="#5A6573",
    )
    fig.savefig(OUT / "acc_itr_compare.png", bbox_inches="tight")
    fig.savefig(OUT / "acc_itr_compare.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
