from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METHOD_ORDER = ("CCA", "FBCCA", "TRCA", "ETRCA", "TDCA")
DEFAULT_COLORS = {
    "CCA": "#5B6C8F",
    "FBCCA": "#2F9C95",
    "TRCA": "#D8893A",
    "ETRCA": "#C84E5A",
    "TDCA": "#6C5BA7",
}


@dataclass(frozen=True)
class AccItrPlotSpec:
    title: str
    accuracy_title: str
    itr_title: str
    footnote: str
    y_accuracy: tuple[float, float] = (0.0, 0.95)
    y_itr: tuple[float, float] = (0.0, 180.0)
    method_order: tuple[str, ...] = DEFAULT_METHOD_ORDER


def load_summaries(paths: list[Path], method_order: tuple[str, ...] = DEFAULT_METHOD_ORDER) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    summary = pd.concat(frames, ignore_index=True)
    summary["method"] = pd.Categorical(summary["method"], method_order, ordered=True)
    return summary.sort_values(["method", "window"]).reset_index(drop=True)


def plot_acc_itr(summary: pd.DataFrame, out_dir: Path, spec: AccItrPlotSpec) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)

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
    _plot_metric(axes[0], summary, "accuracy", "accuracy_sem", "Accuracy", spec)
    _plot_metric(axes[1], summary, "itr", "itr_sem", "ITR (bits/min)", spec)
    axes[0].set_ylim(*spec.y_accuracy)
    axes[1].set_ylim(*spec.y_itr)
    axes[0].set_title(spec.accuracy_title)
    axes[1].set_title(spec.itr_title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle(spec.title, y=1.03, fontsize=14)
    fig.text(0.5, -0.025, spec.footnote, ha="center", va="top", fontsize=9, color="#5A6573")
    fig.savefig(out_dir / "acc_itr_compare.png", bbox_inches="tight")
    fig.savefig(out_dir / "acc_itr_compare.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_metric(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    sem: str,
    ylabel: str,
    spec: AccItrPlotSpec,
) -> None:
    for method in spec.method_order:
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
            color=DEFAULT_COLORS.get(method, "#444444"),
            label=method,
        )
    windows = sorted(float(x) for x in summary["window"].unique())
    ax.set_xlabel("Window length (s)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(windows)
    ax.grid(True, axis="y", color="#D8DEE9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
