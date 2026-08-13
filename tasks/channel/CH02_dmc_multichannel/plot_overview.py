from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


TASK_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = (
    TASK_ROOT.parent
    / "CH01_dmc_benchmark"
    / "results"
    / "extended"
    / "multichannel"
)

METHODS = ("CCA", "FBCCA", "ECCA", "TRCA", "ETRCA")
CONFIGS = (
    "occipital9",
    "posterior21",
    "posterior32",
    "wholehead32",
    "full64",
)
CONFIG_LABELS = {
    "occipital9": "Occipital 9",
    "posterior21": "Posterior 21",
    "posterior32": "Posterior 32",
    "wholehead32": "Whole-head 32",
    "full64": "Full 64",
}
CONFIG_STYLES = {
    "occipital9": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "posterior21": {"color": "#E69F00", "linestyle": "-", "marker": "s"},
    "posterior32": {"color": "#009E73", "linestyle": "-", "marker": "^"},
    "wholehead32": {"color": "#CC79A7", "linestyle": "--", "marker": "D"},
    "full64": {"color": "#4D4D4D", "linestyle": ":", "marker": "v"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RESULT_ROOT / "decision_channel" / "capacity_aggregate.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULT_ROOT / "figures",
    )
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"method", "channel_config", "window", "accuracy", "c_ba"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    unknown_methods = sorted(set(frame["method"]) - set(METHODS))
    unknown_configs = sorted(set(frame["channel_config"]) - set(CONFIGS))
    if unknown_methods or unknown_configs:
        raise ValueError(
            f"Unexpected result groups: methods={unknown_methods}, configs={unknown_configs}"
        )

    key_columns = ["method", "channel_config", "window"]
    duplicates = int(frame.duplicated(key_columns).sum())
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate method/config/window rows")

    expected_windows = np.round(np.arange(0.1, 5.01, 0.1), 1)
    for method in METHODS:
        for config in CONFIGS:
            rows = frame[
                (frame["method"] == method) & (frame["channel_config"] == config)
            ].sort_values("window")
            observed = rows["window"].to_numpy(dtype=float)
            if observed.shape != expected_windows.shape or not np.allclose(
                observed, expected_windows
            ):
                raise ValueError(f"Incomplete window grid for {method}/{config}")

    frame = frame.copy()
    frame["ba_rate_bpm"] = 60.0 * frame["c_ba"] / frame["window"]
    return frame


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=CONFIG_STYLES[config]["color"],
            linestyle=CONFIG_STYLES[config]["linestyle"],
            marker=CONFIG_STYLES[config]["marker"],
            linewidth=2.0,
            markersize=5,
            label=CONFIG_LABELS[config],
        )
        for config in CONFIGS
    ]


def style_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0.1, 5.0)
    axis.set_xticks(np.arange(0.0, 5.1, 1.0))
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.75, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=9)


def plot_facets(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    subtitle: str,
    output_path: Path,
    *,
    percent: bool = False,
    symlog_scale: bool = False,
) -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.4, 8.0),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.84, wspace=0.08, hspace=0.18)
    plot_axes = list(axes.flat[:5])

    for axis, method in zip(plot_axes, METHODS):
        method_rows = frame[frame["method"] == method]
        for config in CONFIGS:
            rows = method_rows[method_rows["channel_config"] == config].sort_values(
                "window"
            )
            style = CONFIG_STYLES[config]
            axis.plot(
                rows["window"],
                rows[metric],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=5,
                markersize=3.8,
                linewidth=1.9,
            )
        axis.set_title(method, fontsize=12, fontweight="bold", pad=8)
        style_axis(axis)
        if percent:
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            axis.set_ylim(0.0, 1.02)
            axis.axhline(1.0 / 40.0, color="#888888", linewidth=0.9, linestyle="--")
        if symlog_scale:
            axis.set_yscale("symlog", linthresh=10.0, linscale=1.0, base=10)

    if symlog_scale:
        plot_axes[0].set_ylim(0.0, float(frame[metric].max()) * 1.25)

    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel, fontsize=10)
    for axis in axes[1, :2]:
        axis.set_xlabel("Decision window (s)", fontsize=10)

    legend_axis = axes.flat[5]
    legend_axis.axis("off")
    legend_axis.legend(
        handles=legend_handles(),
        loc="center",
        frameon=False,
        fontsize=10,
        handlelength=3.0,
        labelspacing=1.0,
    )
    fig.suptitle(title, y=0.975, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=9.5, color="#555555")
    fig.savefig(output_path, dpi=210, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = load_results(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common_subtitle = "Pooled confusion matrices across 35 subjects; 40 targets; 0.1-5.0 s"
    outputs = (
        args.output_dir / "overview_accuracy_vs_window.png",
        args.output_dir / "overview_ba_capacity_vs_window.png",
        args.output_dir / "overview_ba_rate_vs_window.png",
    )
    plot_facets(
        frame,
        "accuracy",
        "Accuracy",
        "Benchmark multichannel accuracy",
        common_subtitle,
        outputs[0],
        percent=True,
    )
    plot_facets(
        frame,
        "c_ba",
        "BA capacity (bits/decision)",
        "Benchmark multichannel decision-channel capacity",
        common_subtitle,
        outputs[1],
    )
    plot_facets(
        frame,
        "ba_rate_bpm",
        "60 / T x C_BA (bits/min)",
        "Benchmark multichannel BA capacity rate",
        common_subtitle + "; zero-overhead time normalization; symlog y-axis",
        outputs[2],
        symlog_scale=True,
    )

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
