from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TASK = Path(__file__).resolve().parent
DEFAULT_SUMMARY = TASK / "results" / "formal_full_20260717" / "summary.csv"

EXPERIMENT_TITLES = {
    "exp1": "Experiment 1: 6-target code optimization",
    "exp2": "Experiment 2: 40-target code comparison",
    "exp3": "Experiment 3: offline dual-frequency comparison",
    "exp4": "Experiment 4: online-style dual vs single frequency",
}
CONDITION_LABELS = {
    "method1_fast_gradient_descent": "M1 fast gradient descent",
    "method2_fast_optimization": "M2 fast optimization",
    "method3_global_search": "M3 global search",
    "method4_random_worst": "M4 random worst",
    "method4_random_median": "M4 random median",
    "method4_random_best": "M4 random best",
    "longest_frequency_distance_zero_phase": "Max frequency distance, zero phase",
    "method1_frequency_pairs_zero_phase": "M1 frequency pair, zero phase",
    "method1_dual_frequency_phase": "M1 dual frequency + phase",
    "single_frequency_phase_jfpm": "Single frequency phase JFPM",
}
COLORS = ("#007C91", "#D06D2F", "#6B5CA5", "#A34B70")
METHOD_STYLES = {
    "TRCA": {"linestyle": "--", "marker": "o"},
    "ETRCA": {"linestyle": "-", "marker": "s"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Liang2020 accuracy and ITR window curves.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "experiment",
        "condition",
        "condition_label",
        "method",
        "window",
        "accuracy",
        "itr_bits_per_min",
        "subject",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Summary lacks required columns: {sorted(missing)}")

    grouped = (
        summary.groupby(
            ["experiment", "condition", "condition_label", "method", "window"],
            sort=True,
            observed=True,
        )
        .agg(
            subjects=("subject", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            itr_mean=("itr_bits_per_min", "mean"),
            itr_std=("itr_bits_per_min", "std"),
        )
        .reset_index()
    )
    grouped["accuracy_sem"] = grouped["accuracy_std"] / np.sqrt(grouped["subjects"])
    grouped["itr_sem"] = grouped["itr_std"] / np.sqrt(grouped["subjects"])
    return grouped


def plot_metric(curves: pd.DataFrame, metric: str, sem: str, ylabel: str, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9.5), constrained_layout=True)
    for axis, experiment in zip(axes.flat, ("exp1", "exp2", "exp3", "exp4"), strict=True):
        rows = curves[curves["experiment"] == experiment]
        conditions = sorted(rows["condition"].unique())
        for color, condition in zip(COLORS, conditions, strict=False):
            condition_rows = rows[rows["condition"] == condition]
            label = CONDITION_LABELS.get(condition_rows["condition_label"].iloc[0], condition_rows["condition_label"].iloc[0])
            for method in ("TRCA", "ETRCA"):
                line = condition_rows[condition_rows["method"] == method].sort_values("window")
                if line.empty:
                    continue
                style = METHOD_STYLES[method]
                axis.plot(
                    line["window"],
                    line[metric],
                    color=color,
                    linewidth=2.0,
                    markersize=4.5,
                    label=f"{label} | {method}",
                    **style,
                )
                axis.fill_between(
                    line["window"].to_numpy(),
                    (line[metric] - line[sem]).to_numpy(),
                    (line[metric] + line[sem]).to_numpy(),
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
        axis.set_title(EXPERIMENT_TITLES[experiment])
        axis.set_xlabel("Analysis window (s)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(sorted(rows["window"].unique()))
        axis.grid(True, axis="y", color="#D8DEE9", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(fontsize=7.7, frameon=False, loc="best")

    figure.suptitle("Liang2020: subject-mean window curves (shaded: SEM)", fontsize=15)
    figure.text(
        0.5,
        -0.01,
        "9 channels, 1000 Hz, 140 ms latency removal, five filter bands. "
        "Exp3: leave-one-block-out; Exp4: runs 1-6 train, 7-9 test. ITR denominator: window + 0.5 s.",
        ha="center",
        va="top",
        fontsize=9,
        color="#4F5B66",
    )
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    summary_path = args.summary.resolve()
    output = args.output or summary_path.parent / "figures"
    output.mkdir(parents=True, exist_ok=True)

    curves = aggregate(pd.read_csv(summary_path))
    curves.to_csv(output / "window_curve_summary.csv", index=False)
    plot_metric(curves, "accuracy_mean", "accuracy_sem", "Accuracy", output / "accuracy_window_curves")
    plot_metric(curves, "itr_mean", "itr_sem", "ITR (bits/min)", output / "itr_window_curves")


if __name__ == "__main__":
    main()
