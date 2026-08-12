from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vep_arena.channel.asymmetry import asymmetry_score
from vep_arena.channel.capacity import capacity_ba, capacity_c0, capacity_c1, mutual_info_uniform
from vep_arena.channel.confusion import confusion_counts, normalize_confusion
from vep_arena.data.embc_jbhi import dataset_spec, load_target_frequency_pairs


@dataclass(frozen=True)
class StudyDataset:
    key: str
    label: str
    task: str
    evidence_role: str
    protocol_status: str

    @property
    def result_dir(self) -> Path:
        return TASK / "results" / self.task / "full"


DATASETS = (
    StudyDataset(
        "embc9",
        "EMBC 9",
        "BS01_embc_9t",
        "published-code-audited-reproduction",
        "legacy-code-verified-block-cv",
    ),
    StudyDataset("jbhi16", "JBHI 16", "BS02_16t", "paper-reproduction", "paper-verified"),
    StudyDataset("jbhi35", "JBHI 35", "BS03_35t", "diagnostic-extension", "operator-confirmed"),
)

METHODS = ("TRCA", "ETRCA", "BPRCA", "EBPRCA", "FUSIONCA", "EFUSIONCA")
ENSEMBLE_METHODS = ("ETRCA", "EBPRCA", "EFUSIONCA")
METHOD_COLORS = {"ETRCA": "#377eb8", "EBPRCA": "#ff9f1c", "EFUSIONCA": "#d62728"}
METHOD_MARKERS = {"ETRCA": "o", "EBPRCA": "s", "EFUSIONCA": "^"}
ERROR_TYPES = ("eye_swap", "void_structure", "one_eye_preserved", "both_units_wrong")
ERROR_LABELS = {
    "eye_swap": "Eye swap",
    "void_structure": "Void structure",
    "one_eye_preserved": "One eye preserved",
    "both_units_wrong": "Both units wrong",
}


def sem(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0


def empirical_mutual_information(true_values: np.ndarray, pred_values: np.ndarray) -> float:
    true_values = np.asarray(true_values)
    pred_values = np.asarray(pred_values)
    true_levels, true_inverse = np.unique(true_values, return_inverse=True)
    pred_levels, pred_inverse = np.unique(pred_values, return_inverse=True)
    joint = np.zeros((true_levels.size, pred_levels.size), dtype=np.float64)
    np.add.at(joint, (true_inverse, pred_inverse), 1.0)
    joint /= joint.sum()
    input_prob = joint.sum(axis=1, keepdims=True)
    output_prob = joint.sum(axis=0, keepdims=True)
    expected = input_prob @ output_prob
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / expected[mask])))


def code_category(pair: tuple[float, float]) -> str:
    left, right = pair
    if left == 0 and right == 0:
        return "void_void"
    if (left == 0) != (right == 0):
        return "monocular"
    if left == right:
        return "binocular_same"
    return "binocular_mixed"


def read_complete_predictions(config: StudyDataset) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest_path = config.result_dir / "manifest.json"
    predictions_path = config.result_dir / "predictions.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"{config.key} manifest is not complete")
    if int(manifest.get("completed_predictions", -1)) != int(manifest.get("expected_predictions", -2)):
        raise RuntimeError(f"{config.key} prediction count is incomplete")
    if int(manifest.get("error_count", -1)) != 0:
        raise RuntimeError(f"{config.key} manifest contains errors")
    predictions = pd.read_csv(predictions_path)
    required = {"dataset", "subject", "method", "window_seconds", "true", "pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{config.key} predictions missing columns: {missing}")
    spec = dataset_spec(config.key)
    if len(predictions) != int(manifest["expected_predictions"]):
        raise RuntimeError(f"{config.key} predictions.csv row count disagrees with manifest")
    if predictions["true"].min() != 1 or predictions["true"].max() != spec.targets:
        raise ValueError(f"{config.key} true labels are not 1-{spec.targets}")
    if predictions["pred"].min() != 1 or predictions["pred"].max() != spec.targets:
        raise ValueError(f"{config.key} predicted labels are not 1-{spec.targets}")
    return predictions, manifest


def channel_row(group: pd.DataFrame, config: StudyDataset, subject: str) -> dict[str, object]:
    spec = dataset_spec(config.key)
    true_zero = group["true"].to_numpy(dtype=np.int64) - 1
    pred_zero = group["pred"].to_numpy(dtype=np.int64) - 1
    counts = confusion_counts(true_zero, pred_zero, spec.targets)
    transition = normalize_confusion(counts)
    accuracy = float(np.mean(true_zero == pred_zero))
    ba = capacity_ba(transition)
    i_uniform = mutual_info_uniform(transition)
    window = float(group["window_seconds"].iloc[0])
    c0 = capacity_c0(spec.targets)
    return {
        "dataset": config.key,
        "dataset_label": config.label,
        "evidence_role": config.evidence_role,
        "protocol_status": config.protocol_status,
        "subject": subject,
        "method": str(group["method"].iloc[0]),
        "window_seconds": window,
        "targets": spec.targets,
        "samples": int(len(group)),
        "accuracy": accuracy,
        "c0": c0,
        "c1": capacity_c1(spec.targets, accuracy),
        "mi_uniform": i_uniform,
        "c_ba": float(ba.capacity),
        "mi_utilization": i_uniform / c0,
        "ba_utilization": float(ba.capacity / c0),
        "ba_minus_mi": float(ba.capacity - i_uniform),
        "asymmetry": asymmetry_score(transition),
        "ct_eeg_bpm": float(60.0 * ba.capacity / window),
        "ct_practical_bpm": float(60.0 * ba.capacity / (window + spec.itr_shift_seconds)),
        "itr_shift_seconds": spec.itr_shift_seconds,
        "ba_converged": bool(ba.converged),
        "ba_gap": float(ba.gap),
    }


def factor_row(
    group: pd.DataFrame,
    config: StudyDataset,
    pairs: tuple[tuple[float, float], ...],
    subject: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    true_idx = group["true"].to_numpy(dtype=np.int64) - 1
    pred_idx = group["pred"].to_numpy(dtype=np.int64) - 1
    pair_array = np.asarray(pairs, dtype=np.float64)
    true_pair = pair_array[true_idx]
    pred_pair = pair_array[pred_idx]
    true_left, true_right = true_pair[:, 0], true_pair[:, 1]
    pred_left, pred_right = pred_pair[:, 0], pred_pair[:, 1]
    exact = true_idx == pred_idx
    left_correct = true_left == pred_left
    right_correct = true_right == pred_right
    swapped = (~exact) & (true_left == pred_right) & (true_right == pred_left)
    true_voids = (true_left == 0).astype(int) + (true_right == 0).astype(int)
    pred_voids = (pred_left == 0).astype(int) + (pred_right == 0).astype(int)
    errors = ~exact
    taxonomy = np.full(len(group), "correct", dtype=object)
    taxonomy[errors] = "both_units_wrong"
    taxonomy[errors & (left_correct | right_correct)] = "one_eye_preserved"
    taxonomy[errors & (true_voids != pred_voids)] = "void_structure"
    taxonomy[swapped] = "eye_swap"
    error_count = int(errors.sum())
    row = {
        "dataset": config.key,
        "dataset_label": config.label,
        "subject": subject,
        "method": str(group["method"].iloc[0]),
        "window_seconds": float(group["window_seconds"].iloc[0]),
        "samples": int(len(group)),
        "accuracy": float(exact.mean()),
        "left_accuracy": float(left_correct.mean()),
        "right_accuracy": float(right_correct.mean()),
        "either_eye_accuracy": float((left_correct | right_correct).mean()),
        "eye_swap_rate_all": float(swapped.mean()),
        "eye_swap_share_errors": float(swapped.sum() / error_count) if error_count else 0.0,
        "joint_mi": empirical_mutual_information(true_idx, pred_idx),
        "left_mi": empirical_mutual_information(true_left, pred_left),
        "right_mi": empirical_mutual_information(true_right, pred_right),
        "left_to_right_hat_mi": empirical_mutual_information(true_left, pred_right),
        "right_to_left_hat_mi": empirical_mutual_information(true_right, pred_left),
    }
    row["joint_minus_marginal_sum"] = float(row["joint_mi"] - row["left_mi"] - row["right_mi"])
    taxonomy_rows = []
    for error_type in ERROR_TYPES:
        count = int(np.sum(taxonomy == error_type))
        taxonomy_rows.append(
            {
                "dataset": config.key,
                "dataset_label": config.label,
                "subject": subject,
                "method": row["method"],
                "window_seconds": row["window_seconds"],
                "error_type": error_type,
                "count": count,
                "share_all": count / len(group),
                "share_errors": count / error_count if error_count else 0.0,
            }
        )
    categories = tuple(dict.fromkeys(code_category(pair) for pair in pairs))
    category_to_index = {category: index for index, category in enumerate(categories)}
    true_categories = np.asarray([category_to_index[code_category(tuple(pair))] for pair in true_pair])
    pred_categories = np.asarray([category_to_index[code_category(tuple(pair))] for pair in pred_pair])
    category_counts = confusion_counts(true_categories, pred_categories, len(categories))
    category_rows = []
    for true_category, true_category_idx in category_to_index.items():
        for pred_category, pred_category_idx in category_to_index.items():
            category_rows.append(
                {
                    "dataset": config.key,
                    "subject": subject,
                    "method": row["method"],
                    "window_seconds": row["window_seconds"],
                    "true_category": true_category,
                    "pred_category": pred_category,
                    "count": int(category_counts[true_category_idx, pred_category_idx]),
                }
            )
    return row, taxonomy_rows, category_rows


def aggregate_subject_rows(
    subject_rows: pd.DataFrame,
    metric_columns: tuple[str, ...],
    extra_group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["dataset", "dataset_label", "method", "window_seconds", *extra_group_columns]
    for keys, group in subject_rows.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row["subjects"] = int(group["subject"].nunique())
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=np.float64)
            row[metric] = float(values.mean())
            row[f"{metric}_sem"] = sem(values)
        rows.append(row)
    return pd.DataFrame(rows)


def receiver_gain_table(channel_subject: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ("accuracy", "mi_uniform", "c_ba", "ba_utilization", "ct_practical_bpm")
    for dataset in channel_subject["dataset"].unique():
        dataset_rows = channel_subject[channel_subject["dataset"] == dataset]
        for window in sorted(dataset_rows["window_seconds"].unique()):
            baseline = dataset_rows[(dataset_rows["window_seconds"] == window) & (dataset_rows["method"] == "ETRCA")]
            for method in ("EBPRCA", "EFUSIONCA"):
                comparison = dataset_rows[(dataset_rows["window_seconds"] == window) & (dataset_rows["method"] == method)]
                merged = baseline.merge(comparison, on="subject", suffixes=("_baseline", "_comparison"))
                if merged.empty:
                    continue
                for metric in metrics:
                    baseline_values = merged[f"{metric}_baseline"].to_numpy(dtype=np.float64)
                    comparison_values = merged[f"{metric}_comparison"].to_numpy(dtype=np.float64)
                    delta = comparison_values - baseline_values
                    test = ttest_rel(comparison_values, baseline_values)
                    rows.append(
                        {
                            "dataset": dataset,
                            "window_seconds": float(window),
                            "baseline": "ETRCA",
                            "method": method,
                            "metric": metric,
                            "subjects": int(delta.size),
                            "baseline_mean": float(baseline_values.mean()),
                            "comparison_mean": float(comparison_values.mean()),
                            "delta_mean": float(delta.mean()),
                            "delta_sem": sem(delta),
                            "paired_t_pvalue": float(test.pvalue),
                        }
                    )
    return pd.DataFrame(rows)


def pooled_receiver_gain_table(channel_pooled: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ("accuracy", "mi_uniform", "c_ba", "ba_utilization", "ct_practical_bpm")
    for dataset in channel_pooled["dataset"].unique():
        dataset_rows = channel_pooled[channel_pooled["dataset"] == dataset]
        for window in sorted(dataset_rows["window_seconds"].unique()):
            baseline = dataset_rows[(dataset_rows["window_seconds"] == window) & (dataset_rows["method"] == "ETRCA")].iloc[0]
            for method in ("EBPRCA", "EFUSIONCA"):
                comparison = dataset_rows[(dataset_rows["window_seconds"] == window) & (dataset_rows["method"] == method)].iloc[0]
                for metric in metrics:
                    rows.append(
                        {
                            "dataset": dataset,
                            "window_seconds": float(window),
                            "baseline": "ETRCA",
                            "method": method,
                            "metric": metric,
                            "baseline_value": float(baseline[metric]),
                            "comparison_value": float(comparison[metric]),
                            "delta": float(comparison[metric] - baseline[metric]),
                        }
                    )
    return pd.DataFrame(rows)


def plot_codebooks(codebook: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, config in zip(axes, DATASETS):
        group = codebook[codebook["dataset"] == config.key]
        units = sorted(set(group["left_freq_hz"]) | set(group["right_freq_hz"]))
        unit_index = {unit: index for index, unit in enumerate(units)}
        left_positions = group["left_freq_hz"].map(unit_index)
        right_positions = group["right_freq_hz"].map(unit_index)
        ax.scatter(left_positions, right_positions, s=52, color="#377eb8", edgecolor="black", lw=0.4)
        for row in group.itertuples():
            ax.annotate(str(row.target), (unit_index[row.left_freq_hz], unit_index[row.right_freq_hz]), xytext=(2, 3), textcoords="offset points", fontsize=6)
        limit = len(units) - 1
        labels = ["void" if unit == 0 else f"{unit:g}" for unit in units]
        ax.plot([0, limit], [0, limit], color="gray", lw=0.8, ls="--")
        ax.set(title=f"{config.label}: {len(group)} ordered pairs", xlabel="Left-eye unit", ylabel="Right-eye unit", xlim=(-0.5, limit + 0.5), ylim=(-0.5, limit + 0.5), xticks=range(len(units)), yticks=range(len(units)), xticklabels=labels, yticklabels=labels)
        ax.grid(alpha=0.22)
    fig.suptitle("Binocular Cartesian-product codebooks")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig01_codebook_lattices.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_curves(channel_pooled: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex="row")
    for row_index, config in enumerate(DATASETS):
        dataset_rows = channel_pooled[channel_pooled["dataset"] == config.key]
        for method in ENSEMBLE_METHODS:
            group = dataset_rows[dataset_rows["method"] == method].sort_values("window_seconds")
            style = dict(color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], lw=1.7, ms=4, label=method)
            axes[row_index, 0].plot(group["window_seconds"], group["mi_uniform"], **style)
            axes[row_index, 1].plot(group["window_seconds"], 100 * group["ba_utilization"], **style)
        axes[row_index, 0].set(ylabel="MI (bits/selection)", title=f"{config.label}: recoverable information")
        axes[row_index, 1].set(ylabel="C_BA / log2(M) (%)", title=f"{config.label}: codebook utilization")
        for ax in axes[row_index]:
            ax.grid(alpha=0.22)
            ax.legend(frameon=False)
            ax.set_xlabel("EEG window (s)")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig02_capacity_utilization_curves.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_time(channel_pooled: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex="row")
    for row_index, config in enumerate(DATASETS):
        dataset_rows = channel_pooled[channel_pooled["dataset"] == config.key]
        for method in ENSEMBLE_METHODS:
            group = dataset_rows[dataset_rows["method"] == method].sort_values("window_seconds")
            style = dict(color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], lw=1.7, ms=4, label=method)
            axes[row_index, 0].plot(group["window_seconds"], group["ct_eeg_bpm"], **style)
            axes[row_index, 1].plot(group["window_seconds"], group["ct_practical_bpm"], **style)
        axes[row_index, 0].set(ylabel="60 C_BA / T_EEG", title=f"{config.label}: EEG-only C_T")
        axes[row_index, 1].set(ylabel="60 C_BA / T_selection", title=f"{config.label}: protocol C_T")
        for ax in axes[row_index]:
            ax.grid(alpha=0.22)
            ax.legend(frameon=False)
            ax.set_xlabel("EEG window (s)")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig03_capacity_time_curves.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_receiver_gain(receiver_gain_pooled: pd.DataFrame, figure_dir: Path) -> None:
    selected = receiver_gain_pooled[(receiver_gain_pooled["metric"] == "mi_uniform") & (receiver_gain_pooled["window_seconds"].isin([1.0, 2.0]))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, config in zip(axes, DATASETS):
        group = selected[selected["dataset"] == config.key]
        x = np.arange(len(group["window_seconds"].unique()))
        width = 0.34
        for method_index, method in enumerate(("EBPRCA", "EFUSIONCA")):
            rows = group[group["method"] == method].sort_values("window_seconds")
            ax.bar(x + (method_index - 0.5) * width, rows["delta"], width, label=f"{method} - ETRCA", color=METHOD_COLORS[method])
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(title=config.label, xticks=x, xticklabels=[f"{window:g}s" for window in sorted(group["window_seconds"].unique())], xlabel="Window", ylabel="Paired ΔMI (bits/selection)")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Receiver gain at matched codebook and protocol")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig04_receiver_gain.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_factor_information(factor_pooled: pd.DataFrame, figure_dir: Path) -> None:
    selected = factor_pooled[(factor_pooled["window_seconds"] == 2.0) & (factor_pooled["method"].isin(ENSEMBLE_METHODS))]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=False)
    metrics = ("left_mi", "right_mi", "joint_mi")
    labels = ("Left", "Right", "Joint target")
    for ax, config in zip(axes, DATASETS):
        group = selected[selected["dataset"] == config.key]
        x = np.arange(len(metrics))
        width = 0.24
        for method_index, method in enumerate(ENSEMBLE_METHODS):
            row = group[group["method"] == method].iloc[0]
            values = [row[metric] for metric in metrics]
            ax.bar(x + (method_index - 1) * width, values, width, color=METHOD_COLORS[method], label=method)
        ax.set(title=f"{config.label} at 2.0 s", xticks=x, xticklabels=labels, ylabel="Mutual information (bits)")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Ordered binocular factors and joint target information")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig05_factor_information.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_error_topology(error_pooled: pd.DataFrame, figure_dir: Path) -> None:
    selected = error_pooled[(error_pooled["window_seconds"] == 2.0) & (error_pooled["method"].isin(ENSEMBLE_METHODS))]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    colors = ("#9467bd", "#17becf", "#ff7f0e", "#7f7f7f")
    for ax, config in zip(axes, DATASETS):
        group = selected[selected["dataset"] == config.key]
        x = np.arange(len(ENSEMBLE_METHODS))
        bottom = np.zeros(len(ENSEMBLE_METHODS))
        for error_type, color in zip(ERROR_TYPES, colors):
            values = np.asarray([
                group[(group["method"] == method) & (group["error_type"] == error_type)]["share_errors"].iloc[0]
                for method in ENSEMBLE_METHODS
            ])
            ax.bar(x, 100 * values, bottom=100 * bottom, color=color, label=ERROR_LABELS[error_type])
            bottom += values
        ax.set(title=f"{config.label} at 2.0 s", xticks=x, xticklabels=ENSEMBLE_METHODS, ylabel="Share of errors (%)")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Error topology in ordered binocular code space")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig06_error_topology.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_subject_accuracy(channel_subject: pd.DataFrame, figure_dir: Path) -> None:
    selected = channel_subject[(channel_subject["window_seconds"] == 2.0) & (channel_subject["method"] == "EFUSIONCA")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    rng = np.random.default_rng(13)
    for ax, config in zip(axes, DATASETS):
        values = selected[selected["dataset"] == config.key]["accuracy"].to_numpy(dtype=np.float64)
        x = rng.normal(0, 0.035, size=values.size)
        ax.scatter(x, 100 * values, s=35, color="#d62728", alpha=0.8)
        ax.boxplot(100 * values, positions=[0], widths=0.25, showfliers=False)
        ax.set(title=f"{config.label}: n={len(values)}", xlim=(-0.35, 0.35), xticks=[], ylabel="Subject accuracy (%)")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Subject heterogeneity under eFusionCA at 2.0 s")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig07_subject_accuracy.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_category_confusion(category_rows: pd.DataFrame, figure_dir: Path) -> None:
    selected = category_rows[(category_rows["subject"] == "all") & (category_rows["window_seconds"] == 2.0) & (category_rows["method"] == "EFUSIONCA")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    image = None
    for ax, config in zip(axes, DATASETS):
        group = selected[selected["dataset"] == config.key]
        categories = list(dict.fromkeys(group["true_category"]))
        matrix = group.pivot(index="true_category", columns="pred_category", values="count").reindex(index=categories, columns=categories).fillna(0).to_numpy(dtype=float).copy()
        matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(column, row, f"{100 * matrix[row, column]:.0f}", ha="center", va="center", fontsize=8)
        labels = [category.replace("binocular_", "bin-").replace("void_void", "void") for category in categories]
        ax.set(title=config.label, xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Predicted family", ylabel="True family")
        ax.tick_params(axis="x", rotation=25)
    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), shrink=0.72, pad=0.02, label="Conditional probability")
    fig.suptitle("Code-family decision channels under eFusionCA at 2.0 s")
    fig.savefig(figure_dir / "fig08_category_confusion.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze EMBC/JBHI binocular codebooks as empirical decision channels.")
    parser.add_argument("--output", type=Path, default=TASK / "results" / "BS04_codebook" / "full")
    args = parser.parse_args()
    output = args.output.resolve()
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    codebook_rows: list[dict[str, object]] = []
    channel_subject_rows: list[dict[str, object]] = []
    channel_aggregate_direct_rows: list[dict[str, object]] = []
    factor_subject_rows: list[dict[str, object]] = []
    factor_aggregate_direct_rows: list[dict[str, object]] = []
    taxonomy_subject_rows: list[dict[str, object]] = []
    taxonomy_aggregate_direct_rows: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []
    source_manifests: dict[str, object] = {}

    for config in DATASETS:
        predictions, manifest = read_complete_predictions(config)
        source_manifests[config.key] = {
            "path": str(config.result_dir / "manifest.json"),
            "status": manifest["status"],
            "predictions": int(manifest["completed_predictions"]),
            "units": int(manifest["completed_units"]),
            "errors": int(manifest["error_count"]),
        }
        pairs = load_target_frequency_pairs(config.key)
        spec = dataset_spec(config.key)
        if len(pairs) != spec.targets or len(set(pairs)) != spec.targets:
            raise ValueError(f"{config.key} codebook is not a unique {spec.targets}-target mapping")
        for target, pair in enumerate(pairs, start=1):
            codebook_rows.append(
                {
                    "dataset": config.key,
                    "dataset_label": config.label,
                    "target": target,
                    "left_freq_hz": pair[0],
                    "right_freq_hz": pair[1],
                    "category": code_category(pair),
                }
            )
        for (method, window, subject), group in predictions.groupby(["method", "window_seconds", "subject"], sort=True):
            channel_subject_rows.append(channel_row(group, config, str(subject)))
            factor, taxonomy, _ = factor_row(group, config, pairs, str(subject))
            factor_subject_rows.append(factor)
            taxonomy_subject_rows.extend(taxonomy)
        for (method, window), group in predictions.groupby(["method", "window_seconds"], sort=True):
            channel_aggregate_direct_rows.append(channel_row(group, config, "all"))
            factor, taxonomy, categories = factor_row(group, config, pairs, "all")
            factor_aggregate_direct_rows.append(factor)
            taxonomy_aggregate_direct_rows.extend(taxonomy)
            category_rows.extend(categories)

    codebook = pd.DataFrame(codebook_rows)
    channel_subject = pd.DataFrame(channel_subject_rows)
    factor_subject = pd.DataFrame(factor_subject_rows)
    taxonomy_subject = pd.DataFrame(taxonomy_subject_rows)
    channel_aggregate = aggregate_subject_rows(
        channel_subject,
        ("accuracy", "c1", "mi_uniform", "c_ba", "mi_utilization", "ba_utilization", "ba_minus_mi", "asymmetry", "ct_eeg_bpm", "ct_practical_bpm"),
    )
    factor_aggregate = aggregate_subject_rows(
        factor_subject,
        ("accuracy", "left_accuracy", "right_accuracy", "either_eye_accuracy", "eye_swap_rate_all", "eye_swap_share_errors", "joint_mi", "left_mi", "right_mi", "left_to_right_hat_mi", "right_to_left_hat_mi", "joint_minus_marginal_sum"),
    )
    error_aggregate = aggregate_subject_rows(
        taxonomy_subject,
        ("share_all", "share_errors"),
        ("error_type",),
    )
    receiver_gain = receiver_gain_table(channel_subject)
    channel_pooled = pd.DataFrame(channel_aggregate_direct_rows)
    factor_pooled = pd.DataFrame(factor_aggregate_direct_rows)
    error_pooled = pd.DataFrame(taxonomy_aggregate_direct_rows)
    receiver_gain_pooled = pooled_receiver_gain_table(channel_pooled)

    codebook.to_csv(table_dir / "codebook.csv", index=False)
    channel_subject.to_csv(table_dir / "channel_metrics_subject.csv", index=False)
    channel_aggregate.to_csv(table_dir / "channel_metrics_group.csv", index=False)
    channel_pooled.to_csv(table_dir / "channel_metrics_pooled.csv", index=False)
    factor_subject.to_csv(table_dir / "factor_metrics_subject.csv", index=False)
    factor_aggregate.to_csv(table_dir / "factor_metrics_group.csv", index=False)
    factor_pooled.to_csv(table_dir / "factor_metrics_pooled.csv", index=False)
    taxonomy_subject.to_csv(table_dir / "error_taxonomy_subject.csv", index=False)
    error_aggregate.to_csv(table_dir / "error_taxonomy_group.csv", index=False)
    error_pooled.to_csv(table_dir / "error_taxonomy_pooled.csv", index=False)
    category_frame = pd.DataFrame(category_rows)
    category_frame.to_csv(table_dir / "category_confusion_pooled.csv", index=False)
    receiver_gain.to_csv(table_dir / "receiver_gain.csv", index=False)
    receiver_gain_pooled.to_csv(table_dir / "receiver_gain_pooled.csv", index=False)

    plot_codebooks(codebook, figure_dir)
    plot_capacity_curves(channel_pooled, figure_dir)
    plot_capacity_time(channel_pooled, figure_dir)
    plot_receiver_gain(receiver_gain_pooled, figure_dir)
    plot_factor_information(factor_pooled, figure_dir)
    plot_error_topology(error_pooled, figure_dir)
    plot_subject_accuracy(channel_subject, figure_dir)
    plot_category_confusion(category_frame, figure_dir)

    manifest = {
        "status": "complete",
        "study": "BS04_codebook",
        "source_manifests": source_manifests,
        "datasets": [config.key for config in DATASETS],
        "methods": list(METHODS),
        "channel_subject_rows": int(len(channel_subject)),
        "channel_group_rows": int(len(channel_aggregate)),
        "factor_subject_rows": int(len(factor_subject)),
        "receiver_gain_rows": int(len(receiver_gain)),
        "figures": sorted(path.name for path in figure_dir.glob("*.png")),
        "tables": sorted(path.name for path in table_dir.glob("*.csv")),
        "interpretation": {
            "decision_channel": "receiver- and protocol-conditioned empirical channel",
            "ct_eeg_bpm": "60*C_BA/EEG_window",
            "ct_practical_bpm": "60*C_BA/(EEG_window+dataset_itr_shift)",
            "joint_minus_marginal_sum": "descriptive non-additivity, not a formal PID synergy estimate",
            "primary_group_estimator": "pooled confusion counts across subjects",
            "subject_capacity_warning": "subject-level plug-in MI and C_BA are exploratory because each class has few trials, especially JBHI35",
        },
    }
    (output / "study_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
