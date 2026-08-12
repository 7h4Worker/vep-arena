from __future__ import annotations

import json
import sys
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

from vep_arena.channel.capacity import capacity_ba, capacity_c0, mutual_info_uniform  # noqa: E402
from vep_arena.channel.confusion import confusion_counts, normalize_confusion  # noqa: E402
from vep_arena.data.embc_jbhi import dataset_spec  # noqa: E402


BASELINES = {
    "embc9": "BS01_embc_9t",
    "jbhi16": "BS02_16t",
    "jbhi35": "BS03_35t",
}
METHODS = ("ETRCA", "EEG_TDCA", "SPECTRAL_TDCA")


def combined_predictions() -> pd.DataFrame:
    tdca = pd.read_csv(TASK / "results" / "tdca_full" / "predictions.csv")
    frames = [tdca]
    for dataset, task_name in BASELINES.items():
        frame = pd.read_csv(TASK / "results" / task_name / "full" / "predictions.csv")
        frame = frame[frame["method"] == "ETRCA"].copy()
        frame["dataset"] = dataset
        frames.append(frame[["dataset", "subject", "window_seconds", "method", "fold", "true", "pred", "correct"]])
    return pd.concat(frames, ignore_index=True)


def channel_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method, window), group in predictions.groupby(["dataset", "method", "window_seconds"], sort=True):
        spec = dataset_spec(str(dataset))
        counts = confusion_counts(group["true"].to_numpy(dtype=int) - 1, group["pred"].to_numpy(dtype=int) - 1, spec.targets)
        transition = normalize_confusion(counts)
        ba = capacity_ba(transition)
        c0 = capacity_c0(spec.targets)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "window_seconds": window,
                "accuracy": float(group["correct"].mean()),
                "mi_uniform": mutual_info_uniform(transition),
                "c_ba": float(ba.capacity),
                "ba_utilization": float(ba.capacity / c0),
                "ct_practical_bpm": float(60 * ba.capacity / (float(window) + spec.itr_shift_seconds)),
                "predictions": len(group),
            }
        )
    return pd.DataFrame(rows)


def paired_gains(predictions: pd.DataFrame) -> pd.DataFrame:
    subject = predictions.groupby(["dataset", "subject", "method", "window_seconds"], as_index=False)["correct"].mean()
    rows = []
    for (dataset, window), group in subject.groupby(["dataset", "window_seconds"]):
        pivot = group.pivot(index="subject", columns="method", values="correct")
        for method in ("EEG_TDCA", "SPECTRAL_TDCA"):
            delta = pivot[method] - pivot["ETRCA"]
            test = ttest_rel(pivot[method], pivot["ETRCA"])
            rows.append(
                {
                    "dataset": dataset,
                    "window_seconds": window,
                    "method": method,
                    "subjects": len(delta),
                    "mean_accuracy_gain": float(delta.mean()),
                    "median_accuracy_gain": float(delta.median()),
                    "subjects_improved": int((delta > 0).sum()),
                    "subjects_tied": int((delta == 0).sum()),
                    "paired_t_pvalue": float(test.pvalue),
                }
            )
    return pd.DataFrame(rows)


def discovery_excluded_gains(predictions: pd.DataFrame) -> pd.DataFrame:
    subject = predictions.groupby(["dataset", "subject", "method", "window_seconds"], as_index=False)["correct"].mean()
    subject = subject[~subject["subject"].isin(["S04", "S06"])]
    rows = []
    for (dataset, window), group in subject.groupby(["dataset", "window_seconds"]):
        pivot = group.pivot(index="subject", columns="method", values="correct")
        delta = pivot["SPECTRAL_TDCA"] - pivot["ETRCA"]
        test = ttest_rel(pivot["SPECTRAL_TDCA"], pivot["ETRCA"])
        rows.append(
            {
                "dataset": dataset,
                "window_seconds": window,
                "subjects": len(delta),
                "etrca_accuracy": float(pivot["ETRCA"].mean()),
                "spectral_tdca_accuracy": float(pivot["SPECTRAL_TDCA"].mean()),
                "mean_accuracy_gain": float(delta.mean()),
                "subjects_improved": int((delta > 0).sum()),
                "paired_t_pvalue": float(test.pvalue),
                "excluded_discovery_subjects": "S04,S06",
            }
        )
    return pd.DataFrame(rows)


def plot_capacity(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True, sharey=True)
    for axis, dataset in zip(axes, BASELINES):
        group = metrics[metrics["dataset"] == dataset]
        for method, method_group in group.groupby("method"):
            axis.plot(method_group["window_seconds"], 100 * method_group["ba_utilization"], marker="o", label=method)
        axis.set_title(dataset)
        axis.set_xlabel("Window (s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Decision-channel utilization (%)")
    axes[-1].legend()
    fig.savefig(output, dpi=210)
    plt.close(fig)


def plot_subject_gains(predictions: pd.DataFrame, output: Path) -> None:
    subject = predictions.groupby(["dataset", "subject", "method", "window_seconds"], as_index=False)["correct"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True, sharey=True)
    for axis, dataset in zip(axes, BASELINES):
        group = subject[(subject["dataset"] == dataset) & (subject["window_seconds"] == 2.0)]
        pivot = group.pivot(index="subject", columns="method", values="correct")
        for index, method in enumerate(("EEG_TDCA", "SPECTRAL_TDCA")):
            delta = pivot[method] - pivot["ETRCA"]
            axis.scatter(np.full(len(delta), index) + np.linspace(-0.08, 0.08, len(delta)), 100 * delta, label=method)
            axis.plot([index - 0.18, index + 0.18], [100 * delta.mean()] * 2, color="black", lw=2)
        axis.axhline(0, color="gray", lw=1)
        axis.set_xticks((0, 1), ("EEG", "Spectral"))
        axis.set_title(dataset)
        axis.set_xlabel("TDCA reference")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Accuracy gain over eTRCA (percentage points)")
    fig.savefig(output, dpi=210)
    plt.close(fig)


def main() -> None:
    output = TASK / "results" / "tdca_full"
    figures = output / "figures"
    predictions = combined_predictions()
    metrics = channel_metrics(predictions)
    gains = paired_gains(predictions)
    holdout_gains = discovery_excluded_gains(predictions)
    metrics.to_csv(output / "channel_metrics.csv", index=False)
    gains.to_csv(output / "paired_gains.csv", index=False)
    holdout_gains.to_csv(output / "discovery_excluded_gains.csv", index=False)
    plot_capacity(metrics, figures / "tdca_decision_channel_utilization.png")
    plot_subject_gains(predictions, figures / "tdca_subject_gains_2s.png")
    analysis_manifest = {
        "status": "complete",
        "methods": list(METHODS),
        "prediction_rows_analyzed": len(predictions),
        "primary_channel_estimator": "pooled confusion across subjects",
        "paired_test": "subject-level paired t-test; exploratory across windows",
        "figures": ["tdca_decision_channel_utilization.png", "tdca_subject_gains_2s.png"],
        "tables": ["channel_metrics.csv", "paired_gains.csv", "discovery_excluded_gains.csv"],
    }
    (output / "analysis_manifest.json").write_text(json.dumps(analysis_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(analysis_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
