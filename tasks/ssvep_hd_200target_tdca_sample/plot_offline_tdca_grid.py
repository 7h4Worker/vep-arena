from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TASK_DIR / "results" / "offline_tdca_grid"
FIG_DIR = RESULTS_DIR / "figures"
CONF_DIR = RESULTS_DIR / "confusions"
SPATIAL_LABELS = ["right", "down", "left", "up", "center"]


def read_summary() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "summary.csv")
    df["target_set"] = df["target_set"].astype(int)
    df["channel_set"] = df["channel_set"].astype(int)
    df["window_ms"] = df["window_ms"].astype(int)
    return df


def read_trials() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "trials.csv")
    df["target_set"] = df["target_set"].astype(int)
    df["channel_set"] = df["channel_set"].astype(int)
    df["window_ms"] = df["window_ms"].astype(int)
    return df


def plot_metric_vs_window(summary: pd.DataFrame, metric: str, ylabel: str, out_name: str) -> None:
    channels = sorted(summary["channel_set"].unique())
    targets = sorted(summary["target_set"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(targets)))
    for ax, ch in zip(axes.ravel(), channels):
        sub = summary[summary["channel_set"] == ch]
        for color, target in zip(colors, targets):
            rows = sub[sub["target_set"] == target].sort_values("window_ms")
            ax.plot(rows["window_ms"], rows[metric], marker="o", linewidth=1.8, color=color, label=f"{target} targets")
            sem_col = f"{metric.split('_bpm')[0]}_sem" if metric == "itr_bpm" else "accuracy_sem"
            if sem_col in rows:
                y = rows[metric].to_numpy(float)
                e = rows[sem_col].to_numpy(float)
                ax.fill_between(rows["window_ms"], y - e, y + e, color=color, alpha=0.12, linewidth=0)
        ax.set_title(f"{ch} channels")
        ax.set_xlabel("window (ms)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(targets), frameon=False)
    fig.savefig(FIG_DIR / out_name, dpi=180)
    plt.close(fig)


def plot_best_itr_distribution(trials: pd.DataFrame) -> None:
    idx = trials.groupby(["subject", "target_set", "channel_set"])["itr_bpm"].idxmax()
    best = trials.loc[idx].copy()
    counts = best["window_ms"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.bar(counts.index.astype(str), counts.values, color="#40798c")
    ax.set_xlabel("best ITR window (ms)")
    ax.set_ylabel("subject x target-set x channel-set groups")
    ax.set_title("Best ITR Window Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG_DIR / "offline_tdca_best_itr_window_distribution.png", dpi=180)
    plt.close(fig)
    best.to_csv(RESULTS_DIR / "offline_tdca_best_itr_by_subject_config.csv", index=False)


def confusion_slug(subject: str, target_set: int, channel_set: int, window_ms: int) -> str:
    return f"{subject}_{target_set}target_{channel_set}ch_w{window_ms}"


def aggregate_confusion(trials: pd.DataFrame, target_set: int, channel_set: int, window_ms: int) -> np.ndarray:
    mats = []
    rows = trials[
        (trials["target_set"] == target_set)
        & (trials["channel_set"] == channel_set)
        & (trials["window_ms"] == window_ms)
    ]
    for row in rows.itertuples(index=False):
        mats.append(np.load(CONF_DIR / f"confusion_{confusion_slug(row.subject, target_set, channel_set, window_ms)}.npy"))
    return np.sum(mats, axis=0)


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float64)
    row_sum = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)


def plot_representative_confusions(summary: pd.DataFrame, trials: pd.DataFrame) -> None:
    best_200_66 = summary[(summary["target_set"] == 200) & (summary["channel_set"] == 66)].sort_values("itr_bpm").iloc[-1]
    best_overall = summary.sort_values("itr_bpm").iloc[-1]
    configs = [
        (200, 66, 100, "200 targets / 66 ch / 100 ms"),
        (200, 66, 500, "200 targets / 66 ch / 500 ms"),
        (200, 66, int(best_200_66.window_ms), f"200 targets / 66 ch / best ITR {int(best_200_66.window_ms)} ms"),
        (
            int(best_overall.target_set),
            int(best_overall.channel_set),
            int(best_overall.window_ms),
            f"overall best mean ITR: {int(best_overall.target_set)} targets / {int(best_overall.channel_set)} ch / {int(best_overall.window_ms)} ms",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (target_set, channel_set, window_ms, title) in zip(axes.ravel(), configs):
        mat = normalized_rows(aggregate_confusion(trials, target_set, channel_set, window_ms))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="magma", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        ticks = np.linspace(0, target_set - 1, 5, dtype=int)
        ax.set_xticks(ticks, ticks + 1)
        ax.set_yticks(ticks, ticks + 1)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.84, label="row-normalized count")
    fig.savefig(FIG_DIR / "offline_tdca_representative_confusions.png", dpi=180)
    plt.close(fig)


def slot(original_target: int) -> int:
    return (original_target - 1) % 5


def freq_bin(original_target: int) -> int:
    return (original_target - 1) // 5


def error_topology(trials: pd.DataFrame) -> None:
    rows = trials[(trials["target_set"] == 200) & (trials["channel_set"] == 66)]
    out_rows = []
    freq_delta_counts: dict[int, dict[int, int]] = {}
    spatial_mats: dict[int, np.ndarray] = {}
    for window_ms in sorted(rows["window_ms"].unique()):
        mat = aggregate_confusion(trials, 200, 66, int(window_ms))
        errors = mat.copy()
        np.fill_diagonal(errors, 0)
        total = int(mat.sum())
        error_total = int(errors.sum())
        same_freq = 0
        same_slot = 0
        adjacent_freq = 0
        freq_counts: dict[int, int] = {}
        spatial = np.zeros((5, 5), dtype=np.int64)
        for true_idx, pred_idx in np.argwhere(errors > 0):
            count = int(errors[true_idx, pred_idx])
            true_original = true_idx + 1
            pred_original = pred_idx + 1
            fd = freq_bin(pred_original) - freq_bin(true_original)
            ts = slot(true_original)
            ps = slot(pred_original)
            if fd == 0:
                same_freq += count
            if ts == ps:
                same_slot += count
            if abs(fd) == 1:
                adjacent_freq += count
            freq_counts[fd] = freq_counts.get(fd, 0) + count
            spatial[ts, ps] += count
        freq_delta_counts[int(window_ms)] = freq_counts
        spatial_mats[int(window_ms)] = spatial
        out_rows.append(
            {
                "target_set": 200,
                "channel_set": 66,
                "window_ms": int(window_ms),
                "total_trials": total,
                "errors": error_total,
                "accuracy": 1 - error_total / total,
                "same_frequency_error_rate": same_freq / error_total if error_total else 0,
                "same_spatial_slot_error_rate": same_slot / error_total if error_total else 0,
                "adjacent_frequency_error_rate": adjacent_freq / error_total if error_total else 0,
            }
        )

    with (RESULTS_DIR / "offline_tdca_200target_66ch_error_topology.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    topo = pd.DataFrame(out_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    axes[0].plot(topo["window_ms"], topo["same_frequency_error_rate"], marker="o", label="same frequency")
    axes[0].plot(topo["window_ms"], topo["same_spatial_slot_error_rate"], marker="o", label="same spatial slot")
    axes[0].plot(topo["window_ms"], topo["adjacent_frequency_error_rate"], marker="o", label="adjacent frequency")
    axes[0].set_xlabel("window (ms)")
    axes[0].set_ylabel("share of errors")
    axes[0].set_title("200-target / 66ch Error Structure")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    fd = freq_delta_counts[500]
    xs = np.arange(-10, 11)
    axes[1].bar(xs, [fd.get(int(x), 0) for x in xs], color="#6a8d73")
    axes[1].set_xlabel("predicted frequency bin - true frequency bin")
    axes[1].set_ylabel("error count")
    axes[1].set_title("Frequency Error Offsets at 500 ms")
    axes[1].grid(axis="y", alpha=0.25)

    spatial = normalized_rows(spatial_mats[500])
    im = axes[2].imshow(spatial, vmin=0, vmax=max(0.01, float(spatial.max())), cmap="viridis")
    axes[2].set_title("Spatial/Phase Error Topology at 500 ms")
    axes[2].set_xticks(range(5), SPATIAL_LABELS, rotation=30, ha="right")
    axes[2].set_yticks(range(5), SPATIAL_LABELS)
    axes[2].set_xlabel("pred spatial slot")
    axes[2].set_ylabel("true spatial slot")
    fig.colorbar(im, ax=axes[2], shrink=0.8, label="row-normalized errors")
    fig.savefig(FIG_DIR / "offline_tdca_200target_66ch_error_topology.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_summary()
    trials = read_trials()
    plot_metric_vs_window(summary, "accuracy", "accuracy", "offline_tdca_accuracy_vs_window.png")
    plot_metric_vs_window(summary, "itr_bpm", "ITR (bits/min)", "offline_tdca_itr_vs_window.png")
    plot_best_itr_distribution(trials)
    plot_representative_confusions(summary, trials)
    error_topology(trials)
    print(FIG_DIR)


if __name__ == "__main__":
    main()
