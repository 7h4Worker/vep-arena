from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = TASK_DIR / "results" / "reference_tdca_full"
ANALYSIS_DIR = Path(__file__).resolve().parent
FIG_DIR = ANALYSIS_DIR / "figures"
TABLE_DIR = ANALYSIS_DIR / "tables"
N_CLASSES = 160


def mutual_info_uniform(pairs: pd.DataFrame, n_classes: int = N_CLASSES) -> tuple[float, float]:
    counts = np.zeros((n_classes, n_classes), dtype=np.float64)
    true = pairs["true"].to_numpy(dtype=int) - 1
    pred = pairs["pred"].to_numpy(dtype=int) - 1
    for t, p in zip(true, pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            counts[t, p] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    p_y_given_x = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)
    p_x = np.full((n_classes, 1), 1.0 / n_classes, dtype=np.float64)
    p_xy = p_x * p_y_given_x
    p_y = p_xy.sum(axis=0, keepdims=True)
    denom = p_x * p_y
    mask = (p_xy > 0) & (denom > 0)
    mi = float(np.sum(p_xy[mask] * np.log2(p_xy[mask] / denom[mask])))
    acc = float(np.trace(counts) / counts.sum()) if counts.sum() > 0 else 0.0
    return mi, acc


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    trials = pd.read_csv(RESULTS_DIR / "trials.csv")
    preds = pd.read_csv(RESULTS_DIR / "predictions.csv")
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")

    rows: list[dict[str, float | int | str]] = []
    for window, grp in preds.groupby("window"):
        mi, acc = mutual_info_uniform(grp)
        summary_row = summary[summary["window"] == window].iloc[0]
        rows.append(
            {
                "method": "ZENODO_TDCA",
                "window": float(window),
                "folds": int(summary_row["folds"]),
                "subjects": int(summary_row["subjects"]),
                "accuracy": acc,
                "mi_bits": mi,
                "c0_bits": math.log2(N_CLASSES),
                "utilization": mi / math.log2(N_CLASSES),
                "itr_bpm": float(summary_row["itr_bpm"]),
            }
        )
    overall = pd.DataFrame(rows).sort_values("window")

    session_rows = (
        trials.groupby(["subject", "session", "window"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), itr_bpm=("itr_bpm", "mean"))
        .sort_values(["subject", "session", "window"])
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(TABLE_DIR / "overall_summary.csv", index=False)
    session_rows.to_csv(TABLE_DIR / "session_window_summary.csv", index=False)
    return overall, session_rows


def plot_acc_itr(overall: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    axes[0].plot(overall["window"], overall["accuracy"] * 100, "o-", color="#2563eb", lw=2)
    axes[0].set_xlabel("Window (s)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Recognition accuracy")
    axes[0].set_xticks(overall["window"])
    axes[0].set_ylim(45, 101)
    axes[0].grid(alpha=0.2)

    axes[1].plot(overall["window"], overall["itr_bpm"], "o-", color="#dc2626", lw=2)
    axes[1].set_xlabel("Window (s)")
    axes[1].set_ylabel("ITR (bits/min)")
    axes[1].set_title("ITR with trial overhead")
    axes[1].set_xticks(overall["window"])
    axes[1].set_ylim(0, max(overall["itr_bpm"]) * 1.18)
    axes[1].grid(alpha=0.2)

    axes[2].plot(overall["window"], overall["utilization"] * 100, "o-", color="#16a34a", lw=2)
    axes[2].set_xlabel("Window (s)")
    axes[2].set_ylabel("MI / log2(160) (%)")
    axes[2].set_title("Information utilization")
    axes[2].set_xticks(overall["window"])
    axes[2].set_ylim(50, 101)
    axes[2].grid(alpha=0.2)

    fig.suptitle("WN-BCI reference TDCA: accuracy and rate")
    fig.savefig(FIG_DIR / "fig01_acc_itr_window.png", dpi=180)
    plt.close(fig)


def plot_confusion(window: float = 0.3) -> None:
    preds = pd.read_csv(RESULTS_DIR / "predictions.csv")
    sub = preds[preds["window"] == window]
    counts = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    for row in sub.itertuples(index=False):
        counts[int(row.true) - 1, int(row.pred) - 1] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    norm = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)
    acc = float(np.trace(counts) / counts.sum())

    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    ax.set_title(f"160-class confusion matrix, window={window:g}s, Acc={acc:.1%}")
    ax.set_xlabel("Predicted target")
    ax.set_ylabel("True target")
    ticks = [0, 39, 79, 119, 159]
    ax.set_xticks(ticks, [str(t + 1) for t in ticks])
    ax.set_yticks(ticks, [str(t + 1) for t in ticks])
    fig.colorbar(im, ax=ax, shrink=0.82, label="P(pred | true)")
    fig.savefig(FIG_DIR / "fig02_confusion_w03.png", dpi=180)
    plt.close(fig)


def plot_session_heatmap(session_rows: pd.DataFrame) -> None:
    session_rows = session_rows.copy()
    session_rows["session_id"] = session_rows.apply(lambda r: f"S{int(r.subject)}-{int(r.session)}", axis=1)
    pivot = session_rows.pivot(index="session_id", columns="window", values="accuracy").sort_index(
        key=lambda idx: [tuple(int(part) for part in item[1:].split("-")) for item in idx]
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy() * 100, cmap="magma", vmin=0, vmax=100, aspect="auto")
    ax.set_title("Session-level accuracy by window")
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("Session")
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{c:g}" for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            ax.text(x, y, f"{pivot.iat[y, x] * 100:.0f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Accuracy (%)")
    fig.savefig(FIG_DIR / "fig03_session_accuracy_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    overall, session_rows = build_tables()
    plot_acc_itr(overall)
    plot_confusion(window=0.3)
    plot_session_heatmap(session_rows)
    print(f"wrote {FIG_DIR}")
    print(f"wrote {TABLE_DIR}")


if __name__ == "__main__":
    main()
