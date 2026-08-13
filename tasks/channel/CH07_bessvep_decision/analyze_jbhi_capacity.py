"""Decision-channel capacity analysis for JBHI16 and JBHI35 five-subject datasets.

Computes per-subject and aggregate:
- C0 (ideal), C1 (Wolpaw symmetric), C_BA (Blahut-Arimoto), MI_uniform
- Capacity-time tradeoff: 60/T * C_BA(T) to find optimal operating window
- Information accumulation curves MI(T)

Outputs CSV tables and figures to each dataset's analysis/ directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.channel.capacity import (
    capacity_ba,
    capacity_c0,
    capacity_c1,
    mutual_info_uniform,
)
from vep_arena.channel.confusion import confusion_counts, normalize_confusion


TASK_DIR = Path(__file__).resolve().parent
JBHI16_PREDICTIONS = PROJECT_ROOT / "tasks" / "bessvep" / "results" / "BS02_16t" / "full" / "predictions.csv"
JBHI35_ROOT = PROJECT_ROOT / "tasks" / "bessvep" / "results" / "BS03_35t" / "final_five_execution_20260730"
JBHI35_TDCA = JBHI35_ROOT / "tdca_full" / "predictions.csv"
JBHI35_PERIODIC = JBHI35_ROOT / "periodic_receivers_full" / "predictions.csv"
JBHI35_ETRCA = JBHI35_ROOT / "etrca_2s" / "predictions.csv"

OUTPUT_ROOT = TASK_DIR
ITR_SHIFT = 0.5  # gaze shift seconds added to window


def load_jbhi16() -> pd.DataFrame:
    df = pd.read_csv(JBHI16_PREDICTIONS)
    # Columns: dataset, subject, method, window_seconds, window_samples, fold, block, trial_index, true, pred, correct
    # Labels are 1-based in JBHI16; convert to 0-based
    df["true"] = df["true"] - 1
    df["pred"] = df["pred"] - 1
    df = df.rename(columns={"window_seconds": "window"})
    return df[["subject", "method", "window", "true", "pred"]].copy()


def load_jbhi35() -> pd.DataFrame:
    frames = []
    for path in [JBHI35_TDCA, JBHI35_PERIODIC]:
        df = pd.read_csv(path)
        df = df.rename(columns={"true_0based": "true", "pred_0based": "pred", "window_seconds": "window"})
        frames.append(df[["subject", "method", "window", "true", "pred"]])
    # eTRCA
    etrca = pd.read_csv(JBHI35_ETRCA)
    # predictions.csv has: subject, variant, fold, true_0based, pred_0based, correct
    etrca = etrca.rename(columns={"true_0based": "true", "pred_0based": "pred", "variant": "method"})
    etrca["window"] = 2.0
    # Only use arena_independent_signed as the canonical eTRCA variant
    etrca = etrca[etrca["method"] == "arena_independent_signed"].copy()
    etrca["method"] = "eTRCA"
    frames.append(etrca[["subject", "method", "window", "true", "pred"]])
    return pd.concat(frames, ignore_index=True)


def analyze_dataset(preds: pd.DataFrame, classes: int, dataset_name: str) -> pd.DataFrame:
    """Compute capacity metrics per (method, window, subject) and aggregate."""
    rows = []
    # Per-subject
    for (method, window, subject), group in preds.groupby(["method", "window", "subject"], sort=True):
        counts = confusion_counts(group["true"], group["pred"], classes)
        P = normalize_confusion(counts, alpha=0.0)
        accuracy = float(np.trace(counts)) / float(np.sum(counts))
        ba = capacity_ba(P)
        mi_u = mutual_info_uniform(P)
        rows.append({
            "dataset": dataset_name,
            "method": str(method),
            "window": float(window),
            "subject": str(subject),
            "classes": classes,
            "samples": int(np.sum(counts)),
            "accuracy": accuracy,
            "c0": capacity_c0(classes),
            "c1": capacity_c1(classes, accuracy),
            "c_ba": ba.capacity,
            "ba_converged": ba.converged,
            "i_uniform": mi_u,
            "itr_c1_bpm": 60.0 / (float(window) + ITR_SHIFT) * capacity_c1(classes, accuracy),
            "itr_ba_bpm": 60.0 / (float(window) + ITR_SHIFT) * ba.capacity,
            "itr_mi_bpm": 60.0 / (float(window) + ITR_SHIFT) * mi_u,
        })

    # Aggregate (pool all subjects)
    for (method, window), group in preds.groupby(["method", "window"], sort=True):
        counts = confusion_counts(group["true"], group["pred"], classes)
        P = normalize_confusion(counts, alpha=0.0)
        accuracy = float(np.trace(counts)) / float(np.sum(counts))
        ba = capacity_ba(P)
        mi_u = mutual_info_uniform(P)
        rows.append({
            "dataset": dataset_name,
            "method": str(method),
            "window": float(window),
            "subject": "aggregate",
            "classes": classes,
            "samples": int(np.sum(counts)),
            "accuracy": accuracy,
            "c0": capacity_c0(classes),
            "c1": capacity_c1(classes, accuracy),
            "c_ba": ba.capacity,
            "ba_converged": ba.converged,
            "i_uniform": mi_u,
            "itr_c1_bpm": 60.0 / (float(window) + ITR_SHIFT) * capacity_c1(classes, accuracy),
            "itr_ba_bpm": 60.0 / (float(window) + ITR_SHIFT) * ba.capacity,
            "itr_mi_bpm": 60.0 / (float(window) + ITR_SHIFT) * mi_u,
        })

    return pd.DataFrame(rows)


def plot_capacity_curves(df: pd.DataFrame, dataset_name: str, classes: int, output_dir: Path) -> None:
    """Plot MI(T), C_BA(T), and capacity-time tradeoff for aggregate data."""
    agg = df[df["subject"] == "aggregate"].copy()
    methods = sorted(agg["method"].unique())

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(methods), 9)))

    # --- Figure 1: MI_uniform(T) and C_BA(T) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for idx, method in enumerate(methods):
        mdata = agg[agg["method"] == method].sort_values("window")
        axes[0].plot(mdata["window"], mdata["i_uniform"], marker="o", color=colors[idx],
                     label=method, markersize=5, linewidth=1.5)
        axes[1].plot(mdata["window"], mdata["c_ba"], marker="s", color=colors[idx],
                     label=method, markersize=5, linewidth=1.5)

    axes[0].axhline(capacity_c0(classes), color="gray", ls="--", lw=0.8, label=f"C₀ = {capacity_c0(classes):.2f} bits")
    axes[1].axhline(capacity_c0(classes), color="gray", ls="--", lw=0.8, label=f"C₀ = {capacity_c0(classes):.2f} bits")
    axes[0].set(xlabel="Window (s)", ylabel="MI_uniform (bits/symbol)", title=f"{dataset_name}: Information per symbol")
    axes[1].set(xlabel="Window (s)", ylabel="C_BA (bits/symbol)", title=f"{dataset_name}: Channel capacity")
    axes[0].grid(alpha=0.25); axes[1].grid(alpha=0.25)
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.savefig(output_dir / f"{dataset_name}_capacity_vs_window.png", dpi=200)
    plt.close(fig)

    # --- Figure 2: Capacity-time tradeoff (ITR from C_BA) ---
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for idx, method in enumerate(methods):
        mdata = agg[agg["method"] == method].sort_values("window")
        ax.plot(mdata["window"], mdata["itr_ba_bpm"], marker="o", color=colors[idx],
                label=method, markersize=5, linewidth=1.5)
    ax.set(xlabel="Window (s)", ylabel="60/(T+0.5) × C_BA (bits/min)",
           title=f"{dataset_name}: Capacity-time tradeoff (optimal operating point)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / f"{dataset_name}_itr_capacity_tradeoff.png", dpi=200)
    plt.close(fig)

    # --- Figure 3: Per-subject C_BA at select windows ---
    subject_data = df[df["subject"] != "aggregate"].copy()
    windows_to_show = sorted(set(agg["window"].unique()) & {0.4, 0.6, 1.0, 1.4, 2.0})
    if not windows_to_show:
        windows_to_show = sorted(agg["window"].unique())[:5]

    fig, axes = plt.subplots(1, len(windows_to_show), figsize=(4 * len(windows_to_show), 4.5),
                             constrained_layout=True, sharey=True)
    if len(windows_to_show) == 1:
        axes = [axes]
    for wi, window in enumerate(windows_to_show):
        ax = axes[wi]
        wdata = subject_data[subject_data["window"] == window]
        for idx, method in enumerate(methods):
            mdata = wdata[wdata["method"] == method]
            if mdata.empty:
                continue
            subjects = mdata["subject"].values
            x_pos = np.arange(len(subjects)) + idx * 0.12
            ax.bar(x_pos, mdata["c_ba"].values, width=0.1, color=colors[idx], label=method if wi == 0 else None)
        ax.set_xticks(np.arange(len(subjects)) + 0.12 * len(methods) / 2)
        ax.set_xticklabels(subjects, fontsize=8)
        ax.set_title(f"T = {window}s", fontsize=10)
        ax.set_xlabel("Subject")
        ax.grid(alpha=0.2, axis="y")
        if wi == 0:
            ax.set_ylabel("C_BA (bits/symbol)")
    if len(methods) <= 8:
        axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{dataset_name}: Per-subject channel capacity", fontsize=12)
    fig.savefig(output_dir / f"{dataset_name}_capacity_per_subject.png", dpi=200)
    plt.close(fig)

    # --- Figure 4: MI utilization η = MI_uniform / C0 ---
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    c0 = capacity_c0(classes)
    for idx, method in enumerate(methods):
        mdata = agg[agg["method"] == method].sort_values("window")
        ax.plot(mdata["window"], mdata["i_uniform"] / c0, marker="o", color=colors[idx],
                label=method, markersize=5, linewidth=1.5)
    ax.set(xlabel="Window (s)", ylabel="η = MI_uniform / C₀",
           title=f"{dataset_name}: Channel utilization efficiency", ylim=(0, 1.05))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_dir / f"{dataset_name}_utilization.png", dpi=200)
    plt.close(fig)


def print_summary(df: pd.DataFrame, dataset_name: str) -> None:
    agg = df[df["subject"] == "aggregate"].copy()
    print(f"\n{'='*70}")
    print(f" {dataset_name} Decision Channel Summary")
    print(f"{'='*70}")
    print(f"{'Method':<16s} {'Window':<8s} {'Acc':<8s} {'C_BA':<8s} {'MI_u':<8s} {'ITR_BA':<10s} {'ITR_MI':<10s}")
    print("-" * 70)
    for _, r in agg.sort_values(["method", "window"]).iterrows():
        print(f"  {r['method']:<14s} {r['window']:<8.1f} {r['accuracy']:<8.1%} "
              f"{r['c_ba']:<8.3f} {r['i_uniform']:<8.3f} "
              f"{r['itr_ba_bpm']:<10.1f} {r['itr_mi_bpm']:<10.1f}")

    # Best operating point per method
    print(f"\nOptimal operating window (max ITR from C_BA):")
    for method in sorted(agg["method"].unique()):
        mdata = agg[agg["method"] == method]
        best = mdata.loc[mdata["itr_ba_bpm"].idxmax()]
        print(f"  {method:<16s} T*={best['window']:.1f}s  ITR*={best['itr_ba_bpm']:.1f} bpm  "
              f"C_BA={best['c_ba']:.3f} bits  Acc={best['accuracy']:.1%}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    # JBHI16
    print("Loading JBHI16...")
    preds16 = load_jbhi16()
    print(f"  {len(preds16)} predictions, methods: {sorted(preds16['method'].unique())}")
    df16 = analyze_dataset(preds16, classes=16, dataset_name="JBHI16")
    df16.to_csv(OUTPUT_ROOT / "jbhi16_capacity.csv", index=False)
    plot_capacity_curves(df16, "JBHI16", 16, figures_dir)
    print_summary(df16, "JBHI16")

    # JBHI35
    print("\nLoading JBHI35 five-subject...")
    preds35 = load_jbhi35()
    print(f"  {len(preds35)} predictions, methods: {sorted(preds35['method'].unique())}")
    df35 = analyze_dataset(preds35, classes=35, dataset_name="JBHI35")
    df35.to_csv(OUTPUT_ROOT / "jbhi35_capacity.csv", index=False)
    plot_capacity_curves(df35, "JBHI35", 35, figures_dir)
    print_summary(df35, "JBHI35")

    # Combined comparison table at key windows
    print("\n\nCross-dataset comparison (aggregate, select windows):")
    combined = pd.concat([df16, df35], ignore_index=True)
    combined_agg = combined[combined["subject"] == "aggregate"]
    for window in [0.4, 1.0, 2.0]:
        subset = combined_agg[combined_agg["window"] == window]
        if subset.empty:
            continue
        print(f"\n--- T = {window}s ---")
        print(f"  {'Dataset':<10s} {'Method':<16s} {'C0':<6s} {'C_BA':<8s} {'η=MI/C0':<8s} {'ITR_BA':<10s}")
        for _, r in subset.sort_values(["dataset", "itr_ba_bpm"], ascending=[True, False]).iterrows():
            eta = r["i_uniform"] / r["c0"]
            print(f"  {r['dataset']:<10s} {r['method']:<16s} {r['c0']:<6.2f} "
                  f"{r['c_ba']:<8.3f} {eta:<8.3f} {r['itr_ba_bpm']:<10.1f}")

    combined_agg.to_csv(OUTPUT_ROOT / "combined_aggregate_capacity.csv", index=False)
    print(f"\nAll outputs saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
