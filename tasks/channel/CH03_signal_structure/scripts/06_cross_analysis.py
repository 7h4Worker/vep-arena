"""Cross-analysis: signal-level metrics vs decision-level capacity.

Bridges Phase 1 signal profile (SNR, PLV) with the decision channel
capacity results produced by the ``benchmark_decision_channel_capacity``
task (confusion matrices, C_BA, accuracy).

Key questions answered:
1. Does per-subject mean SNR predict classification accuracy / C_BA?
2. Does per-frequency SNR predict per-class accuracy?
3. Does the harmonic interference matrix predict the confusion matrix?
4. Do SNR and PLV explain different portions of capacity variance?

Usage
-----
    python cross_analysis.py
    python cross_analysis.py --capacity-dir ../benchmark_decision_channel_capacity/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vep_arena.config import BENCHMARK_FREQS
from vep_arena.signal.utils import harmonic_interference_matrix

TASK_DIR = Path(__file__).resolve().parent
CAPACITY_TASK = TASK_DIR.parent / "benchmark_decision_channel_capacity"
FREQS = np.asarray(BENCHMARK_FREQS, dtype=np.float64)
FREQ_SORTED_IDX = np.argsort(FREQS)
FREQS_SORTED = FREQS[FREQ_SORTED_IDX]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--signal-dir", type=str,
                   default=str(TASK_DIR / "results"))
    p.add_argument("--capacity-dir", type=str,
                   default=str(CAPACITY_TASK / "results"))
    p.add_argument("--method", type=str, default="ETRCA",
                   help="Method to use for cross-analysis (must match capacity results)")
    p.add_argument("--window", type=float, default=1.0)
    return p.parse_args()


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def load_capacity_data(capacity_dir: Path, method: str, window: float) -> pd.DataFrame:
    cap = pd.read_csv(capacity_dir / "analysis" / "capacity_by_subject_method_window.csv")
    mask = (cap["method"] == method) & (np.isclose(cap["window"], window))
    subset = cap[mask].copy()
    if subset.empty:
        available = cap.groupby(["method", "window"]).size().reset_index(name="n")
        raise ValueError(
            f"No capacity data for method={method}, window={window}.\n"
            f"Available:\n{available.to_string(index=False)}"
        )
    return subset


def load_confusion_aggregate(capacity_dir: Path, method: str, window: float) -> np.ndarray:
    npz_path = capacity_dir / "analysis" / "confusion_counts_method_window_aggregate.npz"
    data = np.load(npz_path, allow_pickle=True)
    w_str = f"{window:g}".replace(".", "p")
    key = f"{method.lower()}_w{w_str}_sall"
    if key not in data:
        raise ValueError(f"Key {key!r} not in {npz_path}. Available: {list(data.keys())[:10]}")
    counts = data[key].astype(np.float64)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


# ── analysis functions ──────────────────────────────────────────────

def subject_level_correlation(
    signal_df: pd.DataFrame,
    cap_df: pd.DataFrame,
) -> pd.DataFrame:
    """Correlate per-subject signal metrics with capacity metrics."""
    sig_by_subj = signal_df.groupby("subject", as_index=False).agg(
        mean_snr=("snr_harm_oz_db", "mean"),
        mean_plv=("plv_oz", "mean"),
        mean_conc=("spectral_concentration", "mean"),
    )
    merged = sig_by_subj.merge(
        cap_df[["subject", "accuracy", "c_ba", "c1", "c_ba_minus_c1", "i_uniform"]],
        on="subject",
    )

    results = []
    for sig_col in ["mean_snr", "mean_plv", "mean_conc"]:
        for cap_col in ["accuracy", "c_ba", "c1", "i_uniform"]:
            r, p = stats.pearsonr(merged[sig_col], merged[cap_col])
            results.append({
                "signal_metric": sig_col,
                "capacity_metric": cap_col,
                "pearson_r": r,
                "p_value": p,
                "n": len(merged),
            })
    return pd.DataFrame(results), merged


def frequency_level_correlation(
    signal_df: pd.DataFrame,
    confusion_P: np.ndarray,
) -> pd.DataFrame:
    """Correlate per-frequency SNR with per-class diagonal accuracy."""
    sig_by_freq = signal_df.groupby("target_idx", as_index=False).agg(
        mean_snr=("snr_harm_oz_db", "mean"),
        mean_plv=("plv_oz", "mean"),
    )
    per_class_acc = np.diag(confusion_P)
    sig_by_freq["class_accuracy"] = per_class_acc

    r_snr, p_snr = stats.pearsonr(sig_by_freq["mean_snr"], per_class_acc)
    r_plv, p_plv = stats.pearsonr(sig_by_freq["mean_plv"], per_class_acc)

    return sig_by_freq, {
        "snr_vs_accuracy": {"r": r_snr, "p": p_snr},
        "plv_vs_accuracy": {"r": r_plv, "p": p_plv},
    }


def interference_vs_confusion(confusion_P: np.ndarray) -> dict:
    """Correlate harmonic interference matrix with off-diagonal confusion."""
    imat = harmonic_interference_matrix(FREQS, n_harmonics=5, resolution_hz=0.25)

    mask = ~np.eye(40, dtype=bool)
    interference_flat = imat[mask]
    confusion_flat = confusion_P[mask]

    has_interference = interference_flat > 0
    no_interference = interference_flat == 0

    mean_err_with = float(np.mean(confusion_flat[has_interference])) if has_interference.any() else 0
    mean_err_without = float(np.mean(confusion_flat[no_interference])) if no_interference.any() else 0

    r, p = stats.pearsonr(interference_flat, confusion_flat)

    return {
        "pearson_r": r,
        "p_value": p,
        "mean_confusion_with_interference": mean_err_with,
        "mean_confusion_without_interference": mean_err_without,
        "ratio": mean_err_with / max(mean_err_without, 1e-15),
        "n_pairs_with_interference": int(has_interference.sum()),
        "n_pairs_without": int(no_interference.sum()),
    }


# ── plotting ────────────────────────────────────────────────────────

def plot_subject_scatter(merged: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, (xcol, xlabel) in zip(axes, [
        ("mean_snr", "Mean harmonic SNR (dB)"),
        ("mean_plv", "Mean PLV"),
        ("mean_conc", "Mean spectral concentration"),
    ]):
        ax.scatter(merged[xcol], merged["c_ba"], s=30, alpha=0.7, edgecolors="k", linewidths=0.3)
        r, p = stats.pearsonr(merged[xcol], merged["c_ba"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("$C_{BA}$ (bit/symbol)")
        ax.set_title(f"r = {r:.3f}, p = {p:.4f}")
        ax.grid(alpha=0.15)
        for _, row in merged.iterrows():
            ax.annotate(f"{int(row['subject'])}", (row[xcol], row["c_ba"]),
                        fontsize=5, alpha=0.6, ha="center", va="bottom")

    fig.suptitle("Signal-level metrics vs BA capacity (per subject)", fontsize=12)
    _save(fig, fig_dir / "fig_cross_subject_capacity.png")


def plot_frequency_scatter(freq_df: pd.DataFrame, corr_results: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, (col, label, cr_key) in zip(axes, [
        ("mean_snr", "Mean harmonic SNR (dB)", "snr_vs_accuracy"),
        ("mean_plv", "Mean PLV", "plv_vs_accuracy"),
    ]):
        colors = [plt.cm.viridis(f / 16.0) for f in (FREQS - 8.0)]
        ax.scatter(freq_df[col], freq_df["class_accuracy"], c=colors, s=40, edgecolors="k", linewidths=0.3)
        cr = corr_results[cr_key]
        ax.set_xlabel(label)
        ax.set_ylabel("Per-class accuracy")
        ax.set_title(f"r = {cr['r']:.3f}, p = {cr['p']:.4f}")
        ax.grid(alpha=0.15)
        for _, row in freq_df.iterrows():
            ax.annotate(f"{row['target_idx']:.0f}", (row[col], row["class_accuracy"]),
                        fontsize=5, alpha=0.5)

    fig.suptitle("Signal-level metrics vs per-class accuracy (40 frequencies)", fontsize=12)
    _save(fig, fig_dir / "fig_cross_frequency_accuracy.png")


def plot_interference_confusion_compare(confusion_P: np.ndarray, fig_dir: Path) -> None:
    imat = harmonic_interference_matrix(FREQS, n_harmonics=5, resolution_hz=0.25)
    imat_s = imat[np.ix_(FREQ_SORTED_IDX, FREQ_SORTED_IDX)]
    conf_s = confusion_P[np.ix_(FREQ_SORTED_IDX, FREQ_SORTED_IDX)]

    conf_offdiag = conf_s.copy()
    np.fill_diagonal(conf_offdiag, 0)
    conf_log = np.log10(np.clip(conf_offdiag, 1e-6, None))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    im0 = axes[0].imshow(imat_s, aspect="equal", cmap="YlOrRd", interpolation="nearest")
    axes[0].set_title("Harmonic interference (predicted)")
    fig.colorbar(im0, ax=axes[0], shrink=0.7, label="# collisions")

    im1 = axes[1].imshow(conf_log, aspect="equal", cmap="YlOrRd", interpolation="nearest")
    axes[1].set_title("Actual confusion (log₁₀, off-diagonal)")
    fig.colorbar(im1, ax=axes[1], shrink=0.7, label="log₁₀(P)")

    for ax in axes:
        ax.set_xticks(range(0, 40, 5))
        ax.set_xticklabels([f"{FREQS_SORTED[i]:.1f}" for i in range(0, 40, 5)], fontsize=7)
        ax.set_yticks(range(0, 40, 5))
        ax.set_yticklabels([f"{FREQS_SORTED[i]:.1f}" for i in range(0, 40, 5)], fontsize=7)

    fig.suptitle("Predicted harmonic interference vs actual confusion matrix", fontsize=12)
    _save(fig, fig_dir / "fig_interference_vs_confusion.png")


# ── main ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    signal_dir = Path(args.signal_dir)
    capacity_dir = Path(args.capacity_dir)
    fig_dir = signal_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Cross-analysis: signal × capacity ===")
    print(f"  method:  {args.method}")
    print(f"  window:  {args.window} s")

    signal_df = pd.read_csv(signal_dir / "signal_profile.csv")
    cap_df = load_capacity_data(capacity_dir, args.method, args.window)
    confusion_P = load_confusion_aggregate(capacity_dir, args.method, args.window)

    print(f"\n--- 1. Subject-level: signal metrics → capacity ---")
    corr_table, merged = subject_level_correlation(signal_df, cap_df)
    print(corr_table.to_string(index=False))
    corr_table.to_csv(signal_dir / "cross_subject_correlations.csv", index=False)
    merged.to_csv(signal_dir / "cross_subject_merged.csv", index=False)
    plot_subject_scatter(merged, fig_dir)

    print(f"\n--- 2. Frequency-level: SNR/PLV → per-class accuracy ---")
    freq_df, freq_corr = frequency_level_correlation(signal_df, confusion_P)
    for k, v in freq_corr.items():
        print(f"  {k}: r={v['r']:.3f}, p={v['p']:.4f}")
    freq_df.to_csv(signal_dir / "cross_frequency_accuracy.csv", index=False)
    plot_frequency_scatter(freq_df, freq_corr, fig_dir)

    print(f"\n--- 3. Harmonic interference → confusion matrix ---")
    ifc = interference_vs_confusion(confusion_P)
    print(f"  Pearson r = {ifc['pearson_r']:.4f} (p={ifc['p_value']:.2e})")
    print(f"  Mean confusion WITH harmonic collision:    {ifc['mean_confusion_with_interference']:.6f}")
    print(f"  Mean confusion WITHOUT harmonic collision: {ifc['mean_confusion_without_interference']:.6f}")
    print(f"  Ratio: {ifc['ratio']:.2f}x")
    print(f"  Pairs with collision: {ifc['n_pairs_with_interference']} / {ifc['n_pairs_with_interference'] + ifc['n_pairs_without']}")

    with open(signal_dir / "cross_interference_results.json", "w") as f:
        json.dump(ifc, f, indent=2)
    plot_interference_confusion_compare(confusion_P, fig_dir)

    print(f"\n  Wrote cross-analysis outputs to {signal_dir}")
    print("  Done.")


if __name__ == "__main__":
    main()
