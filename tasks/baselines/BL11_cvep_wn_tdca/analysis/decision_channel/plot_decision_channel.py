"""Decision channel analysis for Broadband White Noise BCI (Zenodo 8300517).

160 white-noise coded targets, Zenodo reference TDCA, windows 0.1-0.4 s.
Uses the canonical vep_arena.channel library for all MI / capacity math.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

PROJECT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT))

from vep_arena.channel.capacity import (
    capacity_ba,
    capacity_c0,
    capacity_c1,
    mutual_info_uniform,
)
from vep_arena.channel.confusion import confusion_counts, normalize_confusion

# ── paths ────────────────────────────────────────────────────────────────
TASK = Path(__file__).resolve().parents[2]
ANALYSIS = Path(__file__).resolve().parent
FIG_DIR = ANALYSIS / "figures"
TABLE_DIR = ANALYSIS / "tables"
PREDICTIONS = TASK / "results" / "reference_tdca_full" / "predictions.csv"
STI_PATH = Path(
    r"D:\ProjData\datasets\cvep_broadband_white_noise_bci_zenodo8300517"
    r"\extracted\data\stimulation\sweep\STI.mat"
)

# ── constants ────────────────────────────────────────────────────────────
N_CLASSES = 160
C0 = capacity_c0(N_CLASSES)  # log2(160) = 7.322
WINDOWS = [0.1, 0.2, 0.3, 0.4]
WINDOW_COLORS = {0.1: "#ef4444", 0.2: "#f59e0b", 0.3: "#22c55e", 0.4: "#3b82f6"}


# ── data loading ─────────────────────────────────────────────────────────
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS)
    df["true_0"] = df["true"] - 1
    df["pred_0"] = df["pred"] - 1
    return df


def load_wn_matrix() -> np.ndarray:
    mat = sio.loadmat(str(STI_PATH))
    return mat["WN"].astype(np.float64)


# ── channel metrics ──────────────────────────────────────────────────────
def channel_metrics(df: pd.DataFrame, window: float) -> dict | None:
    sub = df[np.isclose(df["window"], window)]
    if len(sub) == 0:
        return None
    counts = confusion_counts(sub["true_0"], sub["pred_0"], N_CLASSES)
    P = normalize_confusion(counts, alpha=1e-6)
    mi = mutual_info_uniform(P)
    ba = capacity_ba(P)
    acc = float((sub["true_0"] == sub["pred_0"]).mean())
    c1 = capacity_c1(N_CLASSES, acc)
    return dict(
        window=window,
        accuracy=acc,
        MI_uniform=mi,
        C_BA=ba.capacity,
        C1=c1,
        C0=C0,
        utilization=mi / C0,
        n_trials=len(sub),
        ba_converged=ba.converged,
        ba_iterations=ba.iterations,
    )


def subject_mi(df: pd.DataFrame, window: float) -> list[dict]:
    sub = df[np.isclose(df["window"], window)]
    out = []
    for sid, sdf in sub.groupby("subject"):
        counts = confusion_counts(sdf["true_0"], sdf["pred_0"], N_CLASSES)
        P = normalize_confusion(counts, alpha=1e-6)
        mi = mutual_info_uniform(P)
        acc = float((sdf["true_0"] == sdf["pred_0"]).mean())
        out.append(dict(subject=sid, window=window, MI=mi, accuracy=acc))
    return out


def build_capacity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in WINDOWS:
        r = channel_metrics(df, w)
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────

def fig01_confusion_grid(df: pd.DataFrame) -> None:
    """160x160 confusion matrices at all 4 windows."""
    print("Fig 01: Confusion matrix grid...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)

    for ax, w in zip(axes.flat, WINDOWS):
        sub = df[np.isclose(df["window"], w)]
        counts = confusion_counts(sub["true_0"], sub["pred_0"], N_CLASSES)
        row_sum = counts.sum(axis=1, keepdims=True)
        P = np.divide(counts, row_sum,
                      out=np.zeros_like(counts, dtype=float),
                      where=row_sum > 0)
        acc = float(np.trace(counts) / counts.sum())
        mi_val = mutual_info_uniform(
            normalize_confusion(counts, alpha=1e-6))

        im = ax.imshow(P, vmin=0, vmax=1, cmap="viridis",
                       interpolation="nearest")
        ax.set_title(f"T={w:g}s  Acc={acc:.1%}  MI={mi_val:.2f}/{C0:.2f}",
                     fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ticks = [0, 39, 79, 119, 159]
        ax.set_xticks(ticks, [str(t + 1) for t in ticks], fontsize=8)
        ax.set_yticks(ticks, [str(t + 1) for t in ticks], fontsize=8)

    fig.colorbar(im, ax=list(axes.flat), shrink=0.6, label="P(pred | true)")
    fig.suptitle("WN-BCI: 160-class Confusion Matrices by Window Length",
                 fontsize=13)
    fig.savefig(FIG_DIR / "fig01_confusion_grid.png", dpi=180)
    plt.close(fig)
    print("  -> fig01_confusion_grid.png")


def fig02_capacity_decomposition(cap: pd.DataFrame) -> None:
    """Four-layer capacity decomposition: C0 >= C_BA >= MI >= C1."""
    print("Fig 02: Capacity decomposition...")
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    layers = [
        ("C0 = log2(160)", "C0", "#94a3b8", "--"),
        ("C_BA (Blahut-Arimoto)", "C_BA", "#8b5cf6", "-"),
        ("MI (uniform input)", "MI_uniform", "#2563eb", "-"),
        ("C1 (symmetric approx)", "C1", "#dc2626", "-"),
    ]
    for label, col, color, ls in layers:
        if col == "C0":
            ax.axhline(C0, color=color, ls=ls, lw=1.5, label=label, alpha=0.7)
        else:
            ax.plot(cap["window"], cap[col], marker="o", color=color, lw=2,
                    ms=6, label=label, ls=ls)

    ax.fill_between(cap["window"], cap["C1"], cap["MI_uniform"],
                    alpha=0.08, color="#2563eb")
    ax.fill_between(cap["window"], cap["MI_uniform"], cap["C_BA"],
                    alpha=0.08, color="#8b5cf6")

    ax.set_xlabel("Window length (s)", fontsize=11)
    ax.set_ylabel("Information (bits/selection)", fontsize=11)
    ax.set_title("WN-BCI Decision Channel: Capacity Decomposition")
    ax.set_xticks(WINDOWS)
    ax.set_ylim(0, C0 + 0.5)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.2)

    for _, row in cap.iterrows():
        ax.annotate(f"{row['MI_uniform']:.2f}",
                    (row["window"], row["MI_uniform"]),
                    textcoords="offset points", xytext=(8, -4),
                    fontsize=8, color="#2563eb")

    fig.savefig(FIG_DIR / "fig02_capacity_decomposition.png", dpi=180)
    plt.close(fig)
    print("  -> fig02_capacity_decomposition.png")


def fig03_per_subject(df: pd.DataFrame) -> None:
    """Per-subject MI box/strip at each window."""
    print("Fig 03: Per-subject MI...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # (a) strip chart by window
    ax = axes[0]
    rng = np.random.default_rng(42)
    for idx, w in enumerate(WINDOWS):
        rows = subject_mi(df, w)
        mis = [r["MI"] for r in rows]
        jitter = rng.normal(0, 0.08, len(mis))
        ax.scatter(idx + jitter, mis, c=WINDOW_COLORS[w], s=35, alpha=0.7,
                   edgecolor="black", lw=0.3, zorder=3)
        ax.plot([idx - 0.25, idx + 0.25], [np.mean(mis)] * 2,
                color=WINDOW_COLORS[w], lw=2.5, zorder=4)
    ax.axhline(C0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks(range(len(WINDOWS)))
    ax.set_xticklabels([f"{w:g}s" for w in WINDOWS])
    ax.set_ylabel("MI (bits/selection)")
    ax.set_title("(a) Per-subject MI by window")
    ax.set_ylim(0, C0 + 0.5)
    ax.grid(alpha=0.2)

    # (b) per-subject MI trajectory
    ax = axes[1]
    all_rows = []
    for w in WINDOWS:
        all_rows.extend(subject_mi(df, w))
    sdf = pd.DataFrame(all_rows)
    for sid in sorted(sdf["subject"].unique()):
        ssub = sdf[sdf["subject"] == sid].sort_values("window")
        ax.plot(ssub["window"], ssub["MI"], "o-", ms=4, lw=1.2, alpha=0.6,
                label=f"S{int(sid)}")
    ax.axhline(C0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("MI (bits/selection)")
    ax.set_title("(b) Individual MI trajectories")
    ax.set_xticks(WINDOWS)
    ax.set_ylim(0, C0 + 0.5)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.2)

    fig.savefig(FIG_DIR / "fig03_per_subject.png", dpi=180)
    plt.close(fig)
    print("  -> fig03_per_subject.png")


def fig04_error_vs_correlation(df: pd.DataFrame, wn: np.ndarray) -> None:
    """Confusion error structure vs WN code cross-correlation."""
    print("Fig 04: Error vs code correlation...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    cc = np.corrcoef(wn)
    np.fill_diagonal(cc, np.nan)

    # (a) WN code correlation matrix
    ax = axes[0]
    im = ax.imshow(cc, vmin=-0.4, vmax=0.4, cmap="RdBu_r",
                   interpolation="nearest")
    ax.set_title("(a) WN code cross-correlation")
    ax.set_xlabel("Code index")
    ax.set_ylabel("Code index")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")

    # (b) off-diagonal confusion vs code correlation (0.1s)
    ax = axes[1]
    sub01 = df[np.isclose(df["window"], 0.1)]
    counts01 = confusion_counts(sub01["true_0"], sub01["pred_0"], N_CLASSES)
    row_sum = counts01.sum(axis=1, keepdims=True)
    P01 = np.divide(counts01, row_sum,
                    out=np.zeros_like(counts01, dtype=float),
                    where=row_sum > 0)
    np.fill_diagonal(P01, np.nan)
    mask = ~np.isnan(cc) & ~np.isnan(P01) & (P01 > 0)
    if mask.sum() > 10:
        ax.scatter(cc[mask], P01[mask], s=1, alpha=0.15, color="#ef4444",
                   rasterized=True)
        r_corr = np.corrcoef(cc[mask].ravel(), P01[mask].ravel())[0, 1]
        ax.set_title(f"(b) T=0.1s off-diag confusion vs |r|\n"
                     f"Pearson={r_corr:.3f}")
    else:
        ax.set_title("(b) T=0.1s: insufficient off-diagonal data")
    ax.set_xlabel("Code cross-correlation")
    ax.set_ylabel("P(pred=j | true=i), i!=j")
    ax.grid(alpha=0.2)

    # (c) same for 0.3s
    ax = axes[2]
    sub03 = df[np.isclose(df["window"], 0.3)]
    counts03 = confusion_counts(sub03["true_0"], sub03["pred_0"], N_CLASSES)
    row_sum3 = counts03.sum(axis=1, keepdims=True)
    P03 = np.divide(counts03, row_sum3,
                    out=np.zeros_like(counts03, dtype=float),
                    where=row_sum3 > 0)
    np.fill_diagonal(P03, np.nan)
    mask3 = ~np.isnan(cc) & ~np.isnan(P03) & (P03 > 0)
    if mask3.sum() > 10:
        ax.scatter(cc[mask3], P03[mask3], s=1, alpha=0.15, color="#22c55e",
                   rasterized=True)
        r_corr3 = np.corrcoef(cc[mask3].ravel(), P03[mask3].ravel())[0, 1]
        ax.set_title(f"(c) T=0.3s off-diag confusion vs |r|\n"
                     f"Pearson={r_corr3:.3f}")
    else:
        ax.set_title("(c) T=0.3s: insufficient off-diagonal data")
    ax.set_xlabel("Code cross-correlation")
    ax.set_ylabel("P(pred=j | true=i), i!=j")
    ax.grid(alpha=0.2)

    fig.suptitle("WN-BCI: Code Correlation Structure and Confusion Errors",
                 fontsize=12)
    fig.savefig(FIG_DIR / "fig04_error_vs_correlation.png", dpi=180)
    plt.close(fig)
    print("  -> fig04_error_vs_correlation.png")


def fig05_utilization_itr(cap: pd.DataFrame) -> None:
    """Utilization curve and marginal ITR."""
    print("Fig 05: Utilization and ITR...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # (a) utilization
    ax = axes[0]
    ax.plot(cap["window"], cap["utilization"] * 100, "o-",
            color="#2563eb", lw=2, ms=7)
    ax.axhline(100, color="gray", ls="--", lw=0.8, alpha=0.5)
    for _, row in cap.iterrows():
        ax.annotate(f"{row['utilization']*100:.1f}%",
                    (row["window"], row["utilization"] * 100),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("MI / C0 (%)")
    ax.set_title("(a) Information utilization")
    ax.set_xticks(WINDOWS)
    ax.set_ylim(40, 105)
    ax.grid(alpha=0.2)

    # (b) ITR = MI / T * 60
    ax = axes[1]
    itr = cap["MI_uniform"].values / cap["window"].values * 60
    ax.plot(cap["window"], itr, "s-", color="#dc2626", lw=2, ms=7)
    best_idx = np.argmax(itr)
    ax.scatter(cap["window"].iloc[best_idx], itr[best_idx],
               s=120, facecolors="none", edgecolors="#dc2626", lw=2, zorder=5)
    for i, (w, v) in enumerate(zip(cap["window"], itr)):
        ax.annotate(f"{v:.0f}", (w, v),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("MI-based ITR (bits/min)")
    ax.set_title("(b) Information transfer rate (no trial overhead)")
    ax.set_xticks(WINDOWS)
    ax.grid(alpha=0.2)

    fig.suptitle("WN-BCI: Utilization and Transfer Rate", fontsize=12)
    fig.savefig(FIG_DIR / "fig05_utilization_itr.png", dpi=180)
    plt.close(fig)
    print("  -> fig05_utilization_itr.png")


def fig06_per_target_accuracy(df: pd.DataFrame) -> None:
    """Per-target accuracy distribution at each window."""
    print("Fig 06: Per-target accuracy...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    for ax, w in zip(axes.flat, WINDOWS):
        sub = df[np.isclose(df["window"], w)]
        per_target = []
        for t in range(N_CLASSES):
            tsub = sub[sub["true_0"] == t]
            if len(tsub) > 0:
                per_target.append(float((tsub["pred_0"] == t).mean()))
            else:
                per_target.append(np.nan)
        per_target = np.array(per_target)
        valid = ~np.isnan(per_target)

        ax.bar(np.arange(N_CLASSES)[valid], per_target[valid] * 100,
               width=1.0, color=WINDOW_COLORS[w], alpha=0.7, edgecolor="none")
        mean_acc = np.nanmean(per_target) * 100
        ax.axhline(mean_acc, color="black", ls="--", lw=1, alpha=0.6)
        std_acc = np.nanstd(per_target) * 100
        ax.set_title(f"T={w:g}s  mean={mean_acc:.1f}%  std={std_acc:.1f}%",
                     fontsize=10)
        ax.set_xlabel("Target index (0-159)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(-1, 160)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("WN-BCI: Per-Target Accuracy Distribution", fontsize=13)
    fig.savefig(FIG_DIR / "fig06_per_target_accuracy.png", dpi=180)
    plt.close(fig)
    print("  -> fig06_per_target_accuracy.png")


def fig07_gap_analysis(cap: pd.DataFrame) -> None:
    """MI gap analysis: how much information is lost at each layer."""
    print("Fig 07: Gap analysis...")
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)

    w = cap["window"].values
    gap_ba = C0 - cap["C_BA"].values
    gap_mi = cap["C_BA"].values - cap["MI_uniform"].values
    gap_c1 = cap["MI_uniform"].values - cap["C1"].values

    ax.bar(w, gap_ba, width=0.06, bottom=cap["C_BA"].values,
           color="#f87171", alpha=0.8, label="C0 - C_BA (channel noise)")
    ax.bar(w, gap_mi, width=0.06, bottom=cap["MI_uniform"].values,
           color="#fbbf24", alpha=0.8, label="C_BA - MI (input non-optimality)")
    ax.bar(w, gap_c1, width=0.06, bottom=cap["C1"].values,
           color="#a78bfa", alpha=0.8, label="MI - C1 (asymmetry loss)")
    ax.bar(w, cap["C1"].values, width=0.06,
           color="#34d399", alpha=0.8, label="C1 (symmetric capacity)")

    ax.set_xlabel("Window (s)")
    ax.set_ylabel("Information (bits)")
    ax.set_title("WN-BCI: Information Loss Decomposition")
    ax.set_xticks(WINDOWS)
    ax.axhline(C0, color="gray", ls=":", lw=1)
    ax.set_ylim(0, C0 + 0.5)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(axis="y", alpha=0.2)

    fig.savefig(FIG_DIR / "fig07_gap_analysis.png", dpi=180)
    plt.close(fig)
    print("  -> fig07_gap_analysis.png")


# ── tables ───────────────────────────────────────────────────────────────

def save_tables(df: pd.DataFrame, cap: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    cap.to_csv(TABLE_DIR / "capacity_decomposition.csv", index=False)

    sub_rows = []
    for w in WINDOWS:
        sub_rows.extend(subject_mi(df, w))
    if sub_rows:
        pd.DataFrame(sub_rows).to_csv(
            TABLE_DIR / "per_subject_mi.csv", index=False)

    # capacity for cross-task integration
    export_rows = []
    for _, row in cap.iterrows():
        itr_raw = row["MI_uniform"] / row["window"] * 60
        export_rows.append(dict(
            dataset="WN-BCI",
            method="ZENODO_TDCA",
            paradigm="WN_sweep",
            n_classes=N_CLASSES,
            window=row["window"],
            C0=C0,
            MI_uniform=row["MI_uniform"],
            C_BA=row["C_BA"],
            C1=row["C1"],
            accuracy=row["accuracy"],
            utilization=row["utilization"],
            itr_bpm=itr_raw,
        ))
    pd.DataFrame(export_rows).to_csv(
        TABLE_DIR / "capacity_for_crosstask.csv", index=False)

    print(f"  Tables saved to {TABLE_DIR}")


# ── summary ──────────────────────────────────────────────────────────────

def print_summary(cap: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("WN-BCI Decision Channel Summary")
    print("=" * 80)
    header = (f"{'Window':>8} {'Acc%':>7} {'MI':>7} {'C_BA':>7} "
              f"{'C1':>7} {'C0':>7} {'Util%':>7} {'ITR':>8}")
    print(header)
    print("-" * len(header))
    for _, r in cap.iterrows():
        itr = r["MI_uniform"] / r["window"] * 60
        print(f"{r['window']:>7.1f}s {r['accuracy']*100:>6.1f}% "
              f"{r['MI_uniform']:>7.3f} {r['C_BA']:>7.3f} "
              f"{r['C1']:>7.3f} {r['C0']:>7.3f} "
              f"{r['utilization']*100:>6.1f}% {itr:>7.1f}")
    print("=" * 80)

    print("\nCapacity gaps at T=0.3s:")
    r03 = cap[np.isclose(cap["window"], 0.3)].iloc[0]
    print(f"  C0 - C_BA  = {C0 - r03['C_BA']:.3f} bits (channel noise)")
    print(f"  C_BA - MI  = {r03['C_BA'] - r03['MI_uniform']:.3f} bits "
          f"(input non-optimality)")
    print(f"  MI  - C1   = {r03['MI_uniform'] - r03['C1']:.3f} bits "
          f"(asymmetry)")


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading predictions...")
    df = load_predictions()
    print(f"  {len(df):,} rows, subjects: {sorted(df['subject'].unique())}")

    print("Loading WN stimulus matrix...")
    wn = load_wn_matrix()
    print(f"  Shape: {wn.shape}")

    print("Building capacity table...")
    cap = build_capacity_table(df)
    print(f"  {len(cap)} windows computed")

    fig01_confusion_grid(df)
    fig02_capacity_decomposition(cap)
    fig03_per_subject(df)
    fig04_error_vs_correlation(df, wn)
    fig05_utilization_itr(cap)
    fig06_per_target_accuracy(df)
    fig07_gap_analysis(cap)
    save_tables(df, cap)
    print_summary(cap)
    print("\nDone.")


if __name__ == "__main__":
    main()
