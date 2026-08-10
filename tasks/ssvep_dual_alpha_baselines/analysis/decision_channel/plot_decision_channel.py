"""Decision channel analysis for the Dual-Alpha SSVEP task (GigaDB 102557).

Three paradigms — Checkerboard Arrangement (CA), Binocular Vision (BV),
Binocular-Swap Vision (BsV) — each with 40 dual-frequency targets.
Reads predictions.csv from results/official_baselines/.

Uses the canonical vep_arena.channel library for all MI / capacity math.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[4]
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
PREDICTIONS = TASK / "results" / "official_baselines" / "predictions.csv"
DATASET = Path(r"D:\ProjData\datasets\ssvep_dual_alpha_gigadb_102557")

# ── constants ────────────────────────────────────────────────────────────
N_TARGETS = 40
C0 = capacity_c0(N_TARGETS)  # log2(40) ≈ 5.322

PARADIGMS = ["Checkerboard_Arrangment", "Binocular_Vision",
             "Binocular-Swap_Vision"]
PARADIGM_SHORT = {
    "Checkerboard_Arrangment": "CA",
    "Binocular_Vision": "BV",
    "Binocular-Swap_Vision": "BsV",
}
PARADIGM_COLORS = {
    "Checkerboard_Arrangment": "#1f77b4",
    "Binocular_Vision": "#2ca02c",
    "Binocular-Swap_Vision": "#d62728",
}

METHODS = ["ETRCA", "FBDCCA"]
METHOD_COLORS = {"ETRCA": "#e41a1c", "FBDCCA": "#377eb8"}
METHOD_MARKERS = {"ETRCA": "o", "FBDCCA": "s"}

# Codebook (from Stimulate_Code.txt)
CODEBOOK = {
    "Checkerboard_Arrangment": {
        "freq1": [8.2,9,9.8,10.6,10.6,11.4,12.2,13.8,8.2,9,9.8,10.6,11.4,12.2,13,13.8,8.2,9,9.8,10.6,11.4,12.2,13,14.6,8.2,9,9.8,10.6,11.4,12.2,13,14.6,8.2,9.8,9.8,10.6,11.4,12.2,13.8,15.4],
        "freq2": [9,9.8,11.4,11.4,15.4,15.4,16.2,15.4,9.8,10.6,13.8,12.2,12.2,13,13.8,16.2,13,11.4,14.6,13,13,13.8,14.6,15.4,15.4,16.2,15.4,13.8,13.8,14.6,16.2,16.2,16.2,10.6,16.2,14.6,14.6,15.4,14.6,16.2],
    },
    "Binocular_Vision": {
        "freq1": [8.2,9,9.8,10.6,10.6,11.4,12.2,13.8,8.2,9,9.8,10.6,11.4,12.2,13,13.8,8.2,9,9.8,10.6,11.4,12.2,13,14.6,8.2,9,9.8,10.6,11.4,12.2,13,14.6,8.2,9.8,9.8,10.6,11.4,12.2,13.8,15.4],
        "freq2": [9,9.8,11.4,11.4,15.4,15.4,16.2,15.4,9.8,10.6,13.8,12.2,12.2,13,13.8,16.2,13,11.4,14.6,13,13,13.8,14.6,15.4,15.4,16.2,15.4,13.8,13.8,14.6,16.2,16.2,16.2,10.6,16.2,14.6,14.6,15.4,14.6,16.2],
    },
    "Binocular-Swap_Vision": {
        "freq1": [14.74,12.42,12.47,15.33,12.88,10.1,8.51,11.67,15.4,12.36,8.81,11.33,11.01,9.52,11.22,11.92,11.62,14.72,8.62,14.43,15.59,10.9,8.57,13.93,15.04,15.97,14.37,8.21,11.97,12.94,9.57,12.45,10.94,13.11,9.49,12.63,12.41,15.91,13.34,10.62],
        "freq2": [10.1,12.63,12.88,11.22,12.47,14.74,8.57,11.62,13.11,12.45,8.21,11.97,10.94,12.41,15.33,13.34,11.67,15.59,10.62,12.94,14.72,13.93,8.51,10.9,14.37,15.91,15.04,8.81,11.33,14.43,9.49,12.36,11.01,15.4,9.57,12.42,9.52,15.97,11.92,8.62],
    },
}


# ── helpers ──────────────────────────────────────────────────────────────
def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS)


def _sel(df: pd.DataFrame, method: str, paradigm: str, window: float
         ) -> pd.DataFrame:
    return df[(df["method"] == method) & (df["paradigm"] == paradigm) &
              (np.isclose(df["window"], window))]


def channel_metrics(df: pd.DataFrame, method: str, paradigm: str,
                    window: float) -> dict | None:
    sub = _sel(df, method, paradigm, window)
    if len(sub) == 0:
        return None
    counts = confusion_counts(sub["true"], sub["pred"], N_TARGETS)
    P = normalize_confusion(counts, alpha=0.5)
    mi = mutual_info_uniform(P)
    ba = capacity_ba(P)
    acc = float((sub["true"] == sub["pred"]).mean())
    return dict(
        method=method, paradigm=paradigm,
        paradigm_short=PARADIGM_SHORT[paradigm],
        window=window,
        accuracy=acc,
        MI_uniform=mi, C_BA=ba.capacity, C1=capacity_c1(N_TARGETS, acc),
        C0=C0, utilization=mi / C0,
        n_trials=len(sub), ba_converged=ba.converged,
    )


def subject_mi_list(df: pd.DataFrame, method: str, paradigm: str,
                    window: float) -> list[dict]:
    sub = _sel(df, method, paradigm, window)
    out = []
    for sid, sdf in sub.groupby("subject"):
        counts = confusion_counts(sdf["true"], sdf["pred"], N_TARGETS)
        # alpha must be small: 5 trials/row vs 40 classes means alpha=0.5
        # would add 20 pseudo-counts and overwhelm the 5 real counts.
        P = normalize_confusion(counts, alpha=1e-6)
        out.append(dict(
            subject=sid, method=method, paradigm=paradigm, window=window,
            MI=mutual_info_uniform(P),
            accuracy=float((sdf["true"] == sdf["pred"]).mean()),
        ))
    return out


def build_capacity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    windows = sorted(df["window"].unique())
    for para in PARADIGMS:
        for method in METHODS:
            for w in windows:
                r = channel_metrics(df, method, para, w)
                if r is not None:
                    rows.append(r)
    return pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────

def fig01_codebook(df: pd.DataFrame) -> None:
    """Dual-frequency codebook: scatter of (freq1, freq2) per paradigm."""
    print("Fig 01: Codebook visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, para in zip(axes, PARADIGMS):
        cb = CODEBOOK[para]
        f1, f2 = np.array(cb["freq1"]), np.array(cb["freq2"])
        ax.scatter(f1, f2, c=PARADIGM_COLORS[para], s=40, alpha=0.7,
                   edgecolor="black", lw=0.5, zorder=3)
        for i in range(N_TARGETS):
            ax.annotate(str(i), (f1[i], f2[i]), fontsize=5.5,
                        ha="center", va="bottom", textcoords="offset points",
                        xytext=(0, 3))
        lo = min(f1.min(), f2.min()) - 0.5
        hi = max(f1.max(), f2.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.5, alpha=0.3)
        ax.set_xlabel("Freq 1 (Hz)")
        ax.set_ylabel("Freq 2 (Hz)")
        ax.set_title(f"{PARADIGM_SHORT[para]}: {para.replace('_', ' ')}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    fig.suptitle("Dual-Frequency Codebook: 40 Targets per Paradigm", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_codebook.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig01_codebook.png")


def fig02_confusion(df: pd.DataFrame) -> None:
    """40x40 confusion matrices per paradigm (ETRCA at 2.0 s)."""
    print("Fig 02: Confusion matrices...")
    method, window = "ETRCA", 2.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, para in zip(axes, PARADIGMS):
        sub = _sel(df, method, para, window)
        if sub.empty:
            ax.set_title(f"{PARADIGM_SHORT[para]}: no data")
            continue
        counts = confusion_counts(sub["true"], sub["pred"], N_TARGETS)
        P = normalize_confusion(counts, alpha=0.0)
        mi = mutual_info_uniform(normalize_confusion(counts, alpha=0.5))
        acc = float((sub["true"] == sub["pred"]).mean())

        im = ax.imshow(P, vmin=0, vmax=1, cmap="Blues", aspect="equal")
        ax.set_title(f"{PARADIGM_SHORT[para]}  Acc={acc:.1%}  "
                     f"MI={mi:.2f}/{C0:.2f}")
        ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("True")

    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="P(Y=j | X=i)")
    fig.suptitle(f"40x40 Confusion Matrices ({method}, {window}s)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_confusion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig02_confusion.png")


def fig03_mi_curves(cap: pd.DataFrame) -> None:
    """MI and BA capacity vs window for each paradigm x method."""
    print("Fig 03: MI / BA capacity curves...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, metric, title in zip(
            axes, ["MI_uniform", "C_BA"], ["MI (uniform input)", "BA capacity"]):
        ax.axhline(C0, color="gray", ls="--", lw=0.8,
                   label=f"C0=log2({N_TARGETS})")
        for para in PARADIGMS:
            for method in METHODS:
                sub = cap[(cap["paradigm"] == para) &
                          (cap["method"] == method)]
                if sub.empty:
                    continue
                sub = sub.sort_values("window")
                ls = "-" if method == "ETRCA" else "--"
                ax.plot(sub["window"], sub[metric],
                        color=PARADIGM_COLORS[para],
                        marker=METHOD_MARKERS[method],
                        ls=ls, ms=4, lw=1.5,
                        label=f"{PARADIGM_SHORT[para]} {method}")
        ax.set_xlabel("Window (s)")
        ax.set_ylabel(f"{title} (bits/selection)")
        ax.set_title(title)
        ax.set_xlim(0, 2.15)
        ax.set_ylim(0, C0 + 0.3)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_mi_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig03_mi_curves.png")


def fig04_per_subject(df: pd.DataFrame) -> None:
    """Per-subject MI distributions + paradigm comparison scatter."""
    print("Fig 04: Per-subject analysis...")
    method, window = "ETRCA", 2.0
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) strip chart
    ax = axes[0]
    for idx, para in enumerate(PARADIGMS):
        mis = [r["MI"] for r in subject_mi_list(df, method, para, window)]
        jitter = rng.normal(0, 0.06, len(mis))
        ax.scatter(idx + jitter, mis, c=PARADIGM_COLORS[para], s=25,
                   alpha=0.6, edgecolor="black", lw=0.3)
        ax.plot([idx - 0.25, idx + 0.25], [np.mean(mis)] * 2,
                color=PARADIGM_COLORS[para], lw=2.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([PARADIGM_SHORT[p] for p in PARADIGMS])
    ax.set_ylabel("MI (bits)")
    ax.set_title(f"(a) Per-subject MI ({method}, {window}s)")
    ax.set_ylim(0, C0 + 0.3)
    ax.grid(alpha=0.3)

    # (b) CA vs BV (same codebook, different display)
    ax = axes[1]
    s_ca = {r["subject"]: r["MI"]
            for r in subject_mi_list(df, method, "Checkerboard_Arrangment", window)}
    s_bv = {r["subject"]: r["MI"]
            for r in subject_mi_list(df, method, "Binocular_Vision", window)}
    common = sorted(set(s_ca) & set(s_bv))
    if common:
        xv = [s_ca[s] for s in common]
        yv = [s_bv[s] for s in common]
        ax.scatter(xv, yv, c="#9467bd", s=35, alpha=0.7,
                   edgecolor="black", lw=0.5)
        hi = max(max(xv), max(yv)) * 1.1
        ax.plot([0, hi], [0, hi], "k--", lw=0.8, alpha=0.5)
        pct_ca = sum(a > b for a, b in zip(xv, yv)) / len(common) * 100
        ax.set_title(f"(b) CA vs BV (same codebook)\n{pct_ca:.0f}% CA > BV")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
    ax.set_xlabel("MI (Checkerboard)")
    ax.set_ylabel("MI (Binocular Vision)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # (c) CA vs BsV (different codebook, different display)
    ax = axes[2]
    s_bsv = {r["subject"]: r["MI"]
             for r in subject_mi_list(df, method, "Binocular-Swap_Vision", window)}
    common2 = sorted(set(s_ca) & set(s_bsv))
    if common2:
        xv2 = [s_ca[s] for s in common2]
        yv2 = [s_bsv[s] for s in common2]
        ax.scatter(xv2, yv2, c="#8c564b", s=35, alpha=0.7,
                   edgecolor="black", lw=0.5)
        hi2 = max(max(xv2), max(yv2)) * 1.1
        ax.plot([0, hi2], [0, hi2], "k--", lw=0.8, alpha=0.5)
        pct = sum(a > b for a, b in zip(xv2, yv2)) / len(common2) * 100
        ax.set_title(f"(c) CA vs BsV (diff codebook)\n{pct:.0f}% CA > BsV")
        ax.set_xlim(0, hi2)
        ax.set_ylim(0, hi2)
    ax.set_xlabel("MI (Checkerboard)")
    ax.set_ylabel("MI (Binocular-Swap)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_per_subject.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("  -> fig04_per_subject.png")


def fig05_method_comparison(df: pd.DataFrame, cap: pd.DataFrame) -> None:
    """ETRCA vs FBDCCA for CA and BV (where both methods exist)."""
    print("Fig 05: Method comparison (ETRCA vs FBDCCA)...")
    window = 2.0
    paras_with_fbdcca = ["Checkerboard_Arrangment", "Binocular_Vision"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) MI bar comparison
    ax = axes[0]
    x = np.arange(len(paras_with_fbdcca))
    width = 0.35
    for i, method in enumerate(METHODS):
        vals = []
        for para in paras_with_fbdcca:
            row = cap[(cap["paradigm"] == para) & (cap["method"] == method) &
                      (np.isclose(cap["window"], window))]
            vals.append(float(row.iloc[0]["MI_uniform"]) if not row.empty
                        else 0.0)
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=method,
                      color=METHOD_COLORS[method], alpha=0.8,
                      edgecolor="black", lw=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([PARADIGM_SHORT[p] for p in paras_with_fbdcca])
    ax.set_ylabel("MI (bits)")
    ax.set_title(f"(a) ETRCA vs FBDCCA ({window}s)")
    ax.axhline(C0, color="gray", ls="--", lw=0.8)
    ax.set_ylim(0, C0 + 0.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # (b) per-subject ETRCA vs FBDCCA scatter
    ax = axes[1]
    colors = [PARADIGM_COLORS[p] for p in paras_with_fbdcca]
    for para, c in zip(paras_with_fbdcca, colors):
        s_etrca = {r["subject"]: r["MI"]
                   for r in subject_mi_list(df, "ETRCA", para, window)}
        s_fbdcca = {r["subject"]: r["MI"]
                    for r in subject_mi_list(df, "FBDCCA", para, window)}
        common = sorted(set(s_etrca) & set(s_fbdcca))
        if common:
            xv = [s_etrca[s] for s in common]
            yv = [s_fbdcca[s] for s in common]
            ax.scatter(xv, yv, c=c, s=30, alpha=0.6,
                       edgecolor="black", lw=0.3,
                       label=PARADIGM_SHORT[para])
    lim = [0, C0 + 0.3]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("MI (ETRCA)")
    ax.set_ylabel("MI (FBDCCA)")
    ax.set_title(f"(b) Per-subject: ETRCA vs FBDCCA ({window}s)")
    ax.set_xlim(0, C0 + 0.3)
    ax.set_ylim(0, C0 + 0.3)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_method_comparison.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig05_method_comparison.png")


def fig06_freq_gap_analysis(df: pd.DataFrame) -> None:
    """Frequency gap distribution and its relation to per-target accuracy."""
    print("Fig 06: Frequency gap analysis...")
    method, window = "ETRCA", 2.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, para in zip(axes, PARADIGMS):
        sub = _sel(df, method, para, window)
        if sub.empty:
            continue
        cb = CODEBOOK[para]
        f1, f2 = np.array(cb["freq1"]), np.array(cb["freq2"])
        gap = np.abs(f2 - f1)

        per_target_acc = []
        for t in range(N_TARGETS):
            tsub = sub[sub["true"] == t]
            per_target_acc.append(
                float((tsub["true"] == tsub["pred"]).mean()) if len(tsub) > 0
                else np.nan)
        per_target_acc = np.array(per_target_acc)

        valid = ~np.isnan(per_target_acc)
        ax.scatter(gap[valid], per_target_acc[valid] * 100,
                   c=PARADIGM_COLORS[para], s=30, alpha=0.7,
                   edgecolor="black", lw=0.3)
        if valid.sum() > 2:
            z = np.polyfit(gap[valid], per_target_acc[valid] * 100, 1)
            xfit = np.linspace(gap[valid].min(), gap[valid].max(), 50)
            ax.plot(xfit, np.polyval(z, xfit), "k--", lw=1, alpha=0.5)
            r = np.corrcoef(gap[valid], per_target_acc[valid])[0, 1]
            ax.text(0.05, 0.05, f"r = {r:.3f}", transform=ax.transAxes,
                    fontsize=9)
        ax.set_xlabel("|Freq2 - Freq1| (Hz)")
        ax.set_ylabel("Per-target accuracy (%)")
        ax.set_title(f"{PARADIGM_SHORT[para]}: gap vs accuracy")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_freq_gap_analysis.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig06_freq_gap_analysis.png")


def save_per_subject_table(df: pd.DataFrame) -> None:
    print("Saving per-subject MI table...")
    rows = []
    for window in [1.0, 2.0]:
        for method in METHODS:
            for para in PARADIGMS:
                rows.extend(subject_mi_list(df, method, para, window))
    if rows:
        pd.DataFrame(rows).to_csv(TABLE_DIR / "per_subject_mi.csv",
                                  index=False)


# ── summary ──────────────────────────────────────────────────────────────

def print_summary(cap: pd.DataFrame) -> None:
    window = 2.0
    print("\n" + "=" * 75)
    print(f"Decision Channel Summary (window={window}s)")
    print("=" * 75)
    header = (f"{'Paradigm':<10} {'Method':<8} {'Acc%':>7} {'MI':>7} "
              f"{'C_BA':>7} {'Util%':>7}")
    print(header)
    print("-" * len(header))
    for para in PARADIGMS:
        for method in METHODS:
            row = cap[(cap["paradigm"] == para) & (cap["method"] == method) &
                      (np.isclose(cap["window"], window))]
            if not row.empty:
                r = row.iloc[0]
                print(f"{PARADIGM_SHORT[para]:<10} {method:<8} "
                      f"{r['accuracy']*100:>6.1f}% {r['MI_uniform']:>7.3f} "
                      f"{r['C_BA']:>7.3f} {r['utilization']*100:>6.1f}%")
    print("=" * 75)


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs()
    print("Loading predictions...")
    df = load_predictions()
    print(f"  {len(df):,} rows | methods: {sorted(df['method'].unique())} "
          f"| paradigms: {sorted(df['paradigm'].unique())}")

    print("Building capacity table...")
    cap = build_capacity_table(df)
    cap.to_csv(TABLE_DIR / "capacity_by_paradigm_method_window.csv",
               index=False)
    print(f"  {len(cap)} rows saved")

    fig01_codebook(df)
    fig02_confusion(df)
    fig03_mi_curves(cap)
    fig04_per_subject(df)
    fig05_method_comparison(df, cap)
    fig06_freq_gap_analysis(df)
    save_per_subject_table(df)
    print_summary(cap)
    print("\nDone.")


if __name__ == "__main__":
    main()
