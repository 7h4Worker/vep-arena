"""Decision channel analysis for the Binocular AR SSVEP task (Ke 2025).

Reads predictions.csv from experiment{1,2,3}_trca and produces:
  - 6 figures covering confusion matrices, MI curves, information
    decomposition, coherence bandwidth, condition comparison, per-subject
  - CSV tables: full capacity table, decomposition, per-subject MI

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

RESULTS = {
    "experiment1": TASK / "results" / "experiment1_trca",
    "experiment2": TASK / "results" / "experiment2_trca",
    "experiment3": TASK / "results" / "experiment3_trca",
}

# ── constants ────────────────────────────────────────────────────────────
N_TARGETS = 8
C0 = capacity_c0(N_TARGETS)  # 3.0 bits

METHODS = ["CCA", "FBCCA", "TRCA", "ETRCA"]
METHOD_COLORS = {
    "CCA": "#999999",
    "FBCCA": "#4daf4a",
    "TRCA": "#377eb8",
    "ETRCA": "#e41a1c",
}
METHOD_MARKERS = {"CCA": "^", "FBCCA": "s", "TRCA": "D", "ETRCA": "o"}

EXP_TASKS = {
    "experiment1": ["LF", "MF"],
    "experiment2": ["SFSP", "SFDP", "DFSP", "DFDP"],
    "experiment3": ["DFDP1", "DFDP3", "DFDP5"],
}
ALL_TASKS = ["LF", "MF", "SFSP", "SFDP", "DFSP", "DFDP",
             "DFDP1", "DFDP3", "DFDP5"]

TASK_LABELS = {
    "LF": "LF (8-15 Hz)",
    "MF": "MF (23-30 Hz)",
    "SFSP": "SFSP",
    "SFDP": "SFDP (+phase)",
    "DFSP": "DFSP (+freq)",
    "DFDP": "DFDP (+both)",
    "DFDP1": "DFDP gap=1",
    "DFDP3": "DFDP gap=3",
    "DFDP5": "DFDP gap=5",
}

TASK_COLORS = {
    "LF": "#1f77b4", "MF": "#ff7f0e",
    "SFSP": "#1f77b4", "SFDP": "#ff7f0e", "DFSP": "#2ca02c", "DFDP": "#d62728",
    "DFDP1": "#1f77b4", "DFDP3": "#2ca02c", "DFDP5": "#d62728",
}

GAPS = {"DFDP1": 1, "DFDP3": 3, "DFDP5": 5}
GAZE_SHIFT = 0.5  # seconds


# ── helpers ──────────────────────────────────────────────────────────────
def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_all_predictions() -> pd.DataFrame:
    frames = []
    for exp_name, result_dir in RESULTS.items():
        df = pd.read_csv(result_dir / "predictions.csv")
        df["experiment"] = exp_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _sel(df: pd.DataFrame, method: str, task: str, window: float
         ) -> pd.DataFrame:
    return df[(df["method"] == method) & (df["task"] == task) &
              (np.isclose(df["window"], window))]


def channel_metrics(df: pd.DataFrame, method: str, task: str,
                    window: float) -> dict | None:
    sub = _sel(df, method, task, window)
    if len(sub) == 0:
        return None
    counts = confusion_counts(sub["true"], sub["pred"], N_TARGETS)
    P = normalize_confusion(counts, alpha=0.5)
    mi = mutual_info_uniform(P)
    ba = capacity_ba(P)
    acc = float((sub["true"] == sub["pred"]).mean())
    return dict(
        method=method, task=task, window=window,
        accuracy=acc,
        MI_uniform=mi, C_BA=ba.capacity, C1=capacity_c1(N_TARGETS, acc),
        C0=C0, utilization=mi / C0,
        n_trials=len(sub), ba_converged=ba.converged,
    )


def subject_mi_list(df: pd.DataFrame, method: str, task: str,
                    window: float) -> list[dict]:
    sub = _sel(df, method, task, window)
    out = []
    for sid, sdf in sub.groupby("subject"):
        counts = confusion_counts(sdf["true"], sdf["pred"], N_TARGETS)
        P = normalize_confusion(counts, alpha=0.5)
        out.append(dict(
            subject=sid, method=method, task=task, window=window,
            MI=mutual_info_uniform(P),
            accuracy=float((sdf["true"] == sdf["pred"]).mean()),
        ))
    return out


def build_capacity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    windows = sorted(df["window"].unique())
    for task in ALL_TASKS:
        for method in METHODS:
            for w in windows:
                r = channel_metrics(df, method, task, w)
                if r is not None:
                    rows.append(r)
    return pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────

def fig01_confusion_exp2(df: pd.DataFrame) -> None:
    """Exp2 controlled comparison: SFSP / SFDP / DFSP / DFDP."""
    print("Fig 01: Confusion matrices (Exp2)...")
    tasks = ["SFSP", "SFDP", "DFSP", "DFDP"]
    method, window = "ETRCA", 1.0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, task in zip(axes, tasks):
        sub = _sel(df, method, task, window)
        counts = confusion_counts(sub["true"], sub["pred"], N_TARGETS)
        P = normalize_confusion(counts, alpha=0.0)
        mi = mutual_info_uniform(normalize_confusion(counts, alpha=0.5))
        acc = float((sub["true"] == sub["pred"]).mean())

        im = ax.imshow(P, vmin=0, vmax=1, cmap="Blues", aspect="equal")
        ax.set_title(f"{TASK_LABELS[task]}\nAcc={acc:.1%}  MI={mi:.3f}/{C0:.1f}")
        ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("True")
        ax.set_xticks(range(N_TARGETS))
        ax.set_yticks(range(N_TARGETS))

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="P(Y=j | X=i)")
    fig.suptitle(f"Exp2: Encoding Dimension Comparison ({method}, {window}s)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_confusion_exp2.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("  -> fig01_confusion_exp2.png")


def fig02_mi_curves(cap: pd.DataFrame) -> None:
    """MI vs window for each experiment (ETRCA + TRCA)."""
    print("Fig 02: MI curves...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    for ax, (exp, tasks) in zip(axes, EXP_TASKS.items()):
        ax.axhline(C0, color="gray", ls="--", lw=0.8)
        for task in tasks:
            for method in ["ETRCA", "TRCA"]:
                sub = cap[(cap["task"] == task) & (cap["method"] == method)]
                if sub.empty:
                    continue
                sub = sub.sort_values("window")
                ls = "-" if method == "ETRCA" else "--"
                ax.plot(sub["window"], sub["MI_uniform"],
                        color=TASK_COLORS[task], marker=METHOD_MARKERS[method],
                        ls=ls, ms=3, lw=1.5, markevery=3,
                        label=f"{TASK_LABELS[task]} {method}")
        ax.set_xlabel("Window (s)")
        ax.set_ylabel("MI (bits/selection)")
        ax.set_title(exp.replace("experiment", "Exp "))
        ax.set_xlim(0, 3.15)
        ax.set_ylim(0, C0 + 0.2)
        ax.legend(fontsize=6.5, ncol=2)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_mi_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig02_mi_curves.png")


def fig03_ba_capacity(cap: pd.DataFrame) -> None:
    """BA capacity curves for all experiments, all 4 methods."""
    print("Fig 03: BA capacity curves...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    for ax, (exp, tasks) in zip(axes, EXP_TASKS.items()):
        ax.axhline(C0, color="gray", ls="--", lw=0.8, label=f"C0=log2({N_TARGETS})")
        for task in tasks:
            for method in METHODS:
                sub = cap[(cap["task"] == task) & (cap["method"] == method)]
                if sub.empty:
                    continue
                sub = sub.sort_values("window")
                ls = {
                    "ETRCA": "-", "TRCA": "--", "FBCCA": "-.", "CCA": ":"
                }[method]
                ax.plot(sub["window"], sub["C_BA"],
                        color=TASK_COLORS[task], ls=ls, lw=1.2,
                        label=f"{task} {method}")
        ax.set_xlabel("Window (s)")
        ax.set_ylabel("BA capacity (bits/selection)")
        ax.set_title(exp.replace("experiment", "Exp "))
        ax.set_xlim(0, 3.15)
        ax.set_ylim(0, C0 + 0.2)
        ax.legend(fontsize=5.5, ncol=2)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_ba_capacity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig03_ba_capacity.png")


def fig04_information_decomposition(cap: pd.DataFrame) -> None:
    """Quantify frequency vs phase contribution from Exp2."""
    print("Fig 04: Information decomposition...")
    method, window = "ETRCA", 1.0
    tasks = ["SFSP", "SFDP", "DFSP", "DFDP"]

    mi = {}
    for t in tasks:
        row = cap[(cap["task"] == t) & (cap["method"] == method) &
                  (np.isclose(cap["window"], window))]
        if row.empty:
            print("  (skipped — missing data)")
            return
        mi[t] = float(row.iloc[0]["MI_uniform"])

    base = mi["SFSP"]
    g_phase = mi["SFDP"] - base
    g_freq = mi["DFSP"] - base
    g_joint = mi["DFDP"] - base
    redundancy = (g_phase + g_freq) - g_joint

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) bar comparison
    ax = axes[0]
    vals = [mi[t] for t in tasks]
    colors = [TASK_COLORS[t] for t in tasks]
    bars = ax.bar([TASK_LABELS[t] for t in tasks], vals,
                  color=colors, alpha=0.8, edgecolor="black", lw=0.5)
    ax.axhline(base, color="gray", ls=":", lw=1)
    ax.set_ylabel("MI (bits/selection)")
    ax.set_title(f"Encoding Comparison ({method}, {window}s)")
    ax.set_ylim(0, C0 + 0.15)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                f"{v:.3f}", ha="center", fontsize=9)

    # (b) waterfall
    ax = axes[1]
    cats = ["Baseline\n(SFSP)", "Phase\nseparation", "Frequency\nseparation",
            "Redundancy"]
    vals_w = [base, g_phase, g_freq, -redundancy]
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#e377c2"]
    bottom = 0.0
    for i, (cat, v, c) in enumerate(zip(cats, vals_w, cols)):
        if v >= 0:
            ax.bar(cat, v, bottom=bottom, color=c, alpha=0.8,
                   edgecolor="black", lw=0.5)
            ax.text(i, bottom + v / 2,
                    f"+{v:.3f}" if i > 0 else f"{v:.3f}",
                    ha="center", va="center", fontsize=9)
            bottom += v
        else:
            ax.bar(cat, -v, bottom=bottom + v, color=c, alpha=0.5,
                   edgecolor="black", lw=0.5, hatch="//")
            ax.text(i, bottom + v / 2, f"{v:.3f}",
                    ha="center", va="center", fontsize=9)
            bottom += v
    ax.axhline(mi["DFDP"], color="#d62728", ls="--", lw=1,
               label=f"DFDP = {mi['DFDP']:.3f}")
    ax.set_ylabel("MI (bits)")
    ax.set_title("Information Decomposition (Waterfall)")
    ax.set_ylim(0, C0 + 0.15)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_information_decomposition.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig04_information_decomposition.png")

    pd.DataFrame([dict(
        method=method, window=window,
        MI_SFSP=base, MI_SFDP=mi["SFDP"],
        MI_DFSP=mi["DFSP"], MI_DFDP=mi["DFDP"],
        phase_gain=g_phase, freq_gain=g_freq,
        joint_gain=g_joint, redundancy=redundancy,
        redundancy_rate=redundancy / (g_phase + g_freq),
    )]).to_csv(TABLE_DIR / "information_decomposition.csv", index=False)


def fig05_coherence_bandwidth(df: pd.DataFrame, cap: pd.DataFrame) -> None:
    """Coherence bandwidth: MI vs inter-eye frequency gap (Exp3)."""
    print("Fig 05: Coherence bandwidth...")
    method, window = "ETRCA", 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) MI vs gap, all methods
    ax = axes[0]
    for m in METHODS:
        pts = []
        for task, gap in GAPS.items():
            row = cap[(cap["task"] == task) & (cap["method"] == m) &
                      (np.isclose(cap["window"], window))]
            if not row.empty:
                pts.append((gap, float(row.iloc[0]["MI_uniform"])))
        if pts:
            g, mi = zip(*sorted(pts))
            ax.plot(g, mi, marker=METHOD_MARKERS[m], color=METHOD_COLORS[m],
                    label=m, lw=2, ms=8)
    ax.set_xlabel("Inter-eye frequency gap (Hz)")
    ax.set_ylabel("MI (bits/selection)")
    ax.set_title(f"(a) MI vs gap ({window}s)")
    ax.set_xticks([1, 3, 5])
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) per-subject DFDP1 vs DFDP5
    ax = axes[1]
    s1 = {r["subject"]: r["MI"]
          for r in subject_mi_list(df, method, "DFDP1", window)}
    s5 = {r["subject"]: r["MI"]
          for r in subject_mi_list(df, method, "DFDP5", window)}
    common = sorted(set(s1) & set(s5))
    if common:
        x = [s1[s] for s in common]
        y = [s5[s] for s in common]
        ax.scatter(x, y, c=METHOD_COLORS[method], s=40, alpha=0.7,
                   edgecolor="black", lw=0.5)
        lim = [0, max(max(x), max(y)) + 0.2]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
        pct = sum(a > b for a, b in zip(x, y)) / len(common) * 100
        ax.set_title(f"(b) gap=1 vs gap=5\n{pct:.0f}% subjects gap=1 > gap=5")
    ax.set_xlabel("MI (gap=1 Hz)")
    ax.set_ylabel("MI (gap=5 Hz)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # (c) P(Y|X) difference
    ax = axes[2]
    sub1 = _sel(df, method, "DFDP1", window)
    sub5 = _sel(df, method, "DFDP5", window)
    if not sub1.empty and not sub5.empty:
        P1 = normalize_confusion(
            confusion_counts(sub1["true"], sub1["pred"], N_TARGETS), alpha=0.5)
        P5 = normalize_confusion(
            confusion_counts(sub5["true"], sub5["pred"], N_TARGETS), alpha=0.5)
        diff = P1 - P5
        vmax = float(np.abs(diff).max())
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("(c) P(Y|X) diff: gap=1 minus gap=5")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(N_TARGETS))
        ax.set_yticks(range(N_TARGETS))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_coherence_bandwidth.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig05_coherence_bandwidth.png")


def fig06_condition_comparison(cap: pd.DataFrame) -> None:
    """All 9 conditions x 4 methods grouped bar chart at 1.0 s."""
    print("Fig 06: Condition comparison...")
    window = 1.0
    sub = cap[np.isclose(cap["window"], window)]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(ALL_TASKS))
    width = 0.19
    offsets = np.array([-1.5, -0.5, 0.5, 1.5])

    for i, m in enumerate(METHODS):
        vals = []
        for task in ALL_TASKS:
            row = sub[(sub["task"] == task) & (sub["method"] == m)]
            vals.append(float(row.iloc[0]["MI_uniform"]) if not row.empty
                        else 0.0)
        ax.bar(x + offsets[i] * width, vals, width, label=m,
               color=METHOD_COLORS[m], alpha=0.8, edgecolor="black", lw=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in ALL_TASKS],
                       rotation=25, ha="right")
    ax.set_ylabel("MI (bits/selection)")
    ax.set_title(f"All Conditions — MI at {window} s")
    ax.axhline(C0, color="gray", ls="--", lw=0.8, label=f"C0={C0:.1f}")
    for xpos in [1.5, 5.5]:
        ax.axvline(xpos, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(0.5, C0 + 0.12, "Exp 1", ha="center", fontsize=8, color="gray")
    ax.text(3.5, C0 + 0.12, "Exp 2", ha="center", fontsize=8, color="gray")
    ax.text(7.0, C0 + 0.12, "Exp 3", ha="center", fontsize=8, color="gray")
    ax.legend(ncol=5, fontsize=8, loc="upper left")
    ax.set_ylim(0, C0 + 0.3)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_condition_comparison.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig06_condition_comparison.png")


def fig07_per_subject(df: pd.DataFrame) -> None:
    """Per-subject MI for Exp2 and Exp3."""
    print("Fig 07: Per-subject analysis...")
    method, window = "ETRCA", 1.0
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Exp2 strip chart
    ax = axes[0]
    for idx, task in enumerate(EXP_TASKS["experiment2"]):
        mis = [r["MI"] for r in subject_mi_list(df, method, task, window)]
        jitter = rng.normal(0, 0.06, len(mis))
        ax.scatter(idx + jitter, mis, c=TASK_COLORS[task], s=30, alpha=0.6,
                   edgecolor="black", lw=0.3)
        ax.plot([idx - 0.2, idx + 0.2], [np.mean(mis)] * 2,
                color=TASK_COLORS[task], lw=2.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(EXP_TASKS["experiment2"])
    ax.set_ylabel("MI (bits)")
    ax.set_title(f"(a) Exp2 per-subject ({method}, {window}s)")
    ax.set_ylim(0, C0 + 0.2)
    ax.grid(alpha=0.3)

    # (b) Exp3 strip chart
    ax = axes[1]
    for idx, task in enumerate(EXP_TASKS["experiment3"]):
        mis = [r["MI"] for r in subject_mi_list(df, method, task, window)]
        jitter = rng.normal(0, 0.06, len(mis))
        ax.scatter(idx + jitter, mis, c=TASK_COLORS[task], s=30, alpha=0.6,
                   edgecolor="black", lw=0.3)
        ax.plot([idx - 0.2, idx + 0.2], [np.mean(mis)] * 2,
                color=TASK_COLORS[task], lw=2.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"gap={g} Hz" for g in [1, 3, 5]])
    ax.set_ylabel("MI (bits)")
    ax.set_title(f"(b) Exp3 per-subject ({method}, {window}s)")
    ax.set_ylim(0, C0 + 0.2)
    ax.grid(alpha=0.3)

    # (c) SFSP vs DFDP scatter
    ax = axes[2]
    s_sfsp = {r["subject"]: r["MI"]
              for r in subject_mi_list(df, method, "SFSP", window)}
    s_dfdp = {r["subject"]: r["MI"]
              for r in subject_mi_list(df, method, "DFDP", window)}
    common = sorted(set(s_sfsp) & set(s_dfdp))
    if common:
        xv = [s_sfsp[s] for s in common]
        yv = [s_dfdp[s] for s in common]
        ax.scatter(xv, yv, c="#d62728", s=40, alpha=0.7,
                   edgecolor="black", lw=0.5)
        lim = [0, C0 + 0.2]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
        pct = sum(b > a for a, b in zip(xv, yv)) / len(common) * 100
        ax.set_title(f"(c) SFSP vs DFDP\n{pct:.0f}% improved by dual-freq")
    ax.set_xlabel("MI (SFSP — single-freq)")
    ax.set_ylabel("MI (DFDP — dual-freq)")
    ax.set_xlim(0, C0 + 0.2)
    ax.set_ylim(0, C0 + 0.2)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_per_subject.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("  -> fig07_per_subject.png")


def save_per_subject_table(df: pd.DataFrame) -> None:
    print("Saving per-subject MI table...")
    rows = []
    window = 1.0
    for method in ["ETRCA", "TRCA"]:
        for task in ALL_TASKS:
            rows.extend(subject_mi_list(df, method, task, window))
    pd.DataFrame(rows).to_csv(TABLE_DIR / "per_subject_mi.csv", index=False)


# ── summary ──────────────────────────────────────────────────────────────

def print_summary(cap: pd.DataFrame) -> None:
    method, window = "ETRCA", 1.0
    print("\n" + "=" * 72)
    print(f"Decision Channel Summary ({method}, {window}s)")
    print("=" * 72)
    header = f"{'Task':<10} {'Acc%':>7} {'MI':>7} {'C_BA':>7} {'Util%':>7}"
    print(header)
    print("-" * len(header))
    for task in ALL_TASKS:
        row = cap[(cap["task"] == task) & (cap["method"] == method) &
                  (np.isclose(cap["window"], window))]
        if not row.empty:
            r = row.iloc[0]
            print(f"{task:<10} {r['accuracy']*100:>6.1f}% "
                  f"{r['MI_uniform']:>7.3f} {r['C_BA']:>7.3f} "
                  f"{r['utilization']*100:>6.1f}%")
    print("=" * 72)


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs()
    print("Loading predictions from 3 experiments...")
    df = load_all_predictions()
    print(f"  {len(df):,} rows | methods: {sorted(df['method'].unique())} "
          f"| tasks: {sorted(df['task'].unique())}")

    print("Building capacity table (all conditions x methods x windows)...")
    cap = build_capacity_table(df)
    cap.to_csv(TABLE_DIR / "capacity_by_condition_method_window.csv",
               index=False)
    print(f"  {len(cap)} rows saved")

    fig01_confusion_exp2(df)
    fig02_mi_curves(cap)
    fig03_ba_capacity(cap)
    fig04_information_decomposition(cap)
    fig05_coherence_bandwidth(df, cap)
    fig06_condition_comparison(cap)
    fig07_per_subject(df)
    save_per_subject_table(df)
    print_summary(cap)
    print("\nDone.")


if __name__ == "__main__":
    main()
