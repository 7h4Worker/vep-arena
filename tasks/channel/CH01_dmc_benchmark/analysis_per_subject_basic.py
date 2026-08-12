"""Per-subject basic descriptive analysis of Benchmark decision channel capacity.

No channel model assumptions (no exp fit, no ITR, no overhead).
Pure descriptive: distributions, variability, method comparison at subject level.
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
TASKS = PROJECT_ROOT / "tasks"
CSV = TASKS / "benchmark_decision_channel_capacity" / "results" / "extended" / "combined" / "analysis" / "capacity_by_subject_method_window.csv"
OUTPUT = TASKS / "benchmark_decision_channel_capacity" / "figures_per_subject_v20260804"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
df = df.rename(columns={"c_ba": "C_BA", "i_uniform": "MI", "accuracy": "acc"})
print(f"Loaded: {df.shape[0]} rows, {df['subject'].nunique()} subjects, "
      f"{df['method'].nunique()} methods, {df['window'].nunique()} windows")

METHODS = sorted(df["method"].unique())
SUBJECTS = sorted(df["subject"].unique())
N_SUB = len(SUBJECTS)
METHOD_COLORS = {"CCA": "#1f77b4", "ECCA": "#ff7f0e", "ETRCA": "#2ca02c",
                 "FBCCA": "#d62728", "TRCA": "#9467bd"}

# Representative windows for detailed analysis
REP_WINDOWS = [0.5, 1.0, 2.0, 3.0, 5.0]

# ═══════════════════ Fig 1: C_BA per subject, all methods, selected windows ═══════════════════

fig, axes = plt.subplots(len(REP_WINDOWS), 1, figsize=(14, 3.2 * len(REP_WINDOWS)),
                         constrained_layout=True, sharex=True)
for i, tw in enumerate(REP_WINDOWS):
    ax = axes[i]
    sub_data = df[np.isclose(df["window"], tw, atol=0.005)]
    pivot = sub_data.pivot_table(index="subject", columns="method", values="C_BA")
    x = np.arange(N_SUB)
    width = 0.15
    for j, m in enumerate(METHODS):
        if m in pivot.columns:
            vals = pivot[m].reindex(SUBJECTS).values
            ax.bar(x + j * width, vals, width, color=METHOD_COLORS[m], alpha=0.8, label=m if i == 0 else "")
    ax.set_ylabel("C_BA (bits)")
    ax.set_title(f"T = {tw}s", fontsize=10, loc="left")
    ax.axhline(np.log2(40), color="black", ls="--", lw=0.7, alpha=0.5)
    ax.set_ylim(0, np.log2(40) + 0.3)
    ax.grid(alpha=0.2, axis="y")
    if i == 0:
        ax.legend(fontsize=8, ncol=5, loc="upper right")

axes[-1].set_xticks(x + width * 2)
axes[-1].set_xticklabels([f"S{s}" for s in SUBJECTS], fontsize=7, rotation=45)
axes[-1].set_xlabel("Subject")
fig.suptitle("Per-Subject C_BA by Method at Representative Windows", fontsize=12)
fig.savefig(OUTPUT / "fig01_cba_per_subject_bars.png", dpi=150)
plt.close(fig)
print("  fig01_cba_per_subject_bars.png")

# ═══════════════════ Fig 2: Boxplot — C_BA distribution across subjects ═══════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

# (a) Boxplot per method at T=2s
ax = axes[0]
tw = 2.0
sub_data = df[np.isclose(df["window"], tw, atol=0.005)]
box_data = [sub_data[sub_data["method"] == m]["C_BA"].values for m in METHODS]
bp = ax.boxplot(box_data, tick_labels=METHODS, patch_artist=True, widths=0.5)
for patch, m in zip(bp["boxes"], METHODS):
    patch.set_facecolor(METHOD_COLORS[m])
    patch.set_alpha(0.6)
ax.axhline(np.log2(40), color="black", ls="--", lw=0.7)
ax.set_ylabel("C_BA (bits/symbol)")
ax.set_title(f"(a) C_BA distribution across {N_SUB} subjects at T={tw}s")
ax.grid(alpha=0.2, axis="y")

# (b) Boxplot per window (best method per subject)
ax = axes[1]
best_per_sub = df.loc[df.groupby(["subject", "window"])["C_BA"].idxmax()]
windows_sel = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
box_data2 = []
labels2 = []
for tw in windows_sel:
    vals = best_per_sub[np.isclose(best_per_sub["window"], tw, atol=0.005)]["C_BA"].values
    if len(vals) > 0:
        box_data2.append(vals)
        labels2.append(f"{tw}s")
bp2 = ax.boxplot(box_data2, tick_labels=labels2, patch_artist=True, widths=0.5)
for patch in bp2["boxes"]:
    patch.set_facecolor("#4C72B0")
    patch.set_alpha(0.5)
ax.axhline(np.log2(40), color="black", ls="--", lw=0.7)
ax.set_xlabel("Window")
ax.set_ylabel("C_BA (bits/symbol)")
ax.set_title(f"(b) Best-method C_BA across subjects (envelope per subject)")
ax.grid(alpha=0.2, axis="y")

fig.suptitle("Benchmark: Subject-Level C_BA Variability", fontsize=12)
fig.savefig(OUTPUT / "fig02_cba_boxplots.png", dpi=150)
plt.close(fig)
print("  fig02_cba_boxplots.png")

# ═══════════════════ Fig 3: C_BA growth curves — per subject (spaghetti) ═══════════════════

fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
axes = axes.flatten()

for idx, method in enumerate(METHODS):
    ax = axes[idx]
    mdata = df[df["method"] == method]
    for sub in SUBJECTS:
        sdata = mdata[mdata["subject"] == sub].sort_values("window")
        ax.plot(sdata["window"], sdata["C_BA"], lw=0.6, alpha=0.4, color=METHOD_COLORS[method])
    # Group mean + std
    grp = mdata.groupby("window")["C_BA"].agg(["mean", "std"]).reset_index()
    ax.plot(grp["window"], grp["mean"], "k-", lw=2, label="Mean")
    ax.fill_between(grp["window"], grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                    alpha=0.2, color="gray", label="±1 SD")
    ax.axhline(np.log2(40), color="black", ls="--", lw=0.7)
    ax.set_title(method, fontsize=11)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("C_BA")
    ax.set_xlim(0, 5.2)
    ax.set_ylim(0, np.log2(40) + 0.5)
    ax.grid(alpha=0.2)
    if idx == 0:
        ax.legend(fontsize=8)

# (f) Best method per subject — spaghetti
ax = axes[5]
for sub in SUBJECTS:
    sdata = best_per_sub[best_per_sub["subject"] == sub].sort_values("window")
    ax.plot(sdata["window"], sdata["C_BA"], lw=0.6, alpha=0.4, color="#4C72B0")
grp = best_per_sub.groupby("window")["C_BA"].agg(["mean", "std"]).reset_index()
ax.plot(grp["window"], grp["mean"], "k-", lw=2, label="Mean")
ax.fill_between(grp["window"], grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                alpha=0.2, color="gray")
ax.axhline(np.log2(40), color="black", ls="--", lw=0.7)
ax.set_title("Best method (per subject)", fontsize=11)
ax.set_xlabel("Window (s)")
ax.set_ylabel("C_BA")
ax.set_xlim(0, 5.2)
ax.set_ylim(0, np.log2(40) + 0.5)
ax.grid(alpha=0.2)
ax.legend(fontsize=8)

fig.suptitle("Benchmark: Per-Subject C_BA Growth Curves (35 subjects, thin lines)", fontsize=12)
fig.savefig(OUTPUT / "fig03_spaghetti_curves.png", dpi=150)
plt.close(fig)
print("  fig03_spaghetti_curves.png")

# ═══════════════════ Fig 4: Subject ranking stability ═══════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

# (a) Subject ranking at short vs long window
ax = axes[0]
tw_short, tw_long = 1.0, 5.0
short_best = best_per_sub[np.isclose(best_per_sub["window"], tw_short, atol=0.005)].set_index("subject")["C_BA"]
long_best = best_per_sub[np.isclose(best_per_sub["window"], tw_long, atol=0.005)].set_index("subject")["C_BA"]
common = short_best.index.intersection(long_best.index)
ax.scatter(short_best[common], long_best[common], c="#4C72B0", s=40, alpha=0.7, edgecolor="black", lw=0.3)
for s in common:
    ax.annotate(str(s), (short_best[s], long_best[s]), fontsize=6, alpha=0.7)
# diagonal
lims = [0, np.log2(40) + 0.3]
ax.plot(lims, lims, "k--", lw=0.7, alpha=0.5)
ax.set_xlabel(f"C_BA at T={tw_short}s")
ax.set_ylabel(f"C_BA at T={tw_long}s")
ax.set_title(f"(a) Subject ranking: T={tw_short}s vs T={tw_long}s")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.grid(alpha=0.2)
r = np.corrcoef(short_best[common], long_best[common])[0, 1]
ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10)

# (b) CV across subjects as a function of window
ax = axes[1]
for method in METHODS:
    mdata = df[df["method"] == method]
    grp = mdata.groupby("window")["C_BA"].agg(["mean", "std"]).reset_index()
    grp["cv"] = grp["std"] / grp["mean"] * 100
    ax.plot(grp["window"], grp["cv"], "-", color=METHOD_COLORS[method], lw=1.5, label=method)
# Best method
grp_best = best_per_sub.groupby("window")["C_BA"].agg(["mean", "std"]).reset_index()
grp_best["cv"] = grp_best["std"] / grp_best["mean"] * 100
ax.plot(grp_best["window"], grp_best["cv"], "k-", lw=2, label="Best/subject")
ax.set_xlabel("Window (s)")
ax.set_ylabel("CV (%) across subjects")
ax.set_title("(b) Inter-subject variability (coefficient of variation)")
ax.legend(fontsize=8)
ax.set_xlim(0, 5.2)
ax.grid(alpha=0.2)

fig.suptitle("Benchmark: Subject Ranking and Variability", fontsize=12)
fig.savefig(OUTPUT / "fig04_subject_ranking.png", dpi=150)
plt.close(fig)
print("  fig04_subject_ranking.png")

# ═══════════════════ Fig 5: Method dominance — which method wins per subject ═══════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

# (a) Heatmap: best method per (subject, window_bin)
ax = axes[0]
window_bins = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
win_labels = []
best_method_matrix = np.zeros((N_SUB, len(window_bins)), dtype=int)
method_to_int = {m: i for i, m in enumerate(METHODS)}

for j, tw in enumerate(window_bins):
    win_labels.append(f"{tw}s")
    for i, sub in enumerate(SUBJECTS):
        sub_at_w = df[(df["subject"] == sub) & (np.isclose(df["window"], tw, atol=0.005))]
        if not sub_at_w.empty:
            best_m = sub_at_w.loc[sub_at_w["C_BA"].idxmax(), "method"]
            best_method_matrix[i, j] = method_to_int[best_m]

im = ax.imshow(best_method_matrix, aspect="auto", cmap=plt.cm.Set1, vmin=0, vmax=len(METHODS)-1)
ax.set_xticks(range(len(window_bins)))
ax.set_xticklabels(win_labels)
ax.set_yticks(range(N_SUB))
ax.set_yticklabels([f"S{s}" for s in SUBJECTS], fontsize=6)
ax.set_xlabel("Window")
ax.set_ylabel("Subject")
ax.set_title("(a) Best method per subject at each window")
# legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=plt.cm.Set1(method_to_int[m] / (len(METHODS)-1)),
                         label=m) for m in METHODS]
ax.legend(handles=legend_elements, fontsize=7, loc="upper right", ncol=2)

# (b) Fraction of subjects where each method is best
ax = axes[1]
fractions = {}
all_windows = sorted(df["window"].unique())
for method in METHODS:
    frac_list = []
    for tw in all_windows:
        counts = 0
        for sub in SUBJECTS:
            sub_at_w = df[(df["subject"] == sub) & (np.isclose(df["window"], tw, atol=0.005))]
            if not sub_at_w.empty and sub_at_w.loc[sub_at_w["C_BA"].idxmax(), "method"] == method:
                counts += 1
        frac_list.append(counts / N_SUB * 100)
    fractions[method] = frac_list
    ax.plot(all_windows, frac_list, "-", color=METHOD_COLORS[method], lw=1.5, label=method)
ax.set_xlabel("Window (s)")
ax.set_ylabel("% of subjects where method is best")
ax.set_title("(b) Method dominance across window lengths")
ax.legend(fontsize=8)
ax.set_xlim(0, 5.2)
ax.set_ylim(0, 100)
ax.grid(alpha=0.2)

fig.suptitle("Benchmark: Method Dominance per Subject", fontsize=12)
fig.savefig(OUTPUT / "fig05_method_dominance.png", dpi=150)
plt.close(fig)
print("  fig05_method_dominance.png")

# ═══════════════════ Summary stats table ═══════════════════

print("\n" + "=" * 90)
print("Per-Subject Descriptive Summary (Benchmark, 35 subjects)")
print("=" * 90)
print(f"{'Window':>7s} {'Method':>7s} {'Mean C_BA':>10s} {'SD':>7s} {'Min':>7s} {'Q25':>7s} "
      f"{'Median':>7s} {'Q75':>7s} {'Max':>7s} {'CV%':>6s}")
print("-" * 90)
for tw in [0.5, 1.0, 2.0, 5.0]:
    for method in METHODS:
        vals = df[(df["method"] == method) & (np.isclose(df["window"], tw, atol=0.005))]["C_BA"]
        if len(vals) == 0:
            continue
        print(f"{tw:7.1f} {method:>7s} {vals.mean():10.3f} {vals.std():7.3f} {vals.min():7.3f} "
              f"{vals.quantile(0.25):7.3f} {vals.median():7.3f} {vals.quantile(0.75):7.3f} "
              f"{vals.max():7.3f} {vals.std()/vals.mean()*100:6.1f}")
    # best per subject
    bvals = best_per_sub[np.isclose(best_per_sub["window"], tw, atol=0.005)]["C_BA"]
    if len(bvals) > 0:
        print(f"{tw:7.1f} {'BEST':>7s} {bvals.mean():10.3f} {bvals.std():7.3f} {bvals.min():7.3f} "
              f"{bvals.quantile(0.25):7.3f} {bvals.median():7.3f} {bvals.quantile(0.75):7.3f} "
              f"{bvals.max():7.3f} {bvals.std()/bvals.mean()*100:6.1f}")
    print()

print(f"\nOutputs: {OUTPUT}")
