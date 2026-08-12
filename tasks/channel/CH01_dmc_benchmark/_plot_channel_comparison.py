"""Visualization: Cross-dataset channel reduction comparison.

Figures:
1. HD200 target×channel interaction heatmap
2. Benchmark vs HD200 η curves by channel config
3. Per-subject loss distribution (violin/swarm)
4. Benchmark spatial topology: posterior vs wholehead
5. Method sensitivity radar/bar chart
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

BENCH_CSV = "tasks/channel/CH01_dmc_benchmark/results/extended/multichannel/decision_channel/capacity_by_subject_method_channels_window.csv"
HD200_CSV = "tasks/baselines/BL05_ssvep_hd_200t/results/offline_tdca_grid/decision_channel/capacity_by_subject_targets_channels_window.csv"
OUTPUT = Path("tasks/channel/CH01_dmc_benchmark/figures_channel_comparison_v20260807")
OUTPUT.mkdir(parents=True, exist_ok=True)

bench = pd.read_csv(BENCH_CSV)
hd200 = pd.read_csv(HD200_CSV)
C0_40 = np.log2(40)

# ═══════════════════ Fig 1: HD200 Target × Channel loss heatmap ═══════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
for idx, w_ms in enumerate([200, 300, 500]):
    ax = axes[idx]
    targets_list = [40, 80, 120, 160, 200]
    channels_list = [9, 21, 32]
    loss_matrix = np.zeros((len(targets_list), len(channels_list)))
    for i, nt in enumerate(targets_list):
        ref = hd200[(hd200["targets"] == nt) & (hd200["channels"] == 66) & (hd200["window_ms"] == w_ms)]["c_ba"].mean()
        for j, ch in enumerate(channels_list):
            val = hd200[(hd200["targets"] == nt) & (hd200["channels"] == ch) & (hd200["window_ms"] == w_ms)]["c_ba"].mean()
            loss_matrix[i, j] = (ref - val) / ref * 100 if ref > 0 else 0

    im = ax.imshow(loss_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=15)
    ax.set_xticks(range(len(channels_list)))
    ax.set_xticklabels([f"{ch}ch" for ch in channels_list])
    ax.set_yticks(range(len(targets_list)))
    ax.set_yticklabels([str(t) for t in targets_list])
    ax.set_xlabel("Channel count")
    ax.set_ylabel("Target count")
    ax.set_title(f"Window = {w_ms}ms")
    for i in range(len(targets_list)):
        for j in range(len(channels_list)):
            color = "white" if loss_matrix[i, j] > 8 else "black"
            ax.text(j, i, f"{loss_matrix[i, j]:.1f}%", ha="center", va="center", fontsize=9, color=color)
    plt.colorbar(im, ax=ax, label="C_BA loss vs 66ch (%)", shrink=0.8)

fig.suptitle("HD200: Channel Reduction Loss (%) — Target Count × Channel Interaction", fontsize=12)
fig.savefig(OUTPUT / "fig01_hd200_target_channel_heatmap.png", dpi=150)
plt.close(fig)
print("  fig01_hd200_target_channel_heatmap.png")

# ═══════════════════ Fig 2: η curves — HD200 vs Benchmark ═══════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

# (a) HD200 η by channel
ax = axes[0]
hd40 = hd200[hd200["targets"] == 40]
ch_colors = {66: "#1f77b4", 32: "#2ca02c", 21: "#ff7f0e", 9: "#d62728"}
for ch in [66, 32, 21, 9]:
    grp = hd40[hd40["channels"] == ch].groupby("window")["c_ba"].mean().reset_index()
    eta = grp["c_ba"] / C0_40 * 100
    ax.plot(grp["window"], eta, "o-", color=ch_colors[ch], ms=5, lw=1.5, label=f"{ch}ch")
ax.axhline(100, color="gray", ls="--", lw=0.7)
ax.set_xlabel("Window (s)")
ax.set_ylabel("η = C_BA / C0 (%)")
ax.set_title("(a) HD200 (40 targets, TDCA)")
ax.legend(fontsize=9)
ax.set_xlim(0, 0.55)
ax.set_ylim(60, 105)
ax.grid(alpha=0.25)

# (b) Benchmark η by channel config (best method per subject)
ax = axes[1]
cfg_colors = {"full64": "#1f77b4", "posterior32": "#2ca02c", "posterior21": "#ff7f0e",
              "occipital9": "#d62728", "wholehead32": "#9467bd"}
cfg_labels = {"full64": "64ch (full)", "posterior32": "32ch (posterior)",
              "posterior21": "21ch (posterior)", "occipital9": "9ch (occipital)",
              "wholehead32": "32ch (wholehead)"}
for cfg in ["full64", "posterior32", "posterior21", "occipital9", "wholehead32"]:
    sub = bench[bench["channel_config"] == cfg]
    grp = sub.groupby(["subject", "window"])["c_ba"].max().reset_index()
    mean_eta = grp.groupby("window")["c_ba"].mean() / C0_40 * 100
    ax.plot(mean_eta.index, mean_eta.values, "o-", color=cfg_colors[cfg], ms=3, lw=1.5, label=cfg_labels[cfg])
ax.axhline(100, color="gray", ls="--", lw=0.7)
ax.set_xlabel("Window (s)")
ax.set_ylabel("η = C_BA / C0 (%)")
ax.set_title("(b) Benchmark (40 targets, best method/subject)")
ax.legend(fontsize=7.5, loc="lower right")
ax.set_xlim(0, 5.2)
ax.set_ylim(40, 105)
ax.grid(alpha=0.25)

fig.suptitle("Channel Efficiency η Curves: HD200 vs Benchmark", fontsize=12)
fig.savefig(OUTPUT / "fig02_eta_curves_by_channel.png", dpi=150)
plt.close(fig)
print("  fig02_eta_curves_by_channel.png")

# ═══════════════════ Fig 3: Per-subject loss distribution ═══════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

# Compute per-subject loss at multiple windows
for ax_idx, w_target in enumerate([0.2, 0.3]):
    ax = axes[ax_idx]

    # HD200
    hd_losses = []
    for sub in sorted(hd200[hd200["targets"] == 40]["subject"].unique()):
        full = hd200[(hd200["targets"] == 40) & (hd200["channels"] == 66) & (hd200["subject"] == sub) & np.isclose(hd200["window"], w_target, atol=0.005)]["c_ba"].values
        ch9 = hd200[(hd200["targets"] == 40) & (hd200["channels"] == 9) & (hd200["subject"] == sub) & np.isclose(hd200["window"], w_target, atol=0.005)]["c_ba"].values
        if len(full) > 0 and len(ch9) > 0 and full[0] > 0:
            hd_losses.append((full[0] - ch9[0]) / full[0] * 100)

    # Benchmark
    b_losses = []
    for sub in sorted(bench["subject"].unique()):
        full = bench[(bench["channel_config"] == "full64") & (bench["subject"] == sub) & np.isclose(bench["window"], w_target, atol=0.005)]
        ch9 = bench[(bench["channel_config"] == "occipital9") & (bench["subject"] == sub) & np.isclose(bench["window"], w_target, atol=0.005)]
        if len(full) > 0 and len(ch9) > 0:
            f_best = full["c_ba"].max()
            c9_best = ch9["c_ba"].max()
            if f_best > 0:
                b_losses.append((f_best - c9_best) / f_best * 100)

    # Box + strip
    data = [hd_losses, b_losses]
    bp = ax.boxplot(data, positions=[1, 2], widths=0.5, patch_artist=True,
                    tick_labels=["HD200\n(TDCA, 14sub)", "Benchmark\n(best, 35sub)"])
    bp["boxes"][0].set_facecolor("#ff7f0e")
    bp["boxes"][0].set_alpha(0.4)
    bp["boxes"][1].set_facecolor("#1f77b4")
    bp["boxes"][1].set_alpha(0.4)

    # Scatter individual points
    np.random.seed(42)
    for i, d in enumerate(data):
        x = np.ones(len(d)) * (i + 1) + np.random.normal(0, 0.05, len(d))
        color = "#ff7f0e" if i == 0 else "#1f77b4"
        ax.scatter(x, d, s=30, alpha=0.6, color=color, edgecolor="black", lw=0.3, zorder=3)

    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_ylabel("C_BA loss: (full - 9ch) / full × 100%")
    ax.set_title(f"Window = {w_target}s")
    ax.grid(alpha=0.2, axis="y")
    ax.text(0.05, 0.95, f"+ = 9ch worse\n– = 9ch better", transform=ax.transAxes,
            fontsize=8, va="top", ha="left", style="italic")

fig.suptitle("Per-Subject Channel Sensitivity: 9ch vs Full (HD200 vs Benchmark)", fontsize=12)
fig.savefig(OUTPUT / "fig03_subject_loss_distribution.png", dpi=150)
plt.close(fig)
print("  fig03_subject_loss_distribution.png")

# ═══════════════════ Fig 4: Benchmark topology effect ═══════════════════

fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
method_colors = {"CCA": "#1f77b4", "ECCA": "#ff7f0e", "ETRCA": "#2ca02c", "FBCCA": "#d62728", "TRCA": "#9467bd"}
for method in ["TRCA", "ECCA", "ETRCA", "FBCCA", "CCA"]:
    gains = []
    ws_valid = []
    windows = sorted(bench["window"].unique())
    for w in windows:
        post = bench[(bench["method"] == method) & (bench["channel_config"] == "posterior32") & np.isclose(bench["window"], w, atol=0.005)]["c_ba"].mean()
        whole = bench[(bench["method"] == method) & (bench["channel_config"] == "wholehead32") & np.isclose(bench["window"], w, atol=0.005)]["c_ba"].mean()
        if whole > 0.5:
            gains.append((post - whole) / whole * 100)
            ws_valid.append(w)
    ax.plot(ws_valid, gains, "-", color=method_colors[method], lw=1.5, label=method)

ax.axhline(0, color="black", ls="--", lw=0.7)
ax.set_xlabel("Window (s)")
ax.set_ylabel("Gain: (posterior32 - wholehead32) / wholehead32 × 100%")
ax.set_title("Benchmark: Spatial Topology Effect — Posterior vs Wholehead (same 32ch count)")
ax.legend(fontsize=9)
ax.set_xlim(0, 5.2)
ax.set_ylim(-5, 25)
ax.grid(alpha=0.25)
fig.savefig(OUTPUT / "fig04_topology_posterior_vs_wholehead.png", dpi=150)
plt.close(fig)
print("  fig04_topology_posterior_vs_wholehead.png")

# ═══════════════════ Fig 5: Method sensitivity bars (at 1s) ═══════════════════

fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
w = 1.0
methods = ["TRCA", "ECCA", "ETRCA", "FBCCA", "CCA"]
configs = ["occipital9", "posterior21", "posterior32", "wholehead32"]
cfg_short = ["occ9", "post21", "post32", "whole32"]
cfg_bar_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]

x = np.arange(len(methods))
bar_w = 0.18
for j, (cfg, clr) in enumerate(zip(configs, cfg_bar_colors)):
    gains = []
    for method in methods:
        ref = bench[(bench["method"] == method) & (bench["channel_config"] == "full64") & np.isclose(bench["window"], w, atol=0.005)]["c_ba"].mean()
        val = bench[(bench["method"] == method) & (bench["channel_config"] == cfg) & np.isclose(bench["window"], w, atol=0.005)]["c_ba"].mean()
        g = (val - ref) / ref * 100 if ref > 0.1 else 0
        gains.append(g)
    ax.bar(x + j * bar_w, gains, bar_w, color=clr, alpha=0.8, label=cfg_short[j], edgecolor="black", lw=0.3)

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x + bar_w * 1.5)
ax.set_xticklabels(methods)
ax.set_ylabel("Δ C_BA vs full64 (%)")
ax.set_xlabel("Method")
ax.set_title(f"Benchmark: Method Sensitivity to Channel Configuration (T={w}s)")
ax.legend(fontsize=9, title="Config")
ax.grid(alpha=0.2, axis="y")
fig.savefig(OUTPUT / "fig05_method_sensitivity_bars.png", dpi=150)
plt.close(fig)
print("  fig05_method_sensitivity_bars.png")

# ═══════════════════ Fig 6: Combined — HD200 multi-target channel curves ═══════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
for idx, w_ms in enumerate([200, 300, 500]):
    ax = axes[idx]
    for nt in [40, 80, 120, 160, 200]:
        c0 = np.log2(nt)
        vals = []
        for ch in [9, 21, 32, 66]:
            v = hd200[(hd200["targets"] == nt) & (hd200["channels"] == ch) & (hd200["window_ms"] == w_ms)]["c_ba"].mean()
            vals.append(v / c0 * 100)
        ax.plot([9, 21, 32, 66], vals, "o-", ms=6, lw=1.5, label=f"{nt} targets")
    ax.axhline(100, color="gray", ls="--", lw=0.7)
    ax.set_xlabel("Channel count")
    ax.set_ylabel("η = C_BA / C0 (%)")
    ax.set_title(f"Window = {w_ms}ms")
    ax.set_xticks([9, 21, 32, 66])
    ax.set_ylim(60, 105)
    ax.grid(alpha=0.25)
    if idx == 0:
        ax.legend(fontsize=8)

fig.suptitle("HD200: Channel Efficiency by Target Count (TDCA)", fontsize=12)
fig.savefig(OUTPUT / "fig06_hd200_eta_vs_channels.png", dpi=150)
plt.close(fig)
print("  fig06_hd200_eta_vs_channels.png")

print(f"\nAll outputs: {OUTPUT}")
