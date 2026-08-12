"""Generate eCCA subset analysis figures from completed run output.

Reads per-subject data from the first run output and summary tables.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK_DIR = Path(__file__).resolve().parent
FIG_DIR = TASK_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [5, 8, 10, 15, 20, 25, 30, 35, 40]

# ── Per-subject full-40 ITR from completed run ──
subjects = list(range(1, 36))
cca_full40 = [53, 79, 141, 162, 84, 50, 49, 12, 50, 65, 10, 132, 52, 102, 37,
              2, 26, 59, 3, 40, 18, 103, 73, 87, 122, 129, 90, 68, 6, 54,
              95, 126, 13, 110, 48]
ecca5_full40 = [206, 206, 210, 204, 210, 204, 192, 189, 172, 201, 102, 210, 204,
                210, 157, 154, 185, 185, 66, 201, 169, 210, 192, 210, 204, 210,
                208, 208, 159, 189, 210, 213, 113, 204, 196]
ecca1_full40 = [189, 192, 206, 201, 196, 172, 169, 139, 157, 170, 43, 197, 163,
                204, 114, 87, 126, 155, 25, 176, 99, 199, 182, 196, 199, 203,
                199, 194, 95, 163, 210, 206, 56, 196, 165]
rt20_full40 = [166, 170, 170, 170, 173, 163, 163, 160, 150, 170, 109, 173, 173,
               170, 164, 146, 152, 155, 93, 164, 160, 170, 163, 173, 170, 173,
               170, 166, 160, 170, 173, 173, 113, 170, 166]

# ── Mean greedy ITR from summary table ──
greedy_mean = {
    'CCA':       [60.0, 72.7, 76.3, 86.9, 87.9, 84.9, 80.7, 74.1, 67.1],
    'eCCA_5blk': [87.1, 109.8, 120.8, 143.6, 159.7, 171.9, 181.2, 186.9, 187.7],
    'eCCA_1blk': [81.5, 101.1, 111.4, 130.3, 144.9, 155.6, 161.7, 164.6, 161.3],
}

# ── Mean even ITR from summary table ──
even_mean = {
    'CCA':       [60.2, 63.3, 71.3, 75.3, 78.3, 71.9, 68.6, 67.6, 67.1],
    'eCCA_5blk': [86.2, 107.7, 121.6, 142.0, 155.4, 165.9, 175.0, 182.2, 187.7],
    'eCCA_1blk': [81.2, 97.0, 113.0, 126.7, 139.3, 144.5, 152.4, 157.3, 161.3],
}


def _save(fig, path):
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")


def main():
    colors = {'CCA': '#94a3b8', 'eCCA_1blk': '#f59e0b', 'eCCA_5blk': '#3b82f6'}
    labels_map = {'CCA': 'CCA (no cal)', 'eCCA_1blk': 'eCCA 1-block',
                  'eCCA_5blk': 'eCCA 5-block'}
    methods = ['CCA', 'eCCA_5blk', 'eCCA_1blk']

    # ── Fig 1: ITR vs K ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    ax = axes[0]
    for m in methods:
        ax.plot(K_VALUES, greedy_mean[m], 'o-', color=colors[m], linewidth=2,
                markersize=7, label=f'{labels_map[m]} greedy')
    ax.plot([20], [np.mean(rt20_full40)], 's', color='#dc2626',
            markersize=10, label=f'eCCA retrained K=20 ({np.mean(rt20_full40):.0f})', zorder=5)
    ax.set_xlabel("Codebook size K")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title("ITR vs codebook size — greedy subset")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    ax.set_xticks(K_VALUES)

    ax = axes[1]
    for m in ['eCCA_5blk', 'CCA']:
        ax.plot(K_VALUES, greedy_mean[m], 'o-', color=colors[m], linewidth=2,
                markersize=7, label=f'{labels_map[m]} greedy')
        ax.plot(K_VALUES, even_mean[m], 'D--', color=colors[m], linewidth=1.5,
                markersize=5, alpha=0.5, label=f'{labels_map[m]} even')
    ax.set_xlabel("Codebook size K")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title("Greedy vs even spacing")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.15)
    ax.set_xticks(K_VALUES)

    _save(fig, FIG_DIR / "fig_ecca_itr_vs_k.png")

    # ── Fig 2: Per-subject full-40 + retrained-20 ──
    fig, ax = plt.subplots(figsize=(16, 5.5), constrained_layout=True)
    x_pos = np.arange(len(subjects))
    w = 0.18
    bar_data = [
        ('CCA full-40', cca_full40, '#d1d5db'),
        ('eCCA-5 full-40', ecca5_full40, '#93c5fd'),
        ('eCCA-1 full-40', ecca1_full40, '#fbbf24'),
        ('eCCA-5 retrain-20', rt20_full40, '#dc2626'),
    ]
    offsets = np.linspace(-1.5 * w, 1.5 * w, len(bar_data))
    for (lbl, vals, col), off in zip(bar_data, offsets):
        ax.bar(x_pos + off, vals, w, color=col, alpha=0.7, label=lbl,
               edgecolor='white', linewidth=0.3)
    ax.set_xlabel("Subject")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title("Per-subject ITR: CCA vs eCCA variants (full-40 and retrained K=20)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects], fontsize=5.5, rotation=90)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(alpha=0.15, axis='y')
    _save(fig, FIG_DIR / "fig_ecca_per_subject_k20.png")

    # ── Fig 3: Subset benefit — eCCA drops, CCA gains ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax = axes[0]
    # CCA: greedy-20 benefit estimated from full-40 * (mean ratio)
    cca_ratio = greedy_mean['CCA'][K_VALUES.index(20)] / greedy_mean['CCA'][-1]
    ecca5_ratio = greedy_mean['eCCA_5blk'][K_VALUES.index(20)] / greedy_mean['eCCA_5blk'][-1]
    ecca1_ratio = greedy_mean['eCCA_1blk'][K_VALUES.index(20)] / greedy_mean['eCCA_1blk'][-1]

    delta_ecca5 = [rt20_full40[i] - ecca5_full40[i] for i in range(35)]
    delta_ecca1 = [rt20_full40[i] - ecca1_full40[i] for i in range(35)]

    off_map = {'eCCA5→rt20': -0.2, 'eCCA1→rt20': 0.2}
    ax.bar(x_pos - 0.2, delta_ecca5, 0.35, color='#3b82f6', alpha=0.7,
           label='eCCA-5 → retrain-20')
    ax.bar(x_pos + 0.2, delta_ecca1, 0.35, color='#f59e0b', alpha=0.7,
           label='eCCA-1 → retrain-20')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Subject")
    ax.set_ylabel("ΔITR (retrain K=20 − full 40)")
    ax.set_title("Retrain K=20 vs full-40: per-subject change")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects], fontsize=5.5, rotation=90)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, axis='y')

    ax = axes[1]
    box_data = [delta_ecca5, delta_ecca1]
    box_labels = ['eCCA-5 → rt20', 'eCCA-1 → rt20']
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5, showmeans=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(box_labels)
    bcolors = ['#3b82f6', '#f59e0b']
    for patch, col in zip(bp['boxes'], bcolors):
        patch.set_facecolor(col)
        patch.set_alpha(0.4)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("ΔITR (retrain K=20 − full 40)")
    ax.set_title("Distribution: eCCA loses from codebook reduction")
    ax.grid(alpha=0.15, axis='y')

    # annotate means
    for i, d in enumerate(box_data, 1):
        mn = np.mean(d)
        ax.annotate(f"mean={mn:.1f}", (i, mn), textcoords="offset points",
                    xytext=(35, 0), fontsize=9, color=bcolors[i - 1],
                    arrowprops=dict(arrowstyle='->', color=bcolors[i - 1], lw=0.8))

    _save(fig, FIG_DIR / "fig_ecca_subset_benefit.png")

    # ── Fig 4: Scatter ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax = axes[0]
    ax.scatter(cca_full40, ecca5_full40, c='#3b82f6', s=40, alpha=0.7,
               edgecolors='white', linewidths=0.5)
    lims = [0, max(max(cca_full40), max(ecca5_full40)) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel("CCA ITR (bpm)")
    ax.set_ylabel("eCCA 5-block ITR (bpm)")
    ax.set_title("eCCA vs CCA: full 40-class ITR")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(alpha=0.15)

    ax = axes[1]
    ax.scatter(ecca5_full40, rt20_full40, c='#dc2626', s=40, alpha=0.7,
               edgecolors='white', linewidths=0.5)
    lims2 = [0, max(max(ecca5_full40), max(rt20_full40)) * 1.1]
    ax.plot(lims2, lims2, 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel("eCCA-5 full-40 ITR (bpm)")
    ax.set_ylabel("eCCA retrain K=20 ITR (bpm)")
    ax.set_title("eCCA: full-40 vs retrained K=20")
    ax.set_xlim(lims2)
    ax.set_ylim(lims2)
    ax.set_aspect('equal')
    ax.grid(alpha=0.15)

    _save(fig, FIG_DIR / "fig_ecca_gain_scatter.png")

    # Summary
    print(f"\n  eCCA full-40: mean={np.mean(ecca5_full40):.1f} bpm")
    print(f"  eCCA retrain-20: mean={np.mean(rt20_full40):.1f} bpm  "
          f"(Δ={np.mean(rt20_full40)-np.mean(ecca5_full40):.1f})")
    print(f"  eCCA 1-block: mean={np.mean(ecca1_full40):.1f} bpm")
    print(f"  CCA full-40: mean={np.mean(cca_full40):.1f} bpm")
    print(f"\n  4 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
