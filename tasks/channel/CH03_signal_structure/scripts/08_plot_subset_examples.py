"""Visualize actual greedy-selected frequency subsets.

Shows which frequencies are selected/rejected for sample subjects,
the full selection heatmap across all subjects, and the universal subset.

Usage
-----
    python plot_subset_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vep_arena.config import (
    BENCHMARK_CHANNELS_9,
    BENCHMARK_FREQS,
    BenchmarkSpec,
    DATA_ROOT,
)
from vep_arena.data.benchmark import load_subject_filterbank
from vep_arena.methods.traditional import CCA, FBCCA, TRCA

TASK_DIR = Path(__file__).resolve().parent
FREQS = np.asarray(BENCHMARK_FREQS, dtype=np.float64)
FREQ_SORTED_IDX = np.argsort(FREQS)
FREQS_SORTED = FREQS[FREQ_SORTED_IDX]
N_FBS = 5


def _save(fig, path):
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def itr_wolpaw(n, p, t):
    if n <= 1 or p <= 0:
        return 0.0
    if p >= 1:
        return np.log2(n) * (60 / t)
    return max(0.0, np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))) * (60 / t)


def mutual_info_uniform(c):
    K = c.shape[0]
    t = c.sum()
    if t == 0:
        return 0.0
    P = c / t
    py = P.sum(axis=0)
    mi = 0.0
    for i in range(K):
        for j in range(K):
            if P[i, j] > 0 and py[j] > 0:
                mi += P[i, j] * np.log2(P[i, j] * K / py[j])
    return max(0.0, mi)


def greedy_best_subset(conf, k):
    selected = [int(np.argmax(np.diag(conf)))]
    while len(selected) < k:
        best_c, best_mi = -1, -1.0
        for c in range(40):
            if c in selected:
                continue
            trial = selected + [c]
            mi = mutual_info_uniform(conf[np.ix_(trial, trial)])
            if mi > best_mi:
                best_mi = mi
                best_c = c
        selected.append(best_c)
    return selected


def run_subject_cca(data_root, subject, window, spec):
    epochs = load_subject_filterbank(data_root, subject, window, N_FBS, BENCHMARK_CHANNELS_9, spec)
    n_classes, n_blocks = spec.classes, spec.blocks
    n_samples = epochs.shape[-1]

    conf = np.zeros((40, 40))
    all_scores = []
    for block in range(n_blocks):
        test_x = epochs[:, block]
        cca = CCA(window=window, harmonics=5, spec=spec)
        _, scores = cca.predict(test_x)
        all_scores.append(scores)
        for cls in range(40):
            conf[cls, np.argmax(scores[cls])] += 1

    per_class_acc = np.diag(conf) / conf.sum(axis=1)
    return conf, per_class_acc


def main():
    spec = BenchmarkSpec()
    data_root = Path(DATA_ROOT)
    window = 1.0
    trial_time = 1.5
    fig_dir = TASK_DIR / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    subjects = list(range(1, 36))
    K = 20

    print("Collecting CCA confusion matrices for all subjects...")
    all_conf = {}
    all_per_class_acc = {}
    for s in subjects:
        conf, pca = run_subject_cca(data_root, s, window, spec)
        all_conf[s] = conf
        all_per_class_acc[s] = pca
        acc = np.trace(conf) / conf.sum()
        print(f"  S{s:02d}: acc={acc:.1%}")

    # compute per-subject greedy subsets
    all_subsets = {}
    all_acc40 = {}
    for s in subjects:
        all_subsets[s] = greedy_best_subset(all_conf[s], K)
        all_acc40[s] = np.trace(all_conf[s]) / all_conf[s].sum()

    # ── Fig 1: Selection heatmap (all subjects × all frequencies) ────
    sel_matrix = np.zeros((len(subjects), 40))
    for i, s in enumerate(subjects):
        for idx in all_subsets[s]:
            sel_matrix[i, idx] = 1
    sel_sorted = sel_matrix[:, FREQ_SORTED_IDX]

    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    im = ax.imshow(sel_sorted, aspect="auto", cmap="Blues", interpolation="nearest",
                   vmin=0, vmax=1)
    ax.set_xlabel("Target frequency (Hz)")
    ax.set_ylabel("Subject")
    ax.set_xticks(range(40))
    ax.set_xticklabels([f"{f:.1f}" for f in FREQS_SORTED], rotation=90, fontsize=6)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([f"S{s:02d} ({all_acc40[s]:.0%})" for s in subjects], fontsize=6)
    ax.set_title(f"CCA greedy subset selection (K={K}) — blue = selected, white = rejected", fontsize=12)

    # mark universal subset (top-K most selected)
    freq_count = sel_matrix.sum(axis=0)
    universal_idx = set(np.argsort(freq_count)[-K:])
    for j_sorted in range(40):
        j_orig = FREQ_SORTED_IDX[j_sorted]
        if j_orig in universal_idx:
            ax.axvline(j_sorted, color="#dc2626", alpha=0.15, linewidth=3)

    _save(fig, fig_dir / "fig_subset_selection_heatmap.png")

    # ── Fig 2: Sample subjects — frequency axis with selection ────
    # pick 6 representative subjects: 2 strong, 2 medium, 2 weak
    sorted_by_acc = sorted(subjects, key=lambda s: all_acc40[s])
    sample_subjects = (
        [sorted_by_acc[-1], sorted_by_acc[-2]] +  # strongest
        [sorted_by_acc[len(sorted_by_acc)//2], sorted_by_acc[len(sorted_by_acc)//2 + 1]] +  # median
        [sorted_by_acc[0], sorted_by_acc[1]]  # weakest
    )

    fig, axes = plt.subplots(len(sample_subjects), 1, figsize=(16, 2.2 * len(sample_subjects)),
                             constrained_layout=True, sharex=True)

    for ax, s in zip(axes, sample_subjects):
        subset = all_subsets[s]
        subset_set = set(subset)
        pca = all_per_class_acc[s]

        # bar chart: per-class accuracy, colored by selected/rejected
        for j_sorted in range(40):
            j_orig = FREQ_SORTED_IDX[j_sorted]
            color = "#3b82f6" if j_orig in subset_set else "#e2e8f0"
            edgecolor = "#1e40af" if j_orig in subset_set else "#94a3b8"
            ax.bar(j_sorted, pca[j_orig], color=color, edgecolor=edgecolor,
                   linewidth=0.5, width=0.8)

        # compute K-class ITR
        conf_k = np.zeros((K, K))
        conf40 = all_conf[s]
        for li, gi in enumerate(subset):
            for lj, gj in enumerate(subset):
                conf_k[li, lj] = conf40[gi, gj]
        acc_k = np.trace(conf_k) / max(conf_k.sum(), 1)
        itr40 = itr_wolpaw(40, all_acc40[s], trial_time)
        itr_k = itr_wolpaw(K, acc_k, trial_time)

        ax.set_ylabel("Class acc", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(
            f"S{s:02d} — 40-class: {all_acc40[s]:.0%} ({itr40:.0f} bpm)  →  "
            f"K={K}: {acc_k:.0%} ({itr_k:.0f} bpm, {'+' if itr_k > itr40 else ''}{itr_k - itr40:.0f})",
            fontsize=9, loc="left"
        )
        ax.grid(alpha=0.1, axis="y")

    axes[-1].set_xticks(range(40))
    axes[-1].set_xticklabels([f"{f:.1f}" for f in FREQS_SORTED], rotation=90, fontsize=6)
    axes[-1].set_xlabel("Target frequency (Hz)")

    # legend
    sel_patch = mpatches.Patch(color="#3b82f6", label="Selected")
    rej_patch = mpatches.Patch(color="#e2e8f0", label="Rejected")
    fig.legend(handles=[sel_patch, rej_patch], loc="upper right", fontsize=9)
    fig.suptitle(f"CCA greedy K={K} subset — per-class accuracy (strong → weak subjects)", fontsize=12, x=0.4)
    _save(fig, fig_dir / "fig_subset_examples.png")

    # ── Fig 3: Universal subset on frequency line ────
    freq_count = sel_matrix.sum(axis=0)
    freq_count_sorted = freq_count[FREQ_SORTED_IDX]
    universal_mask = np.zeros(40, dtype=bool)
    top_k_idx = np.argsort(freq_count)[-K:]
    universal_mask[top_k_idx] = True
    universal_sorted = universal_mask[FREQ_SORTED_IDX]

    fig, axes = plt.subplots(2, 1, figsize=(16, 6), constrained_layout=True)

    # top: selection count
    colors = ["#3b82f6" if universal_sorted[j] else "#e2e8f0" for j in range(40)]
    axes[0].bar(range(40), freq_count_sorted, color=colors, edgecolor="#64748b", linewidth=0.3, width=0.8)
    axes[0].axhline(len(subjects) * K / 40, color="#dc2626", linestyle="--", alpha=0.5,
                    label=f"Random baseline ({len(subjects)*K/40:.0f})")
    axes[0].set_ylabel("Selection count\n(out of 35)")
    axes[0].set_title(f"Universal K={K} subset: frequencies selected by most subjects (blue = in universal set)", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.15, axis="y")

    # bottom: the actual frequency line
    for j in range(40):
        f = FREQS_SORTED[j]
        if universal_sorted[j]:
            axes[1].plot(f, 0, "o", color="#3b82f6", markersize=10, markeredgecolor="#1e40af", zorder=3)
            axes[1].text(f, 0.15, f"{f:.1f}", ha="center", fontsize=5.5, rotation=90)
        else:
            axes[1].plot(f, 0, "x", color="#d1d5db", markersize=6, zorder=2)

    axes[1].axhline(0, color="#94a3b8", linewidth=0.5)
    axes[1].set_xlim(7.8, 16.0)
    axes[1].set_ylim(-0.3, 0.6)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_yticks([])
    axes[1].set_title("● = selected     × = rejected", fontsize=9)

    # show spacings between selected
    sel_freqs = sorted(FREQS_SORTED[universal_sorted])
    for i in range(len(sel_freqs) - 1):
        mid = (sel_freqs[i] + sel_freqs[i + 1]) / 2
        gap = sel_freqs[i + 1] - sel_freqs[i]
        axes[1].annotate(f"{gap:.1f}", xy=(mid, -0.15), fontsize=5, ha="center",
                         color="#64748b", alpha=0.7)

    for ax in axes:
        ax.set_xticks(range(40))
        ax.set_xticklabels([f"{f:.1f}" for f in FREQS_SORTED], rotation=90, fontsize=6)

    _save(fig, fig_dir / "fig_universal_subset.png")

    print(f"\n  Universal K={K}:")
    print(f"    {[f'{f:.1f}' for f in sel_freqs]}")
    print(f"    Spacings: {[f'{d:.1f}' for d in np.diff(sel_freqs)]}")
    print(f"\n  3 figures written to {fig_dir}")


if __name__ == "__main__":
    main()
