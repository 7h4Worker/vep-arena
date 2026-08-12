"""Integrate eCCA into the codebook subset analysis framework.

Core questions:
  1. Does eCCA benefit from frequency subset selection (greedy/even)?
  2. Does eCCA's calibration already fill the dimensions that
     subset optimization was trying to address?
  3. eCCA-retrained on K classes vs eCCA-fixed (40-class, restrict to K)?

Cross-validated: leave-one-block-out (6 folds).
Also tests with 1-block calibration.

Usage
-----
    python analyze_ecca_subset.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vep_arena.config import (
    BENCHMARK_CHANNELS_9,
    BENCHMARK_FREQS,
    BENCHMARK_PHASES_PI,
    BenchmarkSpec,
    DATA_ROOT,
)
from vep_arena.data.benchmark import load_subject_filterbank
from vep_arena.methods.traditional import CCA
from vep_arena.methods.ecca import ECCA

TASK_DIR = Path(__file__).resolve().parent
FREQS = np.asarray(BENCHMARK_FREQS, dtype=np.float64)
PHASES_PI = list(BENCHMARK_PHASES_PI)
FREQ_SORTED_IDX = np.argsort(FREQS)
N_FBS = 5


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


def greedy_subset(conf, k):
    sel = [int(np.argmax(np.diag(conf)))]
    while len(sel) < k:
        best_c, best_mi = -1, -1.0
        for c in range(conf.shape[0]):
            if c in sel:
                continue
            trial = sel + [c]
            mi = mutual_info_uniform(conf[np.ix_(trial, trial)])
            if mi > best_mi:
                best_mi, best_c = mi, c
        sel.append(best_c)
    return sel


def even_subset(k):
    step = 40 / k
    return [int(FREQ_SORTED_IDX[int(i * step)]) for i in range(k)]


def eval_subset(subset, all_scores_list, trial_time=1.5):
    K = len(subset)
    correct = total = 0
    for scores in all_scores_list:
        for gi in subset:
            pred_gi = subset[int(np.argmax(scores[gi][subset]))]
            if pred_gi == gi:
                correct += 1
            total += 1
    acc = correct / max(total, 1)
    return acc, itr_wolpaw(K, acc, trial_time)


def make_train_xy(epochs, train_blocks):
    """epochs: (40, n_blocks, n_fbs, n_ch, n_samples) → train_x, train_y."""
    n_train = len(train_blocks)
    sub = epochs[:, train_blocks]  # (40, n_train, n_fbs, n_ch, n_samp)
    train_x = sub.reshape(-1, *sub.shape[2:])  # (40*n_train, n_fbs, n_ch, n_samp)
    train_y = np.repeat(np.arange(40), n_train)
    return train_x, train_y


def main():
    spec = BenchmarkSpec()
    data_root = Path(DATA_ROOT)
    window = 1.0
    trial_time = 1.5
    fig_dir = TASK_DIR / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    subjects = list(range(1, 36))
    K_VALUES = [5, 8, 10, 15, 20, 25, 30, 35, 40]

    methods = ['CCA', 'eCCA_5blk', 'eCCA_1blk']
    full40_itr = {m: [] for m in methods}
    full40_acc = {m: [] for m in methods}
    greedy_itr = {m: {K: [] for K in K_VALUES} for m in methods}
    even_itr = {m: {K: [] for K in K_VALUES} for m in methods}

    retrain_K = 20
    ecca_retrain_itr = []

    t0 = time.time()
    for si, subj in enumerate(subjects):
        print(f"  [{si+1}/{len(subjects)}] S{subj:02d}", end="", flush=True)
        st = time.time()

        epochs = load_subject_filterbank(data_root, subj, window, N_FBS,
                                         BENCHMARK_CHANNELS_9, spec)

        # ── CCA (no calibration) ──
        cca = CCA(window=window, harmonics=5, spec=spec)
        cca_scores_all = []
        cca_conf = np.zeros((40, 40))
        for block in range(spec.blocks):
            _, scores = cca.predict(epochs[:, block])
            cca_scores_all.append(scores)
            for c in range(40):
                cca_conf[c, np.argmax(scores[c])] += 1

        acc40 = np.trace(cca_conf) / cca_conf.sum()
        full40_itr['CCA'].append(itr_wolpaw(40, acc40, trial_time))
        full40_acc['CCA'].append(acc40)

        for K in K_VALUES:
            if K >= 40:
                greedy_itr['CCA'][K].append(full40_itr['CCA'][-1])
                even_itr['CCA'][K].append(full40_itr['CCA'][-1])
                continue
            _, gi = eval_subset(greedy_subset(cca_conf, K), cca_scores_all, trial_time)
            greedy_itr['CCA'][K].append(gi)
            _, ei = eval_subset(even_subset(K), cca_scores_all, trial_time)
            even_itr['CCA'][K].append(ei)

        # ── eCCA 5-block (leave-one-out) ──
        ecca5_scores = []
        ecca5_conf = np.zeros((40, 40))
        for test_b in range(spec.blocks):
            train_blocks = [b for b in range(spec.blocks) if b != test_b]
            train_x, train_y = make_train_xy(epochs, train_blocks)

            ecca = ECCA(window=window, harmonics=5, n_fbs=N_FBS, spec=spec)
            ecca.fit(train_x, train_y)
            _, scores = ecca.predict(epochs[:, test_b])
            ecca5_scores.append(scores)
            for c in range(40):
                ecca5_conf[c, np.argmax(scores[c])] += 1

        acc40 = np.trace(ecca5_conf) / ecca5_conf.sum()
        full40_itr['eCCA_5blk'].append(itr_wolpaw(40, acc40, trial_time))
        full40_acc['eCCA_5blk'].append(acc40)

        for K in K_VALUES:
            if K >= 40:
                greedy_itr['eCCA_5blk'][K].append(full40_itr['eCCA_5blk'][-1])
                even_itr['eCCA_5blk'][K].append(full40_itr['eCCA_5blk'][-1])
                continue
            sub_g = greedy_subset(ecca5_conf, K)
            _, gi = eval_subset(sub_g, ecca5_scores, trial_time)
            greedy_itr['eCCA_5blk'][K].append(gi)
            _, ei = eval_subset(even_subset(K), ecca5_scores, trial_time)
            even_itr['eCCA_5blk'][K].append(ei)

        # ── eCCA retrained K=20 ──
        sub_g20 = greedy_subset(ecca5_conf, retrain_K)
        fold_itrs_rt = []
        for test_b in range(spec.blocks):
            train_blocks = [b for b in range(spec.blocks) if b != test_b]
            sub_epochs = epochs[sub_g20]  # (K, 6, n_fbs, n_ch, n_samp)
            sub_train = sub_epochs[:, train_blocks]
            train_x_k = sub_train.reshape(-1, *sub_train.shape[2:])
            train_y_k = np.repeat(np.arange(retrain_K), len(train_blocks))

            ecca_k = ECCA(window=window, harmonics=5, n_fbs=N_FBS, spec=spec,
                          frequencies=[float(FREQS[i]) for i in sub_g20],
                          phases_pi=[float(PHASES_PI[i]) for i in sub_g20])
            ecca_k.fit(train_x_k, train_y_k)

            test_x_k = sub_epochs[:, test_b]
            _, scores_k = ecca_k.predict(test_x_k)
            correct = sum(1 for c in range(retrain_K) if np.argmax(scores_k[c]) == c)
            fold_itrs_rt.append(itr_wolpaw(retrain_K, correct / retrain_K, trial_time))
        ecca_retrain_itr.append(np.mean(fold_itrs_rt))

        # ── eCCA 1-block ──
        ecca1_scores = []
        ecca1_conf = np.zeros((40, 40))
        for test_b in range(spec.blocks):
            cal_b = (test_b + 1) % spec.blocks
            train_x = epochs[:, cal_b]  # (40, n_fbs, n_ch, n_samp) — already 4D
            train_y = np.arange(40)

            ecca = ECCA(window=window, harmonics=5, n_fbs=N_FBS, spec=spec)
            ecca.fit(train_x, train_y)
            _, scores = ecca.predict(epochs[:, test_b])
            ecca1_scores.append(scores)
            for c in range(40):
                ecca1_conf[c, np.argmax(scores[c])] += 1

        acc40 = np.trace(ecca1_conf) / ecca1_conf.sum()
        full40_itr['eCCA_1blk'].append(itr_wolpaw(40, acc40, trial_time))
        full40_acc['eCCA_1blk'].append(acc40)

        for K in K_VALUES:
            if K >= 40:
                greedy_itr['eCCA_1blk'][K].append(full40_itr['eCCA_1blk'][-1])
                even_itr['eCCA_1blk'][K].append(full40_itr['eCCA_1blk'][-1])
                continue
            sub_g = greedy_subset(ecca1_conf, K)
            _, gi = eval_subset(sub_g, ecca1_scores, trial_time)
            greedy_itr['eCCA_1blk'][K].append(gi)
            _, ei = eval_subset(even_subset(K), ecca1_scores, trial_time)
            even_itr['eCCA_1blk'][K].append(ei)

        elapsed = time.time() - st
        print(f"  CCA={full40_itr['CCA'][-1]:.0f}"
              f"  eCCA5={full40_itr['eCCA_5blk'][-1]:.0f}"
              f"  eCCA1={full40_itr['eCCA_1blk'][-1]:.0f}"
              f"  rt20={ecca_retrain_itr[-1]:.0f}"
              f"  ({elapsed:.1f}s)")

    total_time = time.time() - t0

    # ── summary ──
    print(f"\n{'='*75}")
    print(f"  eCCA subset analysis  ({len(subjects)} subjects, {total_time:.0f}s)")
    print(f"{'='*75}")

    print(f"\n  Full 40-class baseline:")
    for m in methods:
        a = np.mean(full40_acc[m])
        i = np.mean(full40_itr[m])
        print(f"    {m:15s}: acc={a:.1%}  ITR={i:.1f} bpm")

    print(f"\n  Greedy subset ITR (mean bpm):")
    hdr = f"  {'K':>3s}"
    for m in methods:
        hdr += f"  {m:>12s}"
    hdr += f"  {'eCCA_retrain':>12s}"
    print(hdr)
    for K in K_VALUES:
        row = f"  {K:3d}"
        for m in methods:
            row += f"  {np.mean(greedy_itr[m][K]):12.1f}"
        if K == retrain_K:
            row += f"  {np.mean(ecca_retrain_itr):12.1f}"
        else:
            row += f"  {'—':>12s}"
        print(row)

    print(f"\n  Even spacing ITR (mean bpm):")
    hdr = f"  {'K':>3s}"
    for m in methods:
        hdr += f"  {m:>12s}"
    print(hdr)
    for K in K_VALUES:
        row = f"  {K:3d}"
        for m in methods:
            row += f"  {np.mean(even_itr[m][K]):12.1f}"
        print(row)

    print(f"\n  Greedy K=20 benefit (vs full-40):")
    for m in methods:
        g20 = np.mean(greedy_itr[m][20])
        f40 = np.mean(full40_itr[m])
        pct = (g20 - f40) / f40 * 100 if f40 > 0 else 0
        print(f"    {m:15s}: {f40:.1f} → {g20:.1f}  (Δ={g20-f40:+.1f}, {pct:+.1f}%)")
    rt20 = np.mean(ecca_retrain_itr)
    f40_5 = np.mean(full40_itr['eCCA_5blk'])
    print(f"    {'eCCA_retrain':15s}: {f40_5:.1f} → {rt20:.1f}  (Δ={rt20-f40_5:+.1f})")

    # ── figures ──
    def _save(fig, path):
        fig.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(fig)

    colors = {'CCA': '#94a3b8', 'eCCA_1blk': '#f59e0b', 'eCCA_5blk': '#3b82f6'}
    labels_map = {'CCA': 'CCA (no cal)', 'eCCA_1blk': 'eCCA 1-block',
                  'eCCA_5blk': 'eCCA 5-block'}

    # Fig 1: ITR vs K
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    ax = axes[0]
    for m in methods:
        means = [np.mean(greedy_itr[m][K]) for K in K_VALUES]
        ax.plot(K_VALUES, means, 'o-', color=colors[m], linewidth=2,
                markersize=7, label=f'{labels_map[m]} greedy')
    ax.plot([retrain_K], [np.mean(ecca_retrain_itr)], 's', color='#dc2626',
            markersize=10, label='eCCA retrained K=20', zorder=5)
    ax.set_xlabel("Codebook size K")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title("ITR vs codebook size — greedy subset")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    ax.set_xticks(K_VALUES)

    ax = axes[1]
    for m in ['eCCA_5blk', 'CCA']:
        g = [np.mean(greedy_itr[m][K]) for K in K_VALUES]
        e = [np.mean(even_itr[m][K]) for K in K_VALUES]
        ax.plot(K_VALUES, g, 'o-', color=colors[m], linewidth=2, markersize=7,
                label=f'{labels_map[m]} greedy')
        ax.plot(K_VALUES, e, 'D--', color=colors[m], linewidth=1.5, markersize=5,
                alpha=0.5, label=f'{labels_map[m]} even')
    ax.set_xlabel("Codebook size K")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title("Greedy vs even spacing")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.15)
    ax.set_xticks(K_VALUES)

    _save(fig, fig_dir / "fig_ecca_itr_vs_k.png")

    # Fig 2: per-subject at K=20
    fig, ax = plt.subplots(figsize=(16, 5.5), constrained_layout=True)
    x_pos = np.arange(len(subjects))
    w = 0.15
    bar_data = [
        ('CCA full-40', full40_itr['CCA'], '#d1d5db'),
        ('CCA greedy-20', greedy_itr['CCA'][20], '#94a3b8'),
        ('eCCA-5 full-40', full40_itr['eCCA_5blk'], '#93c5fd'),
        ('eCCA-5 greedy-20', greedy_itr['eCCA_5blk'][20], '#3b82f6'),
        ('eCCA-5 retrain-20', ecca_retrain_itr, '#dc2626'),
    ]
    offsets = np.linspace(-2 * w, 2 * w, len(bar_data))
    for (lbl, vals, col), off in zip(bar_data, offsets):
        ax.bar(x_pos + off, vals, w, color=col, alpha=0.7, label=lbl,
               edgecolor='white', linewidth=0.3)
    ax.set_xlabel("Subject")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title("Per-subject ITR: CCA vs eCCA — full-40 vs greedy K=20")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects], fontsize=5.5, rotation=90)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(alpha=0.15, axis='y')
    _save(fig, fig_dir / "fig_ecca_per_subject_k20.png")

    # Fig 3: subset benefit distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax = axes[0]
    for m in methods:
        delta = [greedy_itr[m][20][i] - full40_itr[m][i] for i in range(len(subjects))]
        off = {'CCA': -0.25, 'eCCA_1blk': 0, 'eCCA_5blk': 0.25}[m]
        ax.bar(x_pos + off, delta, 0.22, color=colors[m], alpha=0.7, label=labels_map[m])
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Subject")
    ax.set_ylabel("ΔITR (greedy K=20 − full 40)")
    ax.set_title("Subset selection benefit per subject")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects], fontsize=5.5, rotation=90)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, axis='y')

    ax = axes[1]
    box_data = []
    box_labels = []
    for m in methods:
        d = [greedy_itr[m][20][i] - full40_itr[m][i] for i in range(len(subjects))]
        box_data.append(d)
        box_labels.append(labels_map[m])
    d_rt = [ecca_retrain_itr[i] - full40_itr['eCCA_5blk'][i] for i in range(len(subjects))]
    box_data.append(d_rt)
    box_labels.append('eCCA retrain-20')
    bp = ax.boxplot(box_data, patch_artist=True,
                    widths=0.5, showmeans=True)
    ax.set_xticks(range(1, len(box_labels) + 1))
    ax.set_xticklabels(box_labels)
    bcolors = [colors['CCA'], colors['eCCA_1blk'], colors['eCCA_5blk'], '#dc2626']
    for patch, col in zip(bp['boxes'], bcolors):
        patch.set_facecolor(col)
        patch.set_alpha(0.4)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("ΔITR (K=20 − full 40)")
    ax.set_title("Distribution of subset benefit")
    ax.tick_params(axis='x', rotation=15)
    ax.grid(alpha=0.15, axis='y')
    _save(fig, fig_dir / "fig_ecca_subset_benefit.png")

    # Fig 4: scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax = axes[0]
    ax.scatter(full40_itr['CCA'], full40_itr['eCCA_5blk'],
               c='#3b82f6', s=40, alpha=0.7, edgecolors='white', linewidths=0.5)
    lims = [0, max(max(full40_itr['CCA']), max(full40_itr['eCCA_5blk'])) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel("CCA ITR (bpm)")
    ax.set_ylabel("eCCA 5-block ITR (bpm)")
    ax.set_title("eCCA vs CCA: 40-class ITR")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(alpha=0.15)

    ax = axes[1]
    cca_gain = [greedy_itr['CCA'][20][i] - full40_itr['CCA'][i] for i in range(len(subjects))]
    ecca_gain = [greedy_itr['eCCA_5blk'][20][i] - full40_itr['eCCA_5blk'][i] for i in range(len(subjects))]
    ax.scatter(cca_gain, ecca_gain, c='#3b82f6', s=40, alpha=0.7,
               edgecolors='white', linewidths=0.5)
    gl = [min(min(cca_gain), min(ecca_gain)) - 5, max(max(cca_gain), max(ecca_gain)) + 5]
    ax.plot(gl, gl, 'k--', alpha=0.3, linewidth=0.8)
    ax.axhline(0, color='#94a3b8', linewidth=0.5)
    ax.axvline(0, color='#94a3b8', linewidth=0.5)
    ax.set_xlabel("CCA: ΔITR (greedy K=20 − full 40)")
    ax.set_ylabel("eCCA: ΔITR (greedy K=20 − full 40)")
    ax.set_title("Subset benefit: CCA vs eCCA")
    ax.grid(alpha=0.15)
    _save(fig, fig_dir / "fig_ecca_gain_scatter.png")

    print(f"\n  4 figures written to {fig_dir}")
    print(f"  Total: {total_time:.0f}s")


if __name__ == "__main__":
    main()
