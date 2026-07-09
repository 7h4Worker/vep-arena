"""Simulate minimum-cost calibration for per-subject frequency subset selection.

Core question: how many trials / frequencies must a subject view before
we can identify a near-optimal K-frequency subset?

Protocols compared (all use CCA, K=20):
  Oracle   — greedy from pooled 6-block confusion matrix (post-hoc upper bound)
  1-block  — 1 sweep (40 freqs × 1 trial = 60 s) → greedy or profile-based
  Sparse   — M < 40 evenly-sampled freqs × 1 trial → interpolate → select
  Adaptive — bisection sweep: anchors first, then fill widest gaps
  Even     — evenly spaced K=20, zero calibration (lower bound)

Cross-validated: for each subject, each of 6 blocks serves as the
calibration block once; the other 5 are test blocks.

Usage
-----
    python simulate_calibration.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vep_arena.config import (
    BENCHMARK_CHANNELS_9,
    BENCHMARK_FREQS,
    BenchmarkSpec,
    DATA_ROOT,
)
from vep_arena.data.benchmark import load_subject_filterbank
from vep_arena.methods.traditional import CCA

TASK_DIR = Path(__file__).resolve().parent
FREQS = np.asarray(BENCHMARK_FREQS, dtype=np.float64)
FREQ_SORTED_IDX = np.argsort(FREQS)
FREQS_SORTED = FREQS[FREQ_SORTED_IDX]


# ── helpers ────────────────────────────────────────────────────────────

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
    """Greedy forward selection maximizing I_uniform."""
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


def profile_subset(response, k):
    """Select K freqs maximizing response × min-distance-to-selected.

    Naturally balances signal quality and spacing — picks strong
    frequencies that are far from already-chosen ones.
    """
    r = np.clip(response, 0, None)
    r = r / (r.max() + 1e-12)  # normalize to [0, 1]
    selected = [int(np.argmax(r))]

    while len(selected) < k:
        best_j, best_score = -1, -1.0
        for j in range(40):
            if j in selected:
                continue
            min_dist = min(abs(FREQS[j] - FREQS[s]) for s in selected)
            score = (0.3 + 0.7 * r[j]) * min_dist
            if score > best_score:
                best_score, best_j = score, j
        if best_j < 0:
            break
        selected.append(best_j)
    return selected


def eval_subset(subset, test_scores_list, trial_time=1.5):
    K = len(subset)
    correct = total = 0
    for scores in test_scores_list:
        for li, gi in enumerate(subset):
            pred_li = int(np.argmax(scores[gi][subset]))
            if subset[pred_li] == gi:
                correct += 1
            total += 1
    acc = correct / max(total, 1)
    return acc, itr_wolpaw(K, acc, trial_time)


def even_subset(k):
    step = 40 / k
    return [int(FREQ_SORTED_IDX[int(i * step)]) for i in range(k)]


def interpolate_profile(tested_orig_idx, cal_scores):
    """From M tested frequencies, interpolate response to all 40."""
    tested_freqs = sorted([(FREQS[gi], cal_scores[gi, gi]) for gi in tested_orig_idx])
    tf = np.array([x[0] for x in tested_freqs])
    tr = np.array([x[1] for x in tested_freqs])
    interp_sorted = np.interp(FREQS_SORTED, tf, tr)
    response = np.zeros(40)
    for j in range(40):
        response[FREQ_SORTED_IDX[j]] = interp_sorted[j]
    return response


def adaptive_sweep_indices(cal_scores, budget):
    """Adaptive bisection: start with anchors, fill widest untested gaps."""
    n_anchor = min(5, budget)
    anchor_sorted = np.round(np.linspace(0, 39, n_anchor)).astype(int)
    tested_sorted = set(anchor_sorted.tolist())
    remaining = budget - n_anchor

    while remaining > 0:
        sorted_list = sorted(tested_sorted)
        best_gap, best_mid = 0, -1
        for i in range(len(sorted_list) - 1):
            gap = sorted_list[i + 1] - sorted_list[i]
            if gap > best_gap:
                mid = (sorted_list[i] + sorted_list[i + 1]) // 2
                if mid not in tested_sorted:
                    best_gap, best_mid = gap, mid
        if best_mid < 0:
            break
        tested_sorted.add(best_mid)
        remaining -= 1

    return [FREQ_SORTED_IDX[j] for j in sorted(tested_sorted)]


# ── main ───────────────────────────────────────────────────────────────

def main():
    spec = BenchmarkSpec()
    data_root = Path(DATA_ROOT)
    window = 1.0
    trial_time = 1.5
    fig_dir = TASK_DIR / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    subjects = list(range(1, 36))
    K = 20
    sweep_sizes = [5, 8, 10, 15, 20, 25, 30, 40]

    # storage
    itrs = {k: [] for k in ['oracle', '1b_greedy', '1b_profile', 'even', 'full40']}
    sparse_profile = {M: [] for M in sweep_sizes}
    sparse_greedy = {M: [] for M in sweep_sizes}
    adaptive_profile = {M: [] for M in sweep_sizes}

    t0 = time.time()
    for si, subj in enumerate(subjects):
        print(f"  [{si+1}/{len(subjects)}] S{subj:02d}", end="", flush=True)

        epochs = load_subject_filterbank(data_root, subj, window, 5, BENCHMARK_CHANNELS_9, spec)
        cca = CCA(window=window, harmonics=5, spec=spec)
        block_scores = []
        for block in range(spec.blocks):
            _, scores = cca.predict(epochs[:, block])
            block_scores.append(scores)

        # baseline: full 40-class
        _, full_itr = eval_subset(list(range(40)), block_scores, trial_time)
        itrs['full40'].append(full_itr)

        # even spacing (no cal)
        _, even_itr = eval_subset(even_subset(K), block_scores, trial_time)
        itrs['even'].append(even_itr)

        # oracle: pool all 6 blocks
        oracle_conf = np.zeros((40, 40))
        for scores in block_scores:
            for c in range(40):
                oracle_conf[c, np.argmax(scores[c])] += 1
        oracle_sub = greedy_subset(oracle_conf, K)
        _, oracle_itr = eval_subset(oracle_sub, block_scores, trial_time)
        itrs['oracle'].append(oracle_itr)

        # leave-one-block-out calibration
        fold = {k: [] for k in ['1b_greedy', '1b_profile']}
        fold_sp = {M: [] for M in sweep_sizes}
        fold_sg = {M: [] for M in sweep_sizes}
        fold_ap = {M: [] for M in sweep_sizes}

        for cal_b in range(spec.blocks):
            test_sc = [block_scores[b] for b in range(spec.blocks) if b != cal_b]
            cal = block_scores[cal_b]

            # ── 1-block full sweep → greedy
            cal_conf = np.zeros((40, 40))
            for c in range(40):
                cal_conf[c, np.argmax(cal[c])] += 1
            sub_g = greedy_subset(cal_conf, K)
            _, itr_g = eval_subset(sub_g, test_sc, trial_time)
            fold['1b_greedy'].append(itr_g)

            # ── 1-block full sweep → profile
            response = np.array([cal[c, c] for c in range(40)])
            sub_p = profile_subset(response, K)
            _, itr_p = eval_subset(sub_p, test_sc, trial_time)
            fold['1b_profile'].append(itr_p)

            # ── sparse even sweep
            for M in sweep_sizes:
                if M >= 40:
                    fold_sp[M].append(itr_p)
                    fold_sg[M].append(itr_g)
                    fold_ap[M].append(itr_p)
                    continue

                # even-sampled M frequencies
                step = 40 / M
                test_sorted = [int(i * step) for i in range(M)]
                test_orig = [FREQ_SORTED_IDX[j] for j in test_sorted]

                # profile-based with interpolation
                resp_interp = interpolate_profile(test_orig, cal)
                sub_pi = profile_subset(resp_interp, K)
                _, itr_pi = eval_subset(sub_pi, test_sc, trial_time)
                fold_sp[M].append(itr_pi)

                # greedy from partial confusion (only if M >= K)
                if M >= K:
                    partial_conf = np.zeros((M, M))
                    for li, gi in enumerate(test_orig):
                        sub_s = cal[gi][test_orig]
                        partial_conf[li, np.argmax(sub_s)] += 1
                    p_sub_idx = greedy_subset(partial_conf, K)
                    p_sub = [test_orig[i] for i in p_sub_idx]
                    _, itr_pg = eval_subset(p_sub, test_sc, trial_time)
                else:
                    _, itr_pg = eval_subset(test_orig, test_sc, trial_time)
                fold_sg[M].append(itr_pg)

                # adaptive bisection sweep
                adapt_orig = adaptive_sweep_indices(cal, M)
                resp_adapt = interpolate_profile(adapt_orig, cal)
                sub_ai = profile_subset(resp_adapt, K)
                _, itr_ai = eval_subset(sub_ai, test_sc, trial_time)
                fold_ap[M].append(itr_ai)

        itrs['1b_greedy'].append(np.mean(fold['1b_greedy']))
        itrs['1b_profile'].append(np.mean(fold['1b_profile']))
        for M in sweep_sizes:
            sparse_profile[M].append(np.mean(fold_sp[M]))
            sparse_greedy[M].append(np.mean(fold_sg[M]))
            adaptive_profile[M].append(np.mean(fold_ap[M]))

        print(f"  oracle={oracle_itr:.0f}  1b_g={np.mean(fold['1b_greedy']):.0f}"
              f"  1b_p={np.mean(fold['1b_profile']):.0f}  even={even_itr:.0f}")

    elapsed = time.time() - t0

    # ── summary ────────────────────────────────────────────────────────
    om = np.mean(itrs['oracle'])
    print(f"\n{'='*65}")
    print(f"  Calibration simulation summary  (K={K}, {len(subjects)} subjects)")
    print(f"{'='*65}")
    print(f"  Full 40-class baseline:          {np.mean(itrs['full40']):6.1f} bpm")
    print(f"  Even K={K} (0 s cal):             {np.mean(itrs['even']):6.1f} bpm  ({np.mean(itrs['even'])/om:.0%} oracle)")
    print(f"  1-block profile (60 s):          {np.mean(itrs['1b_profile']):6.1f} bpm  ({np.mean(itrs['1b_profile'])/om:.0%} oracle)")
    print(f"  1-block greedy  (60 s):          {np.mean(itrs['1b_greedy']):6.1f} bpm  ({np.mean(itrs['1b_greedy'])/om:.0%} oracle)")
    print(f"  Oracle greedy   (post-hoc):      {om:6.1f} bpm  (100%)")

    print(f"\n  Sparse sweep (even-sampled M freqs, profile-based selection):")
    print(f"  {'M':>3s}  {'cal_time':>8s}  {'profile':>8s}  {'adapt':>8s}  {'greedy*':>8s}  {'%oracle':>8s}")
    for M in sweep_sizes:
        ct = f"{M * trial_time:.0f} s"
        sp = np.mean(sparse_profile[M])
        ap = np.mean(adaptive_profile[M])
        sg = np.mean(sparse_greedy[M])
        best = max(sp, ap, sg)
        print(f"  {M:3d}  {ct:>8s}  {sp:8.1f}  {ap:8.1f}  {sg:8.1f}  {best/om:7.0%}")

    print(f"\n  * greedy requires M >= K={K}; for M < K uses all M freqs")
    print(f"  Done in {elapsed:.1f}s")

    # ── figures ────────────────────────────────────────────────────────

    def _save(fig, path):
        fig.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(fig)

    # Fig 1: calibration cost curve
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    ax = axes[0]
    cal_times = [M * trial_time for M in sweep_sizes]
    sp_means = [np.mean(sparse_profile[M]) for M in sweep_sizes]
    ap_means = [np.mean(adaptive_profile[M]) for M in sweep_sizes]
    sg_means = [np.mean(sparse_greedy[M]) for M in sweep_sizes]

    ax.plot(cal_times, sp_means, "o-", color="#3b82f6", linewidth=2, markersize=7,
            label="Sparse even → profile")
    ax.plot(cal_times, ap_means, "D-", color="#8b5cf6", linewidth=2, markersize=6,
            label="Adaptive bisection → profile")
    ax.plot(cal_times, sg_means, "s--", color="#f59e0b", linewidth=1.5, markersize=5,
            label="Sparse even → greedy (M≥K)")

    ax.axhline(om, color="#16a34a", ls="-", alpha=0.5, lw=1.5,
               label=f"Oracle ({om:.0f})")
    ax.axhline(np.mean(itrs['1b_greedy']), color="#16a34a", ls="--", alpha=0.35, lw=1,
               label=f"1-block greedy ({np.mean(itrs['1b_greedy']):.0f})")
    ax.axhline(np.mean(itrs['even']), color="#94a3b8", ls="--", alpha=0.5, lw=1.5,
               label=f"Even / no cal ({np.mean(itrs['even']):.0f})")
    ax.axhline(np.mean(itrs['full40']), color="#dc2626", ls=":", alpha=0.4, lw=1,
               label=f"Full 40-class ({np.mean(itrs['full40']):.0f})")

    ax.set_xlabel("Calibration time (seconds)")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title(f"Calibration efficiency — CCA K={K}")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.15)
    ax.set_xlim(0, 65)

    # right: % of oracle
    ax = axes[1]
    rec_sp = [np.mean(sparse_profile[M]) / om * 100 for M in sweep_sizes]
    rec_ap = [np.mean(adaptive_profile[M]) / om * 100 for M in sweep_sizes]
    even_pct = np.mean(itrs['even']) / om * 100

    ax.plot(sweep_sizes, rec_sp, "o-", color="#3b82f6", linewidth=2, markersize=7,
            label="Sparse even → profile")
    ax.plot(sweep_sizes, rec_ap, "D-", color="#8b5cf6", linewidth=2, markersize=6,
            label="Adaptive bisection → profile")
    ax.axhline(100, color="#16a34a", ls="--", alpha=0.4, label="Oracle (100%)")
    ax.axhline(even_pct, color="#94a3b8", ls="--", alpha=0.5,
               label=f"Even / no cal ({even_pct:.0f}%)")

    for M, r in zip(sweep_sizes, rec_sp):
        ax.annotate(f"{r:.0f}%", (M, r), textcoords="offset points",
                    xytext=(0, 8), fontsize=7, ha="center", color="#3b82f6")

    ax.set_xlabel("Number of calibration frequencies (M)")
    ax.set_ylabel("% of oracle ITR")
    ax.set_title("How many frequencies must you sweep?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    ax.set_xticks(sweep_sizes)
    ax.set_ylim(60, 108)

    _save(fig, fig_dir / "fig_calibration_efficiency.png")

    # Fig 2: per-subject comparison of key protocols
    fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
    x = np.arange(len(subjects))
    w = 0.18
    ax.bar(x - 1.5 * w, itrs['oracle'], w, color="#16a34a", alpha=0.6, label="Oracle")
    ax.bar(x - 0.5 * w, itrs['1b_greedy'], w, color="#3b82f6", alpha=0.6, label="1-block greedy")
    ax.bar(x + 0.5 * w, itrs['even'], w, color="#94a3b8", alpha=0.6, label="Even (no cal)")
    ax.bar(x + 1.5 * w, itrs['full40'], w, color="#dc2626", alpha=0.4, label="Full 40-class")

    ax.set_xlabel("Subject")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title(f"Per-subject ITR comparison — CCA K={K}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:02d}" for s in subjects], fontsize=5.5, rotation=90)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, axis="y")

    _save(fig, fig_dir / "fig_calibration_per_subject.png")

    # Fig 3: what does 1-block profile look like? show 4 example subjects
    np.random.seed(42)
    sorted_by_acc = sorted(subjects, key=lambda s: itrs['full40'][subjects.index(s)])
    examples = [sorted_by_acc[-2], sorted_by_acc[len(sorted_by_acc)//2],
                sorted_by_acc[len(sorted_by_acc)//4], sorted_by_acc[1]]

    fig, axes = plt.subplots(len(examples), 1, figsize=(14, 2.5 * len(examples)),
                             constrained_layout=True, sharex=True)

    for ax, subj in zip(axes, examples):
        si = subjects.index(subj)
        epochs = load_subject_filterbank(data_root, subj, window, 5, BENCHMARK_CHANNELS_9, spec)
        cca = CCA(window=window, harmonics=5, spec=spec)
        _, cal = cca.predict(epochs[:, 0])  # block 0 as example

        response = np.array([cal[c, c] for c in range(40)])
        resp_sorted = response[FREQ_SORTED_IDX]

        # sparse M=10 interpolation
        M = 10
        step = 40 / M
        test_sorted = [int(i * step) for i in range(M)]
        test_orig = [FREQ_SORTED_IDX[j] for j in test_sorted]
        resp_interp = interpolate_profile(test_orig, cal)
        interp_sorted = resp_interp[FREQ_SORTED_IDX]

        ax.bar(range(40), resp_sorted, color="#e2e8f0", edgecolor="#94a3b8", lw=0.3, width=0.8,
               label="True response")
        ax.plot(range(40), interp_sorted, "o-", color="#dc2626", markersize=4, lw=1.5,
                label=f"Interpolated from {M} points")
        for j in test_sorted:
            ax.plot(j, resp_sorted[j], "D", color="#16a34a", markersize=6, zorder=5)

        oitr = itrs['oracle'][si]
        eitr = itrs['even'][si]
        ax.set_ylabel("CCA ρ", fontsize=8)
        ax.set_title(f"S{subj:02d} — oracle {oitr:.0f} bpm, even {eitr:.0f} bpm", fontsize=9, loc="left")
        ax.grid(alpha=0.1, axis="y")
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xticks(range(40))
    axes[-1].set_xticklabels([f"{f:.1f}" for f in FREQS_SORTED], rotation=90, fontsize=6)
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("CCA response profile: true vs interpolated from 10-point sweep", fontsize=11, x=0.35)

    _save(fig, fig_dir / "fig_calibration_profiles.png")

    print(f"  3 figures written to {fig_dir}")


if __name__ == "__main__":
    main()
