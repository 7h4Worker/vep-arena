"""Analyze HD200 TDCA results from the communication-channel perspective.

Uses existing offline_tdca_grid results (CSV) to study:
  1. ITR scaling with target count N — where does it plateau?
  2. Optimal N for each channel configuration
  3. Channel capacity lower bounds
  4. Comparison with Benchmark 40-target results

Usage
-----
    python analyze_hd200_capacity.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK_DIR = Path(__file__).resolve().parent
HD_TASK = TASK_DIR.parent / "ssvep_hd_200target_tdca_sample"
BEST_CSV = HD_TASK / "results" / "offline_tdca_grid" / "offline_tdca_best_itr_by_subject_config.csv"
FIG_DIR = TASK_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SETS = [40, 80, 120, 160, 200]
CHANNEL_SETS = [9, 21, 32, 66]


def itr_wolpaw(n, p, t):
    if n <= 1 or p <= 0:
        return 0.0
    if p >= 1:
        return np.log2(n) * (60 / t)
    return max(0.0, np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))) * (60 / t)


def load_data():
    rows = []
    with open(BEST_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "subject": r["subject"],
                "target_set": int(r["target_set"]),
                "channel_set": int(r["channel_set"]),
                "window_ms": int(r["window_ms"]),
                "accuracy": float(r["accuracy"]),
                "itr_bpm": float(r["itr_bpm"]),
            })
    return rows


def main():
    rows = load_data()
    subjects = sorted(set(r["subject"] for r in rows))
    print(f"  Loaded {len(rows)} rows, {len(subjects)} subjects: {subjects}")

    # ── 1. Per-subject best ITR for each (target_set, channel_set) ──
    best = {}
    for r in rows:
        key = (r["subject"], r["target_set"], r["channel_set"])
        if key not in best or r["itr_bpm"] > best[key]["itr_bpm"]:
            best[key] = r

    # ── 2. ITR vs N table ──
    print(f"\n  Mean best ITR (bpm) by target_set × channel_set:")
    hdr = f"  {'N':>4s}"
    for ch in CHANNEL_SETS:
        hdr += f"  {ch:>5d}ch"
    hdr += f"  {'best_ch':>7s}"
    print(hdr)

    itr_matrix = np.zeros((len(TARGET_SETS), len(CHANNEL_SETS)))
    acc_matrix = np.zeros((len(TARGET_SETS), len(CHANNEL_SETS)))
    win_matrix = np.zeros((len(TARGET_SETS), len(CHANNEL_SETS)))

    for ti, N in enumerate(TARGET_SETS):
        row_str = f"  {N:4d}"
        for ci, ch in enumerate(CHANNEL_SETS):
            vals = [best[(s, N, ch)]["itr_bpm"] for s in subjects if (s, N, ch) in best]
            accs = [best[(s, N, ch)]["accuracy"] for s in subjects if (s, N, ch) in best]
            wins = [best[(s, N, ch)]["window_ms"] for s in subjects if (s, N, ch) in best]
            if vals:
                itr_matrix[ti, ci] = np.mean(vals)
                acc_matrix[ti, ci] = np.mean(accs)
                win_matrix[ti, ci] = np.mean(wins)
                row_str += f"  {np.mean(vals):7.1f}"
            else:
                row_str += f"  {'—':>7s}"
        best_ch_idx = np.argmax(itr_matrix[ti])
        row_str += f"  {CHANNEL_SETS[best_ch_idx]:5d}ch"
        print(row_str)

    # optimal N for each channel set
    print(f"\n  Optimal target count by channel set:")
    for ci, ch in enumerate(CHANNEL_SETS):
        opt_ti = np.argmax(itr_matrix[:, ci])
        opt_N = TARGET_SETS[opt_ti]
        opt_itr = itr_matrix[opt_ti, ci]
        opt_acc = acc_matrix[opt_ti, ci]
        print(f"    {ch:2d}ch: N*={opt_N:3d}  ITR={opt_itr:.1f} bpm  acc={opt_acc:.1%}  (log2 N={np.log2(opt_N):.2f} bits)")

    # ── 3. Per-subject optimal N ──
    print(f"\n  Per-subject optimal N (66ch):")
    for s in subjects:
        best_n, best_itr_val = 40, 0
        for N in TARGET_SETS:
            k = (s, N, 66)
            if k in best and best[k]["itr_bpm"] > best_itr_val:
                best_itr_val = best[k]["itr_bpm"]
                best_n = N
        itr40 = best.get((s, 40, 66), {}).get("itr_bpm", 0)
        itr200 = best.get((s, 200, 66), {}).get("itr_bpm", 0)
        print(f"    {s}: N*={best_n:3d}  ITR*={best_itr_val:.1f}  "
              f"ITR@40={itr40:.1f}  ITR@200={itr200:.1f}  "
              f"ratio200/40={itr200/itr40:.2f}" if itr40 > 0 else f"    {s}: no 40-target data")

    # ── 4. Shannon limit comparison ──
    print(f"\n  Channel capacity lower bounds (peak ITR as C_lower):")
    for ci, ch in enumerate(CHANNEL_SETS):
        peak = np.max(itr_matrix[:, ci])
        peak_N = TARGET_SETS[np.argmax(itr_matrix[:, ci])]
        # bits per trial = ITR * T / 60
        T_opt = win_matrix[np.argmax(itr_matrix[:, ci]), ci] / 1000 + 0.5
        bits_per_trial = peak * T_opt / 60
        print(f"    {ch:2d}ch: C ≥ {peak:.1f} bpm  (N*={peak_N}, T*={T_opt:.2f}s, {bits_per_trial:.2f} bits/trial)")

    # ── 5. Accuracy-ITR tradeoff ──
    print(f"\n  Accuracy vs ITR tradeoff (66ch, mean across subjects):")
    for ti, N in enumerate(TARGET_SETS):
        acc = acc_matrix[ti, -1]
        itr = itr_matrix[ti, -1]
        bits_symbol = np.log2(N)
        efficiency = itr / (bits_symbol * 60 / (win_matrix[ti, -1] / 1000 + 0.5)) if bits_symbol > 0 else 0
        print(f"    N={N:3d}: acc={acc:.1%}  ITR={itr:.1f} bpm  "
              f"log2N={bits_symbol:.2f}  coding_eff={efficiency:.1%}")

    # ── Figures ──
    def _save(fig, path):
        fig.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {path.name}")

    ch_colors = {9: '#94a3b8', 21: '#f59e0b', 32: '#3b82f6', 66: '#16a34a'}

    # Fig 1: ITR vs N (target count) for each channel set
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    ax = axes[0]
    for ci, ch in enumerate(CHANNEL_SETS):
        means = itr_matrix[:, ci]
        stds = []
        for ti, N in enumerate(TARGET_SETS):
            vals = [best[(s, N, ch)]["itr_bpm"] for s in subjects if (s, N, ch) in best]
            stds.append(np.std(vals) if vals else 0)
        ax.errorbar(TARGET_SETS, means, yerr=stds, fmt='o-', color=ch_colors[ch],
                    linewidth=2, markersize=7, capsize=4, label=f'{ch} ch')
    ax.set_xlabel("Target count N")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title("TDCA: ITR vs target count (best window per config)")
    ax.legend()
    ax.grid(alpha=0.15)
    ax.set_xticks(TARGET_SETS)

    ax = axes[1]
    for ci, ch in enumerate(CHANNEL_SETS):
        means = acc_matrix[:, ci]
        ax.plot(TARGET_SETS, means, 'o-', color=ch_colors[ch], linewidth=2, markersize=7,
                label=f'{ch} ch')
    ax.set_xlabel("Target count N")
    ax.set_ylabel("Accuracy")
    ax.set_title("TDCA: Accuracy vs target count")
    ax.legend()
    ax.grid(alpha=0.15)
    ax.set_xticks(TARGET_SETS)
    ax.set_ylim(0.5, 1.02)

    _save(fig, FIG_DIR / "fig_hd200_itr_vs_targets.png")

    # Fig 2: Per-subject ITR at 66ch, all target counts
    fig, ax = plt.subplots(figsize=(16, 5.5), constrained_layout=True)
    x_pos = np.arange(len(subjects))
    w = 0.15
    n_colors = {40: '#6366f1', 80: '#3b82f6', 120: '#06b6d4',
                160: '#10b981', 200: '#f59e0b'}
    offsets = np.linspace(-2 * w, 2 * w, len(TARGET_SETS))
    for (N, off) in zip(TARGET_SETS, offsets):
        vals = [best.get((s, N, 66), {}).get("itr_bpm", 0) for s in subjects]
        ax.bar(x_pos + off, vals, w, color=n_colors[N], alpha=0.7,
               label=f'N={N}', edgecolor='white', linewidth=0.3)
    ax.set_xlabel("Subject")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title("Per-subject best ITR by target count (66 channels)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subjects, fontsize=7, rotation=45)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, axis='y')
    _save(fig, FIG_DIR / "fig_hd200_per_subject_targets.png")

    # Fig 3: ITR gain from increasing N (relative to N=40)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    ax = axes[0]
    for ci, ch in enumerate(CHANNEL_SETS):
        base = itr_matrix[0, ci]
        if base > 0:
            gains = [(itr_matrix[ti, ci] - base) / base * 100 for ti in range(len(TARGET_SETS))]
            ax.plot(TARGET_SETS, gains, 'o-', color=ch_colors[ch], linewidth=2, markersize=7,
                    label=f'{ch} ch')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Target count N")
    ax.set_ylabel("ΔITR vs N=40 (%)")
    ax.set_title("ITR gain from more targets (vs 40-target baseline)")
    ax.legend()
    ax.grid(alpha=0.15)
    ax.set_xticks(TARGET_SETS)

    ax = axes[1]
    log2N = np.log2(TARGET_SETS)
    for ci, ch in enumerate(CHANNEL_SETS):
        ax.plot(log2N, itr_matrix[:, ci], 'o-', color=ch_colors[ch], linewidth=2,
                markersize=7, label=f'{ch} ch')
    # theoretical max ITR if p=1: log2(N) * 60/T
    T_ref = 0.7  # typical optimal T for HD200
    theo_max = [np.log2(N) * 60 / T_ref for N in TARGET_SETS]
    ax.plot(log2N, theo_max, 'k--', alpha=0.3, linewidth=1.5, label=f'Theoretical max (T={T_ref}s)')
    ax.set_xlabel("log₂(N) — bits per correct symbol")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title("ITR vs log₂(N): approaching channel capacity?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    ax.set_xticks(log2N)
    ax.set_xticklabels([f"{v:.1f}\n(N={N})" for v, N in zip(log2N, TARGET_SETS)], fontsize=7)

    _save(fig, FIG_DIR / "fig_hd200_itr_scaling.png")

    # Fig 4: Optimal window vs target count
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    ax = axes[0]
    for ci, ch in enumerate(CHANNEL_SETS):
        wins = win_matrix[:, ci]
        ax.plot(TARGET_SETS, wins, 'o-', color=ch_colors[ch], linewidth=2, markersize=7,
                label=f'{ch} ch')
    ax.set_xlabel("Target count N")
    ax.set_ylabel("Optimal window (ms)")
    ax.set_title("Optimal observation time vs target count")
    ax.legend()
    ax.grid(alpha=0.15)
    ax.set_xticks(TARGET_SETS)

    # Channel count effect
    ax = axes[1]
    for ti, N in enumerate(TARGET_SETS):
        ax.plot(CHANNEL_SETS, itr_matrix[ti, :], 'o-', color=list(n_colors.values())[ti],
                linewidth=2, markersize=7, label=f'N={N}')
    ax.set_xlabel("Channel count")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title("Channel count effect on ITR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    ax.set_xticks(CHANNEL_SETS)
    ax.set_xscale('log', base=2)
    ax.set_xticks(CHANNEL_SETS)
    ax.set_xticklabels(CHANNEL_SETS)

    _save(fig, FIG_DIR / "fig_hd200_window_and_channels.png")

    print(f"\n  4 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
