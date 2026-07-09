"""Codebook pruning analysis for wearable dataset.

Tests whether removing confused targets improves MI/ITR on dry electrodes.
Uses existing predictions — no new experiments needed.

Strategies:
  A. Greedy removal (drop worst-accuracy target iteratively)
  B. Greedy MI maximization (drop target that maximizes remaining MI)
  C. Exhaustive search for small subsets

Usage
-----
    .venv/Scripts/python.exe scripts/24_wearable_codebook_pruning.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import csv

TASK_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = TASK_DIR / "results" / "figures"
RESULT_ROOT = Path("D:/ProjData/proj_python/vep_arena/results")

WET_DIR = RESULT_ROOT / "wearable_wet_cca_fbcca_ecca_trca_etrca_w02_10"
DRY_DIR = RESULT_ROOT / "wearable_dry_cca_fbcca_ecca_trca_etrca_w02_10"

N_TARGETS = 12
FREQS = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
         10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

METHODS = ["CCA", "FBCCA", "TRCA", "ETRCA", "ECCA"]
METHOD_COLORS = {"CCA": "#6c757d", "FBCCA": "#2a9d8f", "TRCA": "#e9c46a",
                 "ETRCA": "#e63946", "ECCA": "#264653"}


def load_predictions(pred_csv, method, window):
    pairs = []
    with open(pred_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["method"] == method and abs(float(row["window"]) - window) < 0.01:
                pairs.append((int(row["true"]), int(row["pred"])))
    return pairs


def subset_mi_acc(pairs, keep_set):
    """Compute MI and accuracy for a target subset."""
    keep = sorted(keep_set)
    n = len(keep)
    if n < 2:
        return 0.0, 0.0, 0.0

    idx_map = {t: i for i, t in enumerate(keep)}
    cm = np.zeros((n, n), dtype=np.int64)
    total = 0
    correct = 0

    for true, pred in pairs:
        if true not in idx_map:
            continue
        total += 1
        if pred in idx_map:
            cm[idx_map[true], idx_map[pred]] += 1
            if true == pred:
                correct += 1
        # If pred not in keep_set, it's a "miss" — row sum stays same,
        # but we need to count it. Actually for pruning analysis:
        # only count trials where TRUE target is in the subset.
        # Prediction outside subset = wrong.

    if total == 0:
        return 0.0, 0.0, 0.0

    acc = correct / total

    # MI from the cm (predictions within subset only)
    # But we need to handle predictions outside subset
    # Add an "other" column for predictions falling outside
    cm_ext = np.zeros((n, n + 1), dtype=np.int64)
    cm_ext[:, :n] = cm
    for true, pred in pairs:
        if true in idx_map and pred not in idx_map:
            cm_ext[idx_map[true], n] += 1

    # MI with the extended matrix (includes "other" as an output)
    p_xy = cm_ext.astype(np.float64)
    tot = p_xy.sum()
    if tot == 0:
        return 0.0, 0.0, 0.0
    p_xy /= tot
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    mi = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

    itr_bits = mi  # bits per trial
    return mi, acc, itr_bits


def greedy_drop_worst_acc(pairs, n_targets=12):
    """Greedily drop target with worst per-target accuracy."""
    keep = set(range(n_targets))
    path = [(n_targets, *subset_mi_acc(pairs, keep))]

    while len(keep) > 2:
        # Find target with worst accuracy
        best_drop = None
        best_mi = -1
        for t in keep:
            trial_keep = keep - {t}
            mi, acc, itr = subset_mi_acc(pairs, trial_keep)
            if mi > best_mi:
                best_mi = mi
                best_drop = t
                best_acc = acc
                best_itr = itr

        keep.remove(best_drop)
        path.append((len(keep), best_mi, best_acc, best_itr))

    return path


def greedy_drop_max_mi(pairs, n_targets=12):
    """Greedily drop target that maximizes remaining MI."""
    keep = set(range(n_targets))
    path = [(n_targets, *subset_mi_acc(pairs, keep))]

    while len(keep) > 2:
        best_drop = None
        best_mi = -1
        for t in keep:
            trial_keep = keep - {t}
            mi, acc, itr = subset_mi_acc(pairs, trial_keep)
            if mi > best_mi:
                best_mi = mi
                best_drop = t
                best_acc = acc

        keep.remove(best_drop)
        path.append((len(keep), best_mi, best_acc, best_mi))

    return path


def exhaustive_best(pairs, subset_size):
    """Exhaustive search for best subset of given size."""
    best_mi = -1
    best_subset = None
    for combo in itertools.combinations(range(N_TARGETS), subset_size):
        mi, acc, _ = subset_mi_acc(pairs, set(combo))
        if mi > best_mi:
            best_mi = mi
            best_acc = acc
            best_subset = combo
    return best_subset, best_mi, best_acc


def fig_pruning_analysis(wet_pairs, dry_pairs, method, window):
    """Comprehensive pruning analysis figure."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    for col, (pairs, label, base_color) in enumerate([
        (wet_pairs, "Wet", "steelblue"),
        (dry_pairs, "Dry", "darkorange"),
    ]):
        # Greedy MI maximization path
        path = greedy_drop_max_mi(pairs)
        sizes = [p[0] for p in path]
        mis = [p[1] for p in path]
        accs = [p[2] for p in path]
        capacities = [np.log2(s) for s in sizes]

        # Panel (row 0, col): MI vs subset size
        ax = axes[0, col]
        ax.plot(sizes, mis, "o-", color=base_color, linewidth=2.5, markersize=8,
                label="贪心裁剪 MI", zorder=5)
        ax.plot(sizes, capacities, "k--", linewidth=1.5, alpha=0.5,
                label="log2(N) 上限")
        ax.fill_between(sizes, mis, capacities, alpha=0.1, color="red")

        # Exhaustive search for sizes 4-8
        ex_mis = []
        ex_sizes = list(range(4, 9))
        for s in ex_sizes:
            _, mi_ex, _ = exhaustive_best(pairs, s)
            ex_mis.append(mi_ex)
        ax.plot(ex_sizes, ex_mis, "s", color="red", markersize=10,
                label="穷举最优", zorder=6)

        ax.set_xlabel("目标数 N", fontsize=12)
        ax.set_ylabel("MI (bits)", fontsize=12)
        ax.set_title(f"{label} — MI vs 码本大小\n{method}, {window}s",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_xlim(1.5, 12.5)
        ax.set_xticks(range(2, 13))

        # Mark the optimal point
        best_idx = np.argmax(mis)
        ax.annotate(f"最优: N={sizes[best_idx]}\nMI={mis[best_idx]:.3f}",
                    xy=(sizes[best_idx], mis[best_idx]),
                    xytext=(sizes[best_idx] - 2, mis[best_idx] + 0.3),
                    fontsize=11, fontweight="bold", color=base_color,
                    arrowprops=dict(arrowstyle="->", color=base_color),
                    bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

        # Panel (row 1, col): Accuracy vs subset size
        ax = axes[1, col]
        ax.plot(sizes, [a * 100 for a in accs], "o-", color=base_color,
                linewidth=2.5, markersize=8, label="贪心裁剪准确率")
        ax.axhline(100 / N_TARGETS, color="gray", linestyle=":", alpha=0.5,
                   label=f"12 目标 chance ({100/N_TARGETS:.1f}%)")

        ex_accs = []
        for s in ex_sizes:
            _, _, acc_ex = exhaustive_best(pairs, s)
            ex_accs.append(acc_ex * 100)
        ax.plot(ex_sizes, ex_accs, "s", color="red", markersize=10,
                label="穷举最优准确率")

        ax.set_xlabel("目标数 N", fontsize=12)
        ax.set_ylabel("准确率 (%)", fontsize=12)
        ax.set_title(f"{label} — 准确率 vs 码本大小",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_xlim(1.5, 12.5)
        ax.set_xticks(range(2, 13))

    # Panel (row 0, col 2): Method comparison for pruning
    ax = axes[0, 2]
    for method_name in METHODS:
        dry_p = load_predictions(DRY_DIR / "predictions.csv", method_name, window)
        if not dry_p:
            continue
        path = greedy_drop_max_mi(dry_p)
        sizes = [p[0] for p in path]
        mis = [p[1] for p in path]
        ax.plot(sizes, mis, "o-", color=METHOD_COLORS[method_name],
                linewidth=2, markersize=6, label=method_name)

    ax.plot(range(2, 13), [np.log2(s) for s in range(2, 13)], "k--",
            linewidth=1.5, alpha=0.3, label="上限")
    ax.set_xlabel("目标数 N", fontsize=12)
    ax.set_ylabel("MI (bits)", fontsize=12)
    ax.set_title(f"Dry — 各方法贪心裁剪 MI\n{window}s 窗口",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_xlim(1.5, 12.5)
    ax.set_xticks(range(2, 13))

    # Panel (row 1, col 2): Best subset frequencies
    ax = axes[1, 2]
    ax.axis("off")

    txt_lines = [f"Dry {method} {window}s — 穷举最优子集\n{'='*40}\n"]
    for s in [4, 6, 8, 10]:
        best_sub, best_mi, best_acc = exhaustive_best(dry_pairs, s)
        freqs_str = ", ".join(f"{FREQS[i]:.2f}" for i in best_sub)
        txt_lines.append(f"N={s:2d}: MI={best_mi:.3f} bits, Acc={best_acc*100:.1f}%")
        txt_lines.append(f"       {freqs_str}")
        txt_lines.append("")

    # Also for wet
    txt_lines.append(f"\nWet {method} {window}s — 穷举最优子集\n{'='*40}\n")
    for s in [4, 6, 8]:
        best_sub, best_mi, best_acc = exhaustive_best(wet_pairs, s)
        freqs_str = ", ".join(f"{FREQS[i]:.2f}" for i in best_sub)
        txt_lines.append(f"N={s:2d}: MI={best_mi:.3f} bits, Acc={best_acc*100:.1f}%")
        txt_lines.append(f"       {freqs_str}")
        txt_lines.append("")

    ax.text(0.05, 0.95, "\n".join(txt_lines), transform=ax.transAxes,
            fontsize=10, fontfamily="Microsoft YaHei", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.suptitle(f"Wearable 码本裁剪分析 — {method}, {window}s\n"
                 f"问题: 减少目标数能否提升 dry 电极的有效信息传递?",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    window = 1.0

    # Main analysis for CCA (best dry method)
    print("=== CCA (best method on dry) ===")
    wet_pairs = load_predictions(WET_DIR / "predictions.csv", "CCA", window)
    dry_pairs = load_predictions(DRY_DIR / "predictions.csv", "CCA", window)
    print(f"Wet trials: {len(wet_pairs)}, Dry trials: {len(dry_pairs)}")

    fig = fig_pruning_analysis(wet_pairs, dry_pairs, "CCA", window)
    fig.savefig(FIG_DIR / "fig_wearable_pruning_cca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> fig_wearable_pruning_cca.png")

    # Also for ETRCA (best wet method)
    print("\n=== ETRCA (best method on wet) ===")
    wet_pairs = load_predictions(WET_DIR / "predictions.csv", "ETRCA", window)
    dry_pairs = load_predictions(DRY_DIR / "predictions.csv", "ETRCA", window)

    fig = fig_pruning_analysis(wet_pairs, dry_pairs, "ETRCA", window)
    fig.savefig(FIG_DIR / "fig_wearable_pruning_etrca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> fig_wearable_pruning_etrca.png")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"码本裁剪效果汇总 (1.0s 窗口)")
    print(f"{'='*70}")
    print(f"{'Method':<8} {'Elec':>4} {'N=12 MI':>8} {'N=8 MI':>8} {'N=6 MI':>8} {'N=4 MI':>8} {'Best N':>6}")

    for method_name in ["CCA", "ETRCA"]:
        for elec, pred_dir in [("wet", WET_DIR), ("dry", DRY_DIR)]:
            pairs = load_predictions(pred_dir / "predictions.csv", method_name, window)
            if not pairs:
                continue

            mi_12, _, _ = subset_mi_acc(pairs, set(range(12)))
            results = {}
            for s in [4, 6, 8]:
                _, mi_s, _ = exhaustive_best(pairs, s)
                results[s] = mi_s

            path = greedy_drop_max_mi(pairs)
            best_n = max(path, key=lambda p: p[1])[0]

            print(f"{method_name:<8} {elec:>4} {mi_12:>7.3f}  {results[8]:>7.3f}  "
                  f"{results[6]:>7.3f}  {results[4]:>7.3f}  {best_n:>5}")

    print("\nDone.")


if __name__ == "__main__":
    main()
