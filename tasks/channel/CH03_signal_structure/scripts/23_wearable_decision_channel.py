"""Wearable dataset decision-channel analysis.

Builds confusion matrices and computes MI from existing runner predictions.
Compares wet vs dry electrodes across methods and window sizes.

Usage
-----
    .venv/Scripts/python.exe scripts/23_wearable_decision_channel.py
"""
from __future__ import annotations

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
CAPACITY = np.log2(N_TARGETS)  # 3.585 bits

METHODS = ["CCA", "FBCCA", "TRCA", "ETRCA", "ECCA"]
METHOD_COLORS = {"CCA": "#6c757d", "FBCCA": "#2a9d8f", "TRCA": "#e9c46a",
                 "ETRCA": "#e63946", "ECCA": "#264653"}


def load_predictions(pred_csv):
    """Load predictions CSV → dict of (method, window) → list of (true, pred)."""
    data = {}
    with open(pred_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["method"], float(row["window"]))
            if key not in data:
                data[key] = []
            data[key].append((int(row["true"]), int(row["pred"])))
    return data


def build_confusion(pairs):
    cm = np.zeros((N_TARGETS, N_TARGETS), dtype=np.int64)
    for true, pred in pairs:
        if 0 <= true < N_TARGETS and 0 <= pred < N_TARGETS:
            cm[true, pred] += 1
    return cm


def compute_mi(cm):
    total = cm.sum()
    if total == 0:
        return 0.0
    p_xy = cm.astype(np.float64) / total
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    mi = 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi


def per_subject_mi(pred_csv, method, window):
    """Compute per-subject MI for one method/window."""
    subjects = {}
    with open(pred_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["method"] != method or abs(float(row["window"]) - window) > 0.01:
                continue
            s = int(row["subject"])
            if s not in subjects:
                subjects[s] = []
            subjects[s].append((int(row["true"]), int(row["pred"])))

    mis = []
    accs = []
    for s in sorted(subjects):
        cm = build_confusion(subjects[s])
        mi = compute_mi(cm)
        acc = np.diag(cm).sum() / max(cm.sum(), 1) * 100
        mis.append(mi)
        accs.append(acc)
    return np.array(mis), np.array(accs)


def fig_confusion_comparison(wet_data, dry_data, method, window):
    """Side-by-side wet vs dry confusion matrices for one method/window."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    for idx, (data, label, cmap) in enumerate([
        (wet_data, "Wet 电极", "Blues"),
        (dry_data, "Dry 电极", "Oranges"),
    ]):
        ax = axes[idx]
        key = (method, window)
        if key not in data:
            ax.text(0.5, 0.5, "无数据", transform=ax.transAxes, ha="center")
            continue

        cm = build_confusion(data[key])
        cm_norm = cm.astype(np.float64)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm /= row_sums

        acc = np.diag(cm).sum() / max(cm.sum(), 1) * 100
        mi = compute_mi(cm)

        im = ax.imshow(cm_norm, aspect="equal", cmap=cmap, vmin=0,
                       vmax=max(0.5, cm_norm.max()), interpolation="nearest")

        for i in range(N_TARGETS):
            for j in range(N_TARGETS):
                val = cm_norm[i, j]
                if val > 0.005:
                    color = "white" if val > 0.4 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold", color=color)

        ax.set_xticks(range(N_TARGETS))
        ax.set_xticklabels([f"{FREQS[i]:.2f}" for i in range(N_TARGETS)],
                           fontsize=7, rotation=45)
        ax.set_yticks(range(N_TARGETS))
        ax.set_yticklabels([f"{FREQS[i]:.2f}" for i in range(N_TARGETS)], fontsize=7)
        ax.set_xlabel("预测频率 (Hz)", fontsize=10)
        ax.set_ylabel("真实频率 (Hz)", fontsize=10)
        ax.set_title(f"{label}\n准确率 = {acc:.1f}%,  MI = {mi:.2f} / {CAPACITY:.2f} bits",
                     fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7)

    # Panel 3: MI text summary
    ax3 = axes[2]
    ax3.axis("off")

    wet_cm = build_confusion(wet_data.get((method, window), []))
    dry_cm = build_confusion(dry_data.get((method, window), []))
    wet_mi = compute_mi(wet_cm)
    dry_mi = compute_mi(dry_cm)
    wet_acc = np.diag(wet_cm).sum() / max(wet_cm.sum(), 1) * 100
    dry_acc = np.diag(dry_cm).sum() / max(dry_cm.sum(), 1) * 100

    txt = (
        f"决策信道对比: {method}, {window}s\n"
        f"{'='*36}\n\n"
        f"目标数: {N_TARGETS}\n"
        f"通道数: 8 (枕区)\n"
        f"被试数: 102\n"
        f"blocks: 10\n"
        f"信道容量上限: {CAPACITY:.3f} bits\n\n"
        f"{'─'*36}\n"
        f"         Wet         Dry\n"
        f"{'─'*36}\n"
        f"准确率   {wet_acc:5.1f}%      {dry_acc:5.1f}%\n"
        f"MI       {wet_mi:5.3f}       {dry_mi:5.3f} bits\n"
        f"利用率   {wet_mi/CAPACITY*100:5.1f}%      {dry_mi/CAPACITY*100:5.1f}%\n"
        f"MI 损失  {CAPACITY-wet_mi:5.3f}       {CAPACITY-dry_mi:5.3f} bits\n\n"
        f"{'─'*36}\n"
        f"Dry/Wet MI 比: {dry_mi/max(wet_mi,1e-10):.2f}\n"
        f"Dry MI 损失:   {wet_mi-dry_mi:.3f} bits\n\n"
        f"{'='*36}\n"
        f"信道矩阵 P(Y|X) = 归一化混淆矩阵\n"
        f"MI = I(X;Y) = H(Y) - H(Y|X)\n"
        f"利用率 = MI / log2({N_TARGETS})"
    )
    ax3.text(0.05, 0.95, txt, transform=ax3.transAxes, fontsize=11,
             fontfamily="Microsoft YaHei", verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.suptitle(f"Wearable SSVEP — Wet vs Dry 决策信道混淆矩阵 ({method}, {window}s, 102 被试)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig_mi_curves(wet_data, dry_data):
    """MI vs window for all methods, wet vs dry."""
    windows = sorted(set(w for _, w in wet_data.keys()))

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    for ax_idx, (data, label, ls) in enumerate([
        (None, "对比", None),
        (wet_data, "Wet 电极", "-"),
        (dry_data, "Dry 电极", "--"),
    ]):
        if ax_idx == 0:
            ax = axes[0]
            for method in METHODS:
                wet_mis = []
                dry_mis = []
                for w in windows:
                    wet_cm = build_confusion(wet_data.get((method, w), []))
                    dry_cm = build_confusion(dry_data.get((method, w), []))
                    wet_mis.append(compute_mi(wet_cm))
                    dry_mis.append(compute_mi(dry_cm))
                ax.plot(windows, wet_mis, "-o", color=METHOD_COLORS[method],
                        linewidth=2, markersize=5, label=f"{method} wet")
                ax.plot(windows, dry_mis, "--s", color=METHOD_COLORS[method],
                        linewidth=1.5, markersize=4, alpha=0.7, label=f"{method} dry")

            ax.axhline(CAPACITY, color="red", linestyle=":", linewidth=1.5,
                       label=f"容量上限 {CAPACITY:.2f} bits")
            ax.set_xlabel("窗口长度 (s)", fontsize=12)
            ax.set_ylabel("MI (bits)", fontsize=12)
            ax.set_title("(a) Wet vs Dry MI 曲线\n实线 = wet, 虚线 = dry",
                         fontsize=13, fontweight="bold")
            ax.legend(fontsize=7, ncol=2, loc="lower right")
            ax.grid(alpha=0.2)
            ax.set_ylim(0, CAPACITY * 1.1)
            continue

        ax = axes[ax_idx]
        for method in METHODS:
            mis = []
            accs = []
            for w in windows:
                cm = build_confusion(data.get((method, w), []))
                mis.append(compute_mi(cm))
                accs.append(np.diag(cm).sum() / max(cm.sum(), 1) * 100)
            ax.plot(windows, mis, f"{ls}o", color=METHOD_COLORS[method],
                    linewidth=2, markersize=5, label=method)

        ax.axhline(CAPACITY, color="red", linestyle=":", linewidth=1.5)
        ax.set_xlabel("窗口长度 (s)", fontsize=12)
        ax.set_ylabel("MI (bits)", fontsize=12)
        ax.set_title(f"({'b' if ax_idx == 1 else 'c'}) {label} — MI vs 窗口",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_ylim(0, CAPACITY * 1.1)

    fig.suptitle("Wearable SSVEP 决策信道容量 vs 窗口长度 (12 目标, 8ch, 102 被试)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig_subject_mi_distribution(wet_csv, dry_csv, method, window):
    """Per-subject MI distribution: wet vs dry."""
    wet_mis, wet_accs = per_subject_mi(wet_csv, method, window)
    dry_mis, dry_accs = per_subject_mi(dry_csv, method, window)

    n_subj = min(len(wet_mis), len(dry_mis))
    wet_mis = wet_mis[:n_subj]
    dry_mis = dry_mis[:n_subj]
    wet_accs = wet_accs[:n_subj]
    dry_accs = dry_accs[:n_subj]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel 1: Scatter wet MI vs dry MI
    ax1 = axes[0]
    ax1.scatter(wet_mis, dry_mis, c=wet_accs, cmap="viridis", s=30, alpha=0.6,
                edgecolors="0.3", linewidth=0.3)
    lim = max(wet_mis.max(), dry_mis.max()) * 1.1
    ax1.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5, label="y=x")
    ax1.set_xlabel(f"Wet MI (bits)", fontsize=12)
    ax1.set_ylabel(f"Dry MI (bits)", fontsize=12)
    ax1.set_title(f"(a) 被试级 MI: Wet vs Dry\n{method}, {window}s, N={n_subj}",
                  fontsize=13, fontweight="bold")
    ax1.set_aspect("equal")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.2)

    pct_below = np.mean(dry_mis < wet_mis) * 100
    ax1.text(0.05, 0.92, f"{pct_below:.0f}% 被试 dry < wet",
             transform=ax1.transAxes, fontsize=11,
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    # Panel 2: Histograms
    ax2 = axes[1]
    bins = np.linspace(0, CAPACITY, 25)
    ax2.hist(wet_mis, bins=bins, alpha=0.5, color="steelblue", label="Wet", edgecolor="white")
    ax2.hist(dry_mis, bins=bins, alpha=0.5, color="darkorange", label="Dry", edgecolor="white")
    ax2.axvline(np.median(wet_mis), color="steelblue", linestyle="--", linewidth=2,
                label=f"Wet 中位数 {np.median(wet_mis):.2f}")
    ax2.axvline(np.median(dry_mis), color="darkorange", linestyle="--", linewidth=2,
                label=f"Dry 中位数 {np.median(dry_mis):.2f}")
    ax2.axvline(CAPACITY, color="red", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("MI (bits)", fontsize=12)
    ax2.set_ylabel("被试数", fontsize=12)
    ax2.set_title(f"(b) MI 分布直方图\n{method}, {window}s",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)

    # Panel 3: MI loss (wet - dry) sorted
    ax3 = axes[2]
    delta = wet_mis - dry_mis
    order = np.argsort(delta)
    colors = ["#e63946" if d > 0 else "#2a9d8f" for d in delta[order]]
    ax3.bar(range(n_subj), delta[order], color=colors, alpha=0.7)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.axhline(np.median(delta), color="purple", linestyle="--", linewidth=1.5,
                label=f"中位数差 {np.median(delta):.3f} bits")
    ax3.set_xlabel("被试 (按 MI 差排序)", fontsize=12)
    ax3.set_ylabel("Wet MI - Dry MI (bits)", fontsize=12)
    ax3.set_title(f"(c) 被试级 MI 差异\n红 = wet 更好, 绿 = dry 更好",
                  fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.15, axis="y")

    fig.suptitle(f"Wearable SSVEP 被试级决策信道分析 ({method}, {window}s, 12 目标)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading wet predictions...")
    wet_data = load_predictions(WET_DIR / "predictions.csv")
    print("Loading dry predictions...")
    dry_data = load_predictions(DRY_DIR / "predictions.csv")

    print(f"Wet: {len(wet_data)} (method, window) combos")
    print(f"Dry: {len(dry_data)} (method, window) combos")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Method':<8} {'Window':>6} {'Wet Acc':>8} {'Wet MI':>8} {'Dry Acc':>8} {'Dry MI':>8} {'MI Gap':>8}")
    print(f"{'='*70}")

    for method in METHODS:
        for window in [0.5, 1.0, 2.0]:
            wet_cm = build_confusion(wet_data.get((method, window), []))
            dry_cm = build_confusion(dry_data.get((method, window), []))
            wet_mi = compute_mi(wet_cm)
            dry_mi = compute_mi(dry_cm)
            wet_acc = np.diag(wet_cm).sum() / max(wet_cm.sum(), 1) * 100
            dry_acc = np.diag(dry_cm).sum() / max(dry_cm.sum(), 1) * 100
            print(f"{method:<8} {window:>5.1f}s {wet_acc:>7.1f}% {wet_mi:>7.3f}  {dry_acc:>7.1f}% {dry_mi:>7.3f}  {wet_mi-dry_mi:>+7.3f}")

    # Fig 1: Confusion matrix comparison (ETRCA, 1.0s)
    fig = fig_confusion_comparison(wet_data, dry_data, "ETRCA", 1.0)
    fig.savefig(FIG_DIR / "fig_wearable_confusion_etrca_1s.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("\n-> fig_wearable_confusion_etrca_1s.png")

    # Fig 2: MI curves
    fig = fig_mi_curves(wet_data, dry_data)
    fig.savefig(FIG_DIR / "fig_wearable_mi_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> fig_wearable_mi_curves.png")

    # Fig 3: Per-subject MI distribution (ETRCA, 1.0s)
    fig = fig_subject_mi_distribution(
        WET_DIR / "predictions.csv", DRY_DIR / "predictions.csv", "ETRCA", 1.0)
    fig.savefig(FIG_DIR / "fig_wearable_subject_mi_etrca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> fig_wearable_subject_mi_etrca.png")

    # Fig 4: Confusion matrix comparison (CCA, 1.0s) — baseline
    fig = fig_confusion_comparison(wet_data, dry_data, "CCA", 1.0)
    fig.savefig(FIG_DIR / "fig_wearable_confusion_cca_1s.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> fig_wearable_confusion_cca_1s.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
