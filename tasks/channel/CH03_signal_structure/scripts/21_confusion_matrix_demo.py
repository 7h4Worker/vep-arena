"""Confusion matrix visualization for decision-channel analysis.

Generates:
  A. Full 200×200 confusion matrix (template matching, LOBO)
  B. Zoomed 5×5 spatial confusion block (within one frequency)
  C. Zoomed 25×25 block (5 adjacent frequencies × 5 positions)
  D. Mutual information computation demo

Usage
-----
    .venv/Scripts/python.exe scripts/21_confusion_matrix_demo.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

TASK_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = TASK_DIR / "results" / "figures"
CACHE_DIR = Path("D:/ProjData/datasets/ssvep_hd_200target/derivatives/tdca_sample/cache")

FS = 250
LATENCY_SAMPLES = 35
N_TARGETS = 200
N_BLOCKS = 18

BASE_FREQS = np.asarray(
    list(range(8, 16))
    + [x + 0.2 for x in range(8, 16)]
    + [x + 0.4 for x in range(8, 16)]
    + [x + 0.6 for x in range(8, 16)]
    + [x + 0.8 for x in range(8, 16)],
    dtype=np.float64,
)
POSITION_NAMES = ["R", "D", "L", "U", "C"]


def load_fb0(subj):
    p = CACHE_DIR / f"filtered_{subj}_200target_66ch_18blocks_float32.npy"
    return np.asarray(np.load(str(p), mmap_mode="r")[:, :, :, :, 0])


def compute_confusion_matrix(data, ch_idx, win_samp):
    """LOBO template matching → 200×200 confusion matrix."""
    n_ch = len(ch_idx)
    feat_dim = n_ch * win_samp

    # Extract and normalize all trial vectors
    all_vecs = np.zeros((N_TARGETS, N_BLOCKS, feat_dim))
    for t in range(N_TARGETS):
        for b in range(N_BLOCKS):
            seg = data[ch_idx, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, t, b]
            v = seg.flatten().astype(np.float64)
            v -= v.mean()
            n = np.linalg.norm(v)
            if n > 1e-12:
                v /= n
            all_vecs[t, b] = v

    # LOBO classification
    cm = np.zeros((N_TARGETS, N_TARGETS), dtype=np.int32)

    for b_test in range(N_BLOCKS):
        # Build templates from other blocks
        mask = np.ones(N_BLOCKS, dtype=bool)
        mask[b_test] = False
        templates = all_vecs[:, mask, :].mean(axis=1)  # (200, feat_dim)
        # Normalize templates
        norms = np.linalg.norm(templates, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1
        templates /= norms

        # Classify test trials
        test_trials = all_vecs[:, b_test, :]  # (200, feat_dim)
        corr = test_trials @ templates.T  # (200, 200)
        predictions = corr.argmax(axis=1)

        for true_k in range(N_TARGETS):
            cm[true_k, predictions[true_k]] += 1

    return cm


def compute_mi(cm_counts):
    """Compute mutual information I(X;Y) from confusion count matrix."""
    cm = cm_counts.astype(np.float64)
    total = cm.sum()
    if total == 0:
        return 0.0

    p_xy = cm / total  # joint distribution
    p_x = p_xy.sum(axis=1)  # marginal P(X)
    p_y = p_xy.sum(axis=0)  # marginal P(Y)

    mi = 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi


def fig_full_confusion(cm, subj, ch_label, acc):
    """Full 200×200 confusion matrix with zoomed regions."""
    fig = plt.figure(figsize=(22, 10))

    # Normalize rows → P(Y|X)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm /= row_sums

    # --- Panel 1: Full 200×200 ---
    ax1 = fig.add_axes([0.04, 0.08, 0.42, 0.82])

    cm_log = np.log10(cm_norm + 1e-6)
    im = ax1.imshow(cm_log, aspect="equal", cmap="YlOrRd",
                    vmin=-4, vmax=0, interpolation="nearest")

    # Draw 5×5 grid lines for frequency blocks
    for i in range(0, 201, 5):
        ax1.axhline(i - 0.5, color="0.3", linewidth=0.2, alpha=0.3)
        ax1.axvline(i - 0.5, color="0.3", linewidth=0.2, alpha=0.3)
    # Heavier lines every 40 targets (8 integer freq groups × 5 positions = 40 per group)
    for i in range(0, 201, 40):
        ax1.axhline(i - 0.5, color="0.2", linewidth=0.8, alpha=0.5)
        ax1.axvline(i - 0.5, color="0.2", linewidth=0.8, alpha=0.5)

    # Mark zoom regions
    # Region A: one 5×5 block (freq_idx=0, 8.0 Hz, targets 0-4)
    rect_a = Rectangle((-0.5, -0.5), 5, 5, linewidth=2.5, edgecolor="#2196F3",
                        facecolor="none", zorder=10, linestyle="-")
    ax1.add_patch(rect_a)
    ax1.text(5, 2, "A", fontsize=14, color="#2196F3", fontweight="bold")

    # Region B: 25×25 block (freq_idx 0-4, targets 0-24)
    rect_b = Rectangle((-0.5, -0.5), 25, 25, linewidth=2.5, edgecolor="#4CAF50",
                        facecolor="none", zorder=10, linestyle="--")
    ax1.add_patch(rect_b)
    ax1.text(26, 12, "B", fontsize=14, color="#4CAF50", fontweight="bold")

    # Frequency labels
    freq_ticks = np.arange(0, 200, 5) + 2  # center of each 5-block
    freq_labels = [f"{BASE_FREQS[i]:.1f}" for i in range(40)]
    ax1.set_xticks(freq_ticks[::4])
    ax1.set_xticklabels(freq_labels[::4], fontsize=8, rotation=45)
    ax1.set_yticks(freq_ticks[::4])
    ax1.set_yticklabels(freq_labels[::4], fontsize=8)

    ax1.set_xlabel("预测目标 (频率 Hz)", fontsize=11)
    ax1.set_ylabel("真实目标 (频率 Hz)", fontsize=11)

    mi = compute_mi(cm)
    capacity = np.log2(N_TARGETS)
    ax1.set_title(f"200×200 混淆矩阵 P(Y|X) — {subj}, {ch_label}\n"
                  f"准确率 = {acc:.1f}%,  MI = {mi:.2f} bits (容量上限 = {capacity:.1f} bits)",
                  fontsize=14, fontweight="bold")

    cb = plt.colorbar(im, ax=ax1, shrink=0.7, pad=0.02)
    cb.set_label("log₁₀ P(Y|X)", fontsize=10)

    # --- Panel 2: Zoomed 5×5 block (Region A: 8.0 Hz spatial confusion) ---
    ax2 = fig.add_axes([0.54, 0.55, 0.20, 0.35])

    block_5x5 = cm_norm[0:5, 0:5]
    im2 = ax2.imshow(block_5x5, aspect="equal", cmap="Blues", vmin=0, vmax=1,
                     interpolation="nearest")

    # Annotate values
    for i in range(5):
        for j in range(5):
            val = block_5x5[i, j]
            color = "white" if val > 0.5 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=11, fontweight="bold", color=color)

    ax2.set_xticks(range(5))
    ax2.set_xticklabels(POSITION_NAMES, fontsize=11)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(POSITION_NAMES, fontsize=11)
    ax2.set_xlabel("预测位置", fontsize=11)
    ax2.set_ylabel("真实位置", fontsize=11)
    ax2.set_title(f"区域 A: 8.0 Hz 内 5 位置混淆\n(频率内空间判别)",
                  fontsize=12, fontweight="bold", color="#2196F3")
    plt.colorbar(im2, ax=ax2, shrink=0.7)

    # --- Panel 3: Zoomed 25×25 block (Region B: 5 frequencies × 5 positions) ---
    ax3 = fig.add_axes([0.54, 0.08, 0.42, 0.38])

    block_25 = cm_norm[0:25, 0:25]
    im3 = ax3.imshow(np.log10(block_25 + 1e-6), aspect="equal", cmap="YlOrRd",
                     vmin=-3, vmax=0, interpolation="nearest")

    # Grid lines for 5×5 blocks
    for i in range(0, 26, 5):
        ax3.axhline(i - 0.5, color="black", linewidth=1.5, alpha=0.7)
        ax3.axvline(i - 0.5, color="black", linewidth=1.5, alpha=0.7)

    # Label frequencies for each block
    for blk in range(5):
        freq_hz = BASE_FREQS[blk]
        ax3.text(blk * 5 + 2, -1.5, f"{freq_hz:.1f}", ha="center", fontsize=9,
                 fontweight="bold")
        ax3.text(-2.5, blk * 5 + 2, f"{freq_hz:.1f}", ha="center", fontsize=9,
                 fontweight="bold", rotation=0)

    # Position labels
    for blk in range(5):
        for p in range(5):
            ax3.text(blk * 5 + p, 25.5, POSITION_NAMES[p], ha="center",
                     fontsize=6, color="0.4")

    ax3.set_xlim(-0.5, 24.5)
    ax3.set_ylim(24.5, -0.5)
    ax3.set_xlabel("预测目标 (频率 × 位置)", fontsize=11)
    ax3.set_ylabel("真实目标 (频率 × 位置)", fontsize=11)
    ax3.set_title(f"区域 B: 前 5 个频率 (8.0–12.0 Hz) 的 25×25 局部混淆\n"
                  f"对角 5×5 块 = 频率内空间混淆, 非对角块 = 跨频率误判",
                  fontsize=12, fontweight="bold", color="#4CAF50")
    plt.colorbar(im3, ax=ax3, shrink=0.7, label="log₁₀ P(Y|X)")

    # --- Panel 4: MI breakdown text ---
    ax4 = fig.add_axes([0.78, 0.55, 0.19, 0.35])
    ax4.axis("off")

    # Compute frequency-only confusion (marginalize over positions)
    cm40 = np.zeros((40, 40))
    for fi in range(40):
        for fj in range(40):
            cm40[fi, fj] = cm[fi*5:(fi+1)*5, fj*5:(fj+1)*5].sum()
    mi_freq = compute_mi(cm40.astype(int))

    # Compute position-only confusion (average across frequencies)
    cm5_total = np.zeros((5, 5))
    for fi in range(40):
        cm5_total += cm[fi*5:(fi+1)*5, fi*5:(fi+1)*5]
    mi_spatial = compute_mi(cm5_total.astype(int))

    mi_text = (
        f"决策信道分析\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"计算方法:\n"
        f"  1. LOBO 模板匹配分类\n"
        f"  2. 统计 200×200 混淆计数\n"
        f"  3. 归一化 → P(Y|X)\n"
        f"  4. 计算 I(X;Y)\n\n"
        f"I(X;Y) = Σ P(x,y) log₂ P(x,y)\n"
        f"                    P(x)P(y)\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"信息分解:\n\n"
        f"  总 MI:     {mi:.2f} bits\n"
        f"  频率 MI:   {mi_freq:.2f} / {np.log2(40):.1f} bits\n"
        f"  空间 MI:   {mi_spatial:.2f} / {np.log2(5):.1f} bits\n"
        f"  容量上限:  {capacity:.1f} bits\n\n"
        f"  频率利用率: {mi_freq/np.log2(40)*100:.0f}%\n"
        f"  空间利用率: {mi_spatial/np.log2(5)*100:.0f}%\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"信道矩阵 = P(Y|X)\n"
        f" = 经验转移概率\n"
        f" = 通信理论中的 DMC"
    )
    ax4.text(0.05, 0.98, mi_text, transform=ax4.transAxes, fontsize=10,
             fontfamily="Microsoft YaHei", verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9,
                       edgecolor="0.6"))

    fig.suptitle(f"{subj} — 混淆矩阵与决策信道分析 ({ch_label}, 500ms, LOBO)",
                 fontsize=16, fontweight="bold", y=0.98)
    return fig


def fig_spatial_confusion_multi(cm, subj, ch_label):
    """5×5 spatial confusion for multiple frequencies."""
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm /= row_sums

    freq_indices = [0, 4, 8, 16, 24, 32]  # 8.0, 12.0, 8.2, 10.0, 11.0, 12.4
    n_show = len(freq_indices)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for idx, fi in enumerate(freq_indices):
        ax = axes[idx]
        freq_hz = BASE_FREQS[fi]
        block = cm_norm[fi*5:(fi+1)*5, fi*5:(fi+1)*5]

        im = ax.imshow(block, aspect="equal", cmap="Blues", vmin=0,
                       vmax=max(0.6, block.max()), interpolation="nearest")

        for i in range(5):
            for j in range(5):
                val = block[i, j]
                color = "white" if val > 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)

        diag_acc = np.diag(block).mean()
        ax.set_xticks(range(5))
        ax.set_xticklabels(POSITION_NAMES, fontsize=11)
        ax.set_yticks(range(5))
        ax.set_yticklabels(POSITION_NAMES, fontsize=11)
        ax.set_title(f"{freq_hz:.1f} Hz\n空间准确率 = {diag_acc*100:.1f}%",
                     fontsize=12, fontweight="bold")

    fig.suptitle(f"{subj} — 频率内 5 位置空间混淆矩阵 ({ch_label})\n"
                 f"对角线 = 正确判别概率, 非对角 = 位置间误判",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def main():
    t0 = time.time()
    win_samp = 125  # 500ms

    ch_sets = {
        "9ch": np.array([48, 54, 55, 56, 57, 58, 61, 62, 63]),
        "66ch": np.arange(66),
    }

    for subj in ["S1", "S9"]:
        print(f"Loading {subj}...")
        data = load_fb0(subj)

        for ch_label, ch_idx in ch_sets.items():
            print(f"  Computing confusion matrix ({ch_label})...")
            cm = compute_confusion_matrix(data, ch_idx, win_samp)

            acc = np.diag(cm).sum() / cm.sum() * 100
            print(f"    Accuracy: {acc:.1f}%")
            mi = compute_mi(cm)
            print(f"    MI: {mi:.2f} bits")

            # Full confusion matrix figure
            fig = fig_full_confusion(cm, subj, ch_label, acc)
            fname = f"fig_confusion_200x200_{subj}_{ch_label}.png"
            fig.savefig(FIG_DIR / fname, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"    -> {fname}")

            # Spatial confusion multi-frequency
            if ch_label == "66ch":
                fig = fig_spatial_confusion_multi(cm, subj, ch_label)
                fname = f"fig_confusion_spatial_5x5_{subj}.png"
                fig.savefig(FIG_DIR / fname, dpi=180, bbox_inches="tight")
                plt.close(fig)
                print(f"    -> {fname}")

        del data

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
