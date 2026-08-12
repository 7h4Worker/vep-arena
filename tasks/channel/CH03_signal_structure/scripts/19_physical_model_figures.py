"""Physical signal model figures for SSVEP-BCI analysis.

Generates:
  A. Codebook spectrum — 40 frequencies × 5 harmonics
  B. Gaze interference model — screen layout + received signal
  C. TDCA augmentation schematic

Usage
-----
    .venv/Scripts/python.exe scripts/19_physical_model_figures.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import numpy as np

TASK_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = TASK_DIR / "results" / "figures"

BASE_FREQS = np.asarray(
    list(range(8, 16))
    + [x + 0.2 for x in range(8, 16)]
    + [x + 0.4 for x in range(8, 16)]
    + [x + 0.6 for x in range(8, 16)]
    + [x + 0.8 for x in range(8, 16)],
    dtype=np.float64,
)

INT_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))[:8]


# =====================================================================
#  Figure A: Codebook Spectrum
# =====================================================================
def fig_codebook():
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), height_ratios=[1, 1])

    # --- Panel 1: Full spectrum with harmonics ---
    ax = axes[0]
    N_HARMONICS = 5

    for i, f in enumerate(BASE_FREQS):
        group = int(f) - 8
        color = INT_COLORS[group]
        for h in range(1, N_HARMONICS + 1):
            amp = 1.0 / h
            ax.vlines(f * h, 0, amp, colors=color, linewidth=1.2, alpha=0.7)

    # Band annotations
    bands = [
        (8, 16, "基频\n8–16 Hz"),
        (16, 32, "二次谐波\n16–32 Hz"),
        (32, 48, "三次\n32–48 Hz"),
        (48, 64, "四次\n48–64 Hz"),
        (64, 80, "五次\n64–80 Hz"),
    ]
    for lo, hi, label in bands:
        ax.axvspan(lo, hi, alpha=0.04, color="gray")
        ax.text((lo + hi) / 2, 1.08, label, ha="center", va="bottom",
                fontsize=9, color="0.4")

    ax.set_xlim(6, 82)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("频率 (Hz)", fontsize=12)
    ax.set_ylabel("相对幅度 (1/m)", fontsize=12)
    ax.set_title("码本全频谱：40 基频 × 5 次谐波 = 200 条谱线", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.1)

    # Legend for frequency groups
    handles = [mpatches.Patch(color=INT_COLORS[g], label=f"{g+8}.x Hz") for g in range(8)]
    ax.legend(handles=handles, ncol=4, fontsize=9, loc="upper right", framealpha=0.9,
              title="基频组")

    # --- Panel 2: Zoom on fundamental band ---
    ax2 = axes[1]
    for i, f in enumerate(BASE_FREQS):
        group = int(f) - 8
        color = INT_COLORS[group]
        ax2.vlines(f, 0, 1.0, colors=color, linewidth=2.5, alpha=0.85)

    # Annotate spacing
    ax2.annotate("", xy=(8.2, 0.5), xytext=(8.0, 0.5),
                 arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
    ax2.text(8.1, 0.55, "Δf = 0.2 Hz", ha="center", fontsize=11, color="red",
             fontweight="bold")

    # Show phase annotations for a few frequencies
    phases_example = [0, 0.5, 1.0, 1.5]  # π radians
    for j, f in enumerate(BASE_FREQS[:4]):
        ax2.text(f, 1.05, f"φ={phases_example[j]:.1f}π", ha="center", fontsize=8,
                 color=INT_COLORS[int(f)-8], rotation=45)

    ax2.set_xlim(7.8, 16.2)
    ax2.set_ylim(0, 1.25)
    ax2.set_xlabel("频率 (Hz)", fontsize=12)
    ax2.set_ylabel("幅度", fontsize=12)
    ax2.set_title("基频带放大：40 个频率密集排列于 8–15.8 Hz (最小间隔 0.2 Hz)",
                  fontsize=14, fontweight="bold")
    ax2.grid(alpha=0.1)

    # Tick at every 0.2 Hz would be too dense; use 1 Hz
    ax2.set_xticks(np.arange(8, 17, 1))
    ax2.set_xticks(np.arange(8, 16.1, 0.2), minor=True)
    ax2.grid(which="minor", alpha=0.05)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_model_A_codebook_spectrum.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_model_A_codebook_spectrum.png")


# =====================================================================
#  Figure B: Gaze Interference Model
# =====================================================================
def fig_gaze_model():
    fig = plt.figure(figsize=(20, 10))

    # ----- Panel 1: Screen layout (8×5 grid of flickers) -----
    ax1 = fig.add_axes([0.03, 0.08, 0.35, 0.82])

    n_cols, n_rows = 8, 5
    target_col, target_row = 3, 2  # fixate on 11.4 Hz

    # Assume screen ~50cm wide, ~30cm tall, viewing distance 60cm
    # Grid cell ~5cm, visual angle per cell ~4.8°
    cell_w, cell_h = 1.0, 1.0
    gap_w, gap_h = 0.15, 0.15

    freq_grid = np.zeros((n_rows, n_cols))
    for r in range(n_rows):
        for c in range(n_cols):
            freq_grid[r, c] = 8 + c + r * 0.2

    fixated_freq = freq_grid[target_row, target_col]
    fix_x = target_col * (cell_w + gap_w) + cell_w / 2
    fix_y = (n_rows - 1 - target_row) * (cell_h + gap_h) + cell_h / 2

    # Draw eccentricity circles first (behind targets)
    for radius, alpha_c, label in [(1.2, 0.08, "~5°"), (2.5, 0.05, "~12°"), (4.0, 0.03, "~20°")]:
        circle = Circle((fix_x, fix_y), radius, fill=True, facecolor="orange",
                        alpha=alpha_c, edgecolor="orange", linewidth=0.8, linestyle="--",
                        zorder=1)
        ax1.add_patch(circle)

    # Draw targets
    for r in range(n_rows):
        for c in range(n_cols):
            x = c * (cell_w + gap_w)
            y = (n_rows - 1 - r) * (cell_h + gap_h)
            freq = freq_grid[r, c]
            group = int(freq) - 8
            color = INT_COLORS[group]

            dist = np.sqrt((c - target_col)**2 + (r - target_row)**2)
            alpha_val = max(0.3, 1.0 - dist * 0.15)

            is_target = (r == target_row and c == target_col)
            lw = 3.0 if is_target else 0.8
            ec = "red" if is_target else "0.3"

            rect = FancyBboxPatch(
                (x, y), cell_w, cell_h,
                boxstyle="round,pad=0.05",
                facecolor=color, alpha=alpha_val,
                edgecolor=ec, linewidth=lw, zorder=3,
            )
            ax1.add_patch(rect)

            # Frequency label
            ax1.text(x + cell_w/2, y + cell_h/2, f"{freq:.1f}",
                     ha="center", va="center", fontsize=7,
                     fontweight="bold" if is_target else "normal",
                     color="white" if is_target else "black",
                     zorder=4)

    # Fixation crosshair
    ax1.plot(fix_x, fix_y, "r+", markersize=20, markeredgewidth=3, zorder=10)

    # Label within-target fixation points (zoomed inset concept)
    ax1.annotate("5个注视点\n(R/D/L/U/C)\n偏移 0.23°",
                 xy=(fix_x + 0.3, fix_y + 0.3),
                 xytext=(fix_x + 2.5, fix_y + 2.5),
                 fontsize=9, color="red",
                 arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                 bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.9))

    ax1.set_xlim(-0.5, n_cols * (cell_w + gap_w))
    ax1.set_ylim(-0.5, n_rows * (cell_h + gap_h))
    ax1.set_aspect("equal")
    ax1.set_title(f"屏幕布局 (8×5 = 40 闪烁目标)\n注视目标: {fixated_freq:.1f} Hz (红框)",
                  fontsize=13, fontweight="bold")
    ax1.axis("off")

    # ----- Panel 2: Signal strength vs eccentricity -----
    ax2 = fig.add_axes([0.42, 0.55, 0.22, 0.35])

    d = np.linspace(0, 25, 200)
    # Cortical magnification factor model (approximate)
    # M(d) ∝ 1 / (1 + d/d0) where d0 ≈ 1-2°
    alpha_foveal = 1.0 / (1 + d / 1.5) ** 1.8

    ax2.fill_between(d, alpha_foveal, alpha=0.15, color="steelblue")
    ax2.plot(d, alpha_foveal, "steelblue", linewidth=2.5)

    # Mark zones
    ax2.axvspan(0, 2, alpha=0.1, color="green")
    ax2.axvspan(2, 5, alpha=0.07, color="orange")
    ax2.text(1.0, 0.9, "中央凹\n(fovea)", ha="center", fontsize=9, color="green")
    ax2.text(3.5, 0.35, "旁中央凹\n(parafovea)", ha="center", fontsize=8, color="orange")
    ax2.text(12, 0.08, "外周\n(periphery)", ha="center", fontsize=8, color="0.5")

    # Mark neighbor distances
    for dist_deg, label in [(4.8, "邻近目标"), (9.6, "次邻近")]:
        y_val = 1.0 / (1 + dist_deg / 1.5) ** 1.8
        ax2.plot(dist_deg, y_val, "ro", markersize=8, zorder=5)
        ax2.annotate(f"{label}\n~{dist_deg:.0f}°\nα={y_val:.2f}",
                     (dist_deg, y_val),
                     textcoords="offset points", xytext=(15, 10),
                     fontsize=8, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=1))

    ax2.set_xlabel("离心率 (°)", fontsize=11)
    ax2.set_ylabel("SSVEP 响应衰减 α(d)", fontsize=11)
    ax2.set_title("视网膜灵敏度衰减", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 25)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.15)

    # ----- Panel 3: Received signal spectrum (conceptual) -----
    ax3 = fig.add_axes([0.42, 0.08, 0.55, 0.38])

    freqs_plot = np.linspace(1, 85, 5000)
    # Background 1/f + alpha
    noise_floor = 0.08 / (freqs_plot ** 0.5)
    alpha_bump = 0.25 * np.exp(-0.5 * ((freqs_plot - 10) / 1.5) ** 2)
    background = noise_floor + alpha_bump

    ax3.fill_between(freqs_plot, background, alpha=0.15, color="gray")
    ax3.plot(freqs_plot, background, "gray", linewidth=1, alpha=0.5, label="背景EEG (1/f + α节律)")

    # Target SSVEP harmonics (11.4 Hz)
    target_f = fixated_freq
    for h in range(1, 6):
        amp = 1.0 / h
        freq_h = target_f * h
        peak = amp * np.exp(-0.5 * ((freqs_plot - freq_h) / 0.15) ** 2)
        ax3.fill_between(freqs_plot, peak, alpha=0.3, color="#e63946")
        ax3.plot(freqs_plot, peak, "#e63946", linewidth=1.5)
        if h <= 3:
            ax3.text(freq_h, amp + 0.04, f"{h}f₀={freq_h:.1f}",
                     ha="center", fontsize=8, color="#e63946", fontweight="bold")

    # Nearest neighbor SSVEP (11.2, 11.6 Hz on same column; 10.4, 12.4 on adj columns)
    neighbors = [
        (11.2, 0.06, "#ff9f43"),  # same column, adjacent row
        (11.6, 0.06, "#ff9f43"),
        (10.4, 0.04, "#feca57"),  # adjacent column
        (12.4, 0.04, "#feca57"),
    ]
    for nf, n_amp, nc in neighbors:
        for h in range(1, 4):
            amp = n_amp / h
            freq_h = nf * h
            peak = amp * np.exp(-0.5 * ((freqs_plot - freq_h) / 0.12) ** 2)
            ax3.fill_between(freqs_plot, peak, alpha=0.2, color=nc)

    # Legend entries
    ax3.plot([], [], color="#e63946", linewidth=3, label=f"目标 SSVEP ({target_f:.1f} Hz)")
    ax3.plot([], [], color="#ff9f43", linewidth=3, label="近邻干扰 (±0.2 Hz)")
    ax3.plot([], [], color="#feca57", linewidth=3, label="次邻干扰 (±1.0 Hz)")

    ax3.set_xlim(5, 75)
    ax3.set_ylim(0, 1.15)
    ax3.set_xlabel("频率 (Hz)", fontsize=12)
    ax3.set_ylabel("幅度", fontsize=12)
    ax3.set_title("接收信号频谱模型：r(t) = 目标SSVEP + 邻近干扰 + 背景EEG + 噪声",
                  fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax3.grid(alpha=0.1)

    # ----- Panel 4: Signal model equation -----
    ax4 = fig.add_axes([0.68, 0.55, 0.30, 0.35])
    ax4.axis("off")

    model_text = (
        "接收信号数学模型\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "注视目标 k 时，通道 c 记录：\n\n"
        "  xc(t) = Σᵢ α(dᵢₖ) · sᵢc(t) + nc(t)\n\n"
        "其中：\n"
        "  sᵢc(t) = Σₘ aₘ sin(2πmfᵢt + φₘ)\n"
        "       → 目标 i 的 SSVEP 响应\n\n"
        "  α(d) = 1/(1+d/d₀)^p\n"
        "       → 离心率衰减 (视网膜灵敏度)\n\n"
        "  dᵢₖ = 目标 i 与注视点的视角距离\n\n"
        "  nc(t) → 背景 EEG 噪声\n\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "标准简化假设：\n"
        "  α(d₀) ≈ 1 (注视目标)\n"
        "  α(dᵢ) ≈ 0 (i ≠ k)\n\n"
        "密集屏幕 (HD200) 实际情况：\n"
        "  邻近目标 α ≈ 0.04–0.06\n"
        "  → 产生频率间干扰 (ICI)"
    )
    ax4.text(0.05, 0.95, model_text, transform=ax4.transAxes,
             fontsize=10, fontfamily="Microsoft YaHei", verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9, edgecolor="0.7"))

    fig.savefig(FIG_DIR / "fig_model_B_gaze_interference.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_model_B_gaze_interference.png")


# =====================================================================
#  Figure C: Data Pipeline & TDCA Augmentation
# =====================================================================
def fig_pipeline():
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ----- Panel 1: Data tensor and feature extraction -----
    ax = axes[0]
    ax.axis("off")

    pipeline_text = (
        "数据处理流程与维度\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "原始缓存数据:\n"
        "  (66, 185, 200, 18, 5)\n"
        "   ch  time tgt  blk  fb\n\n"
        "    ↓ 选择 FB0\n"
        "  (66, 185, 200, 18)\n\n"
        "    ↓ 截取窗口 [35:160] (500ms)\n"
        "  (66, 125, 200, 18)\n\n"
        "    ↓ 展平 + 去均值 + 归一化\n"
        "  (8250,) per trial     ← 特征向量\n\n"
        "    ↓ LOBO 平均 (17 blocks)\n"
        "  (8250,) per class     ← 模板向量\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "分类: k* = argmax_k  corr(x, Tₖ)\n"
        "     = argmax_k  x·Tₖ / (‖x‖‖Tₖ‖)\n\n"
        "特征空间:\n"
        "  名义维度:    8,250\n"
        "  有效秩(频率):  32–33\n"
        "  有效秩(空间): 132–139\n"
        "  有效秩(总):   ~159"
    )
    ax.text(0.05, 0.95, pipeline_text, transform=ax.transAxes,
            fontsize=11, fontfamily="Microsoft YaHei", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="aliceblue", alpha=0.9, edgecolor="steelblue"))
    ax.set_title("数据表示", fontsize=14, fontweight="bold")

    # ----- Panel 2: TDCA augmentation -----
    ax2 = axes[1]
    ax2.axis("off")

    # Draw matrix blocks
    def draw_matrix(ax, x, y, w, h, label, color, text_inside=""):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                              facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text_inside, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
        ax.text(x + w/2, y - 0.03, label, ha="center", va="top", fontsize=9)

    # Original matrix X(t): 66 × T
    draw_matrix(ax2, 0.05, 0.72, 0.12, 0.18, "X(t)", "steelblue", "66×T")

    # Arrow
    ax2.annotate("", xy=(0.22, 0.81), xytext=(0.17, 0.81),
                 arrowprops=dict(arrowstyle="->", lw=2, color="0.3"))
    ax2.text(0.195, 0.85, "时域增广\n(lag=5)", ha="center", fontsize=9, color="0.3")

    # Augmented matrix: 5 stacked blocks
    colors_aug = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    labels_aug = ["X(t)", "X(t−1)", "X(t−2)", "X(t−3)", "X(t−4)"]
    block_h = 0.06
    for i in range(5):
        y_pos = 0.72 + (4 - i) * block_h
        rect = FancyBboxPatch((0.25, y_pos), 0.15, block_h - 0.005,
                              boxstyle="round,pad=0.01",
                              facecolor=colors_aug[i], alpha=0.25,
                              edgecolor=colors_aug[i], linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(0.325, y_pos + block_h/2, labels_aug[i], ha="center", va="center",
                fontsize=8, color=colors_aug[i], fontweight="bold")

    # Dimension label
    ax2.annotate("", xy=(0.24, 0.72), xytext=(0.24, 1.02),
                 arrowprops=dict(arrowstyle="<->", lw=1.5, color="red"))
    ax2.text(0.235, 0.87, "330", ha="right", fontsize=12, color="red", fontweight="bold")

    ax2.annotate("", xy=(0.25, 0.70), xytext=(0.40, 0.70),
                 arrowprops=dict(arrowstyle="<->", lw=1.5, color="blue"))
    ax2.text(0.325, 0.675, "T", ha="center", fontsize=12, color="blue", fontweight="bold")

    ax2.text(0.325, 0.65, "(66×5) × T = 330 × T", ha="center", fontsize=10, fontweight="bold")

    # Arrow to Fisher
    ax2.annotate("", xy=(0.50, 0.81), xytext=(0.42, 0.81),
                 arrowprops=dict(arrowstyle="->", lw=2, color="0.3"))
    ax2.text(0.46, 0.85, "Fisher\n判别", ha="center", fontsize=9, color="0.3")

    # Spatial filter w
    draw_matrix(ax2, 0.52, 0.75, 0.04, 0.12, "w", "#E91E63", "330\n×1")

    # Description text
    tdca_text = (
        "TDCA 时域增广机制\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "原始空间滤波器:\n"
        "  y(t) = wᵀ · x(t)\n"
        "  w ∈ ℝ⁶⁶ → 纯通道加权\n\n"
        "增广后时空滤波器:\n"
        "  y(t) = w̃ᵀ · x̃(t)\n"
        "  w̃ ∈ ℝ³³⁰ → 时空联合优化\n\n"
        "等价于 66 个通道各带 5 抽头 FIR:\n"
        "  yc(t) = Σₗ wc,l · xc(t−l)\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Fisher 判别目标:\n"
        "  max  w̃ᵀ Sb w̃ / w̃ᵀ Sw w̃\n"
        "  → 最大化类间距离/类内散度\n\n"
        "通信类比:\n"
        "  纯空间滤波 = 单抽头波束成形\n"
        "  时域增广    = 多抽头时空均衡器"
    )
    ax2.text(0.05, 0.58, tdca_text, transform=ax2.transAxes,
             fontsize=10, fontfamily="Microsoft YaHei", verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="lavenderblush", alpha=0.9,
                       edgecolor="#E91E63"))
    ax2.set_title("TDCA 时域增广", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_model_C_pipeline_tdca.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_model_C_pipeline_tdca.png")


# =====================================================================
#  Figure D: Analysis Tool Comparison
# =====================================================================
def fig_tools_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Simulate 2D data for visualization
    np.random.seed(42)
    n_per_class = 30
    centers = np.array([[2, 0], [-1, 1.5], [-1, -1.5], [0.5, 0.8], [0.5, -0.8]])
    colors = ["#e63946", "#1d3557", "#2a9d8f", "#e9c46a", "#6c757d"]
    labels = ["R", "D", "L", "U", "C"]

    # Add large common-mode noise (simulates SSVEP: signal << noise)
    noise_common = np.random.randn(n_per_class * 5, 2) * 5
    all_points = []
    all_labels = []
    for i, c in enumerate(centers):
        pts = c + np.random.randn(n_per_class, 2) * 0.5 + noise_common[i*n_per_class:(i+1)*n_per_class]
        all_points.append(pts)
        all_labels.extend([i] * n_per_class)
    X = np.vstack(all_points)
    y = np.array(all_labels)

    # --- Panel 1: Raw data (PCA-like view) ---
    ax = axes[0, 0]
    for i in range(5):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=30, alpha=0.5, label=labels[i])
    ax.set_title("原始数据 (高噪声)\n信号完全被噪声淹没", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.1)
    ax.set_xlabel("维度 1 (噪声主导)")
    ax.set_ylabel("维度 2 (噪声主导)")

    # --- Panel 2: PCA projection ---
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    ax = axes[0, 1]
    for i in range(5):
        mask = y == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], s=30, alpha=0.5, label=labels[i])
    var_exp = pca.explained_variance_ratio_[:2].sum() * 100
    ax.set_title(f"PCA 投影 (方差 {var_exp:.0f}%)\n最大化总方差 → 仍然看噪声", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.1)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")

    # --- Panel 3: LDA projection ---
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)
    ax = axes[1, 0]
    for i in range(5):
        mask = y == i
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1], c=colors[i], s=30, alpha=0.5, label=labels[i])
    ax.set_title("LDA 投影\n最大化类间/类内方差比 → 信号显现", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.1)
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")

    # --- Panel 4: Summary text ---
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        "分析工具对比\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "PCA: max Var(wᵀX)\n"
        " → 找总方差最大的方向\n"
        " → 当噪声 >> 信号时，捕获噪声\n"
        " → 适合降维，不适合判别\n\n"
        "LDA: max wᵀSbw / wᵀSww\n"
        " → 找类间分离最大的方向\n"
        " → 即使噪声 >> 信号，仍能分开\n"
        " → 需要标签 (有监督)\n\n"
        "TDCA ≈ LDA 的 SSVEP 特化版:\n"
        " → TRCA 框架 (最大化同类相关)\n"
        " → 时域增广 (时空联合优化)\n"
        " → 不需要逐类 Fisher 判别\n\n"
        "t-SNE:\n"
        " → 保持局部邻域结构\n"
        " → 非线性嵌入，不可逆\n"
        " → 适合可视化聚类结构\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "核心结论:\n"
        "空间信息存在于特定判别方向上,\n"
        "PCA 看不到 ≠ 不存在,\n"
        "LDA/TDCA 能找到并投影出来。"
    )
    ax.text(0.05, 0.98, summary, transform=ax.transAxes,
            fontsize=10, fontfamily="Microsoft YaHei", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("为什么 PCA 看不到空间信息而 LDA 可以？— 投影方向的选择",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_model_D_tools_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_model_D_tools_comparison.png")


def main():
    t0 = time.time()
    fig_codebook()
    fig_gaze_model()
    fig_pipeline()
    fig_tools_comparison()
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
