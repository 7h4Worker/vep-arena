"""Filter bank frequency response analysis.

Visualizes DSP preprocessing differences across SSVEP decoding methods:
  A. 50 Hz notch filter response
  B. 5-subband Chebyshev Type I filterbank (individual + combined)
  C. Filterbank-weighted combined gain
  D. CCA vs FBCCA effective input comparison
  E. Subband boundary vs SSVEP harmonic alignment
  F. Group delay analysis

Usage
-----
    .venv/Scripts/python.exe scripts/22_filterbank_frequency_response.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

TASK_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = TASK_DIR / "results" / "figures"

FS = 250
NYQUIST = FS / 2

SSVEP_FREQS = np.array([8.0 + 0.2 * i for i in range(40)])
FB_WEIGHTS = np.array([(k + 1) ** (-1.25) + 0.25 for k in range(5)])

SUBBAND_COLORS = ["#e63946", "#2a9d8f", "#e9c46a", "#264653", "#6c757d"]
SUBBAND_LABELS = [
    "FB0: 8-90 Hz",
    "FB1: 16-90 Hz",
    "FB2: 24-90 Hz",
    "FB3: 32-90 Hz",
    "FB4: 40-90 Hz",
]


def design_notch():
    return signal.iircomb(50, 35, ftype="notch", fs=FS)


def design_subband(k):
    """Design subband k (0-based) Chebyshev Type I bandpass."""
    wp = [(8 * (k + 1)) / NYQUIST, 90 / NYQUIST]
    ws = [(8 * (k + 1) - 2) / NYQUIST, 100 / NYQUIST]
    order, wn = signal.cheb1ord(wp, ws, 3, 40)
    b, a = signal.cheby1(order, 0.5, wn, btype="bandpass")
    return b, a, order


def fig_main():
    """Main 6-panel figure."""
    fig = plt.figure(figsize=(24, 18))

    # Compute all frequency responses
    freqs = np.linspace(0, NYQUIST, 4096)

    # Notch filter
    b_n, a_n = design_notch()
    _, h_notch = signal.freqz(b_n, a_n, worN=freqs, fs=FS)

    # Subbands
    subband_responses = []
    subband_orders = []
    subband_delays = []
    for k in range(5):
        b, a, order = design_subband(k)
        _, h = signal.freqz(b, a, worN=freqs, fs=FS)
        subband_responses.append(h)
        subband_orders.append(order)
        _, gd = signal.group_delay((b, a), w=freqs, fs=FS)
        subband_delays.append(gd)

    # ===== Panel A: Notch filter =====
    ax_a = fig.add_axes([0.05, 0.70, 0.42, 0.25])

    mag_notch = 20 * np.log10(np.abs(h_notch) + 1e-15)
    ax_a.plot(freqs, mag_notch, "steelblue", linewidth=2)
    ax_a.axvline(50, color="red", linestyle="--", alpha=0.5, label="50 Hz")
    ax_a.axvline(100, color="red", linestyle=":", alpha=0.3, label="100 Hz")
    ax_a.set_xlim(0, 125)
    ax_a.set_ylim(-50, 5)
    ax_a.set_xlabel("频率 (Hz)", fontsize=11)
    ax_a.set_ylabel("幅度 (dB)", fontsize=11)
    ax_a.set_title("(a) 50 Hz IIR 梳状陷波滤波器 (Q=35)\n所有方法共享, 含 CCA",
                   fontsize=13, fontweight="bold")
    ax_a.legend(fontsize=10)
    ax_a.grid(alpha=0.2)

    # Mark SSVEP harmonics that fall near 50 Hz
    for h in range(1, 7):
        for f in [10.0, 12.5]:
            fh = f * h
            if 45 < fh < 55:
                ax_a.annotate(f"{f}x{h}={fh}", xy=(fh, -40),
                              fontsize=8, color="orange", ha="center")

    # ===== Panel B: 5 subbands individual =====
    ax_b = fig.add_axes([0.55, 0.70, 0.42, 0.25])

    for k in range(5):
        mag = 20 * np.log10(np.abs(subband_responses[k]) + 1e-15)
        ax_b.plot(freqs, mag, color=SUBBAND_COLORS[k], linewidth=2,
                  label=f"{SUBBAND_LABELS[k]} (order {subband_orders[k]})")

    # Mark SSVEP base frequency range
    ax_b.axvspan(8, 15.8, alpha=0.08, color="blue", label="基频带 8-15.8 Hz")
    ax_b.axvspan(16, 31.6, alpha=0.06, color="green")
    ax_b.axvspan(32, 47.4, alpha=0.04, color="orange")

    ax_b.set_xlim(0, 110)
    ax_b.set_ylim(-60, 5)
    ax_b.set_xlabel("频率 (Hz)", fontsize=11)
    ax_b.set_ylabel("幅度 (dB)", fontsize=11)
    ax_b.set_title("(b) 5 子带 Chebyshev I 带通滤波器频响\n通带纹波 0.5 dB, 阻带衰减 40 dB",
                   fontsize=13, fontweight="bold")
    ax_b.legend(fontsize=9, loc="lower right")
    ax_b.grid(alpha=0.2)

    # ===== Panel C: Weighted combined response =====
    ax_c = fig.add_axes([0.05, 0.38, 0.42, 0.25])

    combined_linear = np.zeros_like(freqs)
    for k in range(5):
        combined_linear += FB_WEIGHTS[k] * np.abs(subband_responses[k])

    ax_c.plot(freqs, combined_linear, "k", linewidth=2.5, label="加权合计")

    for k in range(5):
        ax_c.fill_between(freqs, 0, FB_WEIGHTS[k] * np.abs(subband_responses[k]),
                          color=SUBBAND_COLORS[k], alpha=0.15)
        ax_c.plot(freqs, FB_WEIGHTS[k] * np.abs(subband_responses[k]),
                  color=SUBBAND_COLORS[k], linewidth=1, alpha=0.5,
                  label=f"FB{k} x w={FB_WEIGHTS[k]:.3f}")

    # Mark where each harmonic band starts losing subbands
    for h_order, color_h in [(1, "blue"), (2, "green"), (3, "orange"), (4, "red")]:
        f_low = 8.0 * h_order
        f_high = 15.8 * h_order
        ax_c.axvspan(f_low, min(f_high, 90), alpha=0.04, color=color_h)
        ax_c.text((f_low + min(f_high, 90)) / 2, combined_linear.max() * 0.92,
                  f"{h_order}f", fontsize=9, ha="center", color=color_h, fontweight="bold")

    ax_c.set_xlim(0, 95)
    ax_c.set_ylim(0, combined_linear.max() * 1.1)
    ax_c.set_xlabel("频率 (Hz)", fontsize=11)
    ax_c.set_ylabel("加权幅度增益", fontsize=11)
    ax_c.set_title("(c) 滤波器组加权合成增益: w(k) = (k+1)^{-1.25} + 0.25\n"
                   "颜色带 = 各次谐波带所在区间",
                   fontsize=13, fontweight="bold")
    ax_c.legend(fontsize=8, loc="upper right", ncol=2)
    ax_c.grid(alpha=0.2)

    # ===== Panel D: Per-harmonic subband coverage =====
    ax_d = fig.add_axes([0.55, 0.38, 0.42, 0.25])

    harmonics_label = ["1f (基波)", "2f", "3f", "4f", "5f"]
    harmonic_ranges = []
    for h in range(1, 6):
        f_lo = 8.0 * h
        f_hi = 15.8 * h
        harmonic_ranges.append((f_lo, f_hi))

    # For each harmonic, which subbands pass it?
    coverage = np.zeros((5, 5))  # harmonics x subbands
    subband_cutoffs = [8, 16, 24, 32, 40]

    for h_idx in range(5):
        f_lo, f_hi = harmonic_ranges[h_idx]
        for fb_idx in range(5):
            fb_lo = subband_cutoffs[fb_idx]
            fb_hi = 90
            if fb_lo <= f_hi and fb_hi >= f_lo:
                overlap_lo = max(f_lo, fb_lo)
                overlap_hi = min(f_hi, fb_hi)
                if overlap_hi > overlap_lo:
                    full_range = f_hi - f_lo
                    coverage[h_idx, fb_idx] = (overlap_hi - overlap_lo) / full_range

    im = ax_d.imshow(coverage, aspect="auto", cmap="YlGn", vmin=0, vmax=1,
                     interpolation="nearest")

    for i in range(5):
        for j in range(5):
            val = coverage[i, j]
            if val > 0:
                color = "white" if val > 0.6 else "black"
                ax_d.text(j, i, f"{val:.0%}", ha="center", va="center",
                          fontsize=13, fontweight="bold", color=color)
            else:
                ax_d.text(j, i, "-", ha="center", va="center",
                          fontsize=13, color="0.7")

    ax_d.set_xticks(range(5))
    ax_d.set_xticklabels([f"FB{k}\n{subband_cutoffs[k]}-90" for k in range(5)], fontsize=10)
    ax_d.set_yticks(range(5))
    ax_d.set_yticklabels([f"{harmonics_label[h]}\n{harmonic_ranges[h][0]:.0f}-{harmonic_ranges[h][1]:.0f}"
                          for h in range(5)], fontsize=10)
    ax_d.set_xlabel("子带", fontsize=12)
    ax_d.set_ylabel("谐波次数", fontsize=12)
    ax_d.set_title("(d) 各子带对各次谐波的覆盖率\n"
                   "FB0 覆盖所有谐波, FB4 仅覆盖 5f",
                   fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax_d, shrink=0.7, label="覆盖率")

    # ===== Panel E: Group delay =====
    ax_e = fig.add_axes([0.05, 0.06, 0.42, 0.25])

    for k in range(5):
        gd = subband_delays[k]
        gd_ms = gd / FS * 1000
        gd_ms = np.clip(gd_ms, -50, 100)
        ax_e.plot(freqs, gd_ms, color=SUBBAND_COLORS[k], linewidth=1.5,
                  label=f"FB{k} (order {subband_orders[k]})")

    ax_e.axhspan(0, 10, alpha=0.05, color="green")
    ax_e.set_xlim(5, 95)
    ax_e.set_ylim(-10, 60)
    ax_e.set_xlabel("频率 (Hz)", fontsize=11)
    ax_e.set_ylabel("群时延 (ms)", fontsize=11)
    ax_e.set_title("(e) 各子带群时延\n"
                   "高阶滤波器在截止频率附近群时延显著增大",
                   fontsize=13, fontweight="bold")
    ax_e.legend(fontsize=9, loc="upper right")
    ax_e.grid(alpha=0.2)

    # ===== Panel F: CCA vs FBCCA input comparison =====
    ax_f = fig.add_axes([0.55, 0.06, 0.42, 0.25])

    # CCA: only notch
    mag_cca = np.abs(h_notch)

    # FBCCA: notch + best subband at each frequency (envelope)
    mag_fbcca_env = np.zeros_like(freqs)
    for k in range(5):
        fb_mag = np.abs(h_notch) * np.abs(subband_responses[k])
        mag_fbcca_env = np.maximum(mag_fbcca_env, fb_mag)

    ax_f.fill_between(freqs, 0, mag_cca, alpha=0.15, color="steelblue", label="CCA: 仅陷波")
    ax_f.plot(freqs, mag_cca, "steelblue", linewidth=2)

    ax_f.plot(freqs, mag_fbcca_env, "red", linewidth=2, linestyle="--",
              label="FBCCA: 陷波 + 最佳子带包络")

    # Mark SSVEP frequencies
    for i, f in enumerate(SSVEP_FREQS):
        if i % 8 == 0:
            ax_f.axvline(f, color="green", alpha=0.2, linewidth=0.8)

    ax_f.set_xlim(0, 95)
    ax_f.set_ylim(0, 1.15)
    ax_f.set_xlabel("频率 (Hz)", fontsize=11)
    ax_f.set_ylabel("幅度增益", fontsize=11)
    ax_f.set_title("(f) CCA vs FBCCA 有效输入频带对比\n"
                   "CCA 接收全频带 (含低频噪声), FBCCA 各子带逐级切除低频",
                   fontsize=13, fontweight="bold")
    ax_f.legend(fontsize=10, loc="upper right")
    ax_f.grid(alpha=0.2)

    # Annotate the key difference
    ax_f.annotate("CCA 包含 0-8 Hz\n低频 EEG 噪声",
                  xy=(4, 0.95), fontsize=10, color="steelblue",
                  fontweight="bold",
                  bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
    ax_f.annotate("FB0 在 8 Hz 以下\n-40 dB 抑制",
                  xy=(4, 0.3), fontsize=10, color="red",
                  bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    fig.suptitle("SSVEP-BCI 预处理滤波器频响分析\n"
                 "Benchmark / HD200 共享预处理链: 50Hz 陷波 → Chebyshev I 滤波器组 (5 子带)",
                 fontsize=16, fontweight="bold", y=0.99)

    return fig


def fig_fb_impact_on_target():
    """Show how different subbands see the same SSVEP target."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    freqs_dense = np.linspace(0, NYQUIST, 8192)

    # Design all subbands
    subband_h = []
    for k in range(5):
        b, a, _ = design_subband(k)
        _, h = signal.freqz(b, a, worN=freqs_dense, fs=FS)
        subband_h.append(h)

    # Notch
    b_n, a_n = design_notch()
    _, h_notch = signal.freqz(b_n, a_n, worN=freqs_dense, fs=FS)

    target_freqs = [8.0, 10.0, 12.0, 14.0, 8.6, 15.8]

    for idx, f_target in enumerate(target_freqs):
        ax = axes.ravel()[idx]

        # Mark harmonics
        for h in range(1, 6):
            fh = f_target * h
            if fh <= NYQUIST:
                ax.axvline(fh, color="0.3", linewidth=0.8, alpha=0.3)
                ax.text(fh, 1.08, f"{h}f", fontsize=8, ha="center",
                        fontweight="bold", color="0.3")

        # Plot each subband's gain at this target's harmonics
        bar_data = []
        for k in range(5):
            gains = []
            for h in range(1, 6):
                fh = f_target * h
                if fh <= NYQUIST:
                    idx_f = np.argmin(np.abs(freqs_dense - fh))
                    g = np.abs(subband_h[k][idx_f]) * np.abs(h_notch[idx_f])
                    gains.append(g)
                else:
                    gains.append(0)
            bar_data.append(gains)

        bar_data = np.array(bar_data)  # (5, 5)
        x_pos = np.arange(5)
        width = 0.15

        for k in range(5):
            ax.bar(x_pos + k * width - 2 * width, bar_data[k],
                   width=width, color=SUBBAND_COLORS[k], alpha=0.7,
                   label=f"FB{k}" if idx == 0 else None)

        # Weighted sum
        weighted = np.zeros(5)
        for k in range(5):
            weighted += FB_WEIGHTS[k] * bar_data[k]
        ax.plot(x_pos, weighted, "ko-", markersize=6, linewidth=2,
                label="加权合计" if idx == 0 else None)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{f_target*h:.1f}" for h in range(1, 6)], fontsize=9)
        ax.set_ylim(0, 1.3)
        ax.set_xlabel("谐波频率 (Hz)", fontsize=10)
        ax.set_ylabel("通过增益", fontsize=10)
        ax.set_title(f"目标 {f_target:.1f} Hz", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.15, axis="y")

    axes[0, 0].legend(fontsize=9, loc="upper right")

    fig.suptitle("各目标频率在 5 个子带中的谐波通过增益\n"
                 "低频目标 (8 Hz) 在所有子带保留基波; "
                 "高频目标 (15.8 Hz) 的高次谐波被 90 Hz 截止",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig_subband_selectivity():
    """Show how FB changes the effective 'channel' for different frequencies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # For each of 40 target frequencies, compute total weighted harmonic energy
    # passing through the filterbank vs raw (CCA-style)
    target_freqs = np.array([8.0 + 0.2 * i for i in range(40)])

    freqs_dense = np.linspace(0, NYQUIST, 8192)

    subband_h = []
    for k in range(5):
        b, a, _ = design_subband(k)
        _, h = signal.freqz(b, a, worN=freqs_dense, fs=FS)
        subband_h.append(h)

    b_n, a_n = design_notch()
    _, h_notch = signal.freqz(b_n, a_n, worN=freqs_dense, fs=FS)

    # Per-target, per-subband effective gain (sum over 5 harmonics)
    gain_matrix = np.zeros((40, 5))  # targets x subbands
    gain_cca = np.zeros(40)

    for ti, ft in enumerate(target_freqs):
        for k in range(5):
            total = 0
            for h in range(1, 6):
                fh = ft * h
                if fh <= NYQUIST:
                    idx_f = np.argmin(np.abs(freqs_dense - fh))
                    total += np.abs(subband_h[k][idx_f]) * np.abs(h_notch[idx_f])
            gain_matrix[ti, k] = total

        total_cca = 0
        for h in range(1, 6):
            fh = ft * h
            if fh <= NYQUIST:
                idx_f = np.argmin(np.abs(freqs_dense - fh))
                total_cca += np.abs(h_notch[idx_f])
        gain_cca[ti] = total_cca

    # Panel 1: Stacked bar - per-target subband contribution
    bottom = np.zeros(40)
    for k in range(5):
        weighted_gain = FB_WEIGHTS[k] * gain_matrix[:, k]
        ax1.bar(range(40), weighted_gain, bottom=bottom,
                color=SUBBAND_COLORS[k], alpha=0.7, label=f"FB{k} (w={FB_WEIGHTS[k]:.3f})")
        bottom += weighted_gain

    ax1.plot(range(40), gain_cca, "k^-", markersize=4, linewidth=1.5,
             label="CCA (无滤波器组)", zorder=5)

    ax1.set_xticks(range(0, 40, 4))
    ax1.set_xticklabels([f"{target_freqs[i]:.1f}" for i in range(0, 40, 4)],
                        fontsize=9, rotation=45)
    ax1.set_xlabel("目标频率 (Hz)", fontsize=11)
    ax1.set_ylabel("5 次谐波加权增益总和", fontsize=11)
    ax1.set_title("(a) 各目标频率的滤波器组加权增益分解\n"
                  "三角 = CCA 增益 (仅陷波), 堆叠 = FBCCA 各子带贡献",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(alpha=0.15, axis="y")

    # Panel 2: Gain ratio FBCCA/CCA — shows frequency-dependent bias
    total_fb = np.sum(FB_WEIGHTS[:, None] * gain_matrix.T, axis=0)
    ratio = total_fb / (gain_cca + 1e-10)

    ax2.bar(range(40), ratio, color=[SUBBAND_COLORS[0] if r > np.median(ratio)
            else SUBBAND_COLORS[4] for r in ratio], alpha=0.7)
    ax2.axhline(np.median(ratio), color="red", linestyle="--", linewidth=1.5,
                label=f"中位数 = {np.median(ratio):.3f}")
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)

    ax2.set_xticks(range(0, 40, 4))
    ax2.set_xticklabels([f"{target_freqs[i]:.1f}" for i in range(0, 40, 4)],
                        fontsize=9, rotation=45)
    ax2.set_xlabel("目标频率 (Hz)", fontsize=11)
    ax2.set_ylabel("FBCCA / CCA 增益比", fontsize=11)
    ax2.set_title("(b) 滤波器组引入的频率依赖增益偏置\n"
                  "比值 > 1: FB 比 CCA 放大了该频率; < 1: FB 衰减",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.15, axis="y")

    fig.suptitle("滤波器组对不同目标频率的差异化影响\n"
                 "FB 不是中性预处理 — 它改变了各频率的相对增益, 即改变了信道条件",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = fig_main()
    fig.savefig(FIG_DIR / "fig_filterbank_response.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> fig_filterbank_response.png")

    fig = fig_fb_impact_on_target()
    fig.savefig(FIG_DIR / "fig_filterbank_harmonic_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> fig_filterbank_harmonic_coverage.png")

    fig = fig_subband_selectivity()
    fig.savefig(FIG_DIR / "fig_filterbank_selectivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> fig_filterbank_selectivity.png")

    # Print summary table
    print("\n=== Filter Bank Parameter Summary ===")
    print(f"Sampling rate: {FS} Hz, Nyquist: {NYQUIST} Hz")
    print(f"Notch: IIR comb, 50 Hz, Q=35")
    print(f"FB weights: {[f'{w:.4f}' for w in FB_WEIGHTS]}")
    print()
    for k in range(5):
        b, a, order = design_subband(k)
        wp_lo = 8 * (k + 1)
        ws_lo = wp_lo - 2
        print(f"FB{k}: pass {wp_lo}-90 Hz, stop {ws_lo}/100 Hz, "
              f"Cheby I order {order}, "
              f"ripple 0.5 dB, atten 40 dB")


if __name__ == "__main__":
    main()
