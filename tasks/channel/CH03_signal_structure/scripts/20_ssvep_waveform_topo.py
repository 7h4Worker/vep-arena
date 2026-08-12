"""Classic SSVEP time-domain & topographic visualizations.

Generates:
  A. Trial-averaged waveform at Oz (clear periodic oscillation)
  B. Multi-channel butterfly plot
  C. Single-trial ERP image (trials × time heatmap)
  D. Topographic map of SSVEP amplitude (66-ch)
  E. PSD with harmonic peaks

Usage
-----
    .venv/Scripts/python.exe scripts/20_ssvep_waveform_topo.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
from scipy import signal as sig

TASK_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = TASK_DIR / "results" / "figures"
CACHE_DIR = Path("D:/ProjData/datasets/ssvep_hd_200target/derivatives/tdca_sample/cache")

FS = 250
LATENCY_SAMPLES = 35
N_BLOCKS = 18

BASE_FREQS = np.asarray(
    list(range(8, 16))
    + [x + 0.2 for x in range(8, 16)]
    + [x + 0.4 for x in range(8, 16)]
    + [x + 0.6 for x in range(8, 16)]
    + [x + 0.8 for x in range(8, 16)],
    dtype=np.float64,
)

# Channel names for key indices (0-based, from plot_sample_ssvep_grid.py)
CH_NAMES = {48: "Pz", 54: "PO3", 55: "PO5", 56: "PO4", 57: "PO6",
            58: "POz", 61: "O1", 62: "Oz", 63: "O2"}
OZ_IDX = 62
O1_IDX = 61
O2_IDX = 63
POZ_IDX = 58

OCCIPITAL_9 = [48, 54, 55, 56, 57, 58, 61, 62, 63]


def standard_66ch_positions():
    pos = []
    rows = [
        (0.85,  [-0.25, 0.00, 0.25]),
        (0.72,  [-0.30, 0.30]),
        (0.55,  [-0.78, -0.55, -0.35, -0.17, 0.00, 0.17, 0.35, 0.55, 0.78]),
        (0.35,  [-0.82, -0.60, -0.38, -0.18, 0.00, 0.18, 0.38, 0.60, 0.82]),
        (0.12,  [-0.85, -0.62, -0.40, -0.19, 0.00, 0.19, 0.40, 0.62, 0.85]),
        (-0.02, [-0.92]),
        (-0.12, [-0.82, -0.60, -0.38, -0.18, 0.00, 0.18, 0.38, 0.60, 0.82]),
        (-0.02, [0.92]),
        (-0.35, [-0.78, -0.55, -0.35, -0.17, 0.00, 0.17, 0.35, 0.55, 0.78]),
        (-0.55, [-0.65, -0.42, -0.22, 0.00, 0.22, 0.42, 0.65]),
        (-0.72, [-0.50]),
        (-0.72, [-0.22, 0.00, 0.22]),
        (-0.72, [0.50]),
        (-0.82, [-0.35, 0.35]),
    ]
    for y, xs in rows:
        for x in xs:
            pos.append((x, y))
    return np.array(pos[:66])


def load_fb0(subj):
    p = CACHE_DIR / f"filtered_{subj}_200target_66ch_18blocks_float32.npy"
    return np.asarray(np.load(str(p), mmap_mode="r")[:, :, :, :, 0])


def spectral_power_at_freq(segment, freq, n_harmonics=3):
    """Compute spectral power at freq and its harmonics using sin/cos correlation."""
    T = segment.shape[-1]
    t = np.arange(T) / FS
    power = 0.0
    for h in range(1, n_harmonics + 1):
        s = np.sin(2 * np.pi * h * freq * t)
        c = np.cos(2 * np.pi * h * freq * t)
        power += (segment @ s) ** 2 + (segment @ c) ** 2
    return power


def draw_head(ax):
    head = Circle((0, 0), 0.95, fill=False, linewidth=1.5, color="#333")
    ax.add_patch(head)
    ax.plot([-0.08, 0, 0.08], [0.93, 1.05, 0.93], color="#333", lw=1.5)
    ax.plot([-0.97, -1.05, -0.97], [0.12, 0, -0.12], color="#333", lw=1)
    ax.plot([0.97, 1.05, 0.97], [0.12, 0, -0.12], color="#333", lw=1)


def make_figure(data, subj, freq_idx, win_samp):
    freq_hz = BASE_FREQS[freq_idx]
    target_idx = freq_idx * 5 + 4  # CENTER position (most "standard" SSVEP)

    # Extract all trials for this target
    trials_raw = data[:, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, target_idx, :]
    # trials_raw shape: (66, win_samp, 18)

    # Trial-averaged signal
    avg_signal = trials_raw.mean(axis=2).astype(np.float64)  # (66, win_samp)

    t_ms = np.arange(win_samp) / FS * 1000  # time in ms

    # ===== Figure: 5-panel comprehensive view =====
    fig = plt.figure(figsize=(22, 16))

    # ----- Panel A: Oz averaged waveform + individual trials -----
    ax_a = fig.add_axes([0.05, 0.72, 0.42, 0.22])

    ch = OZ_IDX
    ch_name = CH_NAMES.get(ch, f"ch{ch}")

    # Individual trials (gray)
    for blk in range(N_BLOCKS):
        trial = trials_raw[ch, :, blk].astype(np.float64)
        trial -= trial.mean()
        ax_a.plot(t_ms, trial, color="#b0b0b0", alpha=0.35, linewidth=0.6,
                  label="单试次" if blk == 0 else None)

    # Average (bold)
    avg_ch = avg_signal[ch] - avg_signal[ch].mean()
    ax_a.plot(t_ms, avg_ch, color="#174a7c", linewidth=2.5, label=f"18 trial 均值")

    # Reference sine at target frequency
    ref_amp = np.max(np.abs(avg_ch)) * 0.6
    ref_sin = ref_amp * np.sin(2 * np.pi * freq_hz * np.arange(win_samp) / FS)
    ax_a.plot(t_ms, ref_sin, color="#e63946", linewidth=1.2, linestyle="--",
              alpha=0.6, label=f"{freq_hz:.1f} Hz 参考正弦")

    ax_a.set_xlabel("时间 (ms)", fontsize=11)
    ax_a.set_ylabel("幅度 (μV)", fontsize=11)
    ax_a.set_title(f"(a) {ch_name} 通道波形 — {freq_hz:.1f} Hz SSVEP\n"
                   f"灰色 = 18 个单试次, 蓝色 = 均值, 红色虚线 = {freq_hz:.1f} Hz 参考",
                   fontsize=13, fontweight="bold")
    ax_a.legend(fontsize=10, loc="upper right")
    ax_a.grid(alpha=0.15)
    ax_a.set_xlim(0, t_ms[-1])

    # ----- Panel B: Multi-channel butterfly -----
    ax_b = fig.add_axes([0.55, 0.72, 0.42, 0.22])

    cmap_ch = plt.cm.viridis
    for ch_i in range(66):
        s = avg_signal[ch_i] - avg_signal[ch_i].mean()
        color = cmap_ch(ch_i / 66)
        alpha = 0.6 if ch_i in OCCIPITAL_9 else 0.15
        lw = 1.5 if ch_i in OCCIPITAL_9 else 0.4
        ax_b.plot(t_ms, s, color=color, alpha=alpha, linewidth=lw)

    # Highlight Oz
    s_oz = avg_signal[OZ_IDX] - avg_signal[OZ_IDX].mean()
    ax_b.plot(t_ms, s_oz, color="#e63946", linewidth=2.5, label="Oz")

    ax_b.set_xlabel("时间 (ms)", fontsize=11)
    ax_b.set_ylabel("幅度 (μV)", fontsize=11)
    ax_b.set_title(f"(b) 66 通道蝶形图 (trial-averaged)\n"
                   f"高亮 = 枕区 9ch, 红色 = Oz",
                   fontsize=13, fontweight="bold")
    ax_b.legend(fontsize=10, loc="upper right")
    ax_b.grid(alpha=0.15)
    ax_b.set_xlim(0, t_ms[-1])

    # ----- Panel C: Single-trial ERP image (Oz) -----
    ax_c = fig.add_axes([0.05, 0.40, 0.42, 0.24])

    erp_image = np.zeros((N_BLOCKS, win_samp))
    for blk in range(N_BLOCKS):
        trial = trials_raw[OZ_IDX, :, blk].astype(np.float64)
        trial -= trial.mean()
        erp_image[blk] = trial

    vmax = np.percentile(np.abs(erp_image), 95)
    im = ax_c.imshow(erp_image, aspect="auto", cmap="RdBu_r",
                     extent=[0, t_ms[-1], N_BLOCKS + 0.5, 0.5],
                     vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax_c, label="幅度 (μV)", shrink=0.8)

    ax_c.set_xlabel("时间 (ms)", fontsize=11)
    ax_c.set_ylabel("试次 (block)", fontsize=11)
    ax_c.set_title(f"(c) Oz 单试次 ERP 图像 (试次 × 时间)\n"
                   f"颜色 = 幅度, 一致性条纹 = SSVEP 稳定性",
                   fontsize=13, fontweight="bold")

    # ----- Panel D: Topographic map of SSVEP amplitude -----
    ax_d = fig.add_axes([0.55, 0.40, 0.22, 0.24])

    positions = standard_66ch_positions()
    ssvep_amp = np.zeros(66)
    for ch_i in range(66):
        seg = avg_signal[ch_i] - avg_signal[ch_i].mean()
        ssvep_amp[ch_i] = np.sqrt(spectral_power_at_freq(seg, freq_hz, n_harmonics=3))

    ssvep_amp_norm = ssvep_amp / (ssvep_amp.max() + 1e-30)

    draw_head(ax_d)

    from scipy.interpolate import griddata
    xi = np.linspace(-1.0, 1.0, 100)
    yi = np.linspace(-1.0, 1.0, 100)
    Xi, Yi = np.meshgrid(xi, yi)
    mask = Xi**2 + Yi**2 <= 0.92**2

    Zi = griddata(positions, ssvep_amp_norm, (Xi, Yi), method="cubic", fill_value=0)
    Zi[~mask] = np.nan

    ax_d.contourf(Xi, Yi, Zi, levels=20, cmap="YlOrRd", alpha=0.7)
    ax_d.scatter(positions[:, 0], positions[:, 1], c=ssvep_amp_norm,
                 cmap="YlOrRd", s=30, edgecolors="black", linewidth=0.5, zorder=5)

    # Label key channels
    for idx, name in CH_NAMES.items():
        if idx < 66:
            x, y = positions[idx]
            ax_d.text(x, y - 0.07, name, fontsize=7, ha="center", va="top", color="0.2")

    ax_d.set_xlim(-1.15, 1.15)
    ax_d.set_ylim(-1.1, 1.15)
    ax_d.set_aspect("equal")
    ax_d.axis("off")
    ax_d.set_title(f"(d) SSVEP 幅度地形图\n{freq_hz:.1f} Hz (3 次谐波能量)",
                   fontsize=13, fontweight="bold")

    # ----- Panel E: PSD with harmonic peaks -----
    ax_e = fig.add_axes([0.82, 0.40, 0.15, 0.24])

    # PSD of Oz averaged signal
    f_psd, psd = sig.welch(avg_signal[OZ_IDX], fs=FS, nperseg=min(win_samp, 256),
                           noverlap=min(win_samp // 2, 128))

    ax_e.semilogy(f_psd, psd, "steelblue", linewidth=1.5)

    # Mark harmonics
    for h in range(1, 6):
        fh = freq_hz * h
        if fh < f_psd[-1]:
            idx_h = np.argmin(np.abs(f_psd - fh))
            ax_e.plot(fh, psd[idx_h], "rv", markersize=8, zorder=5)
            ax_e.text(fh + 1, psd[idx_h], f"{h}f₀", fontsize=8, color="red")

    ax_e.set_xlabel("频率 (Hz)", fontsize=10)
    ax_e.set_ylabel("PSD", fontsize=10)
    ax_e.set_title("(e) Oz PSD\n(谐波标记)", fontsize=12, fontweight="bold")
    ax_e.grid(alpha=0.15)
    ax_e.set_xlim(0, 80)

    # ----- Panel F: 3-channel comparison (O1, Oz, O2) -----
    ax_f = fig.add_axes([0.05, 0.06, 0.42, 0.26])

    ch_list = [O1_IDX, OZ_IDX, O2_IDX, POZ_IDX]
    ch_colors = ["#2a9d8f", "#e63946", "#1d3557", "#e9c46a"]

    for ch_i, color in zip(ch_list, ch_colors):
        s = avg_signal[ch_i] - avg_signal[ch_i].mean()
        name = CH_NAMES.get(ch_i, f"ch{ch_i}")
        ax_f.plot(t_ms, s, color=color, linewidth=2, label=name, alpha=0.85)

    ax_f.set_xlabel("时间 (ms)", fontsize=11)
    ax_f.set_ylabel("幅度 (μV)", fontsize=11)
    ax_f.set_title(f"(f) 枕区 4 通道 trial-averaged 波形对比\n"
                   f"O1, Oz, O2, POz — {freq_hz:.1f} Hz",
                   fontsize=13, fontweight="bold")
    ax_f.legend(fontsize=10, loc="upper right", ncol=2)
    ax_f.grid(alpha=0.15)
    ax_f.set_xlim(0, t_ms[-1])

    # ----- Panel G: Multi-frequency Oz comparison -----
    ax_g = fig.add_axes([0.55, 0.06, 0.42, 0.26])

    freq_indices = [0, 2, 4, 6]  # 8.0, 10.0, 12.0, 14.0 Hz
    freq_colors = ["#e63946", "#2a9d8f", "#1d3557", "#e9c46a"]

    for fi, color in zip(freq_indices, freq_colors):
        fhz = BASE_FREQS[fi]
        tidx = fi * 5 + 4  # CENTER position
        avg_f = data[OZ_IDX, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, tidx, :].mean(axis=-1).astype(np.float64)
        avg_f -= avg_f.mean()
        # Show only first 100ms to see frequency difference
        n_show = min(int(0.15 * FS), win_samp)
        t_show = np.arange(n_show) / FS * 1000
        ax_g.plot(t_show, avg_f[:n_show], color=color, linewidth=2,
                  label=f"{fhz:.1f} Hz", alpha=0.85)

    ax_g.set_xlabel("时间 (ms)", fontsize=11)
    ax_g.set_ylabel("幅度 (μV)", fontsize=11)
    ax_g.set_title(f"(g) Oz 不同频率 SSVEP 波形对比 (前 150ms)\n"
                   f"清晰可见不同周期的稳态响应",
                   fontsize=13, fontweight="bold")
    ax_g.legend(fontsize=10, loc="upper right")
    ax_g.grid(alpha=0.15)
    ax_g.set_xlim(0, t_show[-1])

    fig.suptitle(f"{subj} — Target {target_idx} ({freq_hz:.1f} Hz, CENTER) — "
                 f"SSVEP 时域特征综合可视化",
                 fontsize=16, fontweight="bold", y=0.98)

    return fig


def main():
    t0 = time.time()
    win_samp = 150  # 600ms for better frequency resolution

    for subj in ["S1", "S9"]:
        print(f"Loading {subj}...")
        data = load_fb0(subj)

        for freq_idx in [0, 4]:  # 8.0 Hz, 12.0 Hz
            fig = make_figure(data, subj, freq_idx, win_samp)
            freq_hz = BASE_FREQS[freq_idx]
            fname = f"fig_waveform_topo_{subj}_{freq_hz:.0f}Hz.png"
            fig.savefig(FIG_DIR / fname, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"  -> {fname}")

        del data

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
