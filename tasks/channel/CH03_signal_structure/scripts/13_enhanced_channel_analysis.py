"""Enhanced SSVEP channel matrix analysis — multi-window, raw exhibits, TRCA.

Extends 12_analyze_channel_matrix.py with:
  A. Raw signal waveform & Pearson correlation step-by-step exhibits
  B. Multi-window constellation evolution (200, 300, 500, 700 ms)
  C. Channel × target-count layered comparison
  D. TRCA vs TDCA decoder comparison from grid results

Usage
-----
    .venv/Scripts/python.exe scripts/13_enhanced_channel_analysis.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svdvals
import pandas as pd

TASK_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = TASK_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HD_ROOT = Path("D:/ProjData/datasets/ssvep_hd_200target")
CACHE_DIR = HD_ROOT / "derivatives" / "tdca_sample" / "cache"
RESULTS_DIR = Path("D:/ProjData/proj_python/vep_arena/tasks/baselines/BL05_ssvep_hd_200t/results")

FS = 250
LATENCY_SAMPLES = 35
N_HARMONICS = 5
N_BASE_FREQS = 40

BASE_FREQS = np.asarray(
    list(range(8, 16))
    + [x + 0.2 for x in range(8, 16)]
    + [x + 0.4 for x in range(8, 16)]
    + [x + 0.6 for x in range(8, 16)]
    + [x + 0.8 for x in range(8, 16)],
    dtype=np.float64,
)

CHANNEL_SETS = {
    "1ch_Oz": np.array([29]),
    "9ch": np.array([21, 27, 28, 29, 31, 32, 33, 59, 63]),
    "32ch": np.array(list(range(17, 36)) + list(range(53, 66))),
    "66ch": np.arange(66),
}

TARGET_TO_BASE_FREQ = np.arange(200) // 5


def make_references(n_samples):
    t = np.arange(n_samples) / FS
    refs = np.zeros((N_BASE_FREQS, 2 * N_HARMONICS, n_samples))
    for fi, freq in enumerate(BASE_FREQS):
        for h in range(N_HARMONICS):
            refs[fi, 2 * h] = np.sin(2 * np.pi * (h + 1) * freq * t)
            refs[fi, 2 * h + 1] = np.cos(2 * np.pi * (h + 1) * freq * t)
    refs -= refs.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(refs, axis=-1, keepdims=True)
    refs /= np.clip(norms, 1e-12, None)
    return refs


def build_channel_matrix(data, channel_idx, refs, window_samples):
    n_ch = len(channel_idx)
    ch_data = data[channel_idx, LATENCY_SAMPLES:LATENCY_SAMPLES + window_samples]

    H_sum = np.zeros((N_BASE_FREQS, N_BASE_FREQS))
    H_count = np.zeros(N_BASE_FREQS)
    trial_vectors = {i: [] for i in range(N_BASE_FREQS)}

    for target_idx in range(200):
        base_fi = TARGET_TO_BASE_FREQ[target_idx]
        seg = ch_data[:, :, target_idx, :].transpose(2, 0, 1)
        seg = seg - seg.mean(axis=-1, keepdims=True)
        seg_norms = np.linalg.norm(seg, axis=-1, keepdims=True)
        seg = seg / np.clip(seg_norms, 1e-12, None)
        corr = np.einsum("bcs,fks->bcfk", seg, refs)
        energy = (corr ** 2).sum(axis=-1).mean(axis=1)
        for block in range(energy.shape[0]):
            vec = energy[block]
            H_sum[base_fi] += vec
            H_count[base_fi] += 1
            trial_vectors[base_fi].append(vec.copy())

    H = H_sum / np.clip(H_count, 1, None)[:, None]
    return H, trial_vectors


def effective_rank(H, threshold=0.95):
    sv = svdvals(H)
    total = sv.sum()
    if total < 1e-12:
        return 1
    cumsum = np.cumsum(sv / total)
    return int(np.searchsorted(cumsum, threshold) + 1)


def mutual_information_gaussian(H, trial_vectors):
    K = H.shape[0]
    all_vecs = []
    labels = []
    for i in range(K):
        for v in trial_vectors[i]:
            all_vecs.append(v)
            labels.append(i)
    if len(all_vecs) < K + 1:
        return 0.0
    all_vecs = np.array(all_vecs)
    labels = np.array(labels)
    noise_covs = []
    for i in range(K):
        vecs_i = all_vecs[labels == i]
        if len(vecs_i) > 1:
            centered = vecs_i - H[i]
            noise_covs.append(centered.T @ centered / len(vecs_i))
    if not noise_covs:
        return 0.0
    Sigma_noise = np.mean(noise_covs, axis=0)
    reg = 1e-8 * np.trace(Sigma_noise) / max(K, 1)
    Sigma_noise += np.eye(K) * reg
    grand_mean = H.mean(axis=0)
    Sigma_signal = (H - grand_mean).T @ (H - grand_mean) / K
    Sigma_total = Sigma_signal + Sigma_noise
    sign_t, logdet_t = np.linalg.slogdet(Sigma_total)
    sign_n, logdet_n = np.linalg.slogdet(Sigma_noise)
    if sign_t <= 0 or sign_n <= 0:
        return 0.0
    mi = 0.5 * (logdet_t - logdet_n) / np.log(2)
    return max(0.0, mi)


def _save(fig, path):
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Part A: Raw signal exhibits & Pearson correlation step-by-step
# ═══════════════════════════════════════════════════════════════════════════

def part_a_raw_exhibits(data, subj_name):
    """Show raw EEG waveforms and step-by-step Pearson correlation for one trial."""
    print(f"\n  [A] Raw signal exhibits for {subj_name}...")

    target_idx = 0  # freq index 0 → 8.0 Hz
    base_fi = 0
    freq = BASE_FREQS[base_fi]
    ch_oz = 29
    block = 0

    win_samples = 125  # 500ms
    seg_raw = data[ch_oz, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samples, target_idx, block]
    t_ms = np.arange(win_samples) / FS * 1000

    # Also get a non-target trial for comparison (target at 9.0 Hz = index 5)
    target_nontarget = 5
    seg_nontarget = data[ch_oz, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samples, target_nontarget, block]

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True)

    # Row 0: Raw EEG waveforms
    ax = axes[0, 0]
    ax.plot(t_ms, seg_raw, "b-", linewidth=0.8, label=f"Trial @ {freq:.1f} Hz (target)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"Raw EEG — {subj_name}, Oz, Block 0\nTarget: {freq:.1f} Hz")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    ax = axes[0, 1]
    nontarget_freq = BASE_FREQS[TARGET_TO_BASE_FREQ[target_nontarget]]
    ax.plot(t_ms, seg_nontarget, "r-", linewidth=0.8,
            label=f"Trial @ {nontarget_freq:.1f} Hz")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"Raw EEG — non-target comparison\n{nontarget_freq:.1f} Hz trial")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    # Row 1: Signal vs reference sin/cos at target frequency
    t_s = np.arange(win_samples) / FS
    ref_sin = np.sin(2 * np.pi * freq * t_s)
    ref_cos = np.cos(2 * np.pi * freq * t_s)

    seg_norm = seg_raw - seg_raw.mean()
    seg_norm = seg_norm / (np.linalg.norm(seg_norm) + 1e-12)
    ref_sin_norm = ref_sin - ref_sin.mean()
    ref_sin_norm = ref_sin_norm / (np.linalg.norm(ref_sin_norm) + 1e-12)
    ref_cos_norm = ref_cos - ref_cos.mean()
    ref_cos_norm = ref_cos_norm / (np.linalg.norm(ref_cos_norm) + 1e-12)

    r_sin = np.dot(seg_norm, ref_sin_norm)
    r_cos = np.dot(seg_norm, ref_cos_norm)

    ax = axes[1, 0]
    ax.plot(t_ms, seg_norm, "b-", linewidth=0.8, alpha=0.7, label="EEG (normalized)")
    ax.plot(t_ms, ref_sin_norm, "g--", linewidth=1.2, alpha=0.8,
            label=f"sin({freq:.0f}Hz), r={r_sin:.3f}")
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Pearson correlation: EEG vs sin(2π·{freq:.0f}·t)\nr = {r_sin:.4f}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.15)

    ax = axes[1, 1]
    ax.plot(t_ms, seg_norm, "b-", linewidth=0.8, alpha=0.7, label="EEG (normalized)")
    ax.plot(t_ms, ref_cos_norm, "m--", linewidth=1.2, alpha=0.8,
            label=f"cos({freq:.0f}Hz), r={r_cos:.3f}")
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Pearson correlation: EEG vs cos(2π·{freq:.0f}·t)\nr = {r_cos:.4f}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.15)

    # Row 2: Full harmonic correlation breakdown
    ax = axes[2, 0]
    harmonics = range(1, N_HARMONICS + 1)
    r_vals_sin = []
    r_vals_cos = []
    for h in harmonics:
        ref_h_sin = np.sin(2 * np.pi * h * freq * t_s)
        ref_h_cos = np.cos(2 * np.pi * h * freq * t_s)
        ref_h_sin = ref_h_sin - ref_h_sin.mean()
        ref_h_sin = ref_h_sin / (np.linalg.norm(ref_h_sin) + 1e-12)
        ref_h_cos = ref_h_cos - ref_h_cos.mean()
        ref_h_cos = ref_h_cos / (np.linalg.norm(ref_h_cos) + 1e-12)
        r_vals_sin.append(np.dot(seg_norm, ref_h_sin))
        r_vals_cos.append(np.dot(seg_norm, ref_h_cos))

    x_h = np.arange(len(harmonics))
    w = 0.35
    ax.bar(x_h - w/2, [abs(r) for r in r_vals_sin], w, color="#3b82f6", alpha=0.7, label="|r_sin|")
    ax.bar(x_h + w/2, [abs(r) for r in r_vals_cos], w, color="#f59e0b", alpha=0.7, label="|r_cos|")
    ax.set_xticks(x_h)
    ax.set_xticklabels([f"h={h}" for h in harmonics])
    ax.set_ylabel("|Pearson r|")
    ax.set_title(f"Harmonic correlation breakdown — target trial\nE = Σ(r²_sin + r²_cos) = "
                 f"{sum(r**2 for r in r_vals_sin) + sum(r**2 for r in r_vals_cos):.4f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, axis="y")

    # Same for non-target
    ax = axes[2, 1]
    seg_nt_norm = seg_nontarget - seg_nontarget.mean()
    seg_nt_norm = seg_nt_norm / (np.linalg.norm(seg_nt_norm) + 1e-12)
    r_nt_sin = []
    r_nt_cos = []
    for h in harmonics:
        ref_h_sin = np.sin(2 * np.pi * h * freq * t_s)
        ref_h_cos = np.cos(2 * np.pi * h * freq * t_s)
        ref_h_sin = ref_h_sin - ref_h_sin.mean()
        ref_h_sin = ref_h_sin / (np.linalg.norm(ref_h_sin) + 1e-12)
        ref_h_cos = ref_h_cos - ref_h_cos.mean()
        ref_h_cos = ref_h_cos / (np.linalg.norm(ref_h_cos) + 1e-12)
        r_nt_sin.append(np.dot(seg_nt_norm, ref_h_sin))
        r_nt_cos.append(np.dot(seg_nt_norm, ref_h_cos))

    ax.bar(x_h - w/2, [abs(r) for r in r_nt_sin], w, color="#3b82f6", alpha=0.7, label="|r_sin|")
    ax.bar(x_h + w/2, [abs(r) for r in r_nt_cos], w, color="#f59e0b", alpha=0.7, label="|r_cos|")
    ax.set_xticks(x_h)
    ax.set_xticklabels([f"h={h}" for h in harmonics])
    ax.set_ylabel("|Pearson r|")
    ax.set_title(f"Harmonic correlation — NON-target ({nontarget_freq:.1f} Hz trial) at {freq:.0f} Hz ref\n"
                 f"E = {sum(r**2 for r in r_nt_sin) + sum(r**2 for r in r_nt_cos):.4f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, axis="y")

    # Row 3: Full 40-frequency energy profile for one trial
    refs_full = make_references(win_samples)
    ax = axes[3, 0]
    energy_profile = np.zeros(N_BASE_FREQS)
    for fj in range(N_BASE_FREQS):
        e = 0.0
        for k in range(2 * N_HARMONICS):
            r = np.dot(seg_norm, refs_full[fj, k])
            e += r ** 2
        energy_profile[fj] = e

    colors_bar = ["#dc2626" if i == base_fi else "#94a3b8" for i in range(N_BASE_FREQS)]
    ax.bar(range(N_BASE_FREQS), energy_profile, color=colors_bar, alpha=0.7)
    ax.set_xlabel("Base frequency index")
    ax.set_ylabel("Spectral energy E(x, f_j)")
    ax.set_title(f"Energy profile of one trial — target at f₀={freq:.1f} Hz\n"
                 f"(red = target freq, peak should be at index {base_fi})")
    ax.grid(alpha=0.15, axis="y")

    # Non-target energy profile
    ax = axes[3, 1]
    energy_nt = np.zeros(N_BASE_FREQS)
    for fj in range(N_BASE_FREQS):
        e = 0.0
        for k in range(2 * N_HARMONICS):
            r = np.dot(seg_nt_norm, refs_full[fj, k])
            e += r ** 2
        energy_nt[fj] = e

    nt_base_fi = TARGET_TO_BASE_FREQ[target_nontarget]
    colors_nt = ["#dc2626" if i == nt_base_fi else "#94a3b8" for i in range(N_BASE_FREQS)]
    ax.bar(range(N_BASE_FREQS), energy_nt, color=colors_nt, alpha=0.7)
    ax.set_xlabel("Base frequency index")
    ax.set_ylabel("Spectral energy E(x, f_j)")
    ax.set_title(f"Energy profile — target at f₀={nontarget_freq:.1f} Hz\n"
                 f"(red = target freq, peak should be at index {nt_base_fi})")
    ax.grid(alpha=0.15, axis="y")

    fig.suptitle(f"Part A: Raw Signal & Pearson Correlation Exhibits — {subj_name}, Oz channel, 500ms",
                 fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_raw_signal_exhibits.png")

    # Print numerical summary
    E_target = sum(r**2 for r in r_vals_sin) + sum(r**2 for r in r_vals_cos)
    E_nontarget = sum(r**2 for r in r_nt_sin) + sum(r**2 for r in r_nt_cos)
    print(f"    Target trial energy at target freq: {E_target:.4f}")
    print(f"    Non-target trial energy at target freq: {E_nontarget:.4f}")
    print(f"    Ratio: {E_target / max(E_nontarget, 1e-8):.1f}x")


# ═══════════════════════════════════════════════════════════════════════════
# Part B: Multi-window constellation evolution
# ═══════════════════════════════════════════════════════════════════════════

def part_b_multiwindow(data_dict, subjects):
    """Compute channel matrices at multiple windows, show constellation evolution."""
    print("\n  [B] Multi-window constellation evolution...")

    windows_ms = [200, 300, 500, 700]
    ch_name = "9ch"
    ch_idx = CHANNEL_SETS[ch_name]

    window_results = {}

    for win_ms in windows_ms:
        win_samp = int(win_ms * FS / 1000)
        usable = 185 - LATENCY_SAMPLES
        if win_samp > usable:
            win_samp = usable
        refs = make_references(win_samp)

        for subj in subjects:
            data = data_dict[subj]
            t0 = time.time()
            H, tvecs = build_channel_matrix(data, ch_idx, refs, win_samp)
            mi = mutual_information_gaussian(H, tvecs)
            er = effective_rank(H)
            diag_mean = np.mean(np.diag(H))
            offdiag = H[~np.eye(N_BASE_FREQS, dtype=bool)]
            sir = 10 * np.log10(diag_mean / max(np.mean(offdiag), 1e-12))
            window_results[(subj, win_ms)] = {
                "H": H, "trial_vecs": tvecs, "mi": mi, "erank": er, "sir": sir,
            }
            print(f"    {subj}/{win_ms}ms: erank={er:2d}  MI={mi:.2f}  SIR={sir:.1f} dB  ({time.time()-t0:.1f}s)")

    # Fig B1: Constellation evolution across windows (2 subjects × 4 windows)
    plot_subjects = ["S1", "S11"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)

    for row, subj in enumerate(plot_subjects):
        # Compute shared PCA basis from 500ms H
        H_ref = window_results[(subj, 500)]["H"]
        H_ref_c = H_ref - H_ref.mean(axis=0)
        _, _, Vt_ref = np.linalg.svd(H_ref_c, full_matrices=False)
        proj_mat = Vt_ref[:2].T

        for col, win_ms in enumerate(windows_ms):
            ax = axes[row, col]
            r = window_results[(subj, win_ms)]
            H = r["H"]
            mean_of_H = H.mean(axis=0)
            H_c = H - mean_of_H

            coords = H_c @ proj_mat
            ax.scatter(coords[:, 0], coords[:, 1],
                       c=BASE_FREQS, cmap="rainbow", s=50,
                       edgecolors="black", linewidths=0.5, zorder=5)

            # Noise clouds (subsample for speed)
            for fi in range(0, N_BASE_FREQS, 4):
                vecs = np.array(r["trial_vecs"][fi])
                if len(vecs) == 0:
                    continue
                tc = (vecs - mean_of_H) @ proj_mat
                color = plt.cm.rainbow(fi / 39.0)
                ax.scatter(tc[:, 0], tc[:, 1], c=[color], s=2, alpha=0.06, rasterized=True)

            ax.set_title(f"{subj} / {win_ms}ms\nMI={r['mi']:.2f} bits, erank={r['erank']}")
            ax.set_xlabel("PC1")
            if col == 0:
                ax.set_ylabel("PC2")
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)

    fig.suptitle(f"Constellation evolution across time windows — {ch_name}\n"
                 f"(PCA basis fixed to 500ms; noise clouds shown for every 4th freq)",
                 fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_constellation_evolution_windows.png")

    # Fig B2: MI / SIR / erank vs window for all subjects
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    subj_colors = {"S1": "#3b82f6", "S3": "#10b981", "S9": "#f59e0b", "S11": "#dc2626"}

    for subj in subjects:
        mis = [window_results[(subj, w)]["mi"] for w in windows_ms]
        sirs = [window_results[(subj, w)]["sir"] for w in windows_ms]
        eranks = [window_results[(subj, w)]["erank"] for w in windows_ms]
        c = subj_colors.get(subj, "gray")

        axes[0].plot(windows_ms, mis, "o-", color=c, linewidth=2, markersize=7, label=subj)
        axes[1].plot(windows_ms, sirs, "o-", color=c, linewidth=2, markersize=7, label=subj)
        axes[2].plot(windows_ms, eranks, "o-", color=c, linewidth=2, markersize=7, label=subj)

    for ax, ylabel, title in zip(axes,
            ["MI (bits)", "SIR (dB)", "Effective rank (95%)"],
            ["Decoder-independent MI", "Signal-to-Interference Ratio", "Channel effective rank"]):
        ax.set_xlabel("Window (ms)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.15)
        ax.set_xticks(windows_ms)

    fig.suptitle(f"Channel metrics vs time window — {ch_name}", fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_metrics_vs_window.png")

    return window_results


# ═══════════════════════════════════════════════════════════════════════════
# Part C: Channel config layered comparison (same subject, multiple configs)
# ═══════════════════════════════════════════════════════════════════════════

def part_c_channel_layers(data_dict, subjects):
    """Show same subject's constellation for 1ch/9ch/32ch/66ch side by side."""
    print("\n  [C] Channel config layered analysis...")

    ch_configs = ["1ch_Oz", "9ch", "32ch", "66ch"]
    win_ms = 500
    win_samp = int(win_ms * FS / 1000)
    refs = make_references(win_samp)

    all_results = {}
    for subj in subjects:
        data = data_dict[subj]
        for ch_name in ch_configs:
            ch_idx = CHANNEL_SETS[ch_name]
            t0 = time.time()
            H, tvecs = build_channel_matrix(data, ch_idx, refs, win_samp)
            mi = mutual_information_gaussian(H, tvecs)
            er = effective_rank(H)
            diag_mean = np.mean(np.diag(H))
            offdiag = H[~np.eye(N_BASE_FREQS, dtype=bool)]
            sir = 10 * np.log10(diag_mean / max(np.mean(offdiag), 1e-12))
            all_results[(subj, ch_name)] = {
                "H": H, "trial_vecs": tvecs, "mi": mi, "erank": er, "sir": sir,
            }
            print(f"    {subj}/{ch_name}: MI={mi:.2f}  erank={er}  SIR={sir:.1f} dB  ({time.time()-t0:.1f}s)")

    # Fig C1: 4 subjects × 4 ch configs constellation matrix
    fig, axes = plt.subplots(len(subjects), len(ch_configs),
                             figsize=(5 * len(ch_configs), 5 * len(subjects)),
                             constrained_layout=True)

    for row, subj in enumerate(subjects):
        # Shared PCA basis from 9ch
        H_ref = all_results[(subj, "9ch")]["H"]
        H_ref_c = H_ref - H_ref.mean(axis=0)
        _, _, Vt_ref = np.linalg.svd(H_ref_c, full_matrices=False)
        proj_mat = Vt_ref[:2].T

        for col, ch_name in enumerate(ch_configs):
            ax = axes[row, col] if len(subjects) > 1 else axes[col]
            r = all_results[(subj, ch_name)]
            H = r["H"]
            mean_of_H = H.mean(axis=0)
            H_c = H - mean_of_H
            coords = H_c @ proj_mat

            # Noise clouds
            for fi in range(0, N_BASE_FREQS, 5):
                vecs = np.array(r["trial_vecs"][fi])
                if len(vecs) == 0:
                    continue
                tc = (vecs - mean_of_H) @ proj_mat
                color = plt.cm.rainbow(fi / 39.0)
                ax.scatter(tc[:, 0], tc[:, 1], c=[color], s=2, alpha=0.05, rasterized=True)

            ax.scatter(coords[:, 0], coords[:, 1],
                       c=BASE_FREQS, cmap="rainbow", s=40,
                       edgecolors="black", linewidths=0.5, zorder=5)

            ax.set_title(f"{subj} / {ch_name}\nMI={r['mi']:.2f}, erank={r['erank']}, SIR={r['sir']:.1f}dB",
                         fontsize=9)
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)
            if col == 0:
                ax.set_ylabel("PC2")
            if row == len(subjects) - 1:
                ax.set_xlabel("PC1")

    fig.suptitle("SSVEP Constellation: channel config comparison\n"
                 "(PCA basis from 9ch; noise clouds every 5th freq)", fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_constellation_channel_layers.png")

    # Fig C2: H matrix diagonal profiles — overlay 4 ch configs for 2 subjects
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    ch_colors = {"1ch_Oz": "#94a3b8", "9ch": "#3b82f6", "32ch": "#f59e0b", "66ch": "#dc2626"}

    for ax_i, subj in enumerate(["S1", "S11"]):
        ax = axes[ax_i]
        for ch_name in ch_configs:
            r = all_results[(subj, ch_name)]
            diag = np.diag(r["H"])
            ax.plot(range(N_BASE_FREQS), diag, "o-", color=ch_colors[ch_name],
                    linewidth=1.5, markersize=4, alpha=0.8, label=ch_name)
        ax.set_xlabel("Base frequency index")
        ax.set_ylabel("Diagonal energy H[i,i]")
        ax.set_title(f"{subj} — H diagonal (signal strength per freq)")
        ax.legend()
        ax.grid(alpha=0.15)

    fig.suptitle("Channel matrix diagonal: signal energy per frequency × channel config",
                 fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_H_diagonal_profiles.png")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Part D: TRCA vs TDCA decoder comparison
# ═══════════════════════════════════════════════════════════════════════════

def part_d_trca_comparison():
    """Load TRCA and TDCA grid results, compare across conditions."""
    print("\n  [D] TRCA vs TDCA decoder comparison...")

    trca_trials = pd.read_csv(RESULTS_DIR / "offline_trca_grid" / "trials.csv")
    tdca_trials = pd.read_csv(RESULTS_DIR / "offline_tdca_grid" / "trials.csv")

    # Compute best ITR per subject/target_set/channel_set (best over windows)
    def best_itr(df):
        return (df.groupby(["subject", "target_set", "channel_set"])
                .agg(best_itr=("itr_bpm", "max"),
                     best_acc=("accuracy", "max"),
                     best_win=("window_ms", lambda x: x.iloc[df.loc[x.index, "itr_bpm"].argmax()]))
                .reset_index())

    trca_best = best_itr(trca_trials)
    tdca_best = best_itr(tdca_trials)

    # Fig D1: TRCA vs TDCA ITR by target count (mean over subjects)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    target_sets = [40, 80, 120, 160, 200]
    ch_sets = [9, 21, 32, 66]
    method_colors = {"TRCA": "#f59e0b", "TDCA": "#3b82f6"}

    # Panel 1: ITR vs targets for 66ch
    ax = axes[0]
    for method_name, df_best, ls in [("TRCA", trca_best, "D--"), ("TDCA", tdca_best, "o-")]:
        sub = df_best[df_best["channel_set"] == 66]
        means = sub.groupby("target_set")["best_itr"].mean()
        sems = sub.groupby("target_set")["best_itr"].sem()
        ax.errorbar(means.index, means.values, yerr=sems.values,
                    fmt=ls, color=method_colors[method_name],
                    linewidth=2, markersize=7, capsize=3, label=method_name)
    ax.set_xlabel("Target count")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title("ITR vs target count — 66ch")
    ax.legend()
    ax.grid(alpha=0.15)
    ax.set_xticks(target_sets)

    # Panel 2: ITR vs channels for 200 targets
    ax = axes[1]
    for method_name, df_best, ls in [("TRCA", trca_best, "D--"), ("TDCA", tdca_best, "o-")]:
        sub = df_best[df_best["target_set"] == 200]
        means = sub.groupby("channel_set")["best_itr"].mean()
        sems = sub.groupby("channel_set")["best_itr"].sem()
        ax.errorbar(means.index, means.values, yerr=sems.values,
                    fmt=ls, color=method_colors[method_name],
                    linewidth=2, markersize=7, capsize=3, label=method_name)
    ax.set_xlabel("Channel count")
    ax.set_ylabel("ITR (bits/min)")
    ax.set_title("ITR vs channel count — 200 targets")
    ax.legend()
    ax.grid(alpha=0.15)
    ax.set_xticks(ch_sets)

    # Panel 3: TDCA gain over TRCA — scatter per subject/condition
    ax = axes[2]
    merged = trca_best.merge(tdca_best, on=["subject", "target_set", "channel_set"],
                             suffixes=("_trca", "_tdca"))
    ax.scatter(merged["best_itr_trca"], merged["best_itr_tdca"],
               c="#3b82f6", s=15, alpha=0.4, edgecolors="none")
    lims = [0, max(merged["best_itr_trca"].max(), merged["best_itr_tdca"].max()) * 1.05]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("TRCA ITR (bpm)")
    ax.set_ylabel("TDCA ITR (bpm)")
    ax.set_title(f"TDCA vs TRCA — all conditions\n(n={len(merged)} points)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)

    fig.suptitle("HD200 Decoder Comparison: TRCA vs TDCA", fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_trca_vs_tdca.png")

    # Fig D2: Per-subject TRCA vs TDCA at 200 targets / 66ch
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)

    for ax_i, (tset, title_suffix) in enumerate([(200, "200 targets"), (40, "40 targets")]):
        ax = axes[ax_i]
        for method_name, df_best, color in [("TRCA", trca_best, "#f59e0b"), ("TDCA", tdca_best, "#3b82f6")]:
            sub = df_best[(df_best["target_set"] == tset) & (df_best["channel_set"] == 66)]
            sub_sorted = sub.sort_values("subject")
            x_pos = np.arange(len(sub_sorted))
            ax.bar(x_pos + (0.2 if method_name == "TDCA" else -0.2),
                   sub_sorted["best_itr"].values, 0.35,
                   color=color, alpha=0.7, label=method_name)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sub_sorted["subject"].values, fontsize=7, rotation=45)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Best ITR (bpm)")
        ax.set_title(f"Per-subject ITR — {title_suffix}, 66ch")
        ax.legend()
        ax.grid(alpha=0.15, axis="y")

    fig.suptitle("Per-subject TRCA vs TDCA comparison (66ch, best window)",
                 fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_trca_vs_tdca_per_subject.png")

    # Fig D3: Optimal window comparison TRCA vs TDCA
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)

    for ax_i, (tset, title_suffix) in enumerate([(200, "200 targets"), (40, "40 targets")]):
        ax = axes[ax_i]
        for method_name, df_best, color, offset in [("TRCA", trca_best, "#f59e0b", -0.15),
                                                     ("TDCA", tdca_best, "#3b82f6", 0.15)]:
            sub = df_best[(df_best["target_set"] == tset) & (df_best["channel_set"] == 66)]
            sub_sorted = sub.sort_values("subject")
            x_pos = np.arange(len(sub_sorted))
            ax.bar(x_pos + offset, sub_sorted["best_win"].values, 0.28,
                   color=color, alpha=0.7, label=method_name)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sub_sorted["subject"].values, fontsize=7, rotation=45)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Optimal window (ms)")
        ax.set_title(f"Optimal window — {title_suffix}, 66ch")
        ax.legend()
        ax.grid(alpha=0.15, axis="y")

    fig.suptitle("Optimal window: TRCA vs TDCA (66ch)", fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_optimal_window_trca_tdca.png")

    # Print summary table
    print("\n  TRCA vs TDCA summary (66ch, mean±sem across subjects):")
    print(f"  {'Targets':>8s}  {'TRCA ITR':>12s}  {'TDCA ITR':>12s}  {'Gain':>8s}")
    for tset in target_sets:
        trca_sub = trca_best[(trca_best["target_set"] == tset) & (trca_best["channel_set"] == 66)]
        tdca_sub = tdca_best[(tdca_best["target_set"] == tset) & (tdca_best["channel_set"] == 66)]
        trca_mean = trca_sub["best_itr"].mean()
        tdca_mean = tdca_sub["best_itr"].mean()
        gain = (tdca_mean - trca_mean) / trca_mean * 100
        print(f"  {tset:>8d}  {trca_mean:>10.1f}±{trca_sub['best_itr'].sem():>3.1f}"
              f"  {tdca_mean:>10.1f}±{tdca_sub['best_itr'].sem():>3.1f}  {gain:>+6.1f}%")

    return trca_best, tdca_best


# ═══════════════════════════════════════════════════════════════════════════
# Part E: Correlation-based phase constellation (polar plot)
# ═══════════════════════════════════════════════════════════════════════════

def part_e_phase_constellation(data_dict, subjects):
    """Create phase-based constellation from sin/cos correlation coefficients."""
    print("\n  [E] Phase-based constellation (polar)...")

    ch_idx = CHANNEL_SETS["9ch"]
    win_samp = 125  # 500ms
    refs = make_references(win_samp)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw={"projection": "polar"}, constrained_layout=True)

    for ax_i, subj in enumerate(["S1", "S11"]):
        ax = axes[ax_i]
        data = data_dict[subj]

        for fi in range(N_BASE_FREQS):
            # Collect correlation with fundamental sin/cos across all trials of this freq
            target_indices = [t for t in range(200) if TARGET_TO_BASE_FREQ[t] == fi]
            r_sin_all = []
            r_cos_all = []

            for tidx in target_indices:
                for block in range(18):
                    seg = data[ch_idx][:, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, tidx, block]
                    seg_mean = seg.mean(axis=0)  # average across channels
                    seg_mean = seg_mean - seg_mean.mean()
                    seg_mean = seg_mean / (np.linalg.norm(seg_mean) + 1e-12)
                    # Fundamental harmonic only
                    r_sin = np.dot(seg_mean, refs[fi, 0])
                    r_cos = np.dot(seg_mean, refs[fi, 1])
                    r_sin_all.append(r_sin)
                    r_cos_all.append(r_cos)

            r_sin_mean = np.mean(r_sin_all)
            r_cos_mean = np.mean(r_cos_all)
            amplitude = np.sqrt(r_sin_mean**2 + r_cos_mean**2)
            phase = np.arctan2(r_cos_mean, r_sin_mean)

            color = plt.cm.rainbow(fi / 39.0)
            ax.scatter(phase, amplitude, c=[color], s=40, edgecolors="black",
                       linewidths=0.3, zorder=5)

            # Noise cloud (subset)
            for rs, rc in zip(r_sin_all[::10], r_cos_all[::10]):
                amp_trial = np.sqrt(rs**2 + rc**2)
                ph_trial = np.arctan2(rc, rs)
                ax.scatter(ph_trial, amp_trial, c=[color], s=2, alpha=0.08, rasterized=True)

        ax.set_title(f"{subj} — Phase constellation (fundamental)\n"
                     f"radius = |r|, angle = atan2(r_cos, r_sin)", fontsize=10)

    fig.suptitle("Phase-based SSVEP constellation (polar)\n"
                 "9ch average, 500ms, fundamental harmonic", fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_phase_constellation.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    sample_subjects = ["S1", "S3", "S9", "S11"]

    print("  Loading HD200 data...")
    data_dict = {}
    for subj in sample_subjects:
        cache_path = CACHE_DIR / f"filtered_{subj}_200target_66ch_18blocks_float32.npy"
        if not cache_path.exists():
            print(f"  SKIP {subj}: cache not found")
            continue
        t0 = time.time()
        data_full = np.load(str(cache_path), mmap_mode="r")
        data_dict[subj] = np.asarray(data_full[:, :, :, :, 0])
        print(f"    {subj}: {data_dict[subj].shape} ({time.time()-t0:.1f}s)")

    # Part A: Raw signal exhibits (use S1)
    if "S1" in data_dict:
        part_a_raw_exhibits(data_dict["S1"], "S1")

    # Part B: Multi-window
    part_b_multiwindow(data_dict, sample_subjects)

    # Part C: Channel config layers
    part_c_channel_layers(data_dict, sample_subjects)

    # Part D: TRCA vs TDCA
    part_d_trca_comparison()

    # Part E: Phase constellation
    part_e_phase_constellation(data_dict, ["S1", "S11"])

    print(f"\n  All figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
