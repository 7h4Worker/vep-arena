"""Spatial × frequency decomposition of HD200 SSVEP.

HD200 = 40 frequencies × 5 spatial positions.
Core question: how does HD-EEG enable spatial position discrimination
when all 5 positions share the SAME frequency?

Analysis:
  A. Channel-level energy topography: same freq, different positions
     → spatial pattern differences only visible with many channels
  B. Within-frequency spatial confusion: 5×5 similarity matrix per freq
  C. Frequency vs spatial discrimination decomposition
  D. Why 66ch helps at 200 targets but not 40 targets

Usage
-----
    .venv/Scripts/python.exe scripts/14_spatial_frequency_decomposition.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svdvals

TASK_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = TASK_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HD_ROOT = Path("D:/ProjData/datasets/ssvep_hd_200target")
CACHE_DIR = HD_ROOT / "derivatives" / "tdca_sample" / "cache"

FS = 250
LATENCY_SAMPLES = 35
N_HARMONICS = 5
N_BASE_FREQS = 40
N_POSITIONS = 5

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


def _save(fig, path):
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.name}")


def compute_channel_energy_profiles(data, refs, win_samp):
    """For each of 200 targets, compute per-channel spectral energy at its own base freq.

    Returns: (200, 66) array — energy[target, channel] at target's own frequency.
    This captures the SPATIAL pattern of the SSVEP response.
    """
    n_blocks = data.shape[3]
    energy_profiles = np.zeros((200, 66))

    for target_idx in range(200):
        base_fi = target_idx // N_POSITIONS
        block_energies = []

        for block in range(n_blocks):
            seg = data[:, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, target_idx, block]
            seg = seg - seg.mean(axis=-1, keepdims=True)
            seg_norms = np.linalg.norm(seg, axis=-1, keepdims=True)
            seg = seg / np.clip(seg_norms, 1e-12, None)

            # Correlation with target frequency references: (66, 10)
            corr = seg @ refs[base_fi].T  # (66, 10)
            energy_per_ch = (corr ** 2).sum(axis=-1)  # (66,)
            block_energies.append(energy_per_ch)

        energy_profiles[target_idx] = np.mean(block_energies, axis=0)

    return energy_profiles


def compute_spatial_templates(data, win_samp):
    """Compute mean EEG response template per target (no freq projection).

    Returns: (200, 66, win_samp) — raw spatial-temporal templates.
    """
    n_blocks = data.shape[3]
    templates = np.zeros((200, 66, win_samp))

    for target_idx in range(200):
        trial_sum = np.zeros((66, win_samp))
        for block in range(n_blocks):
            seg = data[:, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, target_idx, block]
            trial_sum += seg
        templates[target_idx] = trial_sum / n_blocks

    return templates


# ═══════════════════════════════════════════════════════════════════════════
# Part A: Spatial topography at same frequency, different positions
# ═══════════════════════════════════════════════════════════════════════════

def part_a_spatial_topography(energy_profiles, subj_name):
    """Show per-channel energy for 5 positions at the same frequency."""
    print(f"\n  [A] Spatial topography — {subj_name}...")

    # Pick 3 representative frequencies
    freq_indices = [0, 4, 20]  # 8.0 Hz, 12.0 Hz, 10.4 Hz
    freq_labels = [f"{BASE_FREQS[fi]:.1f} Hz" for fi in freq_indices]

    fig, axes = plt.subplots(3, 6, figsize=(24, 12), constrained_layout=True)

    for row, fi in enumerate(freq_indices):
        # 5 positions for this frequency
        target_indices = [fi * 5 + pos for pos in range(5)]
        profiles = energy_profiles[target_indices]  # (5, 66)

        # Show each position's channel energy
        for pos in range(5):
            ax = axes[row, pos]
            ax.bar(range(66), profiles[pos], color=plt.cm.Set2(pos / 4.0), alpha=0.7, width=1.0)
            ax.set_xlim(-1, 67)
            ax.set_title(f"Pos {pos+1}", fontsize=9)
            if pos == 0:
                ax.set_ylabel(f"{freq_labels[row]}\nEnergy")
            if row == 2:
                ax.set_xlabel("Channel index")
            ax.set_ylim(0, profiles.max() * 1.15)
            ax.grid(alpha=0.1, axis="y")

        # 6th column: overlay all 5 positions
        ax = axes[row, 5]
        for pos in range(5):
            ax.plot(range(66), profiles[pos], linewidth=1.2, alpha=0.7,
                    label=f"Pos {pos+1}", color=plt.cm.Set2(pos / 4.0))
        ax.set_title("All 5 positions overlay", fontsize=9)
        ax.legend(fontsize=6, ncol=2)
        ax.set_xlim(-1, 67)
        if row == 2:
            ax.set_xlabel("Channel index")
        ax.grid(alpha=0.15)

    fig.suptitle(f"Part A: Per-channel spectral energy — same frequency, 5 spatial positions\n"
                 f"{subj_name}, 66ch, 500ms, 18-block average",
                 fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_spatial_topography_same_freq.png")

    # Quantify: pairwise correlation between position profiles
    print(f"    Within-frequency spatial profile correlations (66ch):")
    for fi_idx, fi in enumerate(freq_indices):
        target_indices = [fi * 5 + pos for pos in range(5)]
        profiles = energy_profiles[target_indices]
        # Pairwise Pearson correlation
        corr_mat = np.corrcoef(profiles)
        off_diag = corr_mat[np.triu_indices(5, k=1)]
        print(f"      {freq_labels[fi_idx]}: mean r={off_diag.mean():.3f}, "
              f"min r={off_diag.min():.3f}, max r={off_diag.max():.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Part B: Within-frequency spatial confusion matrix
# ═══════════════════════════════════════════════════════════════════════════

def part_b_spatial_confusion(data, subj_name):
    """5×5 spatial similarity within each frequency, across channel configs."""
    print(f"\n  [B] Within-frequency spatial confusion — {subj_name}...")

    win_samp = 125
    n_blocks = data.shape[3]

    ch_configs = ["9ch", "32ch", "66ch"]

    # For each channel config, compute average within-freq spatial correlation
    # Use raw EEG templates (not frequency-projected) for spatial pattern comparison
    all_spatial_corrs = {}

    for ch_name in ch_configs:
        ch_idx = CHANNEL_SETS[ch_name]
        n_ch = len(ch_idx)

        # Build per-target template vectors (flatten channel × time)
        templates = np.zeros((200, n_ch * win_samp))
        for tidx in range(200):
            trial_sum = np.zeros(n_ch * win_samp)
            for blk in range(n_blocks):
                seg = data[ch_idx, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, tidx, blk]
                trial_sum += seg.flatten()
            templates[tidx] = trial_sum / n_blocks

        # Normalize templates
        templates -= templates.mean(axis=-1, keepdims=True)
        norms = np.linalg.norm(templates, axis=-1, keepdims=True)
        templates /= np.clip(norms, 1e-12, None)

        # Within-frequency 5×5 correlation matrices
        within_corrs = []  # average off-diagonal correlation per freq
        between_corrs = []  # average between-frequency correlation

        freq_5x5_list = []
        for fi in range(N_BASE_FREQS):
            idx = [fi * 5 + p for p in range(5)]
            sub_templates = templates[idx]  # (5, features)
            corr_5x5 = sub_templates @ sub_templates.T  # (5, 5) — already normalized
            freq_5x5_list.append(corr_5x5)
            off = corr_5x5[np.triu_indices(5, k=1)]
            within_corrs.append(off.mean())

        # Between-frequency correlation (sample)
        for fi in range(0, N_BASE_FREQS, 5):
            for fj in range(fi + 1, min(fi + 5, N_BASE_FREQS)):
                idx_i = [fi * 5 + p for p in range(5)]
                idx_j = [fj * 5 + p for p in range(5)]
                cross = templates[idx_i] @ templates[idx_j].T
                between_corrs.append(cross.mean())

        all_spatial_corrs[ch_name] = {
            "within_mean": np.mean(within_corrs),
            "within_std": np.std(within_corrs),
            "within_per_freq": within_corrs,
            "between_mean": np.mean(between_corrs),
            "freq_5x5_list": freq_5x5_list,
        }

        print(f"    {ch_name}: within-freq spatial r = {np.mean(within_corrs):.3f} ± {np.std(within_corrs):.3f}"
              f"  |  between-freq r = {np.mean(between_corrs):.3f}")

    # Fig B1: Example 5×5 confusion matrices for 3 frequencies × 3 ch configs
    freq_show = [0, 4, 20]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)

    for col, ch_name in enumerate(ch_configs):
        for row, fi in enumerate(freq_show):
            ax = axes[row, col]
            mat = all_spatial_corrs[ch_name]["freq_5x5_list"][fi]
            im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdYlBu_r", aspect="auto")
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if mat[i,j] > 0.7 else "black")
            ax.set_xticks(range(5))
            ax.set_xticklabels([f"P{p+1}" for p in range(5)], fontsize=8)
            ax.set_yticks(range(5))
            ax.set_yticklabels([f"P{p+1}" for p in range(5)], fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{BASE_FREQS[fi]:.1f} Hz")
            if row == 0:
                ax.set_title(f"{ch_name}", fontsize=11)
            if row == 2:
                ax.set_xlabel("Position")
    plt.colorbar(im, ax=axes, shrink=0.6, label="Template correlation")

    fig.suptitle(f"Part B: Within-frequency spatial confusion — {subj_name}\n"
                 f"(5×5 template correlation at same frequency, different positions)",
                 fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_within_freq_spatial_confusion.png")

    # Fig B2: Within-freq vs between-freq correlation across channel configs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(ch_configs))
    within_means = [all_spatial_corrs[c]["within_mean"] for c in ch_configs]
    within_stds = [all_spatial_corrs[c]["within_std"] for c in ch_configs]
    between_means = [all_spatial_corrs[c]["between_mean"] for c in ch_configs]

    ax.bar(x - 0.15, within_means, 0.28, yerr=within_stds, capsize=3,
           color="#f59e0b", alpha=0.7, label="Within-freq (same freq, diff position)")
    ax.bar(x + 0.15, between_means, 0.28,
           color="#3b82f6", alpha=0.7, label="Between-freq (diff freq)")
    ax.set_xticks(x)
    ax.set_xticklabels(ch_configs)
    ax.set_ylabel("Template correlation")
    ax.set_title("Spatial discriminability gap")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, axis="y")

    # Discrimination ratio
    ax = axes[1]
    gap = [between_means[i] / within_means[i] for i in range(len(ch_configs))]
    ax.bar(x, [1 - w for w in within_means], 0.4, color="#dc2626", alpha=0.7,
           label="1 − within_corr (spatial separability)")
    ax.set_xticks(x)
    ax.set_xticklabels(ch_configs)
    ax.set_ylabel("Spatial separability (1 − r)")
    ax.set_title("How well can same-freq positions be separated?")
    ax.grid(alpha=0.15, axis="y")

    fig.suptitle(f"Spatial vs frequency discrimination — {subj_name}",
                 fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_spatial_vs_freq_discrimination.png")

    return all_spatial_corrs


# ═══════════════════════════════════════════════════════════════════════════
# Part C: Full 200×200 similarity matrix — block structure
# ═══════════════════════════════════════════════════════════════════════════

def part_c_full_200x200(data, subj_name):
    """Show the block structure of the full 200×200 target similarity matrix."""
    print(f"\n  [C] Full 200×200 similarity matrix — {subj_name}...")

    win_samp = 125
    n_blocks = data.shape[3]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    for ax_i, ch_name in enumerate(["9ch", "32ch", "66ch"]):
        ch_idx = CHANNEL_SETS[ch_name]
        n_ch = len(ch_idx)

        templates = np.zeros((200, n_ch * win_samp))
        for tidx in range(200):
            trial_sum = np.zeros(n_ch * win_samp)
            for blk in range(n_blocks):
                seg = data[ch_idx, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, tidx, blk]
                trial_sum += seg.flatten()
            templates[tidx] = trial_sum / n_blocks

        templates -= templates.mean(axis=-1, keepdims=True)
        norms = np.linalg.norm(templates, axis=-1, keepdims=True)
        templates /= np.clip(norms, 1e-12, None)

        sim_mat = templates @ templates.T  # (200, 200)

        ax = axes[ax_i]
        im = ax.imshow(sim_mat, cmap="RdYlBu_r", vmin=-0.1, vmax=0.8, aspect="auto",
                        interpolation="nearest")
        ax.set_title(f"{ch_name} ({len(ch_idx)} channels)", fontsize=11)
        ax.set_xlabel("Target index (0-199)")
        ax.set_ylabel("Target index (0-199)")

        # Mark frequency boundaries every 5 targets
        for i in range(0, 200, 5):
            ax.axhline(i - 0.5, color="white", linewidth=0.2, alpha=0.3)
            ax.axvline(i - 0.5, color="white", linewidth=0.2, alpha=0.3)
        # Mark frequency group boundaries every 40 targets (spatial zone boundaries)
        for i in range(0, 200, 40):
            ax.axhline(i - 0.5, color="red", linewidth=0.8, alpha=0.5)
            ax.axvline(i - 0.5, color="red", linewidth=0.8, alpha=0.5)

        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(f"Full 200×200 template similarity matrix — {subj_name}\n"
                 f"(white lines: 5-target freq groups; red lines: 40-target spatial zones)",
                 fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_200x200_similarity.png")


# ═══════════════════════════════════════════════════════════════════════════
# Part D: Frequency-axis vs spatial-axis effective rank
# ═══════════════════════════════════════════════════════════════════════════

def part_d_dual_axis_rank(data, subj_name):
    """Decompose discrimination capacity into frequency axis and spatial axis."""
    print(f"\n  [D] Dual-axis effective rank — {subj_name}...")

    win_samp = 125
    n_blocks = data.shape[3]

    ch_configs = ["9ch", "32ch", "66ch"]
    results = {}

    for ch_name in ch_configs:
        ch_idx = CHANNEL_SETS[ch_name]
        n_ch = len(ch_idx)

        # Build templates: (200, n_ch * win_samp)
        templates = np.zeros((200, n_ch * win_samp))
        for tidx in range(200):
            trial_sum = np.zeros(n_ch * win_samp)
            for blk in range(n_blocks):
                seg = data[ch_idx, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, tidx, blk]
                trial_sum += seg.flatten()
            templates[tidx] = trial_sum / n_blocks

        # Full 200-class effective rank
        templates_c = templates - templates.mean(axis=0)
        sv_full = svdvals(templates_c)
        total = sv_full.sum()
        cumsum_full = np.cumsum(sv_full / total)
        erank_200 = int(np.searchsorted(cumsum_full, 0.95) + 1)

        # Frequency-axis: average 5 positions → 40 frequency centroids
        freq_centroids = np.zeros((N_BASE_FREQS, n_ch * win_samp))
        for fi in range(N_BASE_FREQS):
            freq_centroids[fi] = templates[fi*5:(fi+1)*5].mean(axis=0)

        freq_c = freq_centroids - freq_centroids.mean(axis=0)
        sv_freq = svdvals(freq_c)
        total_f = sv_freq.sum()
        cumsum_freq = np.cumsum(sv_freq / total_f)
        erank_freq = int(np.searchsorted(cumsum_freq, 0.95) + 1)

        # Spatial-axis: within-frequency residuals (position - freq centroid)
        spatial_residuals = np.zeros((200, n_ch * win_samp))
        for fi in range(N_BASE_FREQS):
            for p in range(5):
                spatial_residuals[fi*5 + p] = templates[fi*5 + p] - freq_centroids[fi]

        spatial_c = spatial_residuals  # already zero-mean within each freq group
        sv_spatial = svdvals(spatial_c)
        total_s = sv_spatial.sum()
        cumsum_spatial = np.cumsum(sv_spatial / total_s)
        erank_spatial = int(np.searchsorted(cumsum_spatial, 0.95) + 1)

        results[ch_name] = {
            "erank_200": erank_200, "erank_freq": erank_freq, "erank_spatial": erank_spatial,
            "sv_full": sv_full, "sv_freq": sv_freq, "sv_spatial": sv_spatial,
            "cumsum_full": cumsum_full, "cumsum_freq": cumsum_freq, "cumsum_spatial": cumsum_spatial,
        }

        # Variance explained
        freq_var = np.sum(sv_freq[:min(40, len(sv_freq))]**2)
        spatial_var = np.sum(sv_spatial[:min(200, len(sv_spatial))]**2)
        full_var = np.sum(sv_full[:min(200, len(sv_full))]**2)

        print(f"    {ch_name}: erank_200={erank_200}  erank_freq={erank_freq}  "
              f"erank_spatial={erank_spatial}")

    # Fig D1: Cumulative eigenspectrum comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    ch_colors = {"9ch": "#3b82f6", "32ch": "#f59e0b", "66ch": "#dc2626"}

    ax = axes[0]
    for ch_name in ch_configs:
        r = results[ch_name]
        n_sv = min(60, len(r["cumsum_full"]))
        ax.plot(range(1, n_sv+1), r["cumsum_full"][:n_sv], "o-",
                color=ch_colors[ch_name], markersize=3, linewidth=1.5, label=ch_name)
    ax.axhline(0.95, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_title("Full 200-target eigenspectrum")
    ax.legend()
    ax.grid(alpha=0.15)

    ax = axes[1]
    for ch_name in ch_configs:
        r = results[ch_name]
        n_sv = min(40, len(r["cumsum_freq"]))
        ax.plot(range(1, n_sv+1), r["cumsum_freq"][:n_sv], "o-",
                color=ch_colors[ch_name], markersize=3, linewidth=1.5, label=ch_name)
    ax.axhline(0.95, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_title("Frequency-axis eigenspectrum\n(40 freq centroids)")
    ax.legend()
    ax.grid(alpha=0.15)

    ax = axes[2]
    for ch_name in ch_configs:
        r = results[ch_name]
        n_sv = min(40, len(r["cumsum_spatial"]))
        ax.plot(range(1, n_sv+1), r["cumsum_spatial"][:n_sv], "o-",
                color=ch_colors[ch_name], markersize=3, linewidth=1.5, label=ch_name)
    ax.axhline(0.95, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_title("Spatial-axis eigenspectrum\n(within-freq position residuals)")
    ax.legend()
    ax.grid(alpha=0.15)

    fig.suptitle(f"Dual-axis rank decomposition — {subj_name}\n"
                 f"200 targets = frequency axis (40 freqs) + spatial axis (5 positions per freq)",
                 fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_dual_axis_eigenspectrum.png")

    # Fig D2: Effective rank bar chart
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    x = np.arange(len(ch_configs))
    w = 0.22
    erank_full = [results[c]["erank_200"] for c in ch_configs]
    erank_freq = [results[c]["erank_freq"] for c in ch_configs]
    erank_spat = [results[c]["erank_spatial"] for c in ch_configs]

    ax.bar(x - w, erank_freq, w, color="#3b82f6", alpha=0.7, label="Frequency axis (40 freqs)")
    ax.bar(x, erank_spat, w, color="#f59e0b", alpha=0.7, label="Spatial axis (5 positions)")
    ax.bar(x + w, erank_full, w, color="#dc2626", alpha=0.7, label="Full 200 targets")

    for i, (ef, es, e200) in enumerate(zip(erank_freq, erank_spat, erank_full)):
        ax.text(i - w, ef + 0.5, str(ef), ha="center", fontsize=9, fontweight="bold")
        ax.text(i, es + 0.5, str(es), ha="center", fontsize=9, fontweight="bold")
        ax.text(i + w, e200 + 0.5, str(e200), ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ch_configs, fontsize=11)
    ax.set_ylabel("Effective rank (95% energy)")
    ax.set_title(f"Effective rank decomposition — {subj_name}\n"
                 f"HD-EEG 增加的维度主要在哪个轴?")
    ax.legend()
    ax.grid(alpha=0.15, axis="y")
    _save(fig, FIG_DIR / "fig_dual_axis_erank.png")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Part E: Multi-subject spatial discrimination comparison
# ═══════════════════════════════════════════════════════════════════════════

def part_e_multisubject(data_dict, subjects):
    """Compare spatial discrimination across subjects and channel configs."""
    print(f"\n  [E] Multi-subject spatial discrimination...")

    win_samp = 125
    ch_configs = ["9ch", "32ch", "66ch"]

    all_results = {}  # (subj, ch_name) → within_corr, between_corr

    for subj in subjects:
        data = data_dict[subj]
        n_blocks = data.shape[3]

        for ch_name in ch_configs:
            ch_idx = CHANNEL_SETS[ch_name]
            n_ch = len(ch_idx)

            templates = np.zeros((200, n_ch * win_samp))
            for tidx in range(200):
                trial_sum = np.zeros(n_ch * win_samp)
                for blk in range(n_blocks):
                    seg = data[ch_idx, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, tidx, blk]
                    trial_sum += seg.flatten()
                templates[tidx] = trial_sum / n_blocks

            templates -= templates.mean(axis=-1, keepdims=True)
            norms = np.linalg.norm(templates, axis=-1, keepdims=True)
            templates /= np.clip(norms, 1e-12, None)

            within_corrs = []
            between_corrs = []
            for fi in range(N_BASE_FREQS):
                idx = [fi * 5 + p for p in range(5)]
                sub_t = templates[idx]
                corr_5x5 = sub_t @ sub_t.T
                off = corr_5x5[np.triu_indices(5, k=1)]
                within_corrs.append(off.mean())

            # Between: sample across frequency pairs
            for fi in range(0, N_BASE_FREQS, 4):
                for fj in range(fi+1, min(fi+4, N_BASE_FREQS)):
                    idx_i = fi * 5
                    idx_j = fj * 5
                    r = np.dot(templates[idx_i], templates[idx_j])
                    between_corrs.append(r)

            all_results[(subj, ch_name)] = {
                "within": np.mean(within_corrs),
                "between": np.mean(between_corrs),
                "spatial_gap": np.mean(within_corrs) - np.mean(between_corrs),
            }
            print(f"    {subj}/{ch_name}: within={np.mean(within_corrs):.3f}  "
                  f"between={np.mean(between_corrs):.3f}  "
                  f"gap={np.mean(within_corrs) - np.mean(between_corrs):.3f}")

    # Fig E1: Spatial gap across subjects × ch configs
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    subj_colors = {"S1": "#3b82f6", "S3": "#10b981", "S9": "#f59e0b", "S11": "#dc2626"}

    ax = axes[0]
    for subj in subjects:
        within_vals = [all_results[(subj, c)]["within"] for c in ch_configs]
        ax.plot(range(len(ch_configs)), within_vals, "o-",
                color=subj_colors.get(subj, "gray"), linewidth=2, markersize=7, label=subj)
    ax.set_xticks(range(len(ch_configs)))
    ax.set_xticklabels(ch_configs)
    ax.set_ylabel("Within-freq spatial correlation")
    ax.set_title("Same frequency, different positions\n(lower = more spatially separable)")
    ax.legend()
    ax.grid(alpha=0.15)

    ax = axes[1]
    for subj in subjects:
        gaps = [all_results[(subj, c)]["spatial_gap"] for c in ch_configs]
        ax.plot(range(len(ch_configs)), gaps, "o-",
                color=subj_colors.get(subj, "gray"), linewidth=2, markersize=7, label=subj)
    ax.set_xticks(range(len(ch_configs)))
    ax.set_xticklabels(ch_configs)
    ax.set_ylabel("Spatial confusion gap\n(within − between)")
    ax.set_title("How much harder is spatial vs frequency discrimination?\n"
                 "(larger gap = spatial harder to resolve)")
    ax.legend()
    ax.grid(alpha=0.15)

    fig.suptitle("Multi-subject spatial discrimination analysis",
                 fontsize=13, fontweight="bold")
    _save(fig, FIG_DIR / "fig_multisubject_spatial_gap.png")

    return all_results


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

    win_samp = 125
    refs = make_references(win_samp)

    # Part A: Spatial topography (S1)
    if "S1" in data_dict:
        print("\n  Computing 200-target channel energy profiles for S1...")
        ep_s1 = compute_channel_energy_profiles(data_dict["S1"], refs, win_samp)
        part_a_spatial_topography(ep_s1, "S1")

    # Part B: Within-frequency spatial confusion (S1)
    if "S1" in data_dict:
        part_b_spatial_confusion(data_dict["S1"], "S1")

    # Part C: Full 200×200 similarity matrix (S1)
    if "S1" in data_dict:
        part_c_full_200x200(data_dict["S1"], "S1")

    # Part D: Dual-axis effective rank (S1)
    if "S1" in data_dict:
        part_d_dual_axis_rank(data_dict["S1"], "S1")

    # Part E: Multi-subject comparison
    part_e_multisubject(data_dict, sample_subjects)

    print(f"\n  All figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
