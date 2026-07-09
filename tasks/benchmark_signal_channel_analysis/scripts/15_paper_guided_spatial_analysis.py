"""Paper-guided spatial analysis of HD200 SSVEP dataset.

Guided by findings from Ming et al. 2026 (the HD200 paper), which reveals:
  - 5 spatial positions (R/D/L/U/C) within each flicker, separated by 0.23 deg
  - Different fixation positions produce different SNR topographies and phases
  - R/L fixation -> lateralized amplitude; U/D -> anterior-posterior differences
  - HD-EEG helps spatial discrimination (+15.53%) far more than frequency (+1.32%)
  - Optimal fixation combos follow "maximum physical distance" principle

Analysis:
  A. Position-pair SNR topography asymmetry
  B. Spatial vs frequency channel-density sensitivity (the paper's core finding)
  C. Pairwise position discriminability matrix
  D. Per-frequency spatial accuracy variation

Usage
-----
    .venv/Scripts/python.exe scripts/15_paper_guided_spatial_analysis.py
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

TASK_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = TASK_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HD_ROOT = Path("D:/ProjData/datasets/ssvep_hd_200target")
CACHE_DIR = HD_ROOT / "derivatives" / "tdca_sample" / "cache"

FS = 250
LATENCY_SAMPLES = 35          # visual latency already applied in cached data
N_HARMONICS = 5
N_BASE_FREQS = 40
N_POSITIONS = 5
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

TARGET_TO_BASE_FREQ = np.arange(N_TARGETS) // N_POSITIONS
TARGET_TO_POSITION = np.arange(N_TARGETS) % N_POSITIONS

POSITION_NAMES = {0: "RIGHT", 1: "DOWN", 2: "LEFT", 3: "UP", 4: "CENTER"}
POSITION_SHORT = {0: "R", 1: "D", 2: "L", 3: "U", 4: "C"}

# Approximate angular positions for distance computation (degrees on unit circle)
POSITION_ANGLES_DEG = {0: 0.0, 1: 270.0, 2: 180.0, 3: 90.0, 4: None}  # C=center

CHANNEL_SETS = {
    "9ch": np.array([21, 27, 28, 29, 31, 32, 33, 59, 63]),
    "32ch": np.array(list(range(17, 36)) + list(range(53, 66))),
    "66ch": np.arange(66),
}

# Hemisphere channel assignments (approximate — left/right of midline)
# With 66-ch layout: lower indices tend to be frontal/left, higher indices posterior/right
# Rough split: channels 0-32 left hemisphere, 33-65 right hemisphere
LEFT_HEMI_CH = np.array([i for i in range(66) if i < 33])
RIGHT_HEMI_CH = np.array([i for i in range(66) if i >= 33])

SUBJECTS_DEFAULT = ["S1", "S3", "S9", "S11"]
ALL_SUBJECTS = [f"S{n}" for n in list(range(1, 10)) + list(range(11, 16))]

POSITION_COLORS = {
    0: "#e63946",   # RIGHT — red
    1: "#457b9d",   # DOWN  — blue
    2: "#2a9d8f",   # LEFT  — teal
    3: "#e9c46a",   # UP    — gold
    4: "#6c757d",   # CENTER — gray
}

SUBJ_MARKERS = {"S1": "o", "S3": "s", "S9": "^", "S11": "D"}
SUBJ_COLORS = {"S1": "#3b82f6", "S3": "#10b981", "S9": "#f59e0b", "S11": "#dc2626"}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {path.name}")


def load_subject_data(subj: str):
    """Load single subject, FB0 only. Returns (66, n_samples, 200, 18) or None."""
    cache_path = CACHE_DIR / f"filtered_{subj}_200target_66ch_18blocks_float32.npy"
    if not cache_path.exists():
        print(f"  SKIP {subj}: cache not found at {cache_path}")
        return None
    t0 = time.time()
    data_full = np.load(str(cache_path), mmap_mode="r")
    data = np.asarray(data_full[:, :, :, :, 0])  # FB0 only
    print(f"  Loaded {subj}: {data.shape}  ({time.time() - t0:.1f}s)")
    return data


def make_references(n_samples: int):
    """Sin/cos reference signals for N_HARMONICS at each of 40 base freqs.
    Returns: (40, 2*N_HARMONICS, n_samples), zero-mean & unit-norm.
    """
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


def compute_channel_energy(data, refs, win_samp):
    """Per-channel spectral energy for each of 200 targets, averaged across blocks.

    For each target, computes Pearson-correlation-based energy at its own
    fundamental + harmonics (via pre-computed sin/cos references).

    Parameters
    ----------
    data : (66, n_samples, 200, 18) array
    refs : (40, 2*N_HARMONICS, win_samp) normalized references
    win_samp : int — window length in samples

    Returns
    -------
    energy : (200, 66) — energy[target, channel]
    """
    n_ch = data.shape[0]
    n_blocks = data.shape[3]
    energy = np.zeros((N_TARGETS, n_ch))

    for target_idx in range(N_TARGETS):
        base_fi = TARGET_TO_BASE_FREQ[target_idx]
        block_energies = np.zeros((n_blocks, n_ch))

        for blk in range(n_blocks):
            seg = data[:, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, target_idx, blk]
            seg = seg - seg.mean(axis=-1, keepdims=True)
            seg_norms = np.linalg.norm(seg, axis=-1, keepdims=True)
            seg = seg / np.clip(seg_norms, 1e-12, None)

            # Correlation with reference: (n_ch, 2*N_HARMONICS)
            corr = seg @ refs[base_fi].T
            block_energies[blk] = (corr ** 2).sum(axis=-1)

        energy[target_idx] = block_energies.mean(axis=0)

    return energy


def pearson_corr_1d(a, b):
    """Pearson correlation between two 1-d vectors."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_c, b_c) / denom)


# ═══════════════════════════════════════════════════════════════════════════════
# Part A: Position-Pair SNR Topography Asymmetry
# ═══════════════════════════════════════════════════════════════════════════════

def part_a(data_dict: dict, refs, win_samp: int):
    """Position-pair SNR topography asymmetry analysis.

    For representative subjects and frequencies:
    - Per-channel spectral energy for each of 5 positions
    - Difference topographies between position pairs
    - Lateralization quantification for R vs L
    """
    print("\n" + "=" * 70)
    print("  Part A: Position-Pair SNR Topography Asymmetry")
    print("=" * 70)
    t0 = time.time()

    repr_subjects = [s for s in ["S1", "S3", "S9", "S11"] if s in data_dict]
    repr_freq_indices = []
    for target_hz in [8.0, 10.0, 12.0, 14.0]:
        idx = int(np.argmin(np.abs(BASE_FREQS - target_hz)))
        repr_freq_indices.append(idx)
    repr_freq_labels = [f"{BASE_FREQS[fi]:.1f} Hz" for fi in repr_freq_indices]

    # --- Fig A1: Per-position energy bar charts for each subject ---
    for subj in repr_subjects:
        print(f"\n  Computing channel energy for {subj}...")
        energy = compute_channel_energy(data_dict[subj], refs, win_samp)

        n_freqs_show = len(repr_freq_indices)
        fig, axes = plt.subplots(
            n_freqs_show, 6, figsize=(26, 4 * n_freqs_show),
            constrained_layout=True,
        )
        if n_freqs_show == 1:
            axes = axes[np.newaxis, :]

        for row, fi in enumerate(repr_freq_indices):
            targets = [fi * N_POSITIONS + pos for pos in range(N_POSITIONS)]
            profiles = energy[targets]  # (5, 66)

            # Individual position bar charts
            for pos in range(N_POSITIONS):
                ax = axes[row, pos]
                ax.bar(
                    range(66), profiles[pos],
                    color=POSITION_COLORS[pos], alpha=0.7, width=1.0,
                )
                ax.set_xlim(-1, 67)
                ax.set_title(
                    f"{POSITION_NAMES[pos]} (pos {pos})", fontsize=9,
                )
                if pos == 0:
                    ax.set_ylabel(f"{repr_freq_labels[row]}\nEnergy (r^2)")
                if row == n_freqs_show - 1:
                    ax.set_xlabel("Channel index")
                ax.set_ylim(0, profiles.max() * 1.15)
                ax.grid(alpha=0.1, axis="y")

            # 6th column: overlay all 5 positions
            ax = axes[row, 5]
            for pos in range(N_POSITIONS):
                ax.plot(
                    range(66), profiles[pos], linewidth=1.2, alpha=0.7,
                    label=POSITION_SHORT[pos], color=POSITION_COLORS[pos],
                )
            ax.set_title("All positions overlay", fontsize=9)
            ax.legend(fontsize=7, ncol=3)
            ax.set_xlim(-1, 67)
            if row == n_freqs_show - 1:
                ax.set_xlabel("Channel index")
            ax.grid(alpha=0.15)

        fig.suptitle(
            f"Part A1: Per-channel spectral energy by fixation position\n"
            f"{subj}, 66ch, {win_samp / FS * 1000:.0f}ms, 18-block average",
            fontsize=13, fontweight="bold",
        )
        _save(fig, FIG_DIR / f"fig_paper_guided_A1_topography_{subj}.png")

        # --- Fig A2: Difference topography heatmaps ---
        pairs = [
            (0, 2, "RIGHT - LEFT"),
            (1, 3, "DOWN - UP"),
            (0, 4, "RIGHT - CENTER"),
            (2, 4, "LEFT - CENTER"),
            (1, 4, "DOWN - CENTER"),
            (3, 4, "UP - CENTER"),
        ]

        fig, axes = plt.subplots(
            n_freqs_show, len(pairs), figsize=(4 * len(pairs), 3.5 * n_freqs_show),
            constrained_layout=True,
        )
        if n_freqs_show == 1:
            axes = axes[np.newaxis, :]

        for row, fi in enumerate(repr_freq_indices):
            targets = [fi * N_POSITIONS + pos for pos in range(N_POSITIONS)]
            profiles = energy[targets]  # (5, 66)

            for col, (p_a, p_b, pair_label) in enumerate(pairs):
                diff = profiles[p_a] - profiles[p_b]
                ax = axes[row, col]
                colors_bar = ["#e63946" if v > 0 else "#457b9d" for v in diff]
                ax.bar(range(66), diff, color=colors_bar, alpha=0.7, width=1.0)
                ax.axhline(0, color="black", linewidth=0.5)
                ax.set_xlim(-1, 67)
                if row == 0:
                    ax.set_title(pair_label, fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{repr_freq_labels[row]}\nDelta energy")
                if row == n_freqs_show - 1:
                    ax.set_xlabel("Channel")
                ax.grid(alpha=0.1, axis="y")

        fig.suptitle(
            f"Part A2: Position-pair energy differences (topography asymmetry)\n"
            f"{subj}, 66ch — red: positive, blue: negative",
            fontsize=13, fontweight="bold",
        )
        _save(fig, FIG_DIR / f"fig_paper_guided_A2_difference_{subj}.png")

        # --- Lateralization quantification for R vs L ---
        print(f"\n  Lateralization analysis (R vs L) for {subj}:")
        for fi in repr_freq_indices:
            targets_r = fi * N_POSITIONS + 0
            targets_l = fi * N_POSITIONS + 2
            e_r = energy[targets_r]  # (66,)
            e_l = energy[targets_l]  # (66,)

            # Energy in left hemi vs right hemi for RIGHT fixation
            r_left_hemi = e_r[LEFT_HEMI_CH].sum()
            r_right_hemi = e_r[RIGHT_HEMI_CH].sum()
            # Energy in left hemi vs right hemi for LEFT fixation
            l_left_hemi = e_l[LEFT_HEMI_CH].sum()
            l_right_hemi = e_l[RIGHT_HEMI_CH].sum()

            lat_r = (r_right_hemi - r_left_hemi) / (r_right_hemi + r_left_hemi + 1e-12)
            lat_l = (l_right_hemi - l_left_hemi) / (l_right_hemi + l_left_hemi + 1e-12)

            print(
                f"    {BASE_FREQS[fi]:5.1f} Hz: "
                f"RIGHT fix lat_idx={lat_r:+.3f} (L_hemi={r_left_hemi:.3f}, R_hemi={r_right_hemi:.3f})  |  "
                f"LEFT  fix lat_idx={lat_l:+.3f} (L_hemi={l_left_hemi:.3f}, R_hemi={l_right_hemi:.3f})"
            )

    print(f"\n  Part A completed in {time.time() - t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# Part B: Spatial vs Frequency Channel-Density Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

def _build_templates_lobo(data, ch_idx, win_samp, leave_out_block):
    """Build template for all targets, leaving out one block.

    Returns: (n_targets, n_ch * win_samp) templates, zero-mean & unit-norm.
    """
    n_ch = len(ch_idx)
    n_blocks = data.shape[3]
    feat_dim = n_ch * win_samp
    templates = np.zeros((N_TARGETS, feat_dim))

    for tidx in range(N_TARGETS):
        trial_sum = np.zeros(feat_dim)
        count = 0
        for blk in range(n_blocks):
            if blk == leave_out_block:
                continue
            seg = data[ch_idx, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, tidx, blk]
            trial_sum += seg.flatten()
            count += 1
        templates[tidx] = trial_sum / count

    templates -= templates.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(templates, axis=-1, keepdims=True)
    templates /= np.clip(norms, 1e-12, None)
    return templates


def _get_test_trial(data, ch_idx, win_samp, target_idx, block_idx):
    """Extract and normalize a single test trial vector."""
    seg = data[ch_idx, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, target_idx, block_idx]
    vec = seg.flatten().astype(np.float64)
    vec -= vec.mean()
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return vec


def part_b(data_dict: dict, win_samp: int):
    """Reproduce the paper's core finding: HD-EEG helps spatial >> frequency.

    For each channel set (9ch, 32ch, 66ch):
      - FREQUENCY task: classify 40 frequencies (templates average across 5 positions)
      - SPATIAL task: for each frequency, classify 5 positions
    Leave-one-block-out CV.
    """
    print("\n" + "=" * 70)
    print("  Part B: Spatial vs Frequency Channel-Density Sensitivity")
    print("=" * 70)
    t0_all = time.time()

    ch_configs = ["9ch", "32ch", "66ch"]
    all_results = {}  # (subj, ch_name) -> {"freq_acc", "spatial_acc"}

    for subj, data in data_dict.items():
        n_blocks = data.shape[3]
        print(f"\n  Processing {subj}...")

        for ch_name in ch_configs:
            ch_idx = CHANNEL_SETS[ch_name]
            n_ch = len(ch_idx)
            feat_dim = n_ch * win_samp
            t0 = time.time()

            freq_correct = 0
            freq_total = 0
            spatial_correct_per_freq = np.zeros(N_BASE_FREQS)
            spatial_total_per_freq = np.zeros(N_BASE_FREQS)

            for blk in range(n_blocks):
                # --- Build templates from (n_blocks - 1) blocks ---
                # Full 200-target templates
                templates_200 = _build_templates_lobo(data, ch_idx, win_samp, blk)

                # FREQUENCY templates: average 5 positions within each freq
                templates_freq = np.zeros((N_BASE_FREQS, feat_dim))
                for fi in range(N_BASE_FREQS):
                    t_indices = [fi * N_POSITIONS + p for p in range(N_POSITIONS)]
                    templates_freq[fi] = templates_200[t_indices].mean(axis=0)
                # Re-normalize frequency templates
                templates_freq -= templates_freq.mean(axis=-1, keepdims=True)
                norms_f = np.linalg.norm(templates_freq, axis=-1, keepdims=True)
                templates_freq /= np.clip(norms_f, 1e-12, None)

                # SPATIAL templates per frequency: (40, 5, feat_dim)
                templates_spatial = np.zeros((N_BASE_FREQS, N_POSITIONS, feat_dim))
                for fi in range(N_BASE_FREQS):
                    for pos in range(N_POSITIONS):
                        tidx = fi * N_POSITIONS + pos
                        templates_spatial[fi, pos] = templates_200[tidx]

                # --- Test on held-out block ---
                for target_idx in range(N_TARGETS):
                    trial = _get_test_trial(data, ch_idx, win_samp, target_idx, blk)
                    true_freq = TARGET_TO_BASE_FREQ[target_idx]
                    true_pos = TARGET_TO_POSITION[target_idx]

                    # FREQUENCY classification: correlate with 40 freq templates
                    corrs_freq = templates_freq @ trial  # (40,)
                    pred_freq = np.argmax(corrs_freq)
                    if pred_freq == true_freq:
                        freq_correct += 1
                    freq_total += 1

                    # SPATIAL classification: within true frequency, classify 5 positions
                    corrs_spatial = templates_spatial[true_freq] @ trial  # (5,)
                    pred_pos = np.argmax(corrs_spatial)
                    if pred_pos == true_pos:
                        spatial_correct_per_freq[true_freq] += 1
                    spatial_total_per_freq[true_freq] += 1

            freq_acc = freq_correct / freq_total if freq_total > 0 else 0
            spatial_acc = (
                np.mean(
                    spatial_correct_per_freq / np.clip(spatial_total_per_freq, 1, None)
                )
            )

            all_results[(subj, ch_name)] = {
                "freq_acc": freq_acc,
                "spatial_acc": spatial_acc,
                "spatial_per_freq": (
                    spatial_correct_per_freq / np.clip(spatial_total_per_freq, 1, None)
                ),
            }

            print(
                f"    {ch_name}: freq_acc={freq_acc:.4f}  spatial_acc={spatial_acc:.4f}"
                f"  ({time.time() - t0:.1f}s)"
            )

    # --- Summary table ---
    print("\n  Summary — accuracy by (subject, channel set):")
    print(f"  {'Subj':>5s}  {'ChSet':>5s}  {'FreqAcc':>8s}  {'SpatAcc':>8s}")
    for subj in data_dict:
        for ch_name in ch_configs:
            r = all_results.get((subj, ch_name))
            if r:
                print(
                    f"  {subj:>5s}  {ch_name:>5s}  {r['freq_acc']:8.4f}  {r['spatial_acc']:8.4f}"
                )

    # --- Fig B1: accuracy vs channel count ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    ch_counts = [len(CHANNEL_SETS[c]) for c in ch_configs]

    # Left panel: raw accuracy
    ax = axes[0]
    for subj in data_dict:
        freq_accs = [all_results[(subj, c)]["freq_acc"] for c in ch_configs]
        spat_accs = [all_results[(subj, c)]["spatial_acc"] for c in ch_configs]
        marker = SUBJ_MARKERS.get(subj, "o")
        color = SUBJ_COLORS.get(subj, "gray")
        ax.plot(
            ch_counts, freq_accs, marker + "-",
            color=color, linewidth=2, markersize=8, alpha=0.7,
            label=f"{subj} freq",
        )
        ax.plot(
            ch_counts, spat_accs, marker + "--",
            color=color, linewidth=2, markersize=8, alpha=0.7,
            label=f"{subj} spatial",
        )
    ax.set_xlabel("Number of channels")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Frequency vs Spatial classification accuracy\n(solid=freq, dashed=spatial)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xticks(ch_counts)
    ax.set_xticklabels([f"{c}\n({n})" for c, n in zip(ch_configs, ch_counts)])
    ax.grid(alpha=0.15)
    ax.set_ylim(0, 1.05)

    # Right panel: gain from 9ch baseline
    ax = axes[1]
    for subj in data_dict:
        freq_base = all_results[(subj, "9ch")]["freq_acc"]
        spat_base = all_results[(subj, "9ch")]["spatial_acc"]
        freq_gains = [
            (all_results[(subj, c)]["freq_acc"] - freq_base) * 100
            for c in ch_configs
        ]
        spat_gains = [
            (all_results[(subj, c)]["spatial_acc"] - spat_base) * 100
            for c in ch_configs
        ]
        marker = SUBJ_MARKERS.get(subj, "o")
        color = SUBJ_COLORS.get(subj, "gray")
        ax.plot(
            ch_counts, freq_gains, marker + "-",
            color=color, linewidth=2, markersize=8, alpha=0.7,
            label=f"{subj} freq",
        )
        ax.plot(
            ch_counts, spat_gains, marker + "--",
            color=color, linewidth=2, markersize=8, alpha=0.7,
            label=f"{subj} spatial",
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Number of channels")
    ax.set_ylabel("Accuracy gain vs 9ch (%)")
    ax.set_title(
        "Channel-density sensitivity\n"
        "(paper finding: spatial >> frequency gain from HD-EEG)"
    )
    ax.legend(fontsize=7, ncol=2)
    ax.set_xticks(ch_counts)
    ax.set_xticklabels([f"{c}\n({n})" for c, n in zip(ch_configs, ch_counts)])
    ax.grid(alpha=0.15)

    fig.suptitle(
        "Part B: Spatial vs Frequency channel-density sensitivity\n"
        "500ms window, FB0 broadband, template matching (Pearson), LOBO CV",
        fontsize=13, fontweight="bold",
    )
    _save(fig, FIG_DIR / "fig_paper_guided_B1_spatial_vs_freq_sensitivity.png")

    # --- Fig B2: Mean across subjects ---
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    mean_freq = np.array([
        np.mean([all_results[(s, c)]["freq_acc"] for s in data_dict])
        for c in ch_configs
    ])
    std_freq = np.array([
        np.std([all_results[(s, c)]["freq_acc"] for s in data_dict])
        for c in ch_configs
    ])
    mean_spat = np.array([
        np.mean([all_results[(s, c)]["spatial_acc"] for s in data_dict])
        for c in ch_configs
    ])
    std_spat = np.array([
        np.std([all_results[(s, c)]["spatial_acc"] for s in data_dict])
        for c in ch_configs
    ])

    x = np.arange(len(ch_configs))
    w = 0.3
    ax.bar(
        x - w / 2, mean_freq, w, yerr=std_freq, capsize=4,
        color="#3b82f6", alpha=0.7, label="Frequency task (40-class)",
    )
    ax.bar(
        x + w / 2, mean_spat, w, yerr=std_spat, capsize=4,
        color="#e63946", alpha=0.7, label="Spatial task (5-class)",
    )
    for i in range(len(ch_configs)):
        ax.text(
            x[i] - w / 2, mean_freq[i] + std_freq[i] + 0.01,
            f"{mean_freq[i]:.1%}", ha="center", fontsize=8, fontweight="bold",
        )
        ax.text(
            x[i] + w / 2, mean_spat[i] + std_spat[i] + 0.01,
            f"{mean_spat[i]:.1%}", ha="center", fontsize=8, fontweight="bold",
        )

    # Annotate gains from 9ch to 66ch
    freq_gain = (mean_freq[-1] - mean_freq[0]) * 100
    spat_gain = (mean_spat[-1] - mean_spat[0]) * 100
    ax.annotate(
        f"Freq gain: +{freq_gain:.1f}%",
        xy=(2, mean_freq[-1]), xytext=(2.3, mean_freq[-1] - 0.08),
        fontsize=9, color="#3b82f6", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#3b82f6"),
    )
    ax.annotate(
        f"Spatial gain: +{spat_gain:.1f}%",
        xy=(2, mean_spat[-1]), xytext=(2.3, mean_spat[-1] + 0.05),
        fontsize=9, color="#e63946", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#e63946"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({len(CHANNEL_SETS[c])} ch)" for c in ch_configs])
    ax.set_ylabel("Classification accuracy")
    ax.set_title(
        "Mean accuracy across subjects\n"
        "Paper finding: HD-EEG benefits spatial >> frequency discrimination"
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.15, axis="y")
    ax.set_ylim(0, 1.1)

    _save(fig, FIG_DIR / "fig_paper_guided_B2_mean_sensitivity.png")

    print(f"\n  Part B completed in {time.time() - t0_all:.1f}s")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Part C: Pairwise Position Discriminability Matrix
# ═══════════════════════════════════════════════════════════════════════════════

def part_c(data_dict: dict, win_samp: int):
    """Build 5x5 pairwise position discriminability matrix.

    For each pair of positions (10 pairs from C(5,2)):
    - For each frequency, do 2-class classification across blocks (LOBO)
    - Average accuracy across 40 frequencies
    """
    print("\n" + "=" * 70)
    print("  Part C: Pairwise Position Discriminability Matrix")
    print("=" * 70)
    t0_all = time.time()

    ch_configs = ["9ch", "32ch", "66ch"]
    all_matrices = {}  # (subj, ch_name) -> (5, 5) matrix

    for subj, data in data_dict.items():
        n_blocks = data.shape[3]
        print(f"\n  Processing {subj}...")

        for ch_name in ch_configs:
            ch_idx = CHANNEL_SETS[ch_name]
            n_ch = len(ch_idx)
            feat_dim = n_ch * win_samp
            t0 = time.time()

            disc_matrix = np.eye(N_POSITIONS)  # diagonal = 1.0
            pair_correct = np.zeros((N_POSITIONS, N_POSITIONS))
            pair_total = np.zeros((N_POSITIONS, N_POSITIONS))

            for blk in range(n_blocks):
                templates_200 = _build_templates_lobo(data, ch_idx, win_samp, blk)

                for fi in range(N_BASE_FREQS):
                    for pos_a in range(N_POSITIONS):
                        for pos_b in range(pos_a + 1, N_POSITIONS):
                            tidx_a = fi * N_POSITIONS + pos_a
                            tidx_b = fi * N_POSITIONS + pos_b
                            tmpl_a = templates_200[tidx_a]
                            tmpl_b = templates_200[tidx_b]

                            for tidx_test, true_label in [
                                (tidx_a, 0),
                                (tidx_b, 1),
                            ]:
                                trial = _get_test_trial(
                                    data, ch_idx, win_samp, tidx_test, blk,
                                )
                                r_a = np.dot(tmpl_a, trial)
                                r_b = np.dot(tmpl_b, trial)
                                pred = 0 if r_a >= r_b else 1
                                if pred == true_label:
                                    pair_correct[pos_a, pos_b] += 1
                                pair_total[pos_a, pos_b] += 1

            for pos_a in range(N_POSITIONS):
                for pos_b in range(pos_a + 1, N_POSITIONS):
                    acc = pair_correct[pos_a, pos_b] / max(pair_total[pos_a, pos_b], 1)
                    disc_matrix[pos_a, pos_b] = acc
                    disc_matrix[pos_b, pos_a] = acc

            all_matrices[(subj, ch_name)] = disc_matrix
            print(f"    {ch_name}: mean off-diag acc = {disc_matrix[np.triu_indices(5, k=1)].mean():.4f}  ({time.time() - t0:.1f}s)")

    # --- Physical distance for comparison ---
    # Positions on a circle: R=0, D=270, L=180, U=90, C=center
    # Distance between points on circle: angular distance
    # C is at center, distance to any edge point ~ radius
    def angular_dist(p_a, p_b):
        if p_a == 4 or p_b == 4:
            return 1.0  # normalized distance center-to-edge
        a = POSITION_ANGLES_DEG[p_a]
        b = POSITION_ANGLES_DEG[p_b]
        d = abs(a - b)
        if d > 180:
            d = 360 - d
        return d / 180.0  # normalize to [0, 2] -> [0, 1]

    dist_matrix = np.zeros((N_POSITIONS, N_POSITIONS))
    for i in range(N_POSITIONS):
        for j in range(N_POSITIONS):
            dist_matrix[i, j] = angular_dist(i, j)

    # --- Fig C1: Discriminability matrices ---
    subjects_plot = [s for s in data_dict]
    n_subj = len(subjects_plot)

    fig, axes = plt.subplots(
        n_subj, len(ch_configs) + 1,
        figsize=(5 * (len(ch_configs) + 1), 5 * n_subj),
        constrained_layout=True,
    )
    if n_subj == 1:
        axes = axes[np.newaxis, :]

    pos_labels = [POSITION_SHORT[p] for p in range(N_POSITIONS)]

    for row, subj in enumerate(subjects_plot):
        for col, ch_name in enumerate(ch_configs):
            ax = axes[row, col]
            mat = all_matrices[(subj, ch_name)]
            im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="YlOrRd", aspect="auto")
            for i in range(N_POSITIONS):
                for j in range(N_POSITIONS):
                    txt = f"{mat[i, j]:.2f}" if i != j else "—"
                    ax.text(
                        j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if mat[i, j] > 0.85 else "black",
                    )
            ax.set_xticks(range(N_POSITIONS))
            ax.set_xticklabels(pos_labels)
            ax.set_yticks(range(N_POSITIONS))
            ax.set_yticklabels(pos_labels)
            if col == 0:
                ax.set_ylabel(f"{subj}\nPosition")
            if row == 0:
                ax.set_title(f"{ch_name}", fontsize=11)
            if row == n_subj - 1:
                ax.set_xlabel("Position")

        # Last column: physical distance matrix
        ax = axes[row, -1]
        im2 = ax.imshow(dist_matrix, vmin=0, vmax=1.0, cmap="Blues", aspect="auto")
        for i in range(N_POSITIONS):
            for j in range(N_POSITIONS):
                ax.text(
                    j, i, f"{dist_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=9,
                )
        ax.set_xticks(range(N_POSITIONS))
        ax.set_xticklabels(pos_labels)
        ax.set_yticks(range(N_POSITIONS))
        ax.set_yticklabels(pos_labels)
        if row == 0:
            ax.set_title("Physical distance\n(normalized)", fontsize=11)
        if row == n_subj - 1:
            ax.set_xlabel("Position")

    fig.suptitle(
        "Part C: Pairwise position discriminability (2-class accuracy)\n"
        "LOBO CV, 500ms, template matching — paper: farthest pairs best",
        fontsize=13, fontweight="bold",
    )
    _save(fig, FIG_DIR / "fig_paper_guided_C1_pairwise_discriminability.png")

    # --- Fig C2: Distance-accuracy correlation ---
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    for subj in subjects_plot:
        for ch_name in ch_configs:
            mat = all_matrices[(subj, ch_name)]
            dists = []
            accs = []
            for i in range(N_POSITIONS):
                for j in range(i + 1, N_POSITIONS):
                    dists.append(dist_matrix[i, j])
                    accs.append(mat[i, j])

            marker = SUBJ_MARKERS.get(subj, "o")
            color = SUBJ_COLORS.get(subj, "gray")
            linestyle = {"9ch": ":", "32ch": "--", "66ch": "-"}[ch_name]
            ax.scatter(
                dists, accs, marker=marker, color=color, s=60, alpha=0.7,
                edgecolors="white", linewidth=0.5,
            )

    # Add labeled pair markers for reference (66ch, S1)
    if ("S1", "66ch") in all_matrices:
        mat = all_matrices[("S1", "66ch")]
        for i in range(N_POSITIONS):
            for j in range(i + 1, N_POSITIONS):
                d = dist_matrix[i, j]
                a = mat[i, j]
                label = f"{POSITION_SHORT[i]}-{POSITION_SHORT[j]}"
                ax.annotate(
                    label, (d, a), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.6,
                )

    ax.set_xlabel("Normalized physical distance between positions")
    ax.set_ylabel("2-class discrimination accuracy")
    ax.set_title(
        "Distance-accuracy correlation\n"
        "Paper finding: maximum physical distance -> best discriminability"
    )
    ax.grid(alpha=0.15)
    ax.set_ylim(0.45, 1.05)

    # Create legend manually
    from matplotlib.lines import Line2D
    legend_elements = []
    for subj in subjects_plot:
        marker = SUBJ_MARKERS.get(subj, "o")
        color = SUBJ_COLORS.get(subj, "gray")
        legend_elements.append(
            Line2D([0], [0], marker=marker, color=color, linestyle="None",
                   markersize=8, label=subj)
        )
    ax.legend(handles=legend_elements, fontsize=9)

    _save(fig, FIG_DIR / "fig_paper_guided_C2_distance_accuracy.png")

    # --- Print best/worst pairs ---
    print("\n  Best and worst position pairs (averaged across subjects):")
    for ch_name in ch_configs:
        print(f"    {ch_name}:")
        pair_accs = {}
        for i in range(N_POSITIONS):
            for j in range(i + 1, N_POSITIONS):
                label = f"{POSITION_SHORT[i]}-{POSITION_SHORT[j]}"
                vals = [
                    all_matrices[(s, ch_name)][i, j]
                    for s in subjects_plot
                ]
                pair_accs[label] = np.mean(vals)
        sorted_pairs = sorted(pair_accs.items(), key=lambda x: x[1], reverse=True)
        for label, acc in sorted_pairs:
            print(f"      {label}: {acc:.4f}")

    print(f"\n  Part C completed in {time.time() - t0_all:.1f}s")
    return all_matrices


# ═══════════════════════════════════════════════════════════════════════════════
# Part D: Per-Frequency Spatial Accuracy Variation
# ═══════════════════════════════════════════════════════════════════════════════

def part_d(data_dict: dict, win_samp: int):
    """Per-frequency 5-position classification accuracy.

    The paper's Fig.4D shows spatial accuracy varies across the 40 flicker
    positions — center flickers may have higher accuracy due to neighboring
    flickers providing additional discrimination cues.
    """
    print("\n" + "=" * 70)
    print("  Part D: Per-Frequency Spatial Accuracy Variation")
    print("=" * 70)
    t0_all = time.time()

    ch_configs = ["9ch", "32ch", "66ch"]
    all_per_freq = {}  # (subj, ch_name) -> (40,) accuracy per frequency

    for subj, data in data_dict.items():
        n_blocks = data.shape[3]
        print(f"\n  Processing {subj}...")

        for ch_name in ch_configs:
            ch_idx = CHANNEL_SETS[ch_name]
            n_ch = len(ch_idx)
            feat_dim = n_ch * win_samp
            t0 = time.time()

            correct_per_freq = np.zeros(N_BASE_FREQS)
            total_per_freq = np.zeros(N_BASE_FREQS)

            for blk in range(n_blocks):
                # Build templates for all 200 targets
                templates_200 = _build_templates_lobo(data, ch_idx, win_samp, blk)

                for fi in range(N_BASE_FREQS):
                    # 5-class spatial templates for this frequency
                    tidx_list = [fi * N_POSITIONS + p for p in range(N_POSITIONS)]
                    spatial_templates = templates_200[tidx_list]  # (5, feat_dim)

                    for pos in range(N_POSITIONS):
                        tidx = fi * N_POSITIONS + pos
                        trial = _get_test_trial(data, ch_idx, win_samp, tidx, blk)

                        corrs = spatial_templates @ trial  # (5,)
                        pred = np.argmax(corrs)
                        if pred == pos:
                            correct_per_freq[fi] += 1
                        total_per_freq[fi] += 1

            per_freq_acc = correct_per_freq / np.clip(total_per_freq, 1, None)
            all_per_freq[(subj, ch_name)] = per_freq_acc

            print(
                f"    {ch_name}: mean={per_freq_acc.mean():.4f}  "
                f"min={per_freq_acc.min():.4f}  max={per_freq_acc.max():.4f}"
                f"  ({time.time() - t0:.1f}s)"
            )

    # --- Fig D1: 8x5 heatmap matching flicker layout ---
    # BASE_FREQS layout: 8 integer bases x 5 fractional offsets
    # Index mapping: fi = col * 8 + row where
    #   col 0: .0, col 1: .2, col 2: .4, col 3: .6, col 4: .8
    #   row 0-7: 8, 9, 10, ..., 15 Hz

    for subj in data_dict:
        fig, axes = plt.subplots(
            1, len(ch_configs), figsize=(6 * len(ch_configs), 8),
            constrained_layout=True,
        )

        for ax_i, ch_name in enumerate(ch_configs):
            acc = all_per_freq[(subj, ch_name)]
            # Reshape into 8 rows (base integers) x 5 cols (fractional offsets)
            heatmap = np.zeros((8, 5))
            for fi in range(N_BASE_FREQS):
                freq = BASE_FREQS[fi]
                base_int = int(freq)
                frac = round((freq - base_int) * 10) / 10
                row = base_int - 8  # 8->0, 9->1, ..., 15->7
                col = int(round(frac / 0.2))  # 0.0->0, 0.2->1, ...0.8->4
                if 0 <= row < 8 and 0 <= col < 5:
                    heatmap[row, col] = acc[fi]

            ax = axes[ax_i]
            im = ax.imshow(
                heatmap, vmin=0.15, vmax=0.65, cmap="YlOrRd", aspect="auto",
            )
            for r in range(8):
                for c in range(5):
                    ax.text(
                        c, r, f"{heatmap[r, c]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if heatmap[r, c] > 0.5 else "black",
                    )
            ax.set_xticks(range(5))
            ax.set_xticklabels(["+0.0", "+0.2", "+0.4", "+0.6", "+0.8"])
            ax.set_yticks(range(8))
            ax.set_yticklabels([f"{b} Hz" for b in range(8, 16)])
            ax.set_xlabel("Fractional offset (Hz)")
            if ax_i == 0:
                ax.set_ylabel("Base frequency (Hz)")
            ax.set_title(f"{ch_name}", fontsize=11)

        plt.colorbar(im, ax=axes, shrink=0.6, label="5-position accuracy")
        fig.suptitle(
            f"Part D1: Per-frequency 5-position spatial accuracy — {subj}\n"
            f"500ms, LOBO CV, template matching",
            fontsize=13, fontweight="bold",
        )
        _save(fig, FIG_DIR / f"fig_paper_guided_D1_perfreq_heatmap_{subj}.png")

    # --- Fig D2: Line plot across frequencies for all subjects ---
    fig, axes = plt.subplots(
        len(ch_configs), 1, figsize=(16, 4 * len(ch_configs)),
        constrained_layout=True, sharex=True,
    )

    for ax_i, ch_name in enumerate(ch_configs):
        ax = axes[ax_i]
        for subj in data_dict:
            acc = all_per_freq[(subj, ch_name)]
            color = SUBJ_COLORS.get(subj, "gray")
            ax.plot(
                BASE_FREQS, acc, "o-", color=color, markersize=4,
                linewidth=1.2, alpha=0.7, label=subj,
            )
        ax.set_ylabel("5-position accuracy")
        ax.set_title(f"{ch_name}", fontsize=11)
        ax.legend(fontsize=8, ncol=4)
        ax.grid(alpha=0.15)
        ax.set_ylim(0, 0.85)
        # Mark integer frequencies
        for f_int in range(8, 16):
            ax.axvline(f_int, color="gray", linewidth=0.3, alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(
        "Part D2: Spatial accuracy across 40 frequencies\n"
        "500ms, LOBO CV, template matching",
        fontsize=13, fontweight="bold",
    )
    _save(fig, FIG_DIR / "fig_paper_guided_D2_perfreq_lines.png")

    # --- Fig D3: Mean across subjects, per ch config ---
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

    ch_colors_map = {"9ch": "#94a3b8", "32ch": "#3b82f6", "66ch": "#16a34a"}

    for ch_name in ch_configs:
        accs_all = np.array([
            all_per_freq[(s, ch_name)] for s in data_dict
        ])  # (n_subj, 40)
        mean_acc = accs_all.mean(axis=0)
        std_acc = accs_all.std(axis=0)
        color = ch_colors_map[ch_name]
        ax.plot(
            BASE_FREQS, mean_acc, "o-", color=color, markersize=4,
            linewidth=1.5, label=f"{ch_name} (mean={mean_acc.mean():.1%})",
        )
        ax.fill_between(
            BASE_FREQS, mean_acc - std_acc, mean_acc + std_acc,
            color=color, alpha=0.15,
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("5-position accuracy (mean +/- std)")
    ax.set_title(
        "Mean per-frequency spatial accuracy across subjects\n"
        "Variation reveals which flicker positions are easier for spatial discrimination"
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.15)
    for f_int in range(8, 16):
        ax.axvline(f_int, color="gray", linewidth=0.3, alpha=0.3)
    ax.set_ylim(0, 0.85)

    _save(fig, FIG_DIR / "fig_paper_guided_D3_perfreq_mean.png")

    # --- Print frequency-range statistics ---
    print("\n  Spatial accuracy by frequency range (66ch, mean across subjects):")
    accs_66 = np.array([all_per_freq[(s, "66ch")] for s in data_dict])
    mean_66 = accs_66.mean(axis=0)
    # Group by integer base
    for base_int in range(8, 16):
        fi_group = [
            fi for fi in range(N_BASE_FREQS)
            if int(BASE_FREQS[fi]) == base_int
        ]
        group_acc = mean_66[fi_group].mean()
        print(f"    {base_int}.x Hz: {group_acc:.4f}")

    print(f"\n  Part D completed in {time.time() - t0_all:.1f}s")
    return all_per_freq


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("  Paper-guided spatial analysis of HD200 SSVEP dataset")
    print("  Ming et al. 2026 findings reproduction")
    print("=" * 70)

    win_samp = 125  # 500ms at 250 Hz
    refs = make_references(win_samp)

    # Load subjects one at a time to manage memory
    data_dict = {}
    for subj in SUBJECTS_DEFAULT:
        data = load_subject_data(subj)
        if data is not None:
            data_dict[subj] = data

    if not data_dict:
        print("  ERROR: No subject data found. Exiting.")
        return

    print(f"\n  Loaded {len(data_dict)} subjects: {list(data_dict.keys())}")
    print(f"  Window: {win_samp} samples = {win_samp / FS * 1000:.0f} ms")
    print(f"  Filter bank: FB0 (broadband [6, 90] Hz)")

    # Part A: Position-pair SNR topography asymmetry
    part_a(data_dict, refs, win_samp)

    # Part B: Spatial vs frequency channel-density sensitivity
    results_b = part_b(data_dict, win_samp)

    # Part C: Pairwise position discriminability matrix
    results_c = part_c(data_dict, win_samp)

    # Part D: Per-frequency spatial accuracy variation
    results_d = part_d(data_dict, win_samp)

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    # Paper claims vs our findings
    print("\n  Paper claim vs reproduction:")

    # B: channel density sensitivity
    if results_b:
        for subj in data_dict:
            if (subj, "9ch") in results_b and (subj, "66ch") in results_b:
                freq_gain = (
                    results_b[(subj, "66ch")]["freq_acc"]
                    - results_b[(subj, "9ch")]["freq_acc"]
                ) * 100
                spat_gain = (
                    results_b[(subj, "66ch")]["spatial_acc"]
                    - results_b[(subj, "9ch")]["spatial_acc"]
                ) * 100
                ratio = spat_gain / freq_gain if abs(freq_gain) > 0.01 else float("inf")
                print(
                    f"    {subj}: freq_gain={freq_gain:+.2f}%  "
                    f"spatial_gain={spat_gain:+.2f}%  "
                    f"ratio={ratio:.1f}x"
                )
        print("    Paper: freq +1.32%, spatial +15.53%, ratio ~11.8x")

    # C: best/worst pairs
    if results_c:
        print("\n  Pairwise discriminability (66ch, mean across subjects):")
        pair_means = {}
        for i in range(N_POSITIONS):
            for j in range(i + 1, N_POSITIONS):
                label = f"{POSITION_SHORT[i]}-{POSITION_SHORT[j]}"
                vals = [
                    results_c[(s, "66ch")][i, j]
                    for s in data_dict
                    if (s, "66ch") in results_c
                ]
                if vals:
                    pair_means[label] = np.mean(vals)
        sorted_pm = sorted(pair_means.items(), key=lambda x: x[1], reverse=True)
        for label, acc in sorted_pm:
            print(f"    {label}: {acc:.4f}")
        print("    Paper: best pairs are farthest (D-U, R-L); R+L+U best 3-point combo")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  All figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
