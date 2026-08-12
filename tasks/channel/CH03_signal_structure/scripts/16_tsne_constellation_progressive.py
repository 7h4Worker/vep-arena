"""Progressive constellation and t-SNE analysis of HD200 spatial encoding.

Shows how the encoding space expands from 40 targets (FDM only) to 200 targets
(FDM + SDM) using PCA constellation diagrams and t-SNE embeddings.

Key question: the spatial information IS there — how does the constellation
geometry change as spatial multiplexing is added?

Parts:
  A. Progressive PCA constellation (40 → 80 → 120 → 160 → 200)
  B. t-SNE of 200-target templates (frequency coloring + position shape)
  C. t-SNE of individual trials with noise clouds
  D. Within-frequency spatial sub-structure zoom

Usage
-----
    .venv/Scripts/python.exe scripts/16_tsne_constellation_progressive.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ═══════════════════════════════════════════════════════════════════════════════
# Constants (consistent with script 15)
# ═══════════════════════════════════════════════════════════════════════════════

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

POSITION_NAMES = {0: "R", 1: "D", 2: "L", 3: "U", 4: "C"}
POSITION_MARKERS = {0: "o", 1: "v", 2: "s", 3: "^", 4: "D"}

CHANNEL_SETS = {
    "9ch": np.array([21, 27, 28, 29, 31, 32, 33, 59, 63]),
    "66ch": np.arange(66),
}

# Paper's optimal fixation subsets (Table 1)
FIXATION_SUBSETS = {
    40:  [3],           # UP only
    80:  [1, 3],        # DOWN + UP
    120: [0, 2, 3],     # RIGHT + LEFT + UP
    160: [0, 1, 2, 3],  # R + D + L + U
    200: [0, 1, 2, 3, 4],  # all
}

# Frequency colormap: 40 frequencies mapped to continuous rainbow
FREQ_CMAP = plt.cm.nipy_spectral(np.linspace(0.05, 0.95, N_BASE_FREQS))


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {path.name}")


def load_subject_fb0(subj: str):
    cache_path = CACHE_DIR / f"filtered_{subj}_200target_66ch_18blocks_float32.npy"
    if not cache_path.exists():
        print(f"  SKIP {subj}: not found")
        return None
    data_full = np.load(str(cache_path), mmap_mode="r")
    return np.asarray(data_full[:, :, :, :, 0])  # (66, 185, 200, 18) FB0


def build_templates(data, ch_idx, win_samp):
    """Build mean templates for all 200 targets across all 18 blocks.
    Returns (200, n_ch * win_samp) zero-mean unit-norm.
    """
    n_ch = len(ch_idx)
    feat_dim = n_ch * win_samp
    templates = np.zeros((N_TARGETS, feat_dim))
    for tidx in range(N_TARGETS):
        block_sum = np.zeros(feat_dim)
        for blk in range(N_BLOCKS):
            seg = data[ch_idx, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, tidx, blk]
            block_sum += seg.flatten()
        templates[tidx] = block_sum / N_BLOCKS
    templates -= templates.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(templates, axis=-1, keepdims=True)
    templates /= np.clip(norms, 1e-12, None)
    return templates


def get_trial_vectors(data, ch_idx, win_samp, target_indices, block_indices):
    """Extract individual trial vectors. Returns (n_trials, feat_dim)."""
    n_ch = len(ch_idx)
    feat_dim = n_ch * win_samp
    trials = []
    labels_freq = []
    labels_pos = []
    for tidx in target_indices:
        for blk in block_indices:
            seg = data[ch_idx, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, tidx, blk]
            vec = seg.flatten().astype(np.float64)
            vec -= vec.mean()
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec /= norm
            trials.append(vec)
            labels_freq.append(TARGET_TO_BASE_FREQ[tidx])
            labels_pos.append(TARGET_TO_POSITION[tidx])
    return np.array(trials), np.array(labels_freq), np.array(labels_pos)


# ═══════════════════════════════════════════════════════════════════════════════
# Part A: Progressive PCA constellation (40 → 200)
# ═══════════════════════════════════════════════════════════════════════════════

def part_a(templates_66, subj: str):
    """Show how the constellation expands as spatial positions are added."""
    print("\n" + "=" * 70)
    print("  Part A: Progressive PCA Constellation")
    print("=" * 70)

    target_counts = [40, 80, 120, 160, 200]

    # --- Fig A1: Side-by-side PCA at each target count ---
    fig, axes = plt.subplots(1, 5, figsize=(28, 5.5), constrained_layout=True)

    for ax_i, n_targets in enumerate(target_counts):
        positions_used = FIXATION_SUBSETS[n_targets]
        tidx_mask = np.isin(TARGET_TO_POSITION, positions_used)
        tidx_selected = np.where(tidx_mask)[0]
        subset = templates_66[tidx_selected]

        pca = PCA(n_components=2)
        proj = pca.fit_transform(subset)

        ax = axes[ax_i]
        freq_labels = TARGET_TO_BASE_FREQ[tidx_selected]
        pos_labels = TARGET_TO_POSITION[tidx_selected]

        for i, tidx in enumerate(tidx_selected):
            fi = freq_labels[i]
            pos = pos_labels[i]
            ax.scatter(
                proj[i, 0], proj[i, 1],
                c=[FREQ_CMAP[fi]],
                marker=POSITION_MARKERS[pos],
                s=30, alpha=0.8, edgecolors="none",
            )

        pos_str = "+".join(POSITION_NAMES[p] for p in positions_used)
        var_exp = pca.explained_variance_ratio_[:2].sum() * 100
        ax.set_title(
            f"{n_targets} targets\n"
            f"positions: {pos_str}\n"
            f"PC1+PC2: {var_exp:.0f}%",
            fontsize=10,
        )
        ax.set_xlabel("PC1")
        if ax_i == 0:
            ax.set_ylabel("PC2")
        ax.grid(alpha=0.1)

    # Legend for position markers
    from matplotlib.lines import Line2D
    legend_pos = [
        Line2D([0], [0], marker=POSITION_MARKERS[p], color="gray",
               linestyle="None", markersize=7, label=POSITION_NAMES[p])
        for p in range(N_POSITIONS)
    ]
    axes[-1].legend(handles=legend_pos, title="Position", fontsize=7,
                    loc="upper right")

    fig.suptitle(
        f"Progressive constellation: {subj}, 66ch, 500ms\n"
        f"Each point = one target template; color = frequency, shape = fixation position",
        fontsize=12, fontweight="bold",
    )
    _save(fig, FIG_DIR / f"fig_tsne_A1_progressive_constellation_{subj}.png")

    # --- Fig A2: 40-target vs 200-target direct comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True)

    # Panel 1: 40-target (frequency centroids = average of 5 positions)
    freq_centroids = np.zeros((N_BASE_FREQS, templates_66.shape[1]))
    for fi in range(N_BASE_FREQS):
        tidx_fi = np.where(TARGET_TO_BASE_FREQ == fi)[0]
        freq_centroids[fi] = templates_66[tidx_fi].mean(axis=0)
    freq_centroids -= freq_centroids.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(freq_centroids, axis=-1, keepdims=True)
    freq_centroids /= np.clip(norms, 1e-12, None)

    pca_40 = PCA(n_components=2)
    proj_40 = pca_40.fit_transform(freq_centroids)
    ax = axes[0]
    for fi in range(N_BASE_FREQS):
        ax.scatter(proj_40[fi, 0], proj_40[fi, 1],
                   c=[FREQ_CMAP[fi]], s=80, edgecolors="white", linewidth=0.5,
                   zorder=3)
        ax.annotate(f"{BASE_FREQS[fi]:.1f}", (proj_40[fi, 0], proj_40[fi, 1]),
                    fontsize=5, alpha=0.6, textcoords="offset points", xytext=(3, 3))
    var40 = pca_40.explained_variance_ratio_[:2].sum() * 100
    ax.set_title(f"40 targets (FDM only)\nPC1+2: {var40:.0f}%", fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.1)

    # Panel 2: 200-target (all 5 positions)
    pca_200 = PCA(n_components=2)
    proj_200 = pca_200.fit_transform(templates_66)
    ax = axes[1]
    for tidx in range(N_TARGETS):
        fi = TARGET_TO_BASE_FREQ[tidx]
        pos = TARGET_TO_POSITION[tidx]
        ax.scatter(proj_200[tidx, 0], proj_200[tidx, 1],
                   c=[FREQ_CMAP[fi]], marker=POSITION_MARKERS[pos],
                   s=25, alpha=0.7, edgecolors="none")
    var200 = pca_200.explained_variance_ratio_[:2].sum() * 100
    ax.set_title(f"200 targets (FDM + SDM)\nPC1+2: {var200:.0f}%", fontsize=11)
    ax.set_xlabel("PC1")
    ax.grid(alpha=0.1)
    ax.legend(handles=legend_pos, title="Position", fontsize=7, loc="upper right")

    # Panel 3: Zoom into one frequency cluster showing 5 positions
    zoom_fi = 0  # 8.0 Hz
    tidx_zoom = np.where(TARGET_TO_BASE_FREQ == zoom_fi)[0]
    ax = axes[2]
    for tidx in tidx_zoom:
        pos = TARGET_TO_POSITION[tidx]
        ax.scatter(proj_200[tidx, 0], proj_200[tidx, 1],
                   c=[FREQ_CMAP[zoom_fi]], marker=POSITION_MARKERS[pos],
                   s=200, edgecolors="black", linewidth=1.0, zorder=5)
        ax.annotate(POSITION_NAMES[pos],
                    (proj_200[tidx, 0], proj_200[tidx, 1]),
                    fontsize=10, fontweight="bold",
                    textcoords="offset points", xytext=(8, 5))
    # Also plot all other targets faintly
    other_mask = TARGET_TO_BASE_FREQ != zoom_fi
    ax.scatter(proj_200[other_mask, 0], proj_200[other_mask, 1],
               c="lightgray", s=8, alpha=0.3, edgecolors="none")
    ax.set_title(
        f"Zoom: {BASE_FREQS[zoom_fi]:.1f} Hz cluster\n"
        f"5 spatial sub-points within one frequency",
        fontsize=11,
    )
    ax.set_xlabel("PC1")
    ax.grid(alpha=0.1)

    fig.suptitle(
        f"Encoding space expansion: FDM (40) → FDM+SDM (200) — {subj}, 66ch",
        fontsize=13, fontweight="bold",
    )
    _save(fig, FIG_DIR / f"fig_tsne_A2_40vs200_constellation_{subj}.png")

    return pca_200, proj_200


# ═══════════════════════════════════════════════════════════════════════════════
# Part B: t-SNE of 200-target templates
# ═══════════════════════════════════════════════════════════════════════════════

def part_b(templates_66, templates_9, subj: str):
    """t-SNE embedding of 200-target templates, colored by freq and position."""
    print("\n" + "=" * 70)
    print("  Part B: t-SNE of 200-target Templates")
    print("=" * 70)

    for ch_label, tmpl in [("66ch", templates_66), ("9ch", templates_9)]:
        t0 = time.time()
        perplexity = min(30, N_TARGETS - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                     max_iter=2000, learning_rate="auto", init="pca")
        proj = tsne.fit_transform(tmpl)
        print(f"  t-SNE {ch_label}: {time.time() - t0:.1f}s")

        fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)

        # Panel 1: color by frequency
        ax = axes[0]
        for tidx in range(N_TARGETS):
            fi = TARGET_TO_BASE_FREQ[tidx]
            ax.scatter(proj[tidx, 0], proj[tidx, 1],
                       c=[FREQ_CMAP[fi]], s=30, alpha=0.8, edgecolors="none")
        ax.set_title("Color = frequency (40 classes)", fontsize=11)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.1)

        # Panel 2: color by position
        pos_colors_map = {
            0: "#e63946", 1: "#457b9d", 2: "#2a9d8f", 3: "#e9c46a", 4: "#6c757d"
        }
        ax = axes[1]
        for tidx in range(N_TARGETS):
            pos = TARGET_TO_POSITION[tidx]
            ax.scatter(proj[tidx, 0], proj[tidx, 1],
                       c=pos_colors_map[pos],
                       marker=POSITION_MARKERS[pos],
                       s=40, alpha=0.8, edgecolors="none")
        from matplotlib.lines import Line2D
        leg = [Line2D([0], [0], marker=POSITION_MARKERS[p], color=pos_colors_map[p],
                       linestyle="None", markersize=8, label=POSITION_NAMES[p])
               for p in range(N_POSITIONS)]
        ax.legend(handles=leg, fontsize=9)
        ax.set_title("Color+shape = position (5 classes)", fontsize=11)
        ax.set_xlabel("t-SNE 1")
        ax.grid(alpha=0.1)

        # Panel 3: both — frequency color + position marker
        ax = axes[2]
        for tidx in range(N_TARGETS):
            fi = TARGET_TO_BASE_FREQ[tidx]
            pos = TARGET_TO_POSITION[tidx]
            ax.scatter(proj[tidx, 0], proj[tidx, 1],
                       c=[FREQ_CMAP[fi]],
                       marker=POSITION_MARKERS[pos],
                       s=35, alpha=0.8, edgecolors="none")
        ax.set_title("Freq (color) + Position (shape)", fontsize=11)
        ax.set_xlabel("t-SNE 1")
        ax.grid(alpha=0.1)
        ax.legend(handles=leg, fontsize=7, loc="upper right")

        fig.suptitle(
            f"t-SNE of 200-target templates — {subj}, {ch_label}, 500ms\n"
            f"Do frequency clusters split by position? (perplexity={perplexity})",
            fontsize=12, fontweight="bold",
        )
        _save(fig, FIG_DIR / f"fig_tsne_B1_templates_{ch_label}_{subj}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Part C: t-SNE of individual trials with noise clouds
# ═══════════════════════════════════════════════════════════════════════════════

def part_c(data, subj: str, win_samp: int):
    """t-SNE of individual trials for a subset of frequencies, showing
    how trial scatter relates to the centroid structure.
    """
    print("\n" + "=" * 70)
    print("  Part C: t-SNE of Individual Trials (noise clouds)")
    print("=" * 70)

    ch_idx = CHANNEL_SETS["66ch"]

    # Select 5 representative frequencies spread across the range
    repr_freq_indices = [0, 8, 16, 24, 32]  # 8.0, 9.0, 10.0, 11.0, 12.0 Hz
    repr_freq_labels = [f"{BASE_FREQS[fi]:.1f}" for fi in repr_freq_indices]

    # Get target indices for these frequencies (5 freq × 5 pos = 25 targets)
    target_subset = []
    for fi in repr_freq_indices:
        for pos in range(N_POSITIONS):
            target_subset.append(fi * N_POSITIONS + pos)
    target_subset = np.array(target_subset)

    # Use 6 blocks for trial scatter (to keep reasonable count: 25 targets × 6 = 150 trials)
    block_subset = np.arange(0, 18, 3)  # blocks 0, 3, 6, 9, 12, 15

    print(f"  Extracting trials: {len(target_subset)} targets × {len(block_subset)} blocks = {len(target_subset) * len(block_subset)} trials")
    trial_vecs, trial_freqs, trial_pos = get_trial_vectors(
        data, ch_idx, win_samp, target_subset, block_subset,
    )

    # Also get centroids for these targets
    templates = build_templates(data, ch_idx, win_samp)
    centroid_vecs = templates[target_subset]
    centroid_freqs = TARGET_TO_BASE_FREQ[target_subset]
    centroid_pos = TARGET_TO_POSITION[target_subset]

    # Combine for joint t-SNE
    all_vecs = np.vstack([centroid_vecs, trial_vecs])
    n_centroids = len(centroid_vecs)

    t0 = time.time()
    perplexity = min(30, len(all_vecs) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                 max_iter=2000, learning_rate="auto", init="pca")
    proj = tsne.fit_transform(all_vecs)
    print(f"  t-SNE: {time.time() - t0:.1f}s  ({len(all_vecs)} points)")

    proj_c = proj[:n_centroids]
    proj_t = proj[n_centroids:]

    # --- Fig C1: Trials colored by frequency ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)

    # Use a discrete colormap for 5 frequencies
    freq_colors = plt.cm.Set1(np.linspace(0, 0.6, len(repr_freq_indices)))
    freq_color_map = {fi: freq_colors[i] for i, fi in enumerate(repr_freq_indices)}

    # Panel 1: trials colored by frequency
    ax = axes[0]
    for i in range(len(trial_vecs)):
        fi = trial_freqs[i]
        ax.scatter(proj_t[i, 0], proj_t[i, 1],
                   c=[freq_color_map[fi]], s=12, alpha=0.4, edgecolors="none")
    for i in range(n_centroids):
        fi = centroid_freqs[i]
        ax.scatter(proj_c[i, 0], proj_c[i, 1],
                   c=[freq_color_map[fi]], s=100, edgecolors="black",
                   linewidth=1.5, zorder=5, marker="*")
    from matplotlib.lines import Line2D
    leg_freq = [
        Line2D([0], [0], marker="o", color=freq_color_map[fi], linestyle="None",
               markersize=8, label=f"{BASE_FREQS[fi]:.1f} Hz")
        for fi in repr_freq_indices
    ]
    ax.legend(handles=leg_freq, fontsize=8)
    ax.set_title("Trials by frequency\n(large * = centroid)", fontsize=11)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.1)

    # Panel 2: trials colored by position
    pos_colors_map = {
        0: "#e63946", 1: "#457b9d", 2: "#2a9d8f", 3: "#e9c46a", 4: "#6c757d"
    }
    ax = axes[1]
    for i in range(len(trial_vecs)):
        pos = trial_pos[i]
        ax.scatter(proj_t[i, 0], proj_t[i, 1],
                   c=pos_colors_map[pos],
                   marker=POSITION_MARKERS[pos],
                   s=15, alpha=0.4, edgecolors="none")
    for i in range(n_centroids):
        pos = centroid_pos[i]
        ax.scatter(proj_c[i, 0], proj_c[i, 1],
                   c=pos_colors_map[pos],
                   marker=POSITION_MARKERS[pos],
                   s=150, edgecolors="black", linewidth=1.5, zorder=5)
    leg_pos = [
        Line2D([0], [0], marker=POSITION_MARKERS[p], color=pos_colors_map[p],
               linestyle="None", markersize=8, label=POSITION_NAMES[p])
        for p in range(N_POSITIONS)
    ]
    ax.legend(handles=leg_pos, fontsize=8)
    ax.set_title("Trials by position\n(large markers = centroids)", fontsize=11)
    ax.set_xlabel("t-SNE 1")
    ax.grid(alpha=0.1)

    # Panel 3: zoom into ONE frequency, showing 5 position clouds
    zoom_fi = repr_freq_indices[2]  # 10.0 Hz
    trial_mask = trial_freqs == zoom_fi
    centroid_mask = centroid_freqs == zoom_fi
    ax = axes[2]

    # Faint background: other frequencies
    ax.scatter(proj_t[~trial_mask, 0], proj_t[~trial_mask, 1],
               c="lightgray", s=5, alpha=0.15, edgecolors="none")

    # This frequency's trials by position
    for i in np.where(trial_mask)[0]:
        pos = trial_pos[i]
        ax.scatter(proj_t[i, 0], proj_t[i, 1],
                   c=pos_colors_map[pos],
                   marker=POSITION_MARKERS[pos],
                   s=25, alpha=0.6, edgecolors="none")
    for i in np.where(centroid_mask)[0]:
        pos = centroid_pos[i]
        ax.scatter(proj_c[i, 0], proj_c[i, 1],
                   c=pos_colors_map[pos],
                   marker=POSITION_MARKERS[pos],
                   s=200, edgecolors="black", linewidth=2, zorder=5)
        ax.annotate(POSITION_NAMES[pos],
                    (proj_c[i, 0], proj_c[i, 1]),
                    fontsize=11, fontweight="bold",
                    textcoords="offset points", xytext=(10, 5))

    ax.legend(handles=leg_pos, fontsize=8)
    ax.set_title(
        f"Zoom: {BASE_FREQS[zoom_fi]:.1f} Hz — 5 position clouds\n"
        f"Question: do noise clouds separate?",
        fontsize=11,
    )
    ax.set_xlabel("t-SNE 1")
    ax.grid(alpha=0.1)

    fig.suptitle(
        f"t-SNE of individual trials — {subj}, 66ch, 500ms\n"
        f"5 frequencies × 5 positions × {len(block_subset)} blocks = {len(trial_vecs)} trials",
        fontsize=12, fontweight="bold",
    )
    _save(fig, FIG_DIR / f"fig_tsne_C1_trial_clouds_{subj}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Part D: Within-frequency spatial sub-structure
# ═══════════════════════════════════════════════════════════════════════════════

def part_d(templates_66, subj: str):
    """For each frequency, show the 5 spatial sub-points and quantify
    inter-position distance vs intra-frequency diameter.
    """
    print("\n" + "=" * 70)
    print("  Part D: Within-frequency Spatial Sub-structure")
    print("=" * 70)

    # Compute pairwise distances in template space for each frequency
    inter_pos_dists = np.zeros((N_BASE_FREQS, N_POSITIONS, N_POSITIONS))
    intra_freq_diameters = np.zeros(N_BASE_FREQS)
    inter_freq_min_dists = np.zeros(N_BASE_FREQS)

    for fi in range(N_BASE_FREQS):
        tidx_fi = [fi * N_POSITIONS + p for p in range(N_POSITIONS)]
        vecs = templates_66[tidx_fi]  # (5, feat_dim)

        # Pairwise cosine distances within this frequency
        # Since templates are unit-norm, distance = sqrt(2(1-cos)) = sqrt(2(1-dot))
        for pa in range(N_POSITIONS):
            for pb in range(N_POSITIONS):
                if pa == pb:
                    inter_pos_dists[fi, pa, pb] = 0
                else:
                    d = np.sqrt(2 * (1 - np.dot(vecs[pa], vecs[pb])))
                    inter_pos_dists[fi, pa, pb] = d

        # Diameter = max pairwise distance within frequency
        intra_freq_diameters[fi] = inter_pos_dists[fi].max()

        # Min distance to nearest OTHER frequency's centroid
        centroid_fi = vecs.mean(axis=0)
        centroid_fi /= np.linalg.norm(centroid_fi)
        min_d = np.inf
        for fj in range(N_BASE_FREQS):
            if fj == fi:
                continue
            tidx_fj = [fj * N_POSITIONS + p for p in range(N_POSITIONS)]
            centroid_fj = templates_66[tidx_fj].mean(axis=0)
            centroid_fj /= np.linalg.norm(centroid_fj)
            d = np.sqrt(2 * (1 - np.dot(centroid_fi, centroid_fj)))
            if d < min_d:
                min_d = d
        inter_freq_min_dists[fi] = min_d

    # --- Fig D1: spatial diameter vs inter-frequency gap ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    ax = axes[0]
    ax.bar(range(N_BASE_FREQS), intra_freq_diameters, color="#e63946", alpha=0.7,
           label="Intra-freq spatial diameter")
    ax.bar(range(N_BASE_FREQS), inter_freq_min_dists, color="#3b82f6", alpha=0.5,
           label="Inter-freq min gap (centroid)")
    ax.set_xlabel("Frequency index")
    ax.set_ylabel("Cosine distance")
    ax.set_title("Spatial diameter vs frequency gap")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, axis="y")

    # Ratio: how much of the frequency gap is "consumed" by spatial spread
    ratio = intra_freq_diameters / np.clip(inter_freq_min_dists, 1e-12, None)
    ax = axes[1]
    ax.bar(range(N_BASE_FREQS), ratio, color="#6c757d", alpha=0.7)
    ax.axhline(1.0, color="red", linewidth=1, linestyle="--", label="overlap threshold")
    ax.set_xlabel("Frequency index")
    ax.set_ylabel("Spatial diameter / frequency gap")
    ax.set_title(
        f"Overlap ratio (mean={ratio.mean():.2f})\n"
        f"<1: spatial sub-points fit within frequency gap"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15, axis="y")

    # Mean pairwise distance matrix across positions
    mean_dist = inter_pos_dists.mean(axis=0)  # (5, 5)
    ax = axes[2]
    im = ax.imshow(mean_dist, cmap="YlOrRd")
    for i in range(N_POSITIONS):
        for j in range(N_POSITIONS):
            ax.text(j, i, f"{mean_dist[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, color="white" if mean_dist[i, j] > mean_dist.max() * 0.6 else "black")
    ax.set_xticks(range(N_POSITIONS))
    ax.set_xticklabels([POSITION_NAMES[p] for p in range(N_POSITIONS)])
    ax.set_yticks(range(N_POSITIONS))
    ax.set_yticklabels([POSITION_NAMES[p] for p in range(N_POSITIONS)])
    ax.set_title("Mean inter-position cosine distance\n(averaged across 40 frequencies)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f"Within-frequency spatial geometry — {subj}, 66ch\n"
        f"Q: Do 5 spatial sub-points fit within each frequency's Voronoi cell?",
        fontsize=12, fontweight="bold",
    )
    _save(fig, FIG_DIR / f"fig_tsne_D1_spatial_geometry_{subj}.png")

    # Print summary
    print(f"\n  Spatial geometry summary ({subj}):")
    print(f"    Intra-freq spatial diameter:  {intra_freq_diameters.mean():.4f} ± {intra_freq_diameters.std():.4f}")
    print(f"    Inter-freq centroid gap:      {inter_freq_min_dists.mean():.4f} ± {inter_freq_min_dists.std():.4f}")
    print(f"    Overlap ratio:                {ratio.mean():.3f} ± {ratio.std():.3f}")
    print(f"    Ratio < 1 for {(ratio < 1).sum()}/{N_BASE_FREQS} frequencies")
    print(f"\n  Mean inter-position distance matrix:")
    for pa in range(N_POSITIONS):
        row_str = "    "
        for pb in range(N_POSITIONS):
            row_str += f"  {POSITION_NAMES[pb]}={mean_dist[pa, pb]:.4f}"
        print(f"    {POSITION_NAMES[pa]}: {row_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# Part E: Multi-subject comparison
# ═══════════════════════════════════════════════════════════════════════════════

def part_e(all_templates: dict):
    """Compare constellation structure across subjects."""
    print("\n" + "=" * 70)
    print("  Part E: Multi-Subject Constellation Comparison")
    print("=" * 70)

    subjects = list(all_templates.keys())
    n_subj = len(subjects)

    fig, axes = plt.subplots(n_subj, 3, figsize=(20, 5.5 * n_subj),
                             constrained_layout=True)
    if n_subj == 1:
        axes = axes[np.newaxis, :]

    pos_colors_map = {
        0: "#e63946", 1: "#457b9d", 2: "#2a9d8f", 3: "#e9c46a", 4: "#6c757d"
    }

    for row, subj in enumerate(subjects):
        tmpl = all_templates[subj]

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                     max_iter=2000, learning_rate="auto", init="pca")
        proj = tsne.fit_transform(tmpl)

        # Panel 1: color by frequency
        ax = axes[row, 0]
        for tidx in range(N_TARGETS):
            fi = TARGET_TO_BASE_FREQ[tidx]
            ax.scatter(proj[tidx, 0], proj[tidx, 1],
                       c=[FREQ_CMAP[fi]], s=20, alpha=0.7, edgecolors="none")
        ax.set_title(f"{subj} — by frequency", fontsize=10)
        if row == n_subj - 1:
            ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.1)

        # Panel 2: color by position
        ax = axes[row, 1]
        for tidx in range(N_TARGETS):
            pos = TARGET_TO_POSITION[tidx]
            ax.scatter(proj[tidx, 0], proj[tidx, 1],
                       c=pos_colors_map[pos],
                       marker=POSITION_MARKERS[pos],
                       s=25, alpha=0.7, edgecolors="none")
        ax.set_title(f"{subj} — by position", fontsize=10)
        if row == n_subj - 1:
            ax.set_xlabel("t-SNE 1")
        ax.grid(alpha=0.1)

        # Panel 3: overlap ratio histogram
        inter_pos_dists_all = np.zeros(N_BASE_FREQS)
        inter_freq_gaps = np.zeros(N_BASE_FREQS)
        for fi in range(N_BASE_FREQS):
            tidx_fi = [fi * N_POSITIONS + p for p in range(N_POSITIONS)]
            vecs = tmpl[tidx_fi]
            # Max intra-freq distance
            max_d = 0
            for pa in range(N_POSITIONS):
                for pb in range(pa + 1, N_POSITIONS):
                    d = np.sqrt(2 * (1 - np.clip(np.dot(vecs[pa], vecs[pb]), -1, 1)))
                    if d > max_d:
                        max_d = d
            inter_pos_dists_all[fi] = max_d

            centroid = vecs.mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            min_gap = np.inf
            for fj in range(N_BASE_FREQS):
                if fj == fi:
                    continue
                tidx_fj = [fj * N_POSITIONS + p for p in range(N_POSITIONS)]
                c_fj = tmpl[tidx_fj].mean(axis=0)
                c_fj /= np.linalg.norm(c_fj)
                gap = np.sqrt(2 * (1 - np.clip(np.dot(centroid, c_fj), -1, 1)))
                if gap < min_gap:
                    min_gap = gap
            inter_freq_gaps[fi] = min_gap

        ratio = inter_pos_dists_all / np.clip(inter_freq_gaps, 1e-12, None)
        ax = axes[row, 2]
        ax.hist(ratio, bins=15, color="#6c757d", alpha=0.7, edgecolor="white")
        ax.axvline(1.0, color="red", linewidth=1.5, linestyle="--")
        ax.axvline(ratio.mean(), color="blue", linewidth=1.5, linestyle="-",
                   label=f"mean={ratio.mean():.2f}")
        ax.set_title(f"{subj} — overlap ratio distribution", fontsize=10)
        if row == n_subj - 1:
            ax.set_xlabel("Spatial diameter / frequency gap")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)

    fig.suptitle(
        "Multi-subject t-SNE + overlap analysis — 200 targets, 66ch\n"
        "Left: frequency structure | Middle: position structure | Right: overlap ratio",
        fontsize=13, fontweight="bold",
    )
    _save(fig, FIG_DIR / "fig_tsne_E1_multisubject_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("  Progressive Constellation & t-SNE Analysis")
    print("=" * 70)

    win_samp = 125
    subjects = ["S1", "S3", "S9", "S11"]
    all_templates_66 = {}

    for subj in subjects:
        print(f"\n  Loading {subj}...")
        data = load_subject_fb0(subj)
        if data is None:
            continue

        print(f"  Building 66ch templates...")
        t66 = build_templates(data, CHANNEL_SETS["66ch"], win_samp)
        all_templates_66[subj] = t66

        print(f"  Building 9ch templates...")
        t9 = build_templates(data, CHANNEL_SETS["9ch"], win_samp)

        # Parts A-D for primary subject (S1)
        if subj == "S1":
            part_a(t66, subj)
            part_b(t66, t9, subj)
            part_c(data, subj, win_samp)
            part_d(t66, subj)

        del data

    # Part E: multi-subject comparison
    if len(all_templates_66) > 1:
        part_e(all_templates_66)

    elapsed = time.time() - t_start
    print(f"\n  Total: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
