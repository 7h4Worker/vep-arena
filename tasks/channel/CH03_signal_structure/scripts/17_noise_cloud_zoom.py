"""Enlarged noise cloud visualization for a single frequency cluster.

Picks 3 representative frequencies, one figure each, showing:
  - 5 position centroids (large markers with labels)
  - All 18 blocks of individual trials as scatter (small markers, same shape/color)
  - Ellipses showing 1-sigma spread per position
  - Clear legend and annotations

Usage
-----
    .venv/Scripts/python.exe scripts/17_noise_cloud_zoom.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.decomposition import PCA

TASK_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = TASK_DIR / "results" / "figures"
HD_ROOT = Path("D:/ProjData/datasets/ssvep_hd_200target")
CACHE_DIR = HD_ROOT / "derivatives" / "tdca_sample" / "cache"

FS = 250
LATENCY_SAMPLES = 35
N_POSITIONS = 5
N_BLOCKS = 18

BASE_FREQS = np.asarray(
    list(range(8, 16))
    + [x + 0.2 for x in range(8, 16)]
    + [x + 0.4 for x in range(8, 16)]
    + [x + 0.6 for x in range(8, 16)]
    + [x + 0.8 for x in range(8, 16)],
    dtype=np.float64,
)

POSITION_NAMES = {0: "RIGHT", 1: "DOWN", 2: "LEFT", 3: "UP", 4: "CENTER"}
POSITION_SHORT = {0: "R", 1: "D", 2: "L", 3: "U", 4: "C"}
POSITION_MARKERS = {0: "o", 1: "v", 2: "s", 3: "^", 4: "D"}
POSITION_COLORS = {
    0: "#e63946",  1: "#457b9d",  2: "#2a9d8f",  3: "#f4a261",  4: "#6c757d",
}

CH_66 = np.arange(66)


def load_fb0(subj):
    p = CACHE_DIR / f"filtered_{subj}_200target_66ch_18blocks_float32.npy"
    return np.asarray(np.load(str(p), mmap_mode="r")[:, :, :, :, 0])


def extract_trials(data, ch_idx, win_samp, freq_idx):
    """Extract all 18-block trials for 5 positions at one frequency.
    Returns dict {pos: (18, feat_dim)} of zero-mean unit-norm vectors.
    """
    out = {}
    for pos in range(N_POSITIONS):
        tidx = freq_idx * N_POSITIONS + pos
        vecs = np.zeros((N_BLOCKS, len(ch_idx) * win_samp))
        for blk in range(N_BLOCKS):
            seg = data[ch_idx, LATENCY_SAMPLES:LATENCY_SAMPLES + win_samp, tidx, blk]
            v = seg.flatten().astype(np.float64)
            v -= v.mean()
            n = np.linalg.norm(v)
            if n > 1e-12:
                v /= n
            vecs[blk] = v
        out[pos] = vecs
    return out


def draw_noise_cloud(ax, proj_trials, proj_centroid, pos, alpha_scatter=0.55):
    """Draw trial scatter + centroid + 1-sigma ellipse for one position."""
    color = POSITION_COLORS[pos]
    marker = POSITION_MARKERS[pos]
    label = POSITION_NAMES[pos]

    # Individual trials
    ax.scatter(
        proj_trials[:, 0], proj_trials[:, 1],
        c=color, marker=marker, s=45, alpha=alpha_scatter,
        edgecolors="white", linewidth=0.4, zorder=3,
    )

    # Centroid
    ax.scatter(
        proj_centroid[0], proj_centroid[1],
        c=color, marker=marker, s=350, edgecolors="black",
        linewidth=2.0, zorder=5, label=label,
    )

    # 1-sigma ellipse
    if len(proj_trials) > 2:
        cov = np.cov(proj_trials.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width = 2 * np.sqrt(eigvals[0])
        height = 2 * np.sqrt(eigvals[1])
        ell = Ellipse(
            xy=proj_centroid, width=width, height=height, angle=angle,
            facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.5,
            linestyle="--", zorder=2,
        )
        ax.add_patch(ell)


def make_figure(data, subj, freq_idx, win_samp):
    freq_hz = BASE_FREQS[freq_idx]
    ch_idx = CH_66

    trials_dict = extract_trials(data, ch_idx, win_samp, freq_idx)

    # Stack everything for joint PCA
    all_vecs = []
    all_pos = []
    for pos in range(N_POSITIONS):
        all_vecs.append(trials_dict[pos])  # (18, feat_dim)
        all_pos.extend([pos] * N_BLOCKS)
    all_vecs = np.vstack(all_vecs)  # (90, feat_dim)
    all_pos = np.array(all_pos)

    # Centroids
    centroids = np.zeros((N_POSITIONS, all_vecs.shape[1]))
    for pos in range(N_POSITIONS):
        centroids[pos] = trials_dict[pos].mean(axis=0)

    # Joint PCA on trials + centroids
    combined = np.vstack([all_vecs, centroids])  # (95, feat_dim)
    pca = PCA(n_components=2)
    proj_all = pca.fit_transform(combined)
    proj_trials = proj_all[:90]
    proj_centroids = proj_all[90:]
    var_exp = pca.explained_variance_ratio_[:2].sum() * 100

    # --- Draw ---
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)

    for pos in range(N_POSITIONS):
        mask = all_pos == pos
        draw_noise_cloud(ax, proj_trials[mask], proj_centroids[pos], pos)

    # Annotate centroids with position names
    for pos in range(N_POSITIONS):
        cx, cy = proj_centroids[pos]
        offsets = {0: (12, -8), 1: (-8, -14), 2: (-18, 8), 3: (8, 12), 4: (12, 5)}
        ox, oy = offsets[pos]
        ax.annotate(
            POSITION_NAMES[pos],
            (cx, cy),
            textcoords="offset points", xytext=(ox, oy),
            fontsize=13, fontweight="bold", color=POSITION_COLORS[pos],
            arrowprops=dict(arrowstyle="-", color=POSITION_COLORS[pos],
                            linewidth=0.8, alpha=0.6),
        )

    # Draw lines between centroids to show inter-position distances
    for pa in range(N_POSITIONS):
        for pb in range(pa + 1, N_POSITIONS):
            ax.plot(
                [proj_centroids[pa, 0], proj_centroids[pb, 0]],
                [proj_centroids[pa, 1], proj_centroids[pb, 1]],
                color="gray", linewidth=0.5, alpha=0.3, linestyle=":",
                zorder=1,
            )

    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.set_title(
        f"{subj}, {freq_hz:.1f} Hz — 5 fixation positions × 18 trials\n"
        f"Large markers = centroids, small markers = individual trials\n"
        f"Dashed ellipses = 1σ noise cloud per position  (PC1+2: {var_exp:.0f}%)",
        fontsize=13, fontweight="bold",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.1)

    return fig


def main():
    t0 = time.time()
    win_samp = 125

    for subj in ["S1", "S9"]:
        print(f"Loading {subj}...")
        data = load_fb0(subj)

        for freq_idx in [0, 16, 32]:  # 8.0, 10.4, 12.8 Hz
            fig = make_figure(data, subj, freq_idx, win_samp)
            freq_hz = BASE_FREQS[freq_idx]
            fname = f"fig_noise_cloud_zoom_{subj}_{freq_hz:.1f}Hz.png"
            fig.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  -> {fname}")

        del data

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
