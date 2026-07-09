"""Improved noise cloud: LDA projection + centroid zoom inset.

Key fix: PCA maximizes total variance → captures noise directions (only 11% explained).
         LDA maximizes class separation → shows the discriminative projection.

Each figure has:
  - Main panel: LDA-2D projection with trial scatter + 1σ ellipses
  - Inset: centroid zoom with distance annotations
  - Title: explains LDA vs PCA and noise calculation

Usage
-----
    .venv/Scripts/python.exe scripts/18_noise_cloud_improved.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
    0: "#e63946", 1: "#1d3557", 2: "#2a9d8f", 3: "#e9c46a", 4: "#6c757d",
}

CH_66 = np.arange(66)


def load_fb0(subj):
    p = CACHE_DIR / f"filtered_{subj}_200target_66ch_18blocks_float32.npy"
    return np.asarray(np.load(str(p), mmap_mode="r")[:, :, :, :, 0])


def extract_trials(data, ch_idx, win_samp, freq_idx):
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


def cov_ellipse(points, n_sigma=1.0):
    center = points.mean(axis=0)
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * n_sigma * np.sqrt(max(eigvals[0], 0))
    height = 2 * n_sigma * np.sqrt(max(eigvals[1], 0))
    return center, width, height, angle


def compute_hdist(centroids):
    n = len(centroids)
    d = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            d[a, b] = np.sqrt(2 * (1 - np.clip(np.dot(centroids[a], centroids[b]), -1, 1)))
    return d


def make_figure(data, subj, freq_idx, win_samp):
    freq_hz = BASE_FREQS[freq_idx]
    ch_idx = CH_66

    trials_dict = extract_trials(data, ch_idx, win_samp, freq_idx)

    all_vecs = np.vstack([trials_dict[p] for p in range(N_POSITIONS)])
    all_pos = np.concatenate([[p] * N_BLOCKS for p in range(N_POSITIONS)])
    centroids = np.array([trials_dict[p].mean(axis=0) for p in range(N_POSITIONS)])
    hdist = compute_hdist(centroids)

    # ----- PCA pre-reduction (LDA needs n_features > n_samples) -----
    pca_pre = PCA(n_components=min(60, all_vecs.shape[0] - 1))
    X_pca = pca_pre.fit_transform(all_vecs)
    total_var_pca60 = pca_pre.explained_variance_ratio_.sum() * 100

    # ----- LDA projection (5 classes → max 4 discriminant axes → take 2) -----
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_pca, all_pos)

    # Project centroids through same pipeline
    C_pca = pca_pre.transform(centroids)
    C_lda = lda.transform(C_pca)

    # Between-class variance ratio in LDA space
    overall_mean = X_lda.mean(axis=0)
    Sb = sum(N_BLOCKS * np.outer(C_lda[p] - overall_mean, C_lda[p] - overall_mean) for p in range(N_POSITIONS))
    St = np.cov(X_lda.T) * (len(X_lda) - 1)
    lda_ratio = np.trace(Sb) / (np.trace(St) + 1e-30) * 100

    # ----- PCA-only projection (for comparison annotation) -----
    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(all_vecs)
    pca2_var = pca2.explained_variance_ratio_[:2].sum() * 100

    # ===== Figure layout: 2 columns =====
    fig, (ax_lda, ax_pca) = plt.subplots(1, 2, figsize=(20, 9))

    for ax, proj_t, proj_c, title_tag, xlabel, ylabel in [
        (ax_lda, X_lda, C_lda,
         f"LDA projection (between-class ratio: {lda_ratio:.0f}%)",
         "LD1", "LD2"),
        (ax_pca, X_pca2, pca2.transform(centroids),
         f"PCA projection (PC1+2 explains {pca2_var:.0f}% of variance)",
         f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)",
         f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)"),
    ]:
        # Trial scatter + ellipses
        for pos in range(N_POSITIONS):
            mask = all_pos == pos
            pts = proj_t[mask]
            color = POSITION_COLORS[pos]
            marker = POSITION_MARKERS[pos]

            ax.scatter(
                pts[:, 0], pts[:, 1],
                c=color, marker=marker, s=50, alpha=0.5,
                edgecolors="white", linewidth=0.3, zorder=3,
            )

            _, w, h, ang = cov_ellipse(pts, n_sigma=1.0)
            ell = Ellipse(
                xy=pts.mean(axis=0), width=w, height=h, angle=ang,
                facecolor=color, alpha=0.08, edgecolor=color,
                linewidth=2.0, linestyle="--", zorder=2,
            )
            ax.add_patch(ell)

        # Centroids
        for pos in range(N_POSITIONS):
            ax.scatter(
                proj_c[pos, 0], proj_c[pos, 1],
                c=POSITION_COLORS[pos], marker=POSITION_MARKERS[pos],
                s=300, edgecolors="black", linewidth=2.5, zorder=6,
                label=POSITION_NAMES[pos],
            )

        # Labels with smart offsets
        for pos in range(N_POSITIONS):
            cx, cy = proj_c[pos]
            others = np.delete(proj_c, pos, axis=0)
            away = np.array([cx, cy]) - others.mean(axis=0)
            away_norm = away / (np.linalg.norm(away) + 1e-30)
            ox, oy = away_norm * 18
            ax.annotate(
                POSITION_NAMES[pos],
                (cx, cy), textcoords="offset points",
                xytext=(ox, oy),
                fontsize=12, fontweight="bold", color=POSITION_COLORS[pos],
                ha="center", va="center",
            )

        ax.legend(fontsize=10, loc="upper left", framealpha=0.92,
                  title="Position", title_fontsize=10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title_tag, fontsize=13, fontweight="bold")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.08)

        # ----- Inset: centroid zoom with distance annotations -----
        cx_range = proj_c[:, 0].max() - proj_c[:, 0].min()
        cy_range = proj_c[:, 1].max() - proj_c[:, 1].min()
        c_center = proj_c.mean(axis=0)
        margin = max(cx_range, cy_range) * 1.0

        ax_ins = inset_axes(ax, width="38%", height="38%", loc="lower right",
                            borderpad=1.0)

        for pos in range(N_POSITIONS):
            ax_ins.scatter(
                proj_c[pos, 0], proj_c[pos, 1],
                c=POSITION_COLORS[pos], marker=POSITION_MARKERS[pos],
                s=200, edgecolors="black", linewidth=1.5, zorder=5,
            )
            away = proj_c[pos] - proj_c.mean(axis=0)
            away_n = away / (np.linalg.norm(away) + 1e-30)
            ax_ins.annotate(
                POSITION_SHORT[pos],
                proj_c[pos], textcoords="offset points",
                xytext=(away_n[0] * 14, away_n[1] * 14),
                fontsize=12, fontweight="bold", color=POSITION_COLORS[pos],
            )

        # Distance lines between all pairs
        for a in range(N_POSITIONS):
            for b in range(a + 1, N_POSITIONS):
                ax_ins.plot(
                    [proj_c[a, 0], proj_c[b, 0]],
                    [proj_c[a, 1], proj_c[b, 1]],
                    color="gray", linewidth=0.6, alpha=0.3, zorder=1,
                )

        # Annotate best and worst pair
        pair_dists = [(a, b, hdist[a, b]) for a in range(N_POSITIONS) for b in range(a + 1, N_POSITIONS)]
        pair_dists.sort(key=lambda x: x[2])
        for (a, b, d), ec, fs in [(pair_dists[-1], "#e63946", 9), (pair_dists[0], "#3b82f6", 9)]:
            mid = (proj_c[a] + proj_c[b]) / 2
            ax_ins.annotate(
                f"{d:.3f}", mid, fontsize=fs, color=ec,
                textcoords="offset points", xytext=(4, 4),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.85, ec=ec, lw=0.8),
            )

        if margin > 0:
            ax_ins.set_xlim(c_center[0] - margin, c_center[0] + margin)
            ax_ins.set_ylim(c_center[1] - margin, c_center[1] + margin)
        ax_ins.set_aspect("equal")
        ax_ins.set_title("Centroid zoom (cosine dist)", fontsize=8, fontweight="bold")
        ax_ins.grid(alpha=0.12)
        ax_ins.tick_params(labelsize=6)
        mark_inset(ax, ax_ins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.8, ls="--")

    # Compute noise-vs-signal metrics for suptitle
    lda_noise_radii = []
    lda_centroid_dists = []
    for pos in range(N_POSITIONS):
        pts = X_lda[all_pos == pos]
        _, w, h, _ = cov_ellipse(pts)
        lda_noise_radii.append((w + h) / 4)
    for a in range(N_POSITIONS):
        for b in range(a + 1, N_POSITIONS):
            lda_centroid_dists.append(np.linalg.norm(C_lda[a] - C_lda[b]))
    mean_noise = np.mean(lda_noise_radii)
    mean_sep = np.mean(lda_centroid_dists)

    fig.suptitle(
        f"{subj},  {freq_hz:.1f} Hz  —  5 positions × 18 trials\n"
        f"Noise ellipse = 1σ of 2D-projected trial distribution  |  "
        f"LDA noise/separation ratio: {mean_noise/mean_sep:.2f}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


def make_summary_figure(data, subj, freq_indices, win_samp):
    """3-row summary: one row per frequency, LDA projection only."""
    n_freq = len(freq_indices)
    fig, axes = plt.subplots(1, n_freq, figsize=(7 * n_freq, 7))
    if n_freq == 1:
        axes = [axes]

    for col, freq_idx in enumerate(freq_indices):
        freq_hz = BASE_FREQS[freq_idx]
        ax = axes[col]
        ch_idx = CH_66

        trials_dict = extract_trials(data, ch_idx, win_samp, freq_idx)
        all_vecs = np.vstack([trials_dict[p] for p in range(N_POSITIONS)])
        all_pos = np.concatenate([[p] * N_BLOCKS for p in range(N_POSITIONS)])
        centroids = np.array([trials_dict[p].mean(axis=0) for p in range(N_POSITIONS)])
        hdist = compute_hdist(centroids)

        pca_pre = PCA(n_components=min(60, all_vecs.shape[0] - 1))
        X_pca = pca_pre.fit_transform(all_vecs)
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_lda = lda.fit_transform(X_pca, all_pos)
        C_pca = pca_pre.transform(centroids)
        C_lda = lda.transform(C_pca)

        for pos in range(N_POSITIONS):
            mask = all_pos == pos
            pts = X_lda[mask]
            color = POSITION_COLORS[pos]
            marker = POSITION_MARKERS[pos]
            ax.scatter(pts[:, 0], pts[:, 1], c=color, marker=marker,
                       s=45, alpha=0.45, edgecolors="white", linewidth=0.3, zorder=3)
            _, w, h, ang = cov_ellipse(pts)
            ell = Ellipse(xy=pts.mean(axis=0), width=w, height=h, angle=ang,
                          facecolor=color, alpha=0.08, edgecolor=color,
                          linewidth=1.8, linestyle="--", zorder=2)
            ax.add_patch(ell)

        for pos in range(N_POSITIONS):
            ax.scatter(C_lda[pos, 0], C_lda[pos, 1],
                       c=POSITION_COLORS[pos], marker=POSITION_MARKERS[pos],
                       s=280, edgecolors="black", linewidth=2, zorder=6,
                       label=POSITION_NAMES[pos] if col == 0 else None)
            away = C_lda[pos] - C_lda.mean(axis=0)
            away_n = away / (np.linalg.norm(away) + 1e-30)
            ax.annotate(POSITION_SHORT[pos], C_lda[pos],
                        textcoords="offset points", xytext=(away_n[0]*16, away_n[1]*16),
                        fontsize=13, fontweight="bold", color=POSITION_COLORS[pos])

        lda_noise = []
        for pos in range(N_POSITIONS):
            pts = X_lda[all_pos == pos]
            _, w, h, _ = cov_ellipse(pts)
            lda_noise.append((w + h) / 4)
        lda_seps = [np.linalg.norm(C_lda[a] - C_lda[b])
                    for a in range(N_POSITIONS) for b in range(a+1, N_POSITIONS)]

        ratio = np.mean(lda_noise) / np.mean(lda_seps)
        ax.set_title(f"{freq_hz:.1f} Hz\nnoise/sep = {ratio:.2f}", fontsize=13, fontweight="bold")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.08)
        ax.set_xlabel("LD1", fontsize=11)
        if col == 0:
            ax.set_ylabel("LD2", fontsize=11)
            ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    fig.suptitle(
        f"{subj} — LDA projection of 5 spatial positions\n"
        f"Ellipse = 1σ covariance of 18 trials projected onto 2 LDA axes\n"
        f"(LDA maximizes between-class separation, unlike PCA)",
        fontsize=14, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    return fig


def main():
    t0 = time.time()
    win_samp = 125

    for subj in ["S1", "S9", "S11"]:
        print(f"Loading {subj}...")
        data = load_fb0(subj)

        # Detailed LDA vs PCA comparison for one frequency
        fig = make_figure(data, subj, 0, win_samp)  # 8.0 Hz
        fname = f"fig_noise_cloud_v2_{subj}_8.0Hz.png"
        fig.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {fname}")

        # Summary across 3 frequencies (LDA only)
        fig = make_summary_figure(data, subj, [0, 4, 8], win_samp)  # 8.0, 12.0, 9.0
        fname = f"fig_noise_cloud_v2_{subj}_summary.png"
        fig.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {fname}")

        del data

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
