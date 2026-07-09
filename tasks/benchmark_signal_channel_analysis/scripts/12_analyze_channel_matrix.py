"""Empirical SSVEP channel matrix — decoder-independent characterization.

For each subject, compute the 40×40 correlation matrix H where:
  H[i,j] = mean spectral energy of trials at freq_i evaluated at freq_j

This is decoder-independent (no spatial filter optimization, just Pearson/DFT).
Analogous to a communication channel's transfer matrix.

From H we derive:
  - SSVEP "constellation diagram" (PCA of H rows)
  - Effective rank (eigenvalue decay)
  - Decoder-independent mutual information I(X;Y)
  - Comparison across channel configs (1ch, 9ch, 32ch, 66ch)

Uses HD200 dataset (14 subjects, 200 targets, 66ch, 18 blocks).

Usage
-----
    python analyze_channel_matrix.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svdvals

TASK_DIR = Path(__file__).resolve().parent
FIG_DIR = TASK_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HD_ROOT = Path("D:/ProjData/datasets/ssvep_hd_200target")
CACHE_DIR = HD_ROOT / "derivatives" / "tdca_sample" / "cache"

FS = 250
LATENCY_SAMPLES = 35
N_HARMONICS = 5
N_BASE_FREQS = 40

# Must match HD200 run.py ordering: integers first, then +0.2, +0.4, +0.6, +0.8
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

# 200 targets = 40 base freqs × 5 spatial slots
# target 0-4 → freq index 0 (8.0 Hz), 5-9 → freq index 1 (8.2 Hz), ...
TARGET_TO_BASE_FREQ = np.arange(200) // 5


def make_references(n_samples):
    """Generate normalized sin/cos reference matrix for 40 base frequencies.

    Returns: (40, 2*N_HARMONICS, n_samples), each component zero-mean, unit-norm.
    """
    t = np.arange(n_samples) / FS
    refs = np.zeros((N_BASE_FREQS, 2 * N_HARMONICS, n_samples))
    for fi, freq in enumerate(BASE_FREQS):
        for h in range(N_HARMONICS):
            refs[fi, 2 * h] = np.sin(2 * np.pi * (h + 1) * freq * t)
            refs[fi, 2 * h + 1] = np.cos(2 * np.pi * (h + 1) * freq * t)

    # Normalize: zero-mean, unit-norm per component
    refs -= refs.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(refs, axis=-1, keepdims=True)
    refs /= np.clip(norms, 1e-12, None)
    return refs


def build_channel_matrix(data, channel_idx, refs, window_samples):
    """Build 40×40 empirical channel matrix H (vectorized).

    H[i,j] = mean spectral energy of trials at base_freq_i evaluated with refs at base_freq_j.
    Spectral energy = sum of squared Pearson correlations with sin/cos components.
    Average across all blocks, spatial slots sharing the same base freq, and channels.

    data: (66, 185, 200, 18, 5) or (66, 185, 200, 18) — use FB0 if 5D.
    """
    n_ch = len(channel_idx)

    # Extract relevant data: (n_ch, win, 200, 18)
    ch_data = data[channel_idx, LATENCY_SAMPLES:LATENCY_SAMPLES + window_samples]

    H_sum = np.zeros((N_BASE_FREQS, N_BASE_FREQS))
    H_count = np.zeros(N_BASE_FREQS)
    trial_vectors = {i: [] for i in range(N_BASE_FREQS)}

    for target_idx in range(200):
        base_fi = TARGET_TO_BASE_FREQ[target_idx]

        # seg: (n_ch, win, 18) → (18, n_ch, win)
        seg = ch_data[:, :, target_idx, :].transpose(2, 0, 1)

        # Normalize: zero-mean, unit-norm per (block, channel)
        seg = seg - seg.mean(axis=-1, keepdims=True)
        seg_norms = np.linalg.norm(seg, axis=-1, keepdims=True)
        seg = seg / np.clip(seg_norms, 1e-12, None)

        # Correlation: einsum('bcs,fks->bcfk', seg, refs)
        # seg: (18, n_ch, win), refs: (40, 10, win)
        corr = np.einsum("bcs,fks->bcfk", seg, refs)  # (18, n_ch, 40, 10)

        # Spectral energy: sum of squared correlations, average over channels
        energy = (corr ** 2).sum(axis=-1)  # (18, n_ch, 40)
        energy_avg = energy.mean(axis=1)   # (18, 40)

        for block in range(18):
            vec = energy_avg[block]
            H_sum[base_fi] += vec
            H_count[base_fi] += 1
            trial_vectors[base_fi].append(vec.copy())

    H = H_sum / np.clip(H_count, 1, None)[:, None]
    return H, trial_vectors


def effective_rank(H, threshold=0.99):
    """Number of singular values capturing threshold fraction of total."""
    sv = svdvals(H)
    total = sv.sum()
    if total < 1e-12:
        return 1
    cumsum = np.cumsum(sv / total)
    return int(np.searchsorted(cumsum, threshold) + 1)


def mutual_information_gaussian(H, trial_vectors):
    """Estimate I(X;Y) assuming Gaussian noise, uniform input.

    Y = H[x,:] + noise, where x is uniform over {0,...,39}.
    I(X;Y) = 0.5 * log2(det(Sigma_total) / det(Sigma_noise))
    """
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

    # Noise covariance: average within-class covariance
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

    # Signal covariance
    grand_mean = H.mean(axis=0)
    Sigma_signal = (H - grand_mean).T @ (H - grand_mean) / K
    Sigma_total = Sigma_signal + Sigma_noise

    sign_t, logdet_t = np.linalg.slogdet(Sigma_total)
    sign_n, logdet_n = np.linalg.slogdet(Sigma_noise)
    if sign_t <= 0 or sign_n <= 0:
        return 0.0
    mi = 0.5 * (logdet_t - logdet_n) / np.log(2)
    return max(0.0, mi)


def main():
    window_ms = 500
    window_samples = int(window_ms * FS / 1000)
    usable = 185 - LATENCY_SAMPLES
    if window_samples > usable:
        window_samples = usable
    print(f"  Window: {window_ms}ms → {window_samples} samples")

    refs = make_references(window_samples)
    print(f"  References: {refs.shape} (40 freqs × {2*N_HARMONICS} components × {window_samples} samples)")

    sample_subjects = ["S1", "S3", "S9", "S11"]
    ch_configs = ["1ch_Oz", "9ch", "32ch", "66ch"]

    results = {}

    for subj in sample_subjects:
        cache_path = CACHE_DIR / f"filtered_{subj}_200target_66ch_18blocks_float32.npy"
        if not cache_path.exists():
            print(f"  SKIP {subj}: cache not found")
            continue

        print(f"\n  Loading {subj}...", end="", flush=True)
        t0 = time.time()
        data_full = np.load(str(cache_path), mmap_mode="r")
        # shape: (66, 185, 200, 18, 5) — use FB0 (broadband 6-90 Hz)
        data = np.asarray(data_full[:, :, :, :, 0])
        print(f" loaded ({time.time()-t0:.1f}s, shape={data.shape})")

        for ch_name in ch_configs:
            ch_idx = CHANNEL_SETS[ch_name]
            print(f"    {ch_name} ({len(ch_idx):2d} ch)...", end="", flush=True)
            t1 = time.time()

            H, trial_vecs = build_channel_matrix(data, ch_idx, refs, window_samples)
            sv = svdvals(H)
            erank95 = effective_rank(H, threshold=0.95)
            erank99 = effective_rank(H, threshold=0.99)
            mi = mutual_information_gaussian(H, trial_vecs)

            diag_mean = np.mean(np.diag(H))
            offdiag = H[~np.eye(N_BASE_FREQS, dtype=bool)]
            offdiag_mean = np.mean(offdiag)
            sir = 10 * np.log10(diag_mean / max(offdiag_mean, 1e-12))

            results[(subj, ch_name)] = {
                "H": H, "trial_vecs": trial_vecs, "sv": sv,
                "erank95": erank95, "erank99": erank99, "mi": mi,
                "diag_mean": diag_mean, "offdiag_mean": offdiag_mean, "sir": sir,
            }

            elapsed = time.time() - t1
            print(f" erank95={erank95:2d}  erank99={erank99:2d}  MI={mi:.2f} bits"
                  f"  SIR={sir:.1f} dB  ({elapsed:.1f}s)")

    # ── Summary table ──
    print(f"\n{'='*80}")
    print(f"  Empirical channel matrix summary  (window={window_ms}ms)")
    print(f"{'='*80}")
    hdr = f"  {'Subject':>8s}"
    for ch in ch_configs:
        hdr += f"  {ch:>14s}"
    print(f"\n  Effective rank (95% energy):")
    print(hdr)
    for subj in sample_subjects:
        row = f"  {subj:>8s}"
        for ch in ch_configs:
            r = results.get((subj, ch))
            row += f"  {r['erank95']:14d}" if r else f"  {'—':>14s}"
        print(row)

    print(f"\n  Mutual information (bits, Gaussian, uniform input):")
    print(hdr)
    for subj in sample_subjects:
        row = f"  {subj:>8s}"
        for ch in ch_configs:
            r = results.get((subj, ch))
            row += f"  {r['mi']:14.2f}" if r else f"  {'—':>14s}"
        print(row)

    print(f"\n  Signal-to-Interference Ratio (dB):")
    print(hdr)
    for subj in sample_subjects:
        row = f"  {subj:>8s}"
        for ch in ch_configs:
            r = results.get((subj, ch))
            row += f"  {r['sir']:14.1f}" if r else f"  {'—':>14s}"
        print(row)

    # ── Figures ──
    def _save(fig, path):
        fig.savefig(path, dpi=190, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {path.name}")

    # Fig 1: Channel matrices (H) for S1 and S11, 1ch vs 66ch
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    for col, ch_name in enumerate(["1ch_Oz", "66ch"]):
        for row, subj in enumerate(["S1", "S11"]):
            ax = axes[row, col]
            r = results.get((subj, ch_name))
            if r is None:
                continue
            H = r["H"]
            im = ax.imshow(H, aspect="auto", cmap="viridis", interpolation="nearest")
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xlabel("Reference freq index (j)")
            ax.set_ylabel("Trial freq index (i)")
            ax.set_title(f"{subj} / {ch_name}\nerank95={r['erank95']}, MI={r['mi']:.1f} bits, SIR={r['sir']:.1f} dB")

            # Add freq labels on axes (every 5th)
            ticks = list(range(0, 40, 5))
            tlabels = [f"{BASE_FREQS[t]:.1f}" for t in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(tlabels, fontsize=6)
            ax.set_yticks(ticks)
            ax.set_yticklabels(tlabels, fontsize=6)

    fig.suptitle("Empirical channel matrix H[i,j]: spectral energy of freq_i trials at freq_j\n"
                 "(decoder-independent, no spatial filtering)", fontsize=12)
    _save(fig, FIG_DIR / "fig_channel_matrix_H.png")

    # Fig 2: SSVEP constellation diagram (PCA of H rows)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    for col, ch_name in enumerate(["1ch_Oz", "66ch"]):
        for row, subj in enumerate(["S1", "S11"]):
            ax = axes[row, col]
            r = results.get((subj, ch_name))
            if r is None:
                continue
            H = r["H"]
            H_centered = H - H.mean(axis=0)
            U, s, Vt = np.linalg.svd(H_centered, full_matrices=False)
            coords = H_centered @ Vt[:2].T

            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=BASE_FREQS, cmap="rainbow", s=60,
                edgecolors="black", linewidths=0.5, zorder=5,
            )
            # Connect in frequency order
            ax.plot(coords[:, 0], coords[:, 1], "k-", alpha=0.12, linewidth=0.5)
            # Label every 5th frequency
            for idx in range(0, 40, 5):
                ax.annotate(
                    f"{BASE_FREQS[idx]:.1f}",
                    (coords[idx, 0], coords[idx, 1]),
                    fontsize=6, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points",
                )
            plt.colorbar(scatter, ax=ax, label="Frequency (Hz)", shrink=0.8)
            ax.set_xlabel(f"PC1 ({s[0]/s.sum()*100:.0f}%)")
            ax.set_ylabel(f"PC2 ({s[1]/s.sum()*100:.0f}%)")
            ax.set_title(f"{subj} / {ch_name}")
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)

    fig.suptitle("SSVEP constellation diagram: PCA of channel matrix rows\n"
                 "(each point = one of 40 base frequencies)", fontsize=12)
    _save(fig, FIG_DIR / "fig_ssvep_constellation.png")

    # Fig 2b: Constellation with per-trial scatter (noise clouds)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    for col, ch_name in enumerate(["1ch_Oz", "66ch"]):
        for row, subj in enumerate(["S1", "S11"]):
            ax = axes[row, col]
            r = results.get((subj, ch_name))
            if r is None:
                continue
            H = r["H"]
            H_centered = H - H.mean(axis=0)
            U, s, Vt = np.linalg.svd(H_centered, full_matrices=False)
            mean_of_H = H.mean(axis=0)
            proj_mat = Vt[:2].T

            # Plot per-trial points (noise clouds)
            for fi in range(N_BASE_FREQS):
                vecs = np.array(r["trial_vecs"][fi])
                if len(vecs) == 0:
                    continue
                trial_coords = (vecs - mean_of_H) @ proj_mat
                color = plt.cm.rainbow(fi / 39.0)
                ax.scatter(trial_coords[:, 0], trial_coords[:, 1],
                           c=[color], s=3, alpha=0.08, rasterized=True)

            # Plot centroids
            coords = H_centered @ proj_mat
            ax.scatter(
                coords[:, 0], coords[:, 1],
                c=BASE_FREQS, cmap="rainbow", s=50,
                edgecolors="black", linewidths=0.8, zorder=5,
            )
            for idx in range(0, 40, 8):
                ax.annotate(
                    f"{BASE_FREQS[idx]:.1f}",
                    (coords[idx, 0], coords[idx, 1]),
                    fontsize=6, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points",
                    fontweight="bold",
                )
            ax.set_xlabel(f"PC1 ({s[0]/s.sum()*100:.0f}%)")
            ax.set_ylabel(f"PC2 ({s[1]/s.sum()*100:.0f}%)")
            ax.set_title(f"{subj} / {ch_name}")
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)

    fig.suptitle("SSVEP constellation with noise clouds\n"
                 "(small dots = individual trials, large = class centroids)", fontsize=12)
    _save(fig, FIG_DIR / "fig_ssvep_constellation_noise.png")

    # Fig 3: Eigenvalue spectrum
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    ax = axes[0]
    for subj in sample_subjects:
        for ch_name in ["1ch_Oz", "66ch"]:
            r = results.get((subj, ch_name))
            if r is None:
                continue
            sv = r["sv"] / r["sv"].sum()
            ls = "-" if ch_name == "66ch" else "--"
            ax.plot(range(1, len(sv) + 1), np.cumsum(sv), ls,
                    linewidth=2 if ch_name == "66ch" else 1,
                    label=f"{subj}/{ch_name}")
    ax.axhline(0.95, color="red", linestyle=":", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_title("Channel matrix eigenvalue spectrum")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.15)
    ax.set_xlim(0, 40)

    ax = axes[1]
    x = np.arange(len(ch_configs))
    w = 0.18
    subj_colors = {"S1": "#3b82f6", "S3": "#10b981", "S9": "#f59e0b", "S11": "#dc2626"}
    for si, subj in enumerate(sample_subjects):
        eranks = [results.get((subj, ch), {}).get("erank95", 0) for ch in ch_configs]
        ax.bar(x + (si - 1.5) * w, eranks, w,
               color=subj_colors.get(subj, "gray"), alpha=0.7, label=subj)
    ax.set_xlabel("Channel configuration")
    ax.set_ylabel("Effective rank (95% energy)")
    ax.set_title("Effective rank vs channel count")
    ax.set_xticks(x)
    ax.set_xticklabels(ch_configs, fontsize=8)
    ax.legend()
    ax.grid(alpha=0.15, axis="y")

    _save(fig, FIG_DIR / "fig_channel_eigenspectrum.png")

    # Fig 4: MI and SIR vs channel count
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

    ax = axes[0]
    for subj in sample_subjects:
        mis = [results.get((subj, ch), {}).get("mi", 0) for ch in ch_configs]
        ax.plot(range(len(ch_configs)), mis, "o-",
                color=subj_colors.get(subj, "gray"), linewidth=2, markersize=7, label=subj)
    ax.set_xlabel("Channel configuration")
    ax.set_ylabel("Mutual information (bits)")
    ax.set_title("Decoder-independent MI vs channel config")
    ax.set_xticks(range(len(ch_configs)))
    ax.set_xticklabels(ch_configs, fontsize=8)
    ax.legend()
    ax.grid(alpha=0.15)

    ax = axes[1]
    for subj in sample_subjects:
        sirs = [results.get((subj, ch), {}).get("sir", 0) for ch in ch_configs]
        ax.plot(range(len(ch_configs)), sirs, "o-",
                color=subj_colors.get(subj, "gray"), linewidth=2, markersize=7, label=subj)
    ax.set_xlabel("Channel configuration")
    ax.set_ylabel("Signal-to-Interference Ratio (dB)")
    ax.set_title("Decoder-independent SIR vs channel config")
    ax.set_xticks(range(len(ch_configs)))
    ax.set_xticklabels(ch_configs, fontsize=8)
    ax.legend()
    ax.grid(alpha=0.15)

    _save(fig, FIG_DIR / "fig_channel_mi_sir.png")

    print(f"\n  5 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
