"""Visualize method score vectors via t-SNE / PCA.

Collects per-trial 40-dim score vectors from all four methods,
projects to 2D, and compares the representation geometry.

Usage
-----
    python visualize_score_space.py                  # 5 subjects, fast
    python visualize_score_space.py --subjects 1-35  # all subjects
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vep_arena.config import (
    BENCHMARK_CHANNELS_9,
    BENCHMARK_FREQS,
    BenchmarkSpec,
    DATA_ROOT,
)
from vep_arena.data.benchmark import load_subject_filterbank
from vep_arena.methods.traditional import CCA, FBCCA, TRCA

TASK_DIR = Path(__file__).resolve().parent
FREQS = np.asarray(BENCHMARK_FREQS, dtype=np.float64)
FREQ_SORTED_IDX = np.argsort(FREQS)
FREQS_SORTED = FREQS[FREQ_SORTED_IDX]
N_FBS = 5
METHODS = ("CCA", "FBCCA", "TRCA", "ETRCA")


def parse_int_set(text: str) -> tuple[int, ...]:
    vals: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            vals.extend(range(int(a), int(b) + 1))
        else:
            vals.append(int(part))
    return tuple(dict.fromkeys(vals))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", type=str, default="1-5")
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def collect_trial_scores(
    data_root: Path,
    subjects: tuple[int, ...],
    window: float,
    spec: BenchmarkSpec,
) -> dict[str, np.ndarray]:
    """Collect per-trial score vectors for all methods.

    Returns dict: method → (n_total_trials, 40) score array,
    plus "labels" → (n_total_trials,) true class indices,
    plus "subjects" → (n_total_trials,) subject ids.
    """
    all_scores = {m: [] for m in METHODS}
    all_labels: list[np.ndarray] = []
    all_subjects: list[np.ndarray] = []

    n_classes = spec.classes

    for subj in subjects:
        epochs = load_subject_filterbank(data_root, subj, window, N_FBS, BENCHMARK_CHANNELS_9, spec)
        n_samples = epochs.shape[-1]

        for block in range(spec.blocks):
            train_mask = [b for b in range(spec.blocks) if b != block]
            train_x = epochs[:, train_mask].reshape(-1, N_FBS, len(BENCHMARK_CHANNELS_9), n_samples)
            train_y = np.repeat(np.arange(n_classes), len(train_mask))
            test_x = epochs[:, block]  # (40, 5, 9, samples)

            cca = CCA(window=window, harmonics=5, spec=spec)
            fbcca = FBCCA(window=window, harmonics=5, n_fbs=N_FBS, spec=spec)
            trca = TRCA(n_fbs=N_FBS, ensemble=False)
            trca.fit(train_x, train_y)
            etrca = TRCA(n_fbs=N_FBS, ensemble=True)
            etrca.templates = trca.templates
            etrca.filters = trca.filters

            models = {"CCA": cca, "FBCCA": fbcca, "TRCA": trca, "ETRCA": etrca}
            for name, model in models.items():
                _, scores = model.predict(test_x)
                all_scores[name].append(scores)

            all_labels.append(np.arange(n_classes))
            all_subjects.append(np.full(n_classes, subj))

    result = {m: np.vstack(all_scores[m]) for m in METHODS}
    result["labels"] = np.concatenate(all_labels)
    result["subjects"] = np.concatenate(all_subjects)
    return result


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def freq_colormap(labels: np.ndarray) -> np.ndarray:
    """Map class labels to colors by frequency (sorted)."""
    freqs_for_labels = FREQS[labels]
    norm = (freqs_for_labels - FREQS.min()) / (FREQS.max() - FREQS.min())
    return plt.cm.rainbow(norm)


def plot_tsne_4methods(data: dict, fig_dir: Path, perplexity: float, seed: int) -> None:
    labels = data["labels"]
    colors = freq_colormap(labels)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)

    for ax, m in zip(axes.ravel(), METHODS):
        scores = data[m]
        # normalize per-method for comparable t-SNE
        scores_norm = (scores - scores.mean(axis=0)) / (scores.std(axis=0) + 1e-10)
        emb = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                   init="pca", learning_rate="auto").fit_transform(scores_norm)
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=3, alpha=0.4, edgecolors="none")
        ax.set_title(f"{m} — t-SNE of 40D score vectors", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # colorbar
    sm = plt.cm.ScalarMappable(cmap="rainbow",
                               norm=plt.Normalize(vmin=FREQS.min(), vmax=FREQS.max()))
    fig.colorbar(sm, ax=axes, shrink=0.6, label="Target frequency (Hz)")
    fig.suptitle("Score space geometry — four methods (colored by frequency)", fontsize=13)
    _save(fig, fig_dir / "fig_tsne_score_4methods.png")


def plot_pca_4methods(data: dict, fig_dir: Path) -> None:
    labels = data["labels"]
    colors = freq_colormap(labels)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)

    for ax, m in zip(axes.ravel(), METHODS):
        scores = data[m]
        pca = PCA(n_components=2)
        emb = pca.fit_transform(scores)
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=3, alpha=0.4, edgecolors="none")
        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax.set_title(f"{m} — PCA of score vectors", fontsize=11)

    sm = plt.cm.ScalarMappable(cmap="rainbow",
                               norm=plt.Normalize(vmin=FREQS.min(), vmax=FREQS.max()))
    fig.colorbar(sm, ax=axes, shrink=0.6, label="Target frequency (Hz)")
    fig.suptitle("Score space geometry — PCA projection (colored by frequency)", fontsize=13)
    _save(fig, fig_dir / "fig_pca_score_4methods.png")


def plot_trca_etrca_comparison(data: dict, fig_dir: Path, perplexity: float, seed: int) -> None:
    """Side-by-side t-SNE of TRCA vs ETRCA, plus their score vector difference."""
    labels = data["labels"]
    colors = freq_colormap(labels)

    # concatenate TRCA and ETRCA scores for joint embedding
    trca_s = data["TRCA"]
    etrca_s = data["ETRCA"]
    trca_norm = (trca_s - trca_s.mean(axis=0)) / (trca_s.std(axis=0) + 1e-10)
    etrca_norm = (etrca_s - etrca_s.mean(axis=0)) / (etrca_s.std(axis=0) + 1e-10)

    joint = np.vstack([trca_norm, etrca_norm])
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
               init="pca", learning_rate="auto").fit_transform(joint)
    n = len(labels)
    emb_trca = emb[:n]
    emb_etrca = emb[n:]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    axes[0].scatter(emb_trca[:, 0], emb_trca[:, 1], c=colors, s=4, alpha=0.4, edgecolors="none")
    axes[0].set_title("TRCA (joint t-SNE)")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].scatter(emb_etrca[:, 0], emb_etrca[:, 1], c=colors, s=4, alpha=0.4, edgecolors="none")
    axes[1].set_title("ETRCA (joint t-SNE)")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # displacement vectors: how each trial moves from TRCA to ETRCA in the joint embedding
    # sample a subset for clarity
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(n, min(400, n), replace=False)
    for i in sample_idx:
        axes[2].annotate("", xy=emb_etrca[i], xytext=emb_trca[i],
                         arrowprops=dict(arrowstyle="->", color=colors[i], alpha=0.3, lw=0.5))
    axes[2].set_title("Trial displacement: TRCA → ETRCA")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    sm = plt.cm.ScalarMappable(cmap="rainbow",
                               norm=plt.Normalize(vmin=FREQS.min(), vmax=FREQS.max()))
    fig.colorbar(sm, ax=axes, shrink=0.7, label="Target frequency (Hz)")
    fig.suptitle("Same filters, different projection — joint t-SNE embedding", fontsize=13)
    _save(fig, fig_dir / "fig_tsne_trca_vs_etrca_joint.png")


def plot_score_correlation_structure(data: dict, fig_dir: Path) -> None:
    """Show how score dimensions (classes) correlate with each other per method."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)

    for ax, m in zip(axes.ravel(), METHODS):
        scores = data[m]
        corr = np.corrcoef(scores.T)  # (40, 40) correlation between score dimensions
        corr_sorted = corr[np.ix_(FREQ_SORTED_IDX, FREQ_SORTED_IDX)]
        im = ax.imshow(corr_sorted, aspect="equal", cmap="RdBu_r", interpolation="nearest",
                       vmin=-1, vmax=1)
        ax.set_title(f"{m} — inter-class score correlation", fontsize=11)
        ax.set_xticks(range(0, 40, 5))
        ax.set_xticklabels([f"{FREQS_SORTED[i]:.0f}" for i in range(0, 40, 5)], fontsize=7)
        ax.set_yticks(range(0, 40, 5))
        ax.set_yticklabels([f"{FREQS_SORTED[i]:.0f}" for i in range(0, 40, 5)], fontsize=7)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Pearson r")
    fig.suptitle("Score dimension correlation — how coupled are class scores?", fontsize=13)
    _save(fig, fig_dir / "fig_score_correlation_structure.png")


def main() -> None:
    args = parse_args()
    subjects = parse_int_set(args.subjects)
    data_root = Path(args.data_root)
    spec = BenchmarkSpec()

    fig_dir = TASK_DIR / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Score space visualization ===")
    print(f"  subjects:   {subjects}")
    print(f"  perplexity: {args.perplexity}")
    print()

    t0 = time.time()
    print("  Collecting trial-level scores...")
    data = collect_trial_scores(data_root, subjects, args.window, spec)
    n_trials = len(data["labels"])
    print(f"  {n_trials} trials collected ({time.time() - t0:.1f}s)")

    print("  Plotting score correlation structure...")
    plot_score_correlation_structure(data, fig_dir)

    print("  Plotting PCA (fast)...")
    plot_pca_4methods(data, fig_dir)

    print("  Plotting t-SNE 4 methods (may take a minute)...")
    plot_tsne_4methods(data, fig_dir, args.perplexity, args.seed)

    print("  Plotting TRCA vs ETRCA joint t-SNE...")
    plot_trca_etrca_comparison(data, fig_dir, args.perplexity, args.seed)

    total = time.time() - t0
    print(f"\n  Done in {total:.1f}s — 4 figures written to {fig_dir}")


if __name__ == "__main__":
    main()
