"""Phase 2: Spatial filter impact analysis.

Compares CCA / FBCCA / TRCA / ETRCA on the same data:
  1. Confusion matrix comparison across four methods
  2. Score matrix structure (mean similarity landscape)
  3. TRCA spatial filter SNR gain (pre- vs post-filtering)
  4. TRCA vs ETRCA ensemble effect (same filters, different projection)
  5. Discrimination margin analysis

Usage
-----
    python analyze_filters.py                        # all 35 subjects
    python analyze_filters.py --subjects 1-5         # subset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vep_arena.config import (
    BENCHMARK_CHANNELS_9,
    BENCHMARK_FREQS,
    BENCHMARK_PHASES_PI,
    DATA_ROOT,
    BenchmarkSpec,
)
from vep_arena.data.benchmark import load_subject_filterbank
from vep_arena.methods.traditional import CCA, FBCCA, TRCA, filterbank_weights
from vep_arena.signal.snr import snr_harmonic
from vep_arena.signal.spectrum import compute_psd

TASK_DIR = Path(__file__).resolve().parent
FREQS = np.asarray(BENCHMARK_FREQS, dtype=np.float64)
FREQ_SORTED_IDX = np.argsort(FREQS)
FREQS_SORTED = FREQS[FREQ_SORTED_IDX]
N_FBS = 5
N_HARMONICS = 5
N_NEIGHBORS = 10
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
    p = argparse.ArgumentParser(description="Spatial filter impact analysis")
    p.add_argument("--subjects", type=str, default="1-35")
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    return p.parse_args()


# ── method factory ──────────────────────────────────────────────────

def build_methods(window: float, spec: BenchmarkSpec) -> dict:
    return {
        "CCA": CCA(window=window, harmonics=5, spec=spec),
        "FBCCA": FBCCA(window=window, harmonics=5, n_fbs=N_FBS, spec=spec),
        "TRCA": TRCA(n_fbs=N_FBS, ensemble=False),
        "ETRCA": TRCA(n_fbs=N_FBS, ensemble=True),
    }


# ── core: per-subject analysis ──────────────────────────────────────

def analyse_subject(
    data_root: Path,
    subject: int,
    window: float,
    spec: BenchmarkSpec,
) -> dict:
    """Run leave-one-block-out CV for all methods, return aggregated results."""

    epochs = load_subject_filterbank(data_root, subject, window, N_FBS, BENCHMARK_CHANNELS_9, spec)
    # (40, 6, 5, 9, samples)
    n_classes, n_blocks = spec.classes, spec.blocks
    n_samples = epochs.shape[-1]

    confusion = {m: np.zeros((n_classes, n_classes), dtype=np.float64) for m in METHODS}
    score_sum = {m: np.zeros((n_classes, n_classes), dtype=np.float64) for m in METHODS}
    trca_filters_sum = np.zeros((N_FBS, n_classes, len(BENCHMARK_CHANNELS_9)), dtype=np.float64)

    # SNR gain: raw vs TRCA-filtered, per class
    snr_raw_acc = np.zeros(n_classes, dtype=np.float64)
    snr_filt_acc = np.zeros(n_classes, dtype=np.float64)
    snr_count = 0

    for block in range(n_blocks):
        train_mask = [b for b in range(n_blocks) if b != block]
        train_x = epochs[:, train_mask].reshape(-1, N_FBS, len(BENCHMARK_CHANNELS_9), n_samples)
        train_y = np.repeat(np.arange(n_classes), len(train_mask))
        test_x = epochs[:, block]  # (40, 5, 9, samples)

        models = build_methods(window, spec)

        # CCA/FBCCA need no training; TRCA/ETRCA share training data
        trca_model = TRCA(n_fbs=N_FBS, ensemble=False)
        trca_model.fit(train_x, train_y)

        etrca_model = TRCA(n_fbs=N_FBS, ensemble=True)
        etrca_model.templates = trca_model.templates
        etrca_model.filters = trca_model.filters

        models["TRCA"] = trca_model
        models["ETRCA"] = etrca_model

        for name, model in models.items():
            if name in ("CCA", "FBCCA"):
                model.fit(train_x, train_y)
            preds, scores = model.predict(test_x)
            for cls in range(n_classes):
                confusion[name][cls, int(preds[cls])] += 1
            score_sum[name] += scores

        trca_filters_sum += trca_model.filters
        snr_count += 1

        # SNR gain: apply TRCA filter (first subband) to test data
        fs = spec.sampling_rate
        nfft = n_samples * 4
        for cls in range(n_classes):
            f0 = float(FREQS[cls])
            raw_trial = test_x[cls, 0]  # (9, samples) — first subband

            # raw: average PSD across channels
            freqs_psd, psd_raw = compute_psd(raw_trial, fs, method="fft", nfft=nfft)
            snr_raw = snr_harmonic(psd_raw, freqs_psd, f0, N_HARMONICS, N_NEIGHBORS, db=True)

            # TRCA-filtered: w[0, cls] @ trial → (samples,)
            w = trca_model.filters[0, cls]
            projected = w @ raw_trial  # (samples,)
            _, psd_filt = compute_psd(projected, fs, method="fft", nfft=nfft)
            snr_filt = snr_harmonic(psd_filt, freqs_psd, f0, N_HARMONICS, N_NEIGHBORS, db=True)

            snr_raw_acc[cls] += snr_raw
            snr_filt_acc[cls] += snr_filt

    # normalise
    for m in METHODS:
        score_sum[m] /= n_blocks

    snr_raw_mean = snr_raw_acc / snr_count
    snr_filt_mean = snr_filt_acc / snr_count
    trca_filters_mean = trca_filters_sum / n_blocks

    return {
        "confusion": confusion,
        "score_mean": score_sum,
        "snr_raw": snr_raw_mean,
        "snr_filt": snr_filt_mean,
        "snr_gain": snr_filt_mean - snr_raw_mean,
        "trca_filters": trca_filters_mean,
    }


# ── aggregation ─────────────────────────────────────────────────────

def aggregate_subjects(results: list[dict], spec: BenchmarkSpec) -> dict:
    n_classes = spec.classes
    agg = {
        "confusion": {m: np.zeros((n_classes, n_classes)) for m in METHODS},
        "score_mean": {m: np.zeros((n_classes, n_classes)) for m in METHODS},
        "snr_raw": np.zeros(n_classes),
        "snr_filt": np.zeros(n_classes),
        "snr_gain": np.zeros(n_classes),
        "snr_gain_all": [],
    }
    n = len(results)
    for r in results:
        for m in METHODS:
            agg["confusion"][m] += r["confusion"][m]
            agg["score_mean"][m] += r["score_mean"][m]
        agg["snr_raw"] += r["snr_raw"]
        agg["snr_filt"] += r["snr_filt"]
        agg["snr_gain"] += r["snr_gain"]
        agg["snr_gain_all"].append(r["snr_gain"])

    for m in METHODS:
        agg["score_mean"][m] /= n
        row_sums = agg["confusion"][m].sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        agg["confusion"][m] /= row_sums
    agg["snr_raw"] /= n
    agg["snr_filt"] /= n
    agg["snr_gain"] /= n
    agg["snr_gain_all"] = np.array(agg["snr_gain_all"])  # (n_subjects, 40)

    # per-method accuracy
    agg["accuracy"] = {}
    for m in METHODS:
        total = sum(r["confusion"][m].sum() for r in results)
        correct = sum(np.trace(r["confusion"][m]) for r in results)
        agg["accuracy"][m] = correct / total if total > 0 else 0

    return agg


# ── tables ──────────────────────────────────────────────────────────

def build_tables(agg: dict, results: list[dict], subjects: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # per-method summary
    method_rows = []
    for m in METHODS:
        diag = np.diag(agg["confusion"][m])
        method_rows.append({
            "method": m,
            "accuracy": agg["accuracy"][m],
            "mean_class_accuracy": float(diag.mean()),
            "min_class_accuracy": float(diag.min()),
            "max_class_accuracy": float(diag.max()),
        })
    method_df = pd.DataFrame(method_rows)

    # SNR gain per frequency
    snr_rows = []
    for cls in range(40):
        snr_rows.append({
            "target_idx": cls + 1,
            "frequency_hz": float(FREQS[cls]),
            "snr_raw_db": float(agg["snr_raw"][cls]),
            "snr_trca_filtered_db": float(agg["snr_filt"][cls]),
            "snr_gain_db": float(agg["snr_gain"][cls]),
        })
    snr_df = pd.DataFrame(snr_rows)

    return method_df, snr_df


# ── discrimination margin ──────────────────────────────────────────

def discrimination_margin(score_matrix: np.ndarray) -> np.ndarray:
    """Per-class margin: score[k,k] - max_{j!=k} score[k,j]."""
    n = score_matrix.shape[0]
    margins = np.zeros(n)
    for k in range(n):
        target_score = score_matrix[k, k]
        off_diag = np.delete(score_matrix[k], k)
        margins[k] = target_score - np.max(off_diag) if len(off_diag) > 0 else target_score
    return margins


# ── plotting ────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_4methods(agg: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    for ax, m in zip(axes.ravel(), METHODS):
        cm = agg["confusion"][m]
        cm_sorted = cm[np.ix_(FREQ_SORTED_IDX, FREQ_SORTED_IDX)]
        off_diag = cm_sorted.copy()
        np.fill_diagonal(off_diag, 0)
        off_log = np.log10(np.clip(off_diag, 1e-6, None))
        im = ax.imshow(off_log, aspect="equal", cmap="YlOrRd", interpolation="nearest",
                       vmin=-5, vmax=-1.5)
        ax.set_title(f"{m}  (acc={agg['accuracy'][m]:.1%})", fontsize=11)
        ax.set_xticks(range(0, 40, 5))
        ax.set_xticklabels([f"{FREQS_SORTED[i]:.0f}" for i in range(0, 40, 5)], fontsize=7)
        ax.set_yticks(range(0, 40, 5))
        ax.set_yticklabels([f"{FREQS_SORTED[i]:.0f}" for i in range(0, 40, 5)], fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.6, label="log₁₀(confusion probability)")
    fig.suptitle("Off-diagonal confusion matrices — four spatial filtering strategies", fontsize=13)
    _save(fig, fig_dir / "fig_confusion_4methods.png")


def plot_score_landscape(agg: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    for ax, m in zip(axes.ravel(), METHODS):
        sm = agg["score_mean"][m]
        sm_sorted = sm[np.ix_(FREQ_SORTED_IDX, FREQ_SORTED_IDX)]
        im = ax.imshow(sm_sorted, aspect="equal", cmap="viridis", interpolation="nearest")
        ax.set_title(f"{m} — mean score matrix", fontsize=11)
        ax.set_xticks(range(0, 40, 5))
        ax.set_xticklabels([f"{FREQS_SORTED[i]:.0f}" for i in range(0, 40, 5)], fontsize=7)
        ax.set_yticks(range(0, 40, 5))
        ax.set_yticklabels([f"{FREQS_SORTED[i]:.0f}" for i in range(0, 40, 5)], fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Mean score matrices — similarity landscape per method", fontsize=13)
    _save(fig, fig_dir / "fig_score_landscape.png")


def plot_trca_vs_etrca(agg: dict, fig_dir: Path) -> None:
    sm_trca = agg["score_mean"]["TRCA"]
    sm_etrca = agg["score_mean"]["ETRCA"]
    diff = sm_etrca - sm_trca
    diff_sorted = diff[np.ix_(FREQ_SORTED_IDX, FREQ_SORTED_IDX)]

    margin_trca = discrimination_margin(sm_trca)
    margin_etrca = discrimination_margin(sm_etrca)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    # score difference matrix
    vabs = max(abs(diff_sorted.min()), abs(diff_sorted.max()))
    im = axes[0].imshow(diff_sorted, aspect="equal", cmap="RdBu_r", interpolation="nearest",
                        vmin=-vabs, vmax=vabs)
    axes[0].set_title("ETRCA − TRCA score difference")
    axes[0].set_xticks(range(0, 40, 5))
    axes[0].set_xticklabels([f"{FREQS_SORTED[i]:.0f}" for i in range(0, 40, 5)], fontsize=7)
    axes[0].set_yticks(range(0, 40, 5))
    axes[0].set_yticklabels([f"{FREQS_SORTED[i]:.0f}" for i in range(0, 40, 5)], fontsize=7)
    fig.colorbar(im, ax=axes[0], shrink=0.7)

    # discrimination margin comparison
    sorted_idx = FREQ_SORTED_IDX
    axes[1].bar(np.arange(40) - 0.2, margin_trca[sorted_idx], 0.35, label="TRCA", color="#3b82f6", alpha=0.8)
    axes[1].bar(np.arange(40) + 0.2, margin_etrca[sorted_idx], 0.35, label="ETRCA", color="#ef4444", alpha=0.8)
    axes[1].set_xlabel("Target frequency (Hz)")
    axes[1].set_ylabel("Discrimination margin")
    axes[1].set_title("Score margin: target − best non-target")
    axes[1].set_xticks(range(0, 40, 5))
    axes[1].set_xticklabels([f"{FREQS_SORTED[i]:.0f}" for i in range(0, 40, 5)], fontsize=7)
    axes[1].legend()
    axes[1].grid(alpha=0.15)

    # margin scatter
    axes[2].scatter(margin_trca, margin_etrca, s=20, alpha=0.7, edgecolors="k", linewidths=0.3)
    lo = min(margin_trca.min(), margin_etrca.min()) - 0.01
    hi = max(margin_trca.max(), margin_etrca.max()) + 0.01
    axes[2].plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=0.8)
    axes[2].set_xlabel("TRCA margin")
    axes[2].set_ylabel("ETRCA margin")
    axes[2].set_title("Ensemble improves margin?")
    axes[2].grid(alpha=0.15)
    above = (margin_etrca > margin_trca).sum()
    axes[2].annotate(f"ETRCA better: {above}/40", xy=(0.05, 0.95), xycoords="axes fraction",
                     fontsize=9, va="top")

    fig.suptitle("TRCA vs ETRCA: same filters, different projection", fontsize=13)
    _save(fig, fig_dir / "fig_trca_vs_etrca.png")


def plot_snr_gain(agg: dict, fig_dir: Path) -> None:
    gain_sorted = agg["snr_gain"][FREQ_SORTED_IDX]
    raw_sorted = agg["snr_raw"][FREQ_SORTED_IDX]
    filt_sorted = agg["snr_filt"][FREQ_SORTED_IDX]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    # pre/post SNR
    axes[0].bar(np.arange(40) - 0.2, raw_sorted, 0.35, label="Raw (Oz avg)", color="#94a3b8")
    axes[0].bar(np.arange(40) + 0.2, filt_sorted, 0.35, label="TRCA filtered", color="#22c55e")
    axes[0].set_ylabel("Harmonic SNR (dB)")
    axes[0].set_title("SNR before and after TRCA spatial filtering (subband 1)")
    axes[0].legend()
    axes[0].grid(alpha=0.15)

    # gain
    colors = ["#22c55e" if g > 0 else "#ef4444" for g in gain_sorted]
    axes[1].bar(range(40), gain_sorted, color=colors, width=0.7)
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_ylabel("SNR gain (dB)")
    axes[1].set_xlabel("Target frequency (Hz)")
    axes[1].set_title(f"TRCA spatial filter SNR gain (mean={float(gain_sorted.mean()):.2f} dB)")
    axes[1].grid(alpha=0.15)

    for ax in axes:
        ax.set_xticks(range(40))
        ax.set_xticklabels([f"{f:.1f}" for f in FREQS_SORTED], rotation=90, fontsize=6)

    _save(fig, fig_dir / "fig_snr_gain_trca.png")


def plot_snr_gain_heatmap(agg: dict, subjects: tuple[int, ...], fig_dir: Path) -> None:
    mat = agg["snr_gain_all"][:, FREQ_SORTED_IDX]  # (n_subjects, 40)
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", interpolation="nearest",
                   vmin=-5, vmax=10)
    ax.set_xlabel("Target frequency (Hz)")
    ax.set_ylabel("Subject")
    ax.set_xticks(range(40))
    ax.set_xticklabels([f"{f:.1f}" for f in FREQS_SORTED], rotation=90, fontsize=6)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(list(subjects), fontsize=7)
    ax.set_title("TRCA spatial filter SNR gain (dB) — by subject × frequency")
    fig.colorbar(im, ax=ax, label="SNR gain (dB)", shrink=0.8)
    _save(fig, fig_dir / "fig_snr_gain_heatmap.png")


def plot_accuracy_comparison(agg: dict, fig_dir: Path) -> None:
    methods = list(METHODS)
    accs = [agg["accuracy"][m] for m in methods]
    colors = ["#94a3b8", "#60a5fa", "#3b82f6", "#1d4ed8"]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = ax.bar(methods, accs, color=colors, width=0.5, edgecolor="k", linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification accuracy comparison (window=1.0s, 35 subjects)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.15, axis="y")
    _save(fig, fig_dir / "fig_accuracy_comparison.png")


# ── main ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    subjects = parse_int_set(args.subjects)
    data_root = Path(args.data_root)
    spec = BenchmarkSpec()
    window = args.window

    out_dir = TASK_DIR / "results"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Spatial filter analysis ===")
    print(f"  subjects: {subjects}")
    print(f"  window:   {window} s")
    print(f"  methods:  {METHODS}")
    print()

    all_results: list[dict] = []
    t0 = time.time()
    for i, subj in enumerate(subjects, 1):
        ts = time.time()
        r = analyse_subject(data_root, subj, window, spec)
        all_results.append(r)
        elapsed = time.time() - ts
        accs = {m: np.trace(r["confusion"][m]) / max(r["confusion"][m].sum(), 1) for m in METHODS}
        print(f"  [{i}/{len(subjects)}] S{subj:02d}  "
              f"CCA={accs['CCA']:.1%}  FBCCA={accs['FBCCA']:.1%}  "
              f"TRCA={accs['TRCA']:.1%}  ETRCA={accs['ETRCA']:.1%}  ({elapsed:.1f}s)")

    agg = aggregate_subjects(all_results, spec)

    print(f"\n=== Aggregate accuracy ===")
    for m in METHODS:
        print(f"  {m:6s}: {agg['accuracy'][m]:.2%}")

    print(f"\n=== TRCA SNR gain ===")
    print(f"  Mean raw SNR:      {agg['snr_raw'].mean():.2f} dB")
    print(f"  Mean filtered SNR: {agg['snr_filt'].mean():.2f} dB")
    print(f"  Mean gain:         {agg['snr_gain'].mean():.2f} dB")

    # tables
    method_df, snr_df = build_tables(agg, all_results, subjects)
    method_df.to_csv(out_dir / "method_comparison.csv", index=False)
    snr_df.to_csv(out_dir / "snr_gain_by_frequency.csv", index=False)
    print(f"\n  Wrote method_comparison.csv, snr_gain_by_frequency.csv")

    # plots
    plot_confusion_4methods(agg, fig_dir)
    plot_score_landscape(agg, fig_dir)
    plot_trca_vs_etrca(agg, fig_dir)
    plot_snr_gain(agg, fig_dir)
    plot_snr_gain_heatmap(agg, subjects, fig_dir)
    plot_accuracy_comparison(agg, fig_dir)
    print(f"  Wrote 6 figures to {fig_dir}")

    total = time.time() - t0
    manifest = {
        "subjects": list(subjects),
        "window_s": window,
        "methods": list(METHODS),
        "accuracy": {m: agg["accuracy"][m] for m in METHODS},
        "mean_snr_gain_trca_db": float(agg["snr_gain"].mean()),
        "elapsed_s": round(total, 1),
    }
    (out_dir / "filter_analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  Done in {total:.1f}s")


if __name__ == "__main__":
    main()
