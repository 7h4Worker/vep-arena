"""Deep analysis of codebook subset properties.

Questions:
1. What does the greedy algorithm actually select? (spacing, SNR, frequency band)
2. Does raw SNR/PLV predict which frequencies are selected?
3. Is there a "universal" good subset across subjects?
4. ETRCA retrained (K-dim projection) vs fixed (40-dim projection)
5. Can a few trials predict the optimal subset? (practical calibration)

Usage
-----
    python analyze_codebook_deep.py
    python analyze_codebook_deep.py --subjects 1-5  # fast test
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
TARGET_K = 20


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
    p.add_argument("--subjects", type=str, default="1-35")
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    return p.parse_args()


# ── ITR ─────────────────────────────────────────────────────────────

def itr_wolpaw(n_classes: int, accuracy: float, trial_time: float) -> float:
    if n_classes <= 1 or accuracy <= 0:
        return 0.0
    if accuracy >= 1:
        return np.log2(n_classes) * (60 / trial_time)
    p = accuracy
    bits = np.log2(n_classes) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n_classes - 1))
    return max(0.0, bits) * (60 / trial_time)


def mutual_info_uniform(counts: np.ndarray) -> float:
    K = counts.shape[0]
    total = counts.sum()
    if total == 0:
        return 0.0
    P = counts / total
    py = P.sum(axis=0)
    mi = 0.0
    for i in range(K):
        for j in range(K):
            if P[i, j] > 0 and py[j] > 0:
                mi += P[i, j] * np.log2(P[i, j] * K / py[j])
    return max(0.0, mi)


# ── greedy subset ───────────────────────────────────────────────────

def greedy_best_subset(conf: np.ndarray, k: int) -> list[int]:
    selected = [int(np.argmax(np.diag(conf)))]
    while len(selected) < k:
        best_c, best_mi = -1, -1.0
        for c in range(40):
            if c in selected:
                continue
            trial = selected + [c]
            mi = mutual_info_uniform(conf[np.ix_(trial, trial)])
            if mi > best_mi:
                best_mi = mi
                best_c = c
        selected.append(best_c)
    return selected


# ── per-subject data collection ─────────────────────────────────────

def run_subject(data_root, subject, window, spec, subset_sizes):
    epochs = load_subject_filterbank(data_root, subject, window, N_FBS, BENCHMARK_CHANNELS_9, spec)
    n_classes, n_blocks = spec.classes, spec.blocks
    n_samples = epochs.shape[-1]
    trial_time = window + 0.5

    # accumulate scores and confusion over folds
    methods_info = {}
    for m in ("CCA", "FBCCA", "TRCA", "ETRCA"):
        methods_info[m] = {"scores": [], "conf40": np.zeros((40, 40))}

    fold_data = []  # for retrained experiments

    for block in range(n_blocks):
        train_mask = [b for b in range(n_blocks) if b != block]
        train_x = epochs[:, train_mask].reshape(-1, N_FBS, len(BENCHMARK_CHANNELS_9), n_samples)
        train_y = np.repeat(np.arange(n_classes), len(train_mask))
        test_x = epochs[:, block]

        cca = CCA(window=window, harmonics=5, spec=spec)
        _, cca_s = cca.predict(test_x)

        fbcca = FBCCA(window=window, harmonics=5, n_fbs=N_FBS, spec=spec)
        _, fbcca_s = fbcca.predict(test_x)

        trca = TRCA(n_fbs=N_FBS, ensemble=False)
        trca.fit(train_x, train_y)
        _, trca_s = trca.predict(test_x)

        etrca = TRCA(n_fbs=N_FBS, ensemble=True)
        etrca.templates = trca.templates
        etrca.filters = trca.filters
        _, etrca_s = etrca.predict(test_x)

        for name, scores in [("CCA", cca_s), ("FBCCA", fbcca_s),
                              ("TRCA", trca_s), ("ETRCA", etrca_s)]:
            methods_info[name]["scores"].append(scores)
            for cls in range(40):
                methods_info[name]["conf40"][cls, np.argmax(scores[cls])] += 1

        fold_data.append((train_x, train_y, test_x))

    # greedy subsets for each method and K
    result = {"subject": subject, "trial_time": trial_time}
    for m in ("CCA", "FBCCA", "TRCA", "ETRCA"):
        result[m] = {}
        conf40 = methods_info[m]["conf40"]
        acc40 = np.trace(conf40) / conf40.sum()
        result[m]["acc40"] = acc40
        result[m]["itr40"] = itr_wolpaw(40, acc40, trial_time)

        for K in subset_sizes:
            sub = greedy_best_subset(conf40, K)
            # evaluate with fixed scores
            correct, total = 0, 0
            for fold_scores in methods_info[m]["scores"]:
                for li, gi in enumerate(sub):
                    pred = sub[int(np.argmax(fold_scores[gi][sub]))]
                    if pred == gi:
                        correct += 1
                    total += 1
            acc_k = correct / total
            itr_k = itr_wolpaw(K, acc_k, trial_time)
            result[m][K] = {
                "subset": sub,
                "subset_freqs": sorted([float(FREQS[i]) for i in sub]),
                "accuracy": acc_k,
                "itr": itr_k,
            }

    # ETRCA retrained: for selected K values, train K-class ETRCA
    result["ETRCA_retrain"] = {}
    for K in subset_sizes:
        sub = greedy_best_subset(methods_info["ETRCA"]["conf40"], K)
        correct, total = 0, 0
        for train_x, train_y, test_x in fold_data:
            keep = np.isin(train_y, sub)
            sub_tx = train_x[keep]
            lmap = {g: l for l, g in enumerate(sub)}
            sub_ty = np.array([lmap[y] for y in train_y[keep]])

            etrca_k = TRCA(n_fbs=N_FBS, ensemble=True)
            etrca_k.fit(sub_tx, sub_ty)
            sub_test = test_x[sub]
            preds, _ = etrca_k.predict(sub_test)
            for li in range(K):
                if int(preds[li]) == li:
                    correct += 1
                total += 1
        acc_k = correct / total
        itr_k = itr_wolpaw(K, acc_k, trial_time)
        result["ETRCA_retrain"][K] = {
            "subset": sub,
            "accuracy": acc_k,
            "itr": itr_k,
        }

    return result


# ── analysis functions ──────────────────────────────────────────────

def analyze_subset_properties(all_results: list[dict], signal_df: pd.DataFrame | None) -> dict:
    """What does the greedy algorithm select?"""
    analysis = {}

    for method in ("CCA", "FBCCA"):
        freq_selection_count = np.zeros(40)
        spacings = []
        selected_snr = []
        rejected_snr = []

        for r in all_results:
            if TARGET_K not in r[method]:
                continue
            sub = r[method][TARGET_K]["subset"]
            for idx in sub:
                freq_selection_count[idx] += 1

            # spacing: min distance between selected frequencies
            sel_freqs = sorted([FREQS[i] for i in sub])
            diffs = np.diff(sel_freqs)
            spacings.extend(diffs.tolist())

            # SNR correlation
            if signal_df is not None:
                subj = r["subject"]
                sig_sub = signal_df[signal_df["subject"] == subj].sort_values("target_idx")
                snr_vals = sig_sub["snr_harm_oz_db"].values
                for idx in range(40):
                    if idx in sub:
                        selected_snr.append(snr_vals[idx])
                    else:
                        rejected_snr.append(snr_vals[idx])

        analysis[method] = {
            "selection_frequency": freq_selection_count,
            "mean_spacing_hz": float(np.mean(spacings)) if spacings else 0,
            "median_spacing_hz": float(np.median(spacings)) if spacings else 0,
            "min_spacing_hz": float(np.min(spacings)) if spacings else 0,
            "selected_snr_mean": float(np.mean(selected_snr)) if selected_snr else 0,
            "rejected_snr_mean": float(np.mean(rejected_snr)) if rejected_snr else 0,
        }

    return analysis


def find_universal_subset(all_results: list[dict], method: str, K: int) -> list[int]:
    """Pool confusion matrices across subjects, then greedy select."""
    pooled = np.zeros((40, 40))
    for r in all_results:
        # reconstruct confusion from scores
        for fold_scores_entry in [r]:
            pass
    # use selection frequency as proxy: pick the K most-selected classes
    freq_count = np.zeros(40)
    for r in all_results:
        if K in r[method]:
            for idx in r[method][K]["subset"]:
                freq_count[idx] += 1
    return list(np.argsort(freq_count)[-K:][::-1])


# ── plotting ────────────────────────────────────────────────────────

def _save(fig, path):
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_selection_frequency(analysis: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

    for ax, method in zip(axes, ("CCA", "FBCCA")):
        counts = analysis[method]["selection_frequency"]
        counts_sorted = counts[FREQ_SORTED_IDX]
        colors = ["#3b82f6" if c > np.median(counts_sorted) else "#94a3b8"
                  for c in counts_sorted]
        ax.bar(range(40), counts_sorted, color=colors, width=0.7)
        ax.set_xlabel("Target frequency (Hz)")
        ax.set_ylabel(f"Selection count (out of {int(counts.sum() / TARGET_K)} subjects)")
        ax.set_title(f"{method} — how often is each frequency selected in K={TARGET_K} subset?")
        ax.set_xticks(range(40))
        ax.set_xticklabels([f"{f:.1f}" for f in FREQS_SORTED], rotation=90, fontsize=6)
        ax.grid(alpha=0.15, axis="y")

    fig.suptitle(f"Greedy subset selection frequency (K={TARGET_K}, per-subject)", fontsize=13)
    _save(fig, fig_dir / "fig_selection_frequency.png")


def plot_selected_vs_rejected_snr(analysis: dict, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    methods = ["CCA", "FBCCA"]
    x = np.arange(len(methods))
    sel = [analysis[m]["selected_snr_mean"] for m in methods]
    rej = [analysis[m]["rejected_snr_mean"] for m in methods]
    ax.bar(x - 0.15, sel, 0.3, label="Selected", color="#22c55e")
    ax.bar(x + 0.15, rej, 0.3, label="Rejected", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean harmonic SNR (dB)")
    ax.set_title(f"SNR of selected vs rejected frequencies (K={TARGET_K})")
    ax.legend()
    ax.grid(alpha=0.15, axis="y")
    _save(fig, fig_dir / "fig_selected_vs_rejected_snr.png")


def plot_etrca_retrain_comparison(all_results: list[dict], subset_sizes: list[int],
                                   fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # ITR
    etrca_fixed = [np.mean([r["ETRCA"][K]["itr"] for r in all_results]) for K in subset_sizes]
    etrca_retrain = [np.mean([r["ETRCA_retrain"][K]["itr"] for r in all_results]) for K in subset_sizes]

    axes[0].plot(subset_sizes, etrca_fixed, "o-", color="#1d4ed8", label="ETRCA (40-class filters, K argmax)", linewidth=1.5)
    axes[0].plot(subset_sizes, etrca_retrain, "s--", color="#dc2626", label="ETRCA (retrained K-class)", linewidth=1.5)
    axes[0].set_xlabel("Codebook size K")
    axes[0].set_ylabel("ITR (bits/min)")
    axes[0].set_title("ITR comparison")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.15)
    axes[0].set_xticks(subset_sizes)

    # Accuracy
    etrca_fixed_acc = [np.mean([r["ETRCA"][K]["accuracy"] for r in all_results]) for K in subset_sizes]
    etrca_retrain_acc = [np.mean([r["ETRCA_retrain"][K]["accuracy"] for r in all_results]) for K in subset_sizes]

    axes[1].plot(subset_sizes, etrca_fixed_acc, "o-", color="#1d4ed8", label="ETRCA fixed", linewidth=1.5)
    axes[1].plot(subset_sizes, etrca_retrain_acc, "s--", color="#dc2626", label="ETRCA retrained", linewidth=1.5)
    axes[1].set_xlabel("Codebook size K")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy comparison")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.15)
    axes[1].set_xticks(subset_sizes)
    axes[1].set_ylim(0.85, 1.01)

    fig.suptitle("ETRCA: 40D fixed projection vs K-dim retrained — does dimension matter?", fontsize=12)
    _save(fig, fig_dir / "fig_etrca_retrain_comparison.png")


def plot_per_subject_subset_overlap(all_results: list[dict], fig_dir: Path) -> None:
    """How much do per-subject optimal subsets overlap?"""
    n = len(all_results)
    for method in ("CCA", "FBCCA"):
        overlap = np.zeros((n, n))
        for i in range(n):
            si = set(all_results[i][method][TARGET_K]["subset"])
            for j in range(n):
                sj = set(all_results[j][method][TARGET_K]["subset"])
                overlap[i, j] = len(si & sj) / TARGET_K

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, method in zip(axes, ("CCA", "FBCCA")):
        mat = np.zeros((n, n))
        for i in range(n):
            si = set(all_results[i][method][TARGET_K]["subset"])
            for j in range(n):
                sj = set(all_results[j][method][TARGET_K]["subset"])
                mat[i, j] = len(si & sj) / TARGET_K
        im = ax.imshow(mat, aspect="equal", cmap="YlGn", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"{method} — pairwise subset overlap (K={TARGET_K})")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Subject")
        subjects = [r["subject"] for r in all_results]
        ax.set_xticks(range(n))
        ax.set_xticklabels(subjects, fontsize=6, rotation=90)
        ax.set_yticks(range(n))
        ax.set_yticklabels(subjects, fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.7, label="Jaccard overlap")

    mean_overlap = float(mat[np.triu_indices(n, k=1)].mean())
    fig.suptitle(f"Per-subject optimal subset overlap (mean off-diag={mean_overlap:.2f})", fontsize=12)
    _save(fig, fig_dir / "fig_subset_overlap.png")


def plot_subset_spacing_histogram(all_results: list[dict], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, method in zip(axes, ("CCA", "FBCCA")):
        all_spacings = []
        for r in all_results:
            if TARGET_K not in r[method]:
                continue
            sel_freqs = sorted([FREQS[i] for i in r[method][TARGET_K]["subset"]])
            all_spacings.extend(np.diff(sel_freqs).tolist())

        # reference: even spacing = (15.8-8.0)/(K-1) for K=20 → 0.41 Hz
        even_spacing = (FREQS.max() - FREQS.min()) / (TARGET_K - 1)

        ax.hist(all_spacings, bins=30, color="#3b82f6", alpha=0.7, edgecolor="k", linewidth=0.3)
        ax.axvline(even_spacing, color="#dc2626", linestyle="--", linewidth=1.5,
                   label=f"Even spacing ({even_spacing:.2f} Hz)")
        ax.axvline(0.2, color="#94a3b8", linestyle=":", linewidth=1,
                   label="Original spacing (0.2 Hz)")
        ax.set_xlabel("Spacing between adjacent selected frequencies (Hz)")
        ax.set_ylabel("Count")
        ax.set_title(f"{method} — spacing distribution (K={TARGET_K})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)

    fig.suptitle("How spaced are the greedy-selected frequencies?", fontsize=13)
    _save(fig, fig_dir / "fig_subset_spacing.png")


# ── main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    subjects = parse_int_set(args.subjects)
    data_root = Path(args.data_root)
    spec = BenchmarkSpec()

    fig_dir = TASK_DIR / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    subset_sizes = [5, 8, 10, 15, 20, 25, 30, 35, 40]

    # load signal profile if available
    sig_path = TASK_DIR / "results" / "signal_profile.csv"
    signal_df = pd.read_csv(sig_path) if sig_path.exists() else None

    print("=== Deep codebook analysis ===")
    print(f"  subjects: {subjects}")
    print(f"  target K for property analysis: {TARGET_K}")
    print()

    all_results = []
    t0 = time.time()
    for i, subj in enumerate(subjects, 1):
        ts = time.time()
        r = run_subject(data_root, subj, args.window, spec, subset_sizes)
        all_results.append(r)
        elapsed = time.time() - ts

        cca_opt_k = max(subset_sizes, key=lambda k: r["CCA"].get(k, {}).get("itr", 0) if k != "acc40" and k != "itr40" else 0)
        print(f"  [{i}/{len(subjects)}] S{subj:02d}  CCA-opt-K={cca_opt_k}  ({elapsed:.1f}s)")

    # ── ETRCA retrain comparison ────
    print("\n=== ETRCA fixed vs retrained ===")
    print(f"  {'K':>3s}  {'ETRCA fixed':>12s}  {'ETRCA retrain':>14s}  {'Δ ITR':>8s}")
    for K in subset_sizes:
        fix_itr = np.mean([r["ETRCA"][K]["itr"] for r in all_results])
        ret_itr = np.mean([r["ETRCA_retrain"][K]["itr"] for r in all_results])
        fix_acc = np.mean([r["ETRCA"][K]["accuracy"] for r in all_results])
        ret_acc = np.mean([r["ETRCA_retrain"][K]["accuracy"] for r in all_results])
        print(f"  {K:3d}  {fix_itr:7.1f} ({fix_acc:.1%})  {ret_itr:9.1f} ({ret_acc:.1%})  {ret_itr - fix_itr:+7.1f}")

    # ── subset property analysis ────
    print(f"\n=== Subset properties (K={TARGET_K}) ===")
    analysis = analyze_subset_properties(all_results, signal_df)
    for m in ("CCA", "FBCCA"):
        a = analysis[m]
        print(f"\n  {m}:")
        print(f"    Mean spacing:      {a['mean_spacing_hz']:.3f} Hz  (even would be {(FREQS.max()-FREQS.min())/(TARGET_K-1):.3f})")
        print(f"    Median spacing:    {a['median_spacing_hz']:.3f} Hz")
        print(f"    Min spacing:       {a['min_spacing_hz']:.3f} Hz")
        if signal_df is not None:
            print(f"    Selected mean SNR: {a['selected_snr_mean']:.2f} dB")
            print(f"    Rejected mean SNR: {a['rejected_snr_mean']:.2f} dB")

    # ── inter-subject overlap ────
    print(f"\n=== Inter-subject subset overlap (K={TARGET_K}) ===")
    for method in ("CCA", "FBCCA"):
        overlaps = []
        n = len(all_results)
        for i in range(n):
            si = set(all_results[i][method][TARGET_K]["subset"])
            for j in range(i + 1, n):
                sj = set(all_results[j][method][TARGET_K]["subset"])
                overlaps.append(len(si & sj) / TARGET_K)
        print(f"  {method}: mean overlap = {np.mean(overlaps):.2%}, std = {np.std(overlaps):.2%}")

    # ── universal subset (most-popular classes) ────
    print(f"\n=== Universal subset test (K={TARGET_K}) ===")
    for method in ("CCA", "FBCCA"):
        freq_count = np.zeros(40)
        for r in all_results:
            for idx in r[method][TARGET_K]["subset"]:
                freq_count[idx] += 1
        universal = list(np.argsort(freq_count)[-TARGET_K:][::-1])
        universal_freqs = sorted([FREQS[i] for i in universal])
        print(f"\n  {method} universal K={TARGET_K}:")
        print(f"    Frequencies: {[f'{f:.1f}' for f in universal_freqs]}")

        # evaluate universal on all subjects
        itrs = []
        for r in all_results:
            correct, total = 0, 0
            # need scores... use from the stored data
            # actually we can approximate from the confusion matrix
            pass

    # ── plots ────
    plot_selection_frequency(analysis, fig_dir)
    plot_etrca_retrain_comparison(all_results, subset_sizes, fig_dir)
    plot_per_subject_subset_overlap(all_results, fig_dir)
    plot_subset_spacing_histogram(all_results, fig_dir)
    if signal_df is not None:
        plot_selected_vs_rejected_snr(analysis, fig_dir)

    total = time.time() - t0
    print(f"\n  Done in {total:.1f}s — figures written to {fig_dir}")


if __name__ == "__main__":
    main()
