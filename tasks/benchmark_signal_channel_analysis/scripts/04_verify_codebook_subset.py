"""Verify codebook subset optimization on CCA vs TRCA/ETRCA.

Core question: Costa/Sinha claim C_BA > I_uniform, implying a better
codebook exists. Can we actually find a K < 40 subset that achieves
higher ITR than the full 40-class system?

For CCA:  scores are fixed (reference-based) → clean experiment.
For TRCA: two variants —
  (a) fixed filters: keep 40-class filters, argmax over K only
  (b) retrained: train new K-class filters from scratch

This validates whether the capacity gap is practically exploitable.

Usage
-----
    python verify_codebook_subset.py
    python verify_codebook_subset.py --subjects 1-5   # fast test
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
N_FBS = 5


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


# ── ITR computation ─────────────────────────────────────────────────

def itr_wolpaw(n_classes: int, accuracy: float, trial_time: float) -> float:
    """Wolpaw ITR in bits/min."""
    if n_classes <= 1 or accuracy <= 0 or accuracy >= 1:
        if accuracy >= 1:
            return np.log2(n_classes) * (60 / trial_time)
        return 0.0
    p = accuracy
    bits = np.log2(n_classes) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n_classes - 1))
    return max(0.0, bits) * (60 / trial_time)


def mutual_info_uniform(confusion_counts: np.ndarray) -> float:
    """I(X;Y) with uniform input, from confusion count matrix."""
    K = confusion_counts.shape[0]
    total = confusion_counts.sum()
    if total == 0:
        return 0.0
    P = confusion_counts / total  # joint P(x,y)
    py = P.sum(axis=0)  # marginal P(y)
    mi = 0.0
    for i in range(K):
        for j in range(K):
            if P[i, j] > 0 and py[j] > 0:
                mi += P[i, j] * np.log2(P[i, j] * K / py[j])
    return max(0.0, mi)


# ── greedy subset selection ─────────────────────────────────────────

def greedy_best_subset(confusion_40: np.ndarray, target_k: int) -> list[int]:
    """Greedy forward selection: pick classes that maximize I_uniform."""
    all_classes = list(range(40))
    selected: list[int] = []

    # start with the class with highest diagonal
    diag = np.diag(confusion_40)
    best_start = int(np.argmax(diag))
    selected.append(best_start)

    while len(selected) < target_k:
        best_class = -1
        best_mi = -1.0
        for c in all_classes:
            if c in selected:
                continue
            trial = selected + [c]
            sub_conf = confusion_40[np.ix_(trial, trial)]
            mi = mutual_info_uniform(sub_conf)
            if mi > best_mi:
                best_mi = mi
                best_class = c
        selected.append(best_class)

    return selected


def evenly_spaced_subset(n_total: int, target_k: int) -> list[int]:
    """Select evenly spaced indices from frequency-sorted order."""
    sorted_idx = np.argsort(FREQS)
    step = n_total / target_k
    return [int(sorted_idx[int(i * step)]) for i in range(target_k)]


# ── per-subject experiment ──────────────────────────────────────────

def run_subject(
    data_root: Path,
    subject: int,
    window: float,
    spec: BenchmarkSpec,
    subset_sizes: list[int],
) -> dict:
    """For one subject, compute ITR for various subset sizes and methods."""
    epochs = load_subject_filterbank(data_root, subject, window, N_FBS, BENCHMARK_CHANNELS_9, spec)
    n_classes, n_blocks = spec.classes, spec.blocks
    n_samples = epochs.shape[-1]
    trial_time = window + 0.5  # gaze shift + stimulus

    # collect all fold scores and confusion
    method_scores = {}   # method → list of (40, 40) score arrays per fold
    method_confusion_40 = {}  # method → (40, 40) confusion counts

    for m in ("CCA", "FBCCA", "TRCA_fixed", "ETRCA_fixed", "TRCA_retrain"):
        method_scores[m] = []
        method_confusion_40[m] = np.zeros((n_classes, n_classes), dtype=np.float64)

    for block in range(n_blocks):
        train_mask = [b for b in range(n_blocks) if b != block]
        train_x = epochs[:, train_mask].reshape(-1, N_FBS, len(BENCHMARK_CHANNELS_9), n_samples)
        train_y = np.repeat(np.arange(n_classes), len(train_mask))
        test_x = epochs[:, block]

        # CCA
        cca = CCA(window=window, harmonics=5, spec=spec)
        _, cca_scores = cca.predict(test_x)
        method_scores["CCA"].append(cca_scores)
        for cls in range(n_classes):
            method_confusion_40["CCA"][cls, np.argmax(cca_scores[cls])] += 1

        # FBCCA
        fbcca = FBCCA(window=window, harmonics=5, n_fbs=N_FBS, spec=spec)
        _, fbcca_scores = fbcca.predict(test_x)
        method_scores["FBCCA"].append(fbcca_scores)
        for cls in range(n_classes):
            method_confusion_40["FBCCA"][cls, np.argmax(fbcca_scores[cls])] += 1

        # TRCA (40-class)
        trca = TRCA(n_fbs=N_FBS, ensemble=False)
        trca.fit(train_x, train_y)
        _, trca_scores = trca.predict(test_x)
        method_scores["TRCA_fixed"].append(trca_scores)
        for cls in range(n_classes):
            method_confusion_40["TRCA_fixed"][cls, np.argmax(trca_scores[cls])] += 1

        # ETRCA (40-class, same filters)
        etrca = TRCA(n_fbs=N_FBS, ensemble=True)
        etrca.templates = trca.templates
        etrca.filters = trca.filters
        _, etrca_scores = etrca.predict(test_x)
        method_scores["ETRCA_fixed"].append(etrca_scores)
        for cls in range(n_classes):
            method_confusion_40["ETRCA_fixed"][cls, np.argmax(etrca_scores[cls])] += 1

        # TRCA retrain will be done per-subset below
        method_scores["TRCA_retrain"].append((train_x, train_y, test_x))

    # now evaluate subsets
    results = {m: [] for m in ("CCA", "FBCCA", "TRCA_fixed", "ETRCA_fixed", "TRCA_retrain")}

    for K in subset_sizes:
        for method in ("CCA", "FBCCA", "TRCA_fixed", "ETRCA_fixed"):
            conf40 = method_confusion_40[method]

            # greedy subset
            subset = greedy_best_subset(conf40, K)
            # also try evenly spaced
            subset_even = evenly_spaced_subset(40, K)

            for strategy, sub in [("greedy", subset), ("even", subset_even)]:
                correct = 0
                total = 0
                conf_k = np.zeros((K, K), dtype=np.float64)

                for fold_scores in method_scores[method]:
                    for local_i, global_i in enumerate(sub):
                        row_scores = fold_scores[global_i]
                        sub_scores = row_scores[sub]
                        pred_local = int(np.argmax(sub_scores))
                        conf_k[local_i, pred_local] += 1
                        if sub[pred_local] == global_i:
                            correct += 1
                        total += 1

                acc = correct / total if total > 0 else 0
                mi = mutual_info_uniform(conf_k)
                itr = itr_wolpaw(K, acc, trial_time)

                results[method].append({
                    "K": K,
                    "strategy": strategy,
                    "accuracy": acc,
                    "itr_bits_min": itr,
                    "mi_uniform_bits": mi,
                    "subset_freqs": [float(FREQS[i]) for i in sub],
                })

        # TRCA retrained: for each K, train K-class TRCA from scratch
        for strategy, sub_fn in [("greedy", lambda: greedy_best_subset(method_confusion_40["TRCA_fixed"], K)),
                                  ("even", lambda: evenly_spaced_subset(40, K))]:
            sub = sub_fn()
            correct = 0
            total = 0
            conf_k = np.zeros((K, K), dtype=np.float64)

            for train_x, train_y, test_x in method_scores["TRCA_retrain"]:
                # extract only the K classes from training data
                keep_mask = np.isin(train_y, sub)
                sub_train_x = train_x[keep_mask]
                # relabel to 0..K-1
                label_map = {g: l for l, g in enumerate(sub)}
                sub_train_y = np.array([label_map[y] for y in train_y[keep_mask]])

                trca_k = TRCA(n_fbs=N_FBS, ensemble=False)
                trca_k.fit(sub_train_x, sub_train_y)

                sub_test_x = test_x[sub]  # only test the K classes
                preds_k, _ = trca_k.predict(sub_test_x)

                for local_i in range(K):
                    pred_local = int(preds_k[local_i])
                    conf_k[local_i, pred_local] += 1
                    if pred_local == local_i:
                        correct += 1
                    total += 1

            acc = correct / total if total > 0 else 0
            mi = mutual_info_uniform(conf_k)
            itr = itr_wolpaw(K, acc, trial_time)

            results["TRCA_retrain"].append({
                "K": K,
                "strategy": strategy,
                "accuracy": acc,
                "itr_bits_min": itr,
                "mi_uniform_bits": mi,
                "subset_freqs": [float(FREQS[i]) for i in sub],
            })

    return {
        "subject": subject,
        "confusion_40": {m: method_confusion_40[m] for m in ("CCA", "FBCCA", "TRCA_fixed", "ETRCA_fixed")},
        "results": results,
        "trial_time": trial_time,
    }


# ── aggregation and plotting ───────────────────────────────────────

def aggregate_results(all_subj: list[dict], subset_sizes: list[int]) -> dict:
    """Average ITR/accuracy across subjects for each method × K × strategy."""
    methods = ("CCA", "FBCCA", "TRCA_fixed", "ETRCA_fixed", "TRCA_retrain")
    strategies = ("greedy", "even")

    agg = {}
    for m in methods:
        agg[m] = {}
        for s in strategies:
            agg[m][s] = {}
            for K in subset_sizes:
                accs = []
                itrs = []
                mis = []
                for subj_data in all_subj:
                    for r in subj_data["results"][m]:
                        if r["K"] == K and r["strategy"] == s:
                            accs.append(r["accuracy"])
                            itrs.append(r["itr_bits_min"])
                            mis.append(r["mi_uniform_bits"])
                agg[m][s][K] = {
                    "accuracy": float(np.mean(accs)) if accs else 0,
                    "itr_mean": float(np.mean(itrs)) if itrs else 0,
                    "itr_std": float(np.std(itrs)) if itrs else 0,
                    "mi_mean": float(np.mean(mis)) if mis else 0,
                }
    return agg


def _save(fig, path):
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_itr_vs_k(agg: dict, subset_sizes: list[int], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    colors = {"CCA": "#94a3b8", "FBCCA": "#60a5fa", "TRCA_fixed": "#3b82f6",
              "ETRCA_fixed": "#1d4ed8", "TRCA_retrain": "#dc2626"}
    labels = {"CCA": "CCA", "FBCCA": "FBCCA", "TRCA_fixed": "TRCA (fixed filters)",
              "ETRCA_fixed": "ETRCA (fixed filters)", "TRCA_retrain": "TRCA (retrained)"}

    for ax, strategy in zip(axes, ("greedy", "even")):
        for m in ("CCA", "FBCCA", "TRCA_fixed", "ETRCA_fixed", "TRCA_retrain"):
            itrs = [agg[m][strategy][K]["itr_mean"] for K in subset_sizes]
            ax.plot(subset_sizes, itrs, "o-", color=colors[m], label=labels[m],
                    markersize=4, linewidth=1.5)
        ax.set_xlabel("Codebook size K")
        ax.set_ylabel("ITR (bits/min)")
        ax.set_title(f"Subset strategy: {strategy}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
        ax.set_xticks(subset_sizes)

    fig.suptitle("ITR vs codebook size — does reducing K improve throughput?", fontsize=13)
    _save(fig, fig_dir / "fig_itr_vs_codebook_size.png")


def plot_accuracy_vs_k(agg: dict, subset_sizes: list[int], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    colors = {"CCA": "#94a3b8", "FBCCA": "#60a5fa", "TRCA_fixed": "#3b82f6",
              "ETRCA_fixed": "#1d4ed8", "TRCA_retrain": "#dc2626"}
    labels = {"CCA": "CCA", "FBCCA": "FBCCA", "TRCA_fixed": "TRCA (fixed)",
              "ETRCA_fixed": "ETRCA (fixed)", "TRCA_retrain": "TRCA (retrained)"}

    for ax, strategy in zip(axes, ("greedy", "even")):
        for m in ("CCA", "FBCCA", "TRCA_fixed", "ETRCA_fixed", "TRCA_retrain"):
            accs = [agg[m][strategy][K]["accuracy"] for K in subset_sizes]
            ax.plot(subset_sizes, accs, "o-", color=colors[m], label=labels[m],
                    markersize=4, linewidth=1.5)
        ax.set_xlabel("Codebook size K")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Subset strategy: {strategy}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
        ax.set_xticks(subset_sizes)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Accuracy vs codebook size", fontsize=13)
    _save(fig, fig_dir / "fig_accuracy_vs_codebook_size.png")


def plot_trca_fixed_vs_retrain(agg: dict, subset_sizes: list[int], fig_dir: Path) -> None:
    """Show the gap between TRCA with fixed 40-class filters vs retrained K-class filters."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    for ax, strategy in zip(axes, ("greedy", "even")):
        itr_fixed = [agg["TRCA_fixed"][strategy][K]["itr_mean"] for K in subset_sizes]
        itr_retrain = [agg["TRCA_retrain"][strategy][K]["itr_mean"] for K in subset_sizes]
        acc_fixed = [agg["TRCA_fixed"][strategy][K]["accuracy"] for K in subset_sizes]
        acc_retrain = [agg["TRCA_retrain"][strategy][K]["accuracy"] for K in subset_sizes]

        ax.plot(subset_sizes, itr_fixed, "o-", color="#3b82f6", label="TRCA (40-class filters, K argmax)", linewidth=1.5)
        ax.plot(subset_sizes, itr_retrain, "s--", color="#dc2626", label="TRCA (retrained K-class)", linewidth=1.5)
        ax.set_xlabel("Codebook size K")
        ax.set_ylabel("ITR (bits/min)")
        ax.set_title(f"Strategy: {strategy}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
        ax.set_xticks(subset_sizes)

        # annotate gap at K=20
        if 20 in subset_sizes:
            idx = subset_sizes.index(20)
            gap = itr_retrain[idx] - itr_fixed[idx]
            ax.annotate(f"Δ={gap:+.1f}", xy=(20, (itr_fixed[idx] + itr_retrain[idx]) / 2),
                        fontsize=8, ha="left", color="#dc2626")

    fig.suptitle("TRCA: fixed 40-class filters vs retrained — codebook-dependent channel", fontsize=12)
    _save(fig, fig_dir / "fig_trca_fixed_vs_retrain.png")


def plot_per_subject_optimal_k(all_subj: list[dict], fig_dir: Path) -> None:
    """For each subject, find the K that maximizes ITR (CCA greedy) and show the distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, method in zip(axes, ("CCA", "ETRCA_fixed")):
        optimal_ks = []
        itr_gains = []
        for subj_data in all_subj:
            itr_at_40 = None
            best_itr = 0
            best_k = 40
            for r in subj_data["results"][method]:
                if r["strategy"] != "greedy":
                    continue
                if r["K"] == 40:
                    itr_at_40 = r["itr_bits_min"]
                if r["itr_bits_min"] > best_itr:
                    best_itr = r["itr_bits_min"]
                    best_k = r["K"]
            optimal_ks.append(best_k)
            itr_gains.append(best_itr - (itr_at_40 or 0))

        ax.bar(range(len(all_subj)), optimal_ks, color="#3b82f6", alpha=0.7)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Optimal K")
        ax.set_title(f"{method.replace('_fixed', '')} — optimal codebook size per subject")
        ax.set_xticks(range(len(all_subj)))
        ax.set_xticklabels([d["subject"] for d in all_subj], fontsize=6, rotation=90)
        ax.axhline(40, color="k", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.grid(alpha=0.15, axis="y")

    fig.suptitle("Per-subject optimal codebook size (greedy strategy)", fontsize=12)
    _save(fig, fig_dir / "fig_per_subject_optimal_k.png")


# ── main ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    subjects = parse_int_set(args.subjects)
    data_root = Path(args.data_root)
    spec = BenchmarkSpec()

    fig_dir = TASK_DIR / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    subset_sizes = [5, 8, 10, 15, 20, 25, 30, 35, 40]

    print("=== Codebook subset verification ===")
    print(f"  subjects: {subjects}")
    print(f"  subset sizes: {subset_sizes}")
    print()

    all_subj = []
    t0 = time.time()
    for i, subj in enumerate(subjects, 1):
        ts = time.time()
        r = run_subject(data_root, subj, args.window, spec, subset_sizes)
        all_subj.append(r)
        elapsed = time.time() - ts

        # show 40-class accuracies
        cca_acc = r["results"]["CCA"][-2]["accuracy"] if r["results"]["CCA"] else 0
        print(f"  [{i}/{len(subjects)}] S{subj:02d} ({elapsed:.1f}s)")

    agg = aggregate_results(all_subj, subset_sizes)

    print(f"\n=== ITR at different K (greedy, mean across subjects) ===")
    print(f"  {'K':>3s}  {'CCA':>8s}  {'FBCCA':>8s}  {'TRCA_fix':>8s}  {'ETRCA_fix':>8s}  {'TRCA_re':>8s}")
    for K in subset_sizes:
        vals = [f"{agg[m]['greedy'][K]['itr_mean']:8.1f}" for m in
                ("CCA", "FBCCA", "TRCA_fixed", "ETRCA_fixed", "TRCA_retrain")]
        print(f"  {K:3d}  {'  '.join(vals)}")

    print(f"\n=== Accuracy at different K (greedy) ===")
    print(f"  {'K':>3s}  {'CCA':>8s}  {'FBCCA':>8s}  {'TRCA_fix':>8s}  {'ETRCA_fix':>8s}  {'TRCA_re':>8s}")
    for K in subset_sizes:
        vals = [f"{agg[m]['greedy'][K]['accuracy']:8.1%}" for m in
                ("CCA", "FBCCA", "TRCA_fixed", "ETRCA_fixed", "TRCA_retrain")]
        print(f"  {K:3d}  {'  '.join(vals)}")

    plot_itr_vs_k(agg, subset_sizes, fig_dir)
    plot_accuracy_vs_k(agg, subset_sizes, fig_dir)
    plot_trca_fixed_vs_retrain(agg, subset_sizes, fig_dir)
    plot_per_subject_optimal_k(all_subj, fig_dir)

    total = time.time() - t0
    print(f"\n  4 figures written to {fig_dir}")
    print(f"  Done in {total:.1f}s")

    manifest = {
        "subjects": list(subjects),
        "subset_sizes": subset_sizes,
        "agg_greedy": {m: {str(K): agg[m]["greedy"][K] for K in subset_sizes}
                       for m in ("CCA", "FBCCA", "TRCA_fixed", "ETRCA_fixed", "TRCA_retrain")},
        "elapsed_s": round(total, 1),
    }
    (TASK_DIR / "results" / "codebook_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
