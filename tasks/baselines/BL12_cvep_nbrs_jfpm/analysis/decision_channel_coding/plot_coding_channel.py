from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

PROJECT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT))

from vep_arena.channel.capacity import capacity_ba, capacity_c0, capacity_c1, mutual_info_uniform
from vep_arena.channel.confusion import normalize_confusion


TASK = Path(__file__).resolve().parents[2]
ANALYSIS = Path(__file__).resolve().parent
DATASET = Path(r"D:\ProjData\datasets\cvep_nbrs_jfpm_tsinghua_2024")
COMPARISON = TASK / "results" / "comparison_occipital9_fbcca_trca_mstrca_v2_20260706"
FBCCA_TRCA = TASK / "results" / "full_occipital9_fbcca_trca_w04_40_20260706"
MSTRCA = TASK / "results" / "full_occipital9_mstrca_v2_paper_preproc_w04_40_20260706"

FIG_DIR = ANALYSIS / "figures"
TABLE_DIR = ANALYSIS / "tables"
CLASSES = 40
GAZE_SHIFT_SECONDS = 0.5


PARADIGM_META = {
    "NBRS-15": {
        "kind": "Narrow-band random sequence",
        "targets": 40,
        "code_file": "NBRS-15.mat",
        "band": "15-25 Hz",
        "display_hz": 120,
        "code_length_frames": 120,
        "repeat_in_4s": 4,
        "reference": "code",
    },
    "NBRS-8": {
        "kind": "Narrow-band random sequence",
        "targets": 40,
        "code_file": "NBRS-8.mat",
        "band": "8-16 Hz",
        "display_hz": 120,
        "code_length_frames": 120,
        "repeat_in_4s": 4,
        "reference": "code",
    },
    "JFPM-8": {
        "kind": "Joint frequency-phase modulation",
        "targets": 40,
        "code_file": "JFPM-8.mat",
        "band": "8.0-15.8 Hz, 0.2 Hz step, 0.5 pi phase step",
        "display_hz": 120,
        "code_length_frames": 120,
        "repeat_in_4s": 4,
        "reference": "sinusoid",
    },
}


METHOD_COLORS = {
    "FBCCA-CODE": "#1f77b4",
    "TRCA": "#2ca02c",
    "MSTRCA": "#d62728",
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_code(paradigm: str) -> np.ndarray:
    path = DATASET / "metadata" / "supplementary_extracted" / str(PARADIGM_META[paradigm]["code_file"])
    return np.asarray(loadmat(path)["code"], dtype=np.float64)


def load_summary() -> pd.DataFrame:
    summary = pd.read_csv(COMPARISON / "summary_all_methods.csv")
    summary["window"] = summary["window"].astype(float)
    return summary


def load_confusion(paradigm: str, method: str, window: float) -> np.ndarray:
    root = MSTRCA if method == "MSTRCA" else FBCCA_TRCA
    candidates = [
        root / "confusions" / f"{paradigm}_{method}_w{window:g}.npy",
        root / "confusions" / f"{paradigm}_{method}_w{window:.1f}.npy",
    ]
    for path in candidates:
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(f"Missing confusion for {paradigm} {method} w={window:g}")


def code_similarity(code: np.ndarray) -> np.ndarray:
    x = code[:, :120].astype(np.float64, copy=False)
    x = x - x.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    x = x / denom
    return x @ x.T


def plot_code_prototypes() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.4), gridspec_kw={"width_ratios": [1.2, 1.0]})
    for row, paradigm in enumerate(["NBRS-15", "NBRS-8", "JFPM-8"]):
        code = load_code(paradigm)
        t = np.arange(120) / float(PARADIGM_META[paradigm]["display_hz"])
        ax = axes[row, 0]
        for idx in range(3):
            ax.plot(t, code[idx, :120] + idx * 1.25, lw=1.2, label=f"target {idx + 1}")
        ax.set_title(f"{paradigm}: first-cycle prototype")
        ax.set_xlabel("Time in one 120-frame code cycle (s)")
        ax.set_ylabel("Normalized code + offset")
        ax.set_yticks([])
        ax.grid(alpha=0.2)
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)

        sim = code_similarity(code)
        ax = axes[row, 1]
        im = ax.imshow(sim, vmin=-1, vmax=1, cmap="coolwarm", aspect="equal")
        ax.set_title(f"{paradigm}: code correlation")
        ax.set_xlabel("Target")
        ax.set_ylabel("Target")
        ax.set_xticks([0, 9, 19, 29, 39], [1, 10, 20, 30, 40])
        ax.set_yticks([0, 9, 19, 29, 39], [1, 10, 20, 30, 40])
    fig.subplots_adjust(left=0.07, right=0.88, top=0.94, bottom=0.07, hspace=0.42, wspace=0.28)
    cbar_ax = fig.add_axes([0.91, 0.18, 0.018, 0.64])
    fig.colorbar(im, cax=cbar_ax, label="Pearson r")
    fig.savefig(FIG_DIR / "fig01_code_prototypes_and_similarity.png", dpi=180)
    plt.close(fig)


def plot_accuracy_itr(summary: pd.DataFrame) -> None:
    paradigms = ["NBRS-15", "NBRS-8", "JFPM-8"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharex=True)
    for col, paradigm in enumerate(paradigms):
        rows_p = summary[summary["paradigm"] == paradigm]
        for method in ["FBCCA-CODE", "TRCA", "MSTRCA"]:
            rows = rows_p[rows_p["method"] == method].sort_values("window")
            if rows.empty:
                continue
            color = METHOD_COLORS.get(method)
            axes[0, col].errorbar(
                rows["window"],
                rows["accuracy"] * 100,
                yerr=rows["accuracy_sem"] * 100,
                marker="o",
                lw=1.8,
                capsize=3,
                label=method,
                color=color,
            )
            axes[1, col].errorbar(
                rows["window"],
                rows["itr"],
                yerr=rows["itr_sem"],
                marker="o",
                lw=1.8,
                capsize=3,
                label=method,
                color=color,
            )
        axes[0, col].set_title(paradigm)
        axes[0, col].set_ylabel("Accuracy (%)")
        axes[1, col].set_ylabel("ITR (bits/min)")
        axes[1, col].set_xlabel("Window (s)")
        axes[0, col].set_ylim(0, 100)
        for ax in axes[:, col]:
            ax.grid(alpha=0.25)
        axes[0, col].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_acc_itr_by_paradigm_method.png", dpi=180)
    plt.close(fig)


def compute_capacity_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, item in summary.iterrows():
        paradigm = str(item["paradigm"])
        method = str(item["method"])
        window = float(item["window"])
        counts = load_confusion(paradigm, method, window)
        P = normalize_confusion(counts, alpha=0.5)
        ba = capacity_ba(P, max_iter=5000, tol=1e-5)
        acc = float(np.trace(counts) / np.sum(counts))
        c1 = capacity_c1(CLASSES, acc)
        rows.append(
            {
                "paradigm": paradigm,
                "method": method,
                "window": window,
                "accuracy": acc,
                "itr": float(item["itr"]),
                "C0": capacity_c0(CLASSES),
                "C1": c1,
                "I_uniform": mutual_info_uniform(P),
                "C_BA": ba.capacity,
                "C_BA_minus_C1": ba.capacity - c1,
                "C_BA_minus_I_uniform": ba.capacity - mutual_info_uniform(P),
                "ba_converged": ba.converged,
                "ba_gap": ba.gap,
            }
        )
    out = pd.DataFrame(rows).sort_values(["paradigm", "method", "window"])
    out.to_csv(TABLE_DIR / "capacity_by_paradigm_method_window.csv", index=False)
    return out


def plot_capacity(capacity: pd.DataFrame) -> None:
    paradigms = ["NBRS-15", "NBRS-8", "JFPM-8"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, paradigm in zip(axes, paradigms):
        rows_p = capacity[capacity["paradigm"] == paradigm]
        for method in ["FBCCA-CODE", "TRCA", "MSTRCA"]:
            rows = rows_p[rows_p["method"] == method].sort_values("window")
            if rows.empty:
                continue
            ax.plot(rows["window"], rows["C_BA"], marker="o", lw=1.8, color=METHOD_COLORS.get(method), label=method)
        ax.axhline(capacity_c0(CLASSES), color="0.4", lw=1.0, ls="--", label="C0=log2(40)" if ax is axes[0] else None)
        ax.set_title(paradigm)
        ax.set_xlabel("Window (s)")
        ax.grid(alpha=0.25)
        ax.set_ylim(0, capacity_c0(CLASSES) + 0.15)
    axes[0].set_ylabel("BA capacity (bits/selection)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_ba_capacity_by_paradigm_method.png", dpi=180)
    plt.close(fig)

    trade = capacity.copy()
    trade["bits_per_min"] = 60.0 / (trade["window"] + GAZE_SHIFT_SECONDS) * trade["C_BA"]
    trade.to_csv(TABLE_DIR / "capacity_time_tradeoff.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, paradigm in zip(axes, paradigms):
        rows_p = trade[trade["paradigm"] == paradigm]
        for method in ["FBCCA-CODE", "TRCA", "MSTRCA"]:
            rows = rows_p[rows_p["method"] == method].sort_values("window")
            if rows.empty:
                continue
            ax.plot(rows["window"], rows["bits_per_min"], marker="o", lw=1.8, color=METHOD_COLORS.get(method), label=method)
        ax.set_title(paradigm)
        ax.set_xlabel("Window (s)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("60 / (T + 0.5) * C_BA (bits/min)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_capacity_time_tradeoff.png", dpi=180)
    plt.close(fig)


def best_rows(summary: pd.DataFrame) -> pd.DataFrame:
    idx = summary.groupby(["paradigm", "method"])["itr"].idxmax()
    out = summary.loc[idx].sort_values(["paradigm", "method"]).copy()
    out.to_csv(TABLE_DIR / "best_itr_by_paradigm_method.csv", index=False)
    return out


def plot_confusions(summary: pd.DataFrame) -> None:
    best = best_rows(summary)
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.2))
    for r, paradigm in enumerate(["NBRS-15", "NBRS-8", "JFPM-8"]):
        for c, method in enumerate(["FBCCA-CODE", "TRCA", "MSTRCA"]):
            match = best[(best["paradigm"] == paradigm) & (best["method"] == method)]
            ax = axes[r, c]
            if match.empty:
                ax.axis("off")
                continue
            window = float(match.iloc[0]["window"])
            counts = load_confusion(paradigm, method, window)
            P = normalize_confusion(counts, alpha=0.0)
            im = ax.imshow(P, vmin=0, vmax=max(0.25, np.percentile(P, 99)), cmap="magma", aspect="equal")
            ax.set_title(f"{paradigm} / {method}\nw={window:g}s, acc={np.trace(counts)/counts.sum()*100:.1f}%")
            ax.set_xticks([0, 9, 19, 29, 39], [1, 10, 20, 30, 40])
            ax.set_yticks([0, 9, 19, 29, 39], [1, 10, 20, 30, 40])
            if r == 2:
                ax.set_xlabel("Predicted target")
            if c == 0:
                ax.set_ylabel("True target")
    fig.subplots_adjust(left=0.07, right=0.89, top=0.93, bottom=0.07, hspace=0.42, wspace=0.25)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
    fig.colorbar(im, cax=cbar_ax, label="P(pred | target)")
    fig.savefig(FIG_DIR / "fig05_best_itr_confusion_matrices.png", dpi=180)
    plt.close(fig)


def plot_code_vs_confusion(summary: pd.DataFrame) -> None:
    rows = []
    for paradigm in ["NBRS-15", "NBRS-8", "JFPM-8"]:
        code = load_code(paradigm)
        sim = code_similarity(code)
        pair_sim = []
        pair_err = []
        best = summary[(summary["paradigm"] == paradigm) & (summary["method"] == "MSTRCA")]
        if best.empty:
            continue
        window = float(best.sort_values("itr", ascending=False).iloc[0]["window"])
        counts = load_confusion(paradigm, "MSTRCA", window)
        P = normalize_confusion(counts, alpha=0.0)
        for i in range(CLASSES):
            for j in range(CLASSES):
                if i == j:
                    continue
                pair_sim.append(sim[i, j])
                pair_err.append(P[i, j])
        pair_sim_arr = np.asarray(pair_sim)
        pair_err_arr = np.asarray(pair_err)
        corr = np.corrcoef(pair_sim_arr, pair_err_arr)[0, 1]
        rows.append({"paradigm": paradigm, "method": "MSTRCA", "best_itr_window": window, "corr_code_similarity_error": corr})
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(TABLE_DIR / "code_similarity_error_correlation.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for ax, paradigm in zip(axes, ["NBRS-15", "NBRS-8", "JFPM-8"]):
        code = load_code(paradigm)
        sim = code_similarity(code)
        best = summary[(summary["paradigm"] == paradigm) & (summary["method"] == "MSTRCA")]
        window = float(best.sort_values("itr", ascending=False).iloc[0]["window"])
        counts = load_confusion(paradigm, "MSTRCA", window)
        P = normalize_confusion(counts, alpha=0.0)
        xs = []
        ys = []
        for i in range(CLASSES):
            for j in range(CLASSES):
                if i != j:
                    xs.append(sim[i, j])
                    ys.append(P[i, j])
        ax.scatter(xs, ys, s=8, alpha=0.35, color="#4c78a8")
        corr = float(np.corrcoef(xs, ys)[0, 1])
        ax.set_title(f"{paradigm}: MSTRCA best ITR w={window:g}s\nr={corr:.3f}")
        ax.set_xlabel("Code correlation")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Off-diagonal confusion probability")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_code_similarity_vs_confusion.png", dpi=180)
    plt.close(fig)


def write_tables(summary: pd.DataFrame, capacity: pd.DataFrame) -> None:
    meta = pd.DataFrame(
        [
            {"paradigm": key, **value}
            for key, value in PARADIGM_META.items()
        ]
    )
    meta.to_csv(TABLE_DIR / "paradigm_encoding_meta.csv", index=False)

    code_rows = []
    for paradigm in ["NBRS-15", "NBRS-8", "JFPM-8"]:
        sim = code_similarity(load_code(paradigm))
        off = sim[~np.eye(sim.shape[0], dtype=bool)]
        code_rows.append(
            {
                "paradigm": paradigm,
                "mean_offdiag_corr": float(np.mean(off)),
                "mean_abs_offdiag_corr": float(np.mean(np.abs(off))),
                "p95_abs_offdiag_corr": float(np.quantile(np.abs(off), 0.95)),
                "max_abs_offdiag_corr": float(np.max(np.abs(off))),
            }
        )
    pd.DataFrame(code_rows).to_csv(TABLE_DIR / "codebook_similarity_stats.csv", index=False)

    paper = pd.read_csv(COMPARISON / "paper_comparison_summary.csv")
    paper.to_csv(TABLE_DIR / "paper_alignment_summary.csv", index=False)

    best_cap_idx = capacity.groupby(["paradigm", "method"])["C_BA"].idxmax()
    capacity.loc[best_cap_idx].sort_values(["paradigm", "method"]).to_csv(
        TABLE_DIR / "best_capacity_by_paradigm_method.csv", index=False
    )


def main() -> None:
    ensure_dirs()
    summary = load_summary()
    plot_code_prototypes()
    plot_accuracy_itr(summary)
    capacity = compute_capacity_table(summary)
    plot_capacity(capacity)
    plot_confusions(summary)
    plot_code_vs_confusion(summary)
    write_tables(summary, capacity)
    print(f"wrote figures to {FIG_DIR}")
    print(f"wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
