"""Full decision-channel capacity analysis for JBHI16 and JBHI35.

Produces the canonical figure set:
  Fig A: Capacity ladder (C0, C_BA, MI_uniform, C1) with fill-between
  Fig B: Information gap decomposition (stacked bar)
  Fig C: Confusion matrix heatmaps at key windows
  Fig D: Capacity-time tradeoff with optimal point marked
  Fig E: Optimal input distribution q* from Blahut-Arimoto
  Fig F: MI accumulation rate (dMI/dT, exponential fit, half-life)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.channel.capacity import (
    capacity_ba,
    capacity_c0,
    capacity_c1,
    mutual_info_uniform,
)
from vep_arena.channel.confusion import confusion_counts, normalize_confusion

TASK_DIR = Path(__file__).resolve().parent
JBHI16_PREDICTIONS = PROJECT_ROOT / "tasks" / "bessvep" / "results" / "BS02_16t" / "full" / "predictions.csv"
JBHI35_ROOT = PROJECT_ROOT / "tasks" / "bessvep" / "results" / "BS03_35t" / "final_five_execution_20260730"
JBHI35_TDCA = JBHI35_ROOT / "tdca_full" / "predictions.csv"
JBHI35_PERIODIC = JBHI35_ROOT / "periodic_receivers_full" / "predictions.csv"
JBHI35_ETRCA = JBHI35_ROOT / "etrca_2s" / "predictions.csv"

OUTPUT_ROOT = TASK_DIR
FIGURES = TASK_DIR / "figures_full"
TABLES = TASK_DIR / "tables"
ITR_SHIFT = 0.5


def load_jbhi16() -> pd.DataFrame:
    df = pd.read_csv(JBHI16_PREDICTIONS)
    df["true"] = df["true"] - 1
    df["pred"] = df["pred"] - 1
    df = df.rename(columns={"window_seconds": "window"})
    return df[["subject", "method", "window", "true", "pred"]].copy()


def load_jbhi35() -> pd.DataFrame:
    frames = []
    for path in [JBHI35_TDCA, JBHI35_PERIODIC]:
        df = pd.read_csv(path)
        df = df.rename(columns={"true_0based": "true", "pred_0based": "pred", "window_seconds": "window"})
        frames.append(df[["subject", "method", "window", "true", "pred"]])
    etrca = pd.read_csv(JBHI35_ETRCA)
    etrca = etrca.rename(columns={"true_0based": "true", "pred_0based": "pred", "variant": "method"})
    etrca["window"] = 2.0
    etrca = etrca[etrca["method"] == "arena_independent_signed"].copy()
    etrca["method"] = "eTRCA"
    frames.append(etrca[["subject", "method", "window", "true", "pred"]])
    return pd.concat(frames, ignore_index=True)


def compute_full_metrics(preds: pd.DataFrame, classes: int, dataset: str) -> pd.DataFrame:
    rows = []
    for (method, window), group in preds.groupby(["method", "window"], sort=True):
        counts = confusion_counts(group["true"], group["pred"], classes)
        P = normalize_confusion(counts, alpha=0.0)
        accuracy = float(np.trace(counts)) / float(np.sum(counts))
        ba = capacity_ba(P)
        mi_u = mutual_info_uniform(P)
        c0 = capacity_c0(classes)
        c1 = capacity_c1(classes, accuracy)
        rows.append({
            "dataset": dataset,
            "method": str(method),
            "window": float(window),
            "classes": classes,
            "accuracy": accuracy,
            "c0": c0,
            "c1": c1,
            "c_ba": ba.capacity,
            "i_uniform": mi_u,
            "itr_ba_bpm": 60.0 / (float(window) + ITR_SHIFT) * ba.capacity,
            "itr_mi_bpm": 60.0 / (float(window) + ITR_SHIFT) * mi_u,
            "q_star": ba.q,
            "ba_converged": ba.converged,
        })
    return pd.DataFrame(rows)


def get_confusion_at(preds: pd.DataFrame, method: str, window: float, classes: int) -> np.ndarray:
    sub = preds[(preds["method"] == method) & (np.isclose(preds["window"], window))]
    if sub.empty:
        return np.zeros((classes, classes))
    counts = confusion_counts(sub["true"], sub["pred"], classes)
    return normalize_confusion(counts, alpha=0.0)


# ── Figure A: Capacity Ladder ──

def fig_capacity_ladder(df: pd.DataFrame, dataset: str, classes: int) -> None:
    methods = sorted(df["method"].unique())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), constrained_layout=True, sharey=True)
    if n == 1:
        axes = [axes]

    c0 = capacity_c0(classes)
    for idx, method in enumerate(methods):
        ax = axes[idx]
        mdata = df[df["method"] == method].sort_values("window")
        w = mdata["window"].values
        c_ba = mdata["c_ba"].values
        mi_u = mdata["i_uniform"].values
        c1_vals = mdata["c1"].values

        ax.fill_between(w, c_ba, c0, alpha=0.15, color="red", label="C₀ − C_BA (noise)")
        ax.fill_between(w, mi_u, c_ba, alpha=0.15, color="orange", label="C_BA − MI (input gap)")
        ax.fill_between(w, c1_vals, mi_u, alpha=0.15, color="blue", label="MI − C₁ (asymmetry)")
        ax.fill_between(w, 0, c1_vals, alpha=0.15, color="green", label="C₁ (symmetric)")

        ax.plot(w, np.full_like(w, c0), "k--", lw=0.8, label=f"C₀ = {c0:.2f}")
        ax.plot(w, c_ba, "s-", color="red", ms=4, lw=1.3, label="C_BA")
        ax.plot(w, mi_u, "o-", color="darkorange", ms=4, lw=1.3, label="MI_uniform")
        ax.plot(w, c1_vals, "^-", color="blue", ms=4, lw=1.3, label="C₁")

        ax.set_xlabel("Window (s)")
        ax.set_title(method, fontsize=10)
        ax.set_ylim(0, c0 + 0.3)
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.set_ylabel("Information (bits/symbol)")
            ax.legend(fontsize=6.5, loc="lower right")

    fig.suptitle(f"{dataset}: Capacity Ladder Decomposition (M={classes}, C₀={c0:.2f} bits)", fontsize=12)
    fig.savefig(FIGURES / f"{dataset}_capacity_ladder.png", dpi=200)
    plt.close(fig)
    print(f"  {dataset}_capacity_ladder.png")


# ── Figure B: Information Gap Stacked Bar ──

def fig_info_gap(df: pd.DataFrame, dataset: str, classes: int) -> None:
    methods = sorted(df["method"].unique())
    key_windows = sorted(set(df["window"].unique()) & {0.4, 0.6, 1.0, 1.4, 2.0})
    if not key_windows:
        key_windows = sorted(df["window"].unique())[-5:]

    c0 = capacity_c0(classes)
    fig, axes = plt.subplots(1, len(key_windows), figsize=(3.5 * len(key_windows), 5),
                             constrained_layout=True, sharey=True)
    if len(key_windows) == 1:
        axes = [axes]

    for wi, window in enumerate(key_windows):
        ax = axes[wi]
        wdata = df[np.isclose(df["window"], window)]
        if wdata.empty:
            continue
        wdata = wdata.sort_values("method")
        labels = wdata["method"].values
        c1_vals = wdata["c1"].values
        mi_vals = wdata["i_uniform"].values
        cba_vals = wdata["c_ba"].values

        gap_symm = c1_vals
        gap_asym = mi_vals - c1_vals
        gap_input = cba_vals - mi_vals
        gap_noise = c0 - cba_vals

        x = np.arange(len(labels))
        ax.bar(x, gap_symm, 0.6, color="#2ca02c", label="C₁" if wi == 0 else None)
        ax.bar(x, gap_asym, 0.6, bottom=gap_symm, color="#1f77b4",
               label="MI−C₁" if wi == 0 else None)
        ax.bar(x, gap_input, 0.6, bottom=gap_symm + gap_asym, color="#ff7f0e",
               label="C_BA−MI" if wi == 0 else None)
        ax.bar(x, gap_noise, 0.6, bottom=gap_symm + gap_asym + gap_input,
               color="#d62728", alpha=0.6, label="C₀−C_BA" if wi == 0 else None)

        ax.axhline(c0, color="black", ls="--", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"T = {window}s", fontsize=10)
        ax.set_ylim(0, c0 + 0.3)
        ax.grid(alpha=0.2, axis="y")
        if wi == 0:
            ax.set_ylabel("bits/symbol")
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"{dataset}: Information Gap Decomposition", fontsize=12)
    fig.savefig(FIGURES / f"{dataset}_info_gap.png", dpi=200)
    plt.close(fig)
    print(f"  {dataset}_info_gap.png")


# ── Figure C: Confusion Matrix Heatmaps ──

def fig_confusion_matrices(preds: pd.DataFrame, dataset: str, classes: int,
                           best_method: str) -> None:
    key_windows = [0.4, 1.0, 2.0]
    available = sorted(set(preds[preds["method"] == best_method]["window"].unique()) &
                       set(key_windows))
    if not available:
        available = sorted(preds[preds["method"] == best_method]["window"].unique())[-3:]

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for idx, window in enumerate(available):
        ax = axes[idx]
        P = get_confusion_at(preds, best_method, window, classes)
        log_P = np.log10(np.clip(P, 1e-6, None))

        im = ax.imshow(log_P, aspect="auto", cmap="viridis", vmin=-4, vmax=0)
        acc = float(np.trace(P)) / classes
        mi_u = mutual_info_uniform(P)
        ax.set_title(f"T={window}s, Acc={acc:.1%}, MI={mi_u:.2f} bits", fontsize=9)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        plt.colorbar(im, ax=ax, label="log₁₀ P(pred|true)", shrink=0.8)

    fig.suptitle(f"{dataset} {best_method}: Confusion Matrix P(y|x)", fontsize=12)
    fig.savefig(FIGURES / f"{dataset}_confusion_{best_method}.png", dpi=200)
    plt.close(fig)
    print(f"  {dataset}_confusion_{best_method}.png")


# ── Figure D: CT Tradeoff with Optimal Point ──

def fig_ct_tradeoff(df: pd.DataFrame, dataset: str) -> None:
    methods = sorted(df["method"].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(methods), 9)))

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for idx, method in enumerate(methods):
        mdata = df[df["method"] == method].sort_values("window")
        w = mdata["window"].values
        itr = mdata["itr_ba_bpm"].values
        ax.plot(w, itr, marker="o", color=colors[idx], label=method,
                markersize=5, linewidth=1.5)
        best_idx = np.argmax(itr)
        ax.annotate(f"T*={w[best_idx]:.1f}s\n{itr[best_idx]:.0f} bpm",
                    xy=(w[best_idx], itr[best_idx]),
                    xytext=(w[best_idx] + 0.1, itr[best_idx] + 5),
                    fontsize=7, color=colors[idx],
                    arrowprops=dict(arrowstyle="-", color=colors[idx], lw=0.5))

    ax.set_xlabel("Window T (s)", fontsize=11)
    ax.set_ylabel("60/(T+0.5) x C_BA (bits/min)", fontsize=11)
    ax.set_title(f"{dataset}: Capacity-Time Tradeoff (optimal operating point)", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(FIGURES / f"{dataset}_ct_tradeoff.png", dpi=200)
    plt.close(fig)
    print(f"  {dataset}_ct_tradeoff.png")


# ── Figure E: Optimal Input Distribution q* ──

def fig_q_star(df: pd.DataFrame, dataset: str, classes: int) -> None:
    methods_to_show = sorted(df["method"].unique())
    key_windows = sorted(set(df["window"].unique()) & {0.4, 1.0, 2.0})
    if not key_windows:
        key_windows = sorted(df["window"].unique())[-3:]

    n = len(key_windows)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), constrained_layout=True, sharey=True)
    if n == 1:
        axes = [axes]

    uniform = 1.0 / classes
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(methods_to_show), 9)))

    for wi, window in enumerate(key_windows):
        ax = axes[wi]
        ax.axhline(uniform, color="gray", ls="--", lw=0.8, label="Uniform" if wi == 0 else None)
        for mi, method in enumerate(methods_to_show):
            row = df[(df["method"] == method) & (np.isclose(df["window"], window))]
            if row.empty:
                continue
            q = row.iloc[0]["q_star"]
            if q is None or not hasattr(q, '__len__'):
                continue
            ax.plot(range(classes), q, color=colors[mi], lw=1.0, alpha=0.8,
                    label=method if wi == 0 else None)
        ax.set_xlabel("Class index")
        ax.set_title(f"T = {window}s", fontsize=10)
        ax.set_xlim(0, classes - 1)
        ax.grid(alpha=0.2)
        if wi == 0:
            ax.set_ylabel("q*(x)")
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"{dataset}: BA Optimal Input Distribution q* (M={classes})", fontsize=12)
    fig.savefig(FIGURES / f"{dataset}_q_star.png", dpi=200)
    plt.close(fig)
    print(f"  {dataset}_q_star.png")


# ── Figure F: MI Accumulation Rate ──

def _exp_saturation(t, mi_inf, tau):
    return mi_inf * (1.0 - np.exp(-t / tau))


def fig_mi_accumulation(df: pd.DataFrame, dataset: str, classes: int) -> None:
    methods = sorted(df["method"].unique())
    colors_map = plt.cm.Set1(np.linspace(0, 1, max(len(methods), 9)))
    c0 = capacity_c0(classes)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    # (a) MI(T) with exponential fit
    ax = axes[0, 0]
    fit_results = {}
    for idx, method in enumerate(methods):
        mdata = df[df["method"] == method].sort_values("window")
        w = mdata["window"].values
        mi = mdata["i_uniform"].values
        ax.plot(w, mi, marker="o", color=colors_map[idx], ms=5, lw=1.5, label=method)

        if len(w) >= 3:
            try:
                popt, _ = curve_fit(_exp_saturation, w, mi,
                                    p0=[mi.max() * 1.2, 0.5],
                                    bounds=([0, 0.01], [c0 * 2, 10.0]),
                                    maxfev=5000)
                mi_inf, tau = popt
                pred = _exp_saturation(w, *popt)
                ss_res = np.sum((mi - pred) ** 2)
                ss_tot = np.sum((mi - mi.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                t_fine = np.linspace(0, w.max() * 1.1, 100)
                ax.plot(t_fine, _exp_saturation(t_fine, *popt),
                        color=colors_map[idx], ls="--", lw=0.8, alpha=0.6)
                fit_results[method] = {"MI_inf": mi_inf, "tau": tau, "R2": r2}
            except (RuntimeError, ValueError):
                fit_results[method] = {"MI_inf": np.nan, "tau": np.nan, "R2": np.nan}

    ax.axhline(c0, color="black", ls="--", lw=0.8, label=f"C_0 = {c0:.2f}")
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("MI_uniform (bits/symbol)")
    ax.set_title("(a) Information accumulation MI(T)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)

    # (b) dMI/dT
    ax = axes[0, 1]
    for idx, method in enumerate(methods):
        mdata = df[df["method"] == method].sort_values("window")
        w = mdata["window"].values
        mi = mdata["i_uniform"].values
        if len(w) >= 3:
            rate = np.gradient(mi, w)
            ax.plot(w, rate, marker="o", color=colors_map[idx], ms=4, lw=1.3, label=method)
    ax.axhline(0, color="gray", ls=":", lw=0.7)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("dMI/dT (bits/s)")
    ax.set_title("(b) Instantaneous information rate")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)

    # (c) Exponential fit parameters
    ax = axes[1, 0]
    fit_df = pd.DataFrame(fit_results).T
    if not fit_df.empty and not fit_df["tau"].isna().all():
        valid = fit_df.dropna(subset=["tau"])
        x_pos = np.arange(len(valid))
        bars = ax.barh(x_pos, valid["tau"].values, color=[colors_map[methods.index(m)] for m in valid.index],
                       alpha=0.8, edgecolor="black", lw=0.5)
        for i, (m, row) in enumerate(valid.iterrows()):
            ax.text(row["tau"] + 0.01, i,
                    f"tau={row['tau']:.3f}s, MI_inf={row['MI_inf']:.2f}, R2={row['R2']:.3f}",
                    va="center", fontsize=7)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(valid.index, fontsize=8)
        ax.set_xlabel("Time constant tau (s)")
        ax.set_title("(c) Exponential fit: MI(T) = MI_inf * (1 - exp(-T/tau))")
        ax.invert_yaxis()
        ax.grid(alpha=0.25, axis="x")
    else:
        ax.text(0.5, 0.5, "Insufficient data for fit", ha="center", va="center",
                transform=ax.transAxes)

    # (d) Half-life T_50 and utilization at T_max
    ax = axes[1, 1]
    t50_data = []
    for idx, method in enumerate(methods):
        mdata = df[df["method"] == method].sort_values("window")
        w = mdata["window"].values
        mi = mdata["i_uniform"].values
        mi_max = mi[-1]
        target = 0.5 * mi_max
        t50_idx = np.searchsorted(mi, target)
        if t50_idx == 0:
            t50 = w[0]
        elif t50_idx >= len(mi):
            t50 = w[-1]
        else:
            t0, t1 = w[t50_idx - 1], w[t50_idx]
            m0, m1 = mi[t50_idx - 1], mi[t50_idx]
            t50 = t0 + (t1 - t0) * (target - m0) / (m1 - m0) if m1 != m0 else t0
        eta_max = mi_max / c0
        t50_data.append({"method": method, "T_50": t50, "eta_max": eta_max})

    t50_df = pd.DataFrame(t50_data)
    for idx, row in t50_df.iterrows():
        mi_idx = methods.index(row["method"])
        ax.scatter(row["T_50"], row["eta_max"] * 100, color=colors_map[mi_idx],
                   s=80, marker="o", edgecolor="black", lw=0.5, zorder=3)
        ax.annotate(row["method"], (row["T_50"], row["eta_max"] * 100),
                    fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Half-life T_50 (s)")
    ax.set_ylabel("Max utilization eta = MI(T_max)/C_0 (%)")
    ax.set_title("(d) Speed-efficiency: T_50 vs asymptotic utilization")
    ax.grid(alpha=0.25)

    fig.suptitle(f"{dataset}: Information Accumulation Rate Analysis", fontsize=13)
    fig.savefig(FIGURES / f"{dataset}_mi_accumulation.png", dpi=200)
    plt.close(fig)
    print(f"  {dataset}_mi_accumulation.png")

    return fit_results


# ── Main ──

def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    # JBHI16
    print("=" * 60)
    print("JBHI16 (16 targets)")
    print("=" * 60)
    preds16 = load_jbhi16()
    df16 = compute_full_metrics(preds16, 16, "JBHI16")
    df16.to_csv(TABLES / "jbhi16_full_metrics.csv", index=False,
                columns=[c for c in df16.columns if c != "q_star"])

    print("\nGenerating figures...")
    fig_capacity_ladder(df16, "JBHI16", 16)
    fig_info_gap(df16, "JBHI16", 16)
    fig_confusion_matrices(preds16, "JBHI16", 16, "EFUSIONCA")
    fig_ct_tradeoff(df16, "JBHI16")
    fig_q_star(df16, "JBHI16", 16)
    fit16 = fig_mi_accumulation(df16, "JBHI16", 16)

    # JBHI35
    print("\n" + "=" * 60)
    print("JBHI35 (35 targets)")
    print("=" * 60)
    preds35 = load_jbhi35()
    df35 = compute_full_metrics(preds35, 35, "JBHI35")
    df35.to_csv(TABLES / "jbhi35_full_metrics.csv", index=False,
                columns=[c for c in df35.columns if c != "q_star"])

    print("\nGenerating figures...")
    fig_capacity_ladder(df35, "JBHI35", 35)
    fig_info_gap(df35, "JBHI35", 35)
    fig_confusion_matrices(preds35, "JBHI35", 35, "SPECTRAL_TDCA")
    fig_ct_tradeoff(df35, "JBHI35")
    fig_q_star(df35, "JBHI35", 35)
    fit35 = fig_mi_accumulation(df35, "JBHI35", 35)

    # Print fit summary
    print("\n" + "=" * 60)
    print("Exponential Fit Summary: MI(T) = MI_inf * (1 - exp(-T/tau))")
    print("=" * 60)
    print(f"{'Dataset':<8s} {'Method':<16s} {'tau (s)':<10s} {'MI_inf':<10s} {'R2':<10s}")
    print("-" * 60)
    for dataset, fits in [("JBHI16", fit16), ("JBHI35", fit35)]:
        for method, params in fits.items():
            tau_s = f"{params['tau']:.3f}" if not np.isnan(params['tau']) else "--"
            mi_s = f"{params['MI_inf']:.3f}" if not np.isnan(params['MI_inf']) else "--"
            r2_s = f"{params['R2']:.4f}" if not np.isnan(params['R2']) else "--"
            print(f"{dataset:<8s} {method:<16s} {tau_s:<10s} {mi_s:<10s} {r2_s:<10s}")

    print(f"\nAll outputs saved to: {FIGURES}")
    print(f"Tables saved to: {TABLES}")


if __name__ == "__main__":
    main()
