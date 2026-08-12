"""Extended cross-paradigm envelope: adds HD200 + Benchmark optimal channel config.

Extends crossparadigm_envelope.py with:
- HD200_40t: 40 targets, 66ch, TDCA (same C0 as Benchmark)
- HD200_200t: 200 targets, 66ch, TDCA (highest source complexity)
- Benchmark_opt: 40 targets, occipital 9ch (best channel config), all methods

Key addition: shows that Benchmark's true channel capacity is higher when measured
at optimal electrode placement, and HD200's 200-target paradigm reveals a fundamentally
different capacity curve (lower η but higher absolute bits at long windows).
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

TASKS = PROJECT_ROOT / "tasks"
OUTPUT = TASKS / "ssvep_jbhi_decision_channel" / "figures_envelope_extended_v20260808"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Original data sources
CSV_BENCHMARK = TASKS / "channel" / "CH01_dmc_benchmark" / "results" / "extended" / "combined" / "analysis" / "capacity_by_method_window_aggregate.csv"
CSV_BINOCULAR = TASKS / "ssvep_binocular_ar_trca" / "analysis" / "decision_channel" / "tables" / "capacity_by_condition_method_window.csv"
CSV_DUAL_ALPHA = TASKS / "ssvep_dual_alpha_baselines" / "analysis" / "decision_channel" / "tables" / "capacity_by_paradigm_method_window.csv"
CSV_JFPM = TASKS / "cvep_nbrs_jfpm_tsinghua_2024_baselines" / "analysis" / "decision_channel_coding" / "tables" / "capacity_by_paradigm_method_window.csv"
CSV_JBHI = TASKS / "ssvep_jbhi_decision_channel" / "combined_aggregate_capacity.csv"

# New data sources
CSV_HD200 = TASKS / "ssvep_hd_200target_tdca_sample" / "results" / "offline_tdca_grid" / "decision_channel" / "capacity_by_subject_targets_channels_window.csv"
CSV_BENCH_MULTI = TASKS / "channel" / "CH01_dmc_benchmark" / "results" / "extended" / "multichannel" / "decision_channel" / "capacity_by_subject_method_channels_window.csv"

PARADIGM_META = {
    "Benchmark": {"C0": np.log2(40), "M": 40, "color": "#1f77b4", "label": "Benchmark 64ch (40cls, SSVEP)"},
    "Benchmark_opt": {"C0": np.log2(40), "M": 40, "color": "#17becf", "label": "Benchmark 9ch-opt (40cls, SSVEP)"},
    "HD200_40t": {"C0": np.log2(40), "M": 40, "color": "#e377c2", "label": "HD200 40cls (TDCA, 66ch)"},
    "HD200_200t": {"C0": np.log2(200), "M": 200, "color": "#7f7f7f", "label": "HD200 200cls (TDCA, 66ch)"},
    "BinoAR_DFDP": {"C0": np.log2(8), "M": 8, "color": "#d62728", "label": "BinoAR DFDP (8cls)"},
    "DualAlpha_CA": {"C0": np.log2(40), "M": 40, "color": "#9467bd", "label": "DualAlpha CA (40cls)"},
    "JFPM8": {"C0": np.log2(40), "M": 40, "color": "#8c564b", "label": "JFPM-8 (40cls, code-VEP)"},
    "JBHI35": {"C0": np.log2(35), "M": 35, "color": "#ff7f0e", "label": "JBHI35 (35cls, dual-freq)"},
    "JBHI16": {"C0": np.log2(16), "M": 16, "color": "#2ca02c", "label": "JBHI16 (16cls)"},
}

PARADIGM_ORDER = ["HD200_200t", "Benchmark", "Benchmark_opt", "HD200_40t",
                  "JBHI35", "DualAlpha_CA", "JBHI16", "BinoAR_DFDP", "JFPM8"]


def load_all_methods():
    """Load all data including new HD200 and Benchmark multichannel entries."""
    records = []

    # Original Benchmark (64ch default, aggregate level)
    df = pd.read_csv(CSV_BENCHMARK)
    for _, row in df[df.iloc[:, -1] == "all"].iterrows():
        records.append({"paradigm": "Benchmark", "method": row.iloc[-3], "window": float(row.iloc[-2]),
                        "C_BA": float(row.iloc[8]), "MI": float(row.iloc[12]), "accuracy": float(row.iloc[1])})

    # Benchmark_opt: occipital 9ch, aggregate across subjects per method
    bmulti = pd.read_csv(CSV_BENCH_MULTI)
    b_occ = bmulti[bmulti["channel_config"] == "occipital9"]
    for method in b_occ["method"].unique():
        mdata = b_occ[b_occ["method"] == method]
        grp = mdata.groupby("window").agg(
            c_ba=("c_ba", "mean"), i_uniform=("i_uniform", "mean"), accuracy=("accuracy", "mean")
        ).reset_index()
        for _, row in grp.iterrows():
            records.append({"paradigm": "Benchmark_opt", "method": method, "window": float(row["window"]),
                            "C_BA": float(row["c_ba"]), "MI": float(row["i_uniform"]), "accuracy": float(row["accuracy"])})

    # HD200: 40 targets, 66ch
    hd = pd.read_csv(CSV_HD200)
    hd40_66 = hd[(hd["targets"] == 40) & (hd["channels"] == 66)]
    grp = hd40_66.groupby("window").agg(
        c_ba=("c_ba", "mean"), i_uniform=("i_uniform", "mean"), accuracy=("accuracy", "mean")
    ).reset_index()
    for _, row in grp.iterrows():
        records.append({"paradigm": "HD200_40t", "method": "TDCA", "window": float(row["window"]),
                        "C_BA": float(row["c_ba"]), "MI": float(row["i_uniform"]), "accuracy": float(row["accuracy"])})

    # HD200: 200 targets, 66ch
    hd200_66 = hd[(hd["targets"] == 200) & (hd["channels"] == 66)]
    grp = hd200_66.groupby("window").agg(
        c_ba=("c_ba", "mean"), i_uniform=("i_uniform", "mean"), accuracy=("accuracy", "mean")
    ).reset_index()
    for _, row in grp.iterrows():
        records.append({"paradigm": "HD200_200t", "method": "TDCA", "window": float(row["window"]),
                        "C_BA": float(row["c_ba"]), "MI": float(row["i_uniform"]), "accuracy": float(row["accuracy"])})

    # BinoAR
    df = pd.read_csv(CSV_BINOCULAR)
    sub = df[df["task"] == "DFDP"]
    for _, row in sub.iterrows():
        records.append({"paradigm": "BinoAR_DFDP", "method": row["method"], "window": float(row["window"]),
                        "C_BA": float(row["C_BA"]), "MI": float(row["MI_uniform"]), "accuracy": float(row["accuracy"])})

    # DualAlpha
    df = pd.read_csv(CSV_DUAL_ALPHA)
    sub = df[df["paradigm"] == "Checkerboard_Arrangment"]
    for _, row in sub.iterrows():
        records.append({"paradigm": "DualAlpha_CA", "method": row["method"], "window": float(row["window"]),
                        "C_BA": float(row["C_BA"]), "MI": float(row["MI_uniform"]), "accuracy": float(row["accuracy"])})

    # JFPM
    df = pd.read_csv(CSV_JFPM)
    sub = df[df["paradigm"] == "JFPM-8"]
    for _, row in sub.iterrows():
        records.append({"paradigm": "JFPM8", "method": row["method"], "window": float(row["window"]),
                        "C_BA": float(row["C_BA"]), "MI": float(row["I_uniform"]), "accuracy": float(row["accuracy"])})

    # JBHI
    df = pd.read_csv(CSV_JBHI)
    for _, row in df.iterrows():
        paradigm = "JBHI35" if row["dataset"] == "JBHI35" else "JBHI16"
        records.append({"paradigm": paradigm, "method": row["method"], "window": float(row["window"]),
                        "C_BA": float(row["c_ba"]), "MI": float(row["i_uniform"]), "accuracy": float(row["accuracy"])})

    return pd.DataFrame(records)


def compute_envelope(all_data):
    envelope = all_data.groupby(["paradigm", "window"], as_index=False).agg(
        C_BA_max=("C_BA", "max"),
        MI_max=("MI", "max"),
        best_method=("C_BA", lambda x: all_data.loc[x.idxmax(), "method"]),
        n_methods=("method", "nunique"),
    )
    return envelope


def _exp_sat(t, mi_inf, tau):
    return mi_inf * (1.0 - np.exp(-t / tau))


def fit_exponential(w, y):
    try:
        if len(w) < 3 or y.max() <= 0:
            return np.nan, np.nan, np.nan
        popt, _ = curve_fit(_exp_sat, w, y, p0=[y.max() * 1.3, 0.3],
                            bounds=([0, 0.01], [y.max() * 5, 20.0]), maxfev=10000)
        pred = _exp_sat(w, *popt)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return popt[0], popt[1], r2
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan


# ═══════════════════ Figure 1: Absolute C_BA envelope ═══════════════════

def fig01_envelope_absolute(envelope):
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    for paradigm in PARADIGM_ORDER:
        meta = PARADIGM_META[paradigm]
        penv = envelope[envelope["paradigm"] == paradigm].sort_values("window")
        if penv.empty:
            continue
        ls = "--" if "opt" in paradigm or "HD200" in paradigm else "-"
        ms = 4 if len(penv) < 20 else 2
        ax.plot(penv["window"], penv["C_BA_max"], f"o{ls}", color=meta["color"],
                ms=ms, lw=1.5, label=f"{meta['label']} (C0={meta['C0']:.1f})")
    ax.set_xlabel("Window (s)", fontsize=11)
    ax.set_ylabel("max C_BA (bits/symbol)", fontsize=11)
    ax.set_title("Cross-Paradigm Channel Capacity Envelope (including HD200 + optimal config)", fontsize=11)
    ax.set_xlim(0, 5.2)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT / "fig01_envelope_absolute.png", dpi=200)
    plt.close(fig)
    print("  fig01_envelope_absolute.png")


# ═══════════════════ Figure 2: Normalized η envelope ═══════════════════

def fig02_envelope_eta(envelope):
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    for paradigm in PARADIGM_ORDER:
        meta = PARADIGM_META[paradigm]
        penv = envelope[envelope["paradigm"] == paradigm].sort_values("window")
        if penv.empty:
            continue
        eta = penv["C_BA_max"].values / meta["C0"] * 100
        ls = "--" if "opt" in paradigm or "HD200" in paradigm else "-"
        ms = 4 if len(penv) < 20 else 2
        ax.plot(penv["window"], eta, f"o{ls}", color=meta["color"],
                ms=ms, lw=1.5, label=meta["label"])
    ax.axhline(100, color="gray", ls="--", lw=0.7)
    ax.set_xlabel("Window (s)", fontsize=11)
    ax.set_ylabel("max(C_BA) / C0 (%)", fontsize=11)
    ax.set_title("Cross-Paradigm Channel Utilization Envelope (normalized)", fontsize=11)
    ax.set_xlim(0, 5.2)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT / "fig02_envelope_eta.png", dpi=200)
    plt.close(fig)
    print("  fig02_envelope_eta.png")


# ═══════════════════ Figure 3: ITR envelope ═══════════════════

def fig03_envelope_itr(envelope):
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    for paradigm in PARADIGM_ORDER:
        meta = PARADIGM_META[paradigm]
        penv = envelope[envelope["paradigm"] == paradigm].sort_values("window")
        if penv.empty:
            continue
        w = penv["window"].values
        itr = 60.0 / (w + 0.5) * penv["C_BA_max"].values
        ls = "--" if "opt" in paradigm or "HD200" in paradigm else "-"
        ms = 4 if len(penv) < 20 else 2
        ax.plot(w, itr, f"o{ls}", color=meta["color"], ms=ms, lw=1.5, label=meta["label"])
        best_idx = np.argmax(itr)
        ax.annotate(f"{itr[best_idx]:.0f}", xy=(w[best_idx], itr[best_idx]),
                    xytext=(4, 6), textcoords="offset points", fontsize=7, color=meta["color"])
    ax.set_xlabel("Window (s)", fontsize=11)
    ax.set_ylabel("60/(T+0.5) × max(C_BA) (bits/min)", fontsize=11)
    ax.set_title("Cross-Paradigm: Channel Throughput Envelope (CT Tradeoff)", fontsize=11)
    ax.set_xlim(0, 5.2)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT / "fig03_envelope_itr.png", dpi=200)
    plt.close(fig)
    print("  fig03_envelope_itr.png")


# ═══════════════════ Figure 4: Exponential fit — speed vs capacity ═══════════════════

def fig04_fit_comparison(envelope):
    fit_results = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    ax = axes[0]
    for paradigm in PARADIGM_ORDER:
        meta = PARADIGM_META[paradigm]
        penv = envelope[envelope["paradigm"] == paradigm].sort_values("window")
        if penv.empty:
            continue
        w = penv["window"].values
        cba = penv["C_BA_max"].values
        c0 = meta["C0"]
        ax.scatter(w, cba / c0 * 100, color=meta["color"], s=15, zorder=3, alpha=0.7)
        mi_inf, tau, r2 = fit_exponential(w, cba)
        fit_results[paradigm] = {"CBA_inf": mi_inf, "tau": tau, "R2": r2, "C0": c0}
        if not np.isnan(tau):
            t_fine = np.linspace(0, max(w.max(), 2.0), 100)
            fit_line = _exp_sat(t_fine, mi_inf, tau) / c0 * 100
            ls = "--" if "opt" in paradigm or "HD200" in paradigm else "-"
            ax.plot(t_fine, fit_line, color=meta["color"], ls=ls, lw=1.0, alpha=0.7)
    ax.axhline(100, color="gray", ls="--", lw=0.7)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("max(C_BA) / C0 (%)")
    ax.set_title("(a) Envelope saturation fit")
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 115)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for paradigm in PARADIGM_ORDER:
        meta = PARADIGM_META[paradigm]
        fit = fit_results.get(paradigm, {})
        if not fit or np.isnan(fit.get("tau", np.nan)):
            continue
        eta = fit["CBA_inf"] / fit["C0"] * 100
        marker = "s" if "HD200" in paradigm else ("D" if "opt" in paradigm else "o")
        ax.scatter(fit["tau"], eta, color=meta["color"], s=120, marker=marker,
                   edgecolor="black", lw=0.5, zorder=3)
        short = paradigm
        ax.annotate(short, (fit["tau"], eta), fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Channel time constant τ (s)")
    ax.set_ylabel("Asymptotic capacity CBA_∞ / C0 (%)")
    ax.set_title("(b) Channel speed vs. achievable capacity")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, None)
    ax.set_ylim(50, 110)

    fig.suptitle("Cross-Paradigm: Envelope Fit with HD200 + Optimal Config", fontsize=12)
    fig.savefig(OUTPUT / "fig04_fit_comparison.png", dpi=200)
    plt.close(fig)
    print("  fig04_fit_comparison.png")
    return fit_results


# ═══════════════════ Figure 5: HD200 channel sweep envelope ═══════════════════

def fig05_hd200_channel_envelope(hd_data):
    """Shows HD200's capacity at different channel configs — a unique within-paradigm channel sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    ch_colors = {66: "#1f77b4", 32: "#2ca02c", 21: "#ff7f0e", 9: "#d62728"}

    # (a) Absolute C_BA: 200 targets, all channel configs
    ax = axes[0]
    for ch in [66, 32, 21, 9]:
        sub = hd_data[(hd_data["targets"] == 200) & (hd_data["channels"] == ch)]
        grp = sub.groupby("window")["c_ba"].mean().reset_index()
        ax.plot(grp["window"], grp["c_ba"], "o-", color=ch_colors[ch], ms=5, lw=1.5, label=f"{ch}ch")
    ax.axhline(np.log2(200), color="black", ls="--", lw=0.7, label="C0=log2(200)")
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("C_BA (bits/symbol)")
    ax.set_title("(a) HD200: 200 targets, channel configs")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_ylim(4, 8)

    # (b) Same but as η
    ax = axes[1]
    c0_200 = np.log2(200)
    for ch in [66, 32, 21, 9]:
        sub = hd_data[(hd_data["targets"] == 200) & (hd_data["channels"] == ch)]
        grp = sub.groupby("window")["c_ba"].mean().reset_index()
        ax.plot(grp["window"], grp["c_ba"] / c0_200 * 100, "o-", color=ch_colors[ch], ms=5, lw=1.5, label=f"{ch}ch")
    ax.axhline(100, color="gray", ls="--", lw=0.7)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("η = C_BA / C0 (%)")
    ax.set_title("(b) HD200: 200 targets, normalized")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_ylim(60, 100)

    fig.suptitle("HD200: Within-Paradigm Channel Sweep (channel degradation at fixed source)", fontsize=12)
    fig.savefig(OUTPUT / "fig05_hd200_channel_sweep.png", dpi=200)
    plt.close(fig)
    print("  fig05_hd200_channel_sweep.png")


# ═══════════════════ Main ═══════════════════

def main():
    print("Loading all paradigms (original + HD200 + Benchmark optimal)...")
    all_data = load_all_methods()
    print(f"  Total records: {len(all_data)}")
    for p in PARADIGM_ORDER:
        sub = all_data[all_data["paradigm"] == p]
        methods = sub["method"].unique()
        n_wins = sub["window"].nunique()
        print(f"  {p}: {len(methods)} methods, {n_wins} windows ({', '.join(methods[:5])})")

    print("\nComputing envelope...")
    envelope = compute_envelope(all_data)

    print("\nFigure 1: Absolute envelope...")
    fig01_envelope_absolute(envelope)

    print("Figure 2: Normalized η envelope...")
    fig02_envelope_eta(envelope)

    print("Figure 3: ITR envelope...")
    fig03_envelope_itr(envelope)

    print("Figure 4: Exponential fit comparison...")
    fit_results = fig04_fit_comparison(envelope)

    print("Figure 5: HD200 within-paradigm channel sweep...")
    hd_data = pd.read_csv(CSV_HD200)
    fig05_hd200_channel_envelope(hd_data)

    # Summary table
    print("\n" + "=" * 110)
    print("Extended Cross-Paradigm Envelope Summary")
    print("=" * 110)
    fmt = "{:<16s} {:>4s} {:>5s} {:>7s} {:>7s} {:>8s} {:>8s} {:>10s} {:>7s}"
    print(fmt.format("Paradigm", "M", "#Rx", "tau", "eta%", "T*", "CBA@0.5", "Peak ITR", "R2"))
    print("-" * 110)
    for paradigm in PARADIGM_ORDER:
        meta = PARADIGM_META[paradigm]
        penv = envelope[envelope["paradigm"] == paradigm].sort_values("window")
        if penv.empty:
            continue
        w = penv["window"].values
        cba = penv["C_BA_max"].values
        itr = 60.0 / (w + 0.5) * cba
        fit = fit_results.get(paradigm, {})
        tau_s = f"{fit['tau']:.3f}" if not np.isnan(fit.get("tau", np.nan)) else "--"
        eta_s = f"{fit['CBA_inf']/fit['C0']*100:.1f}" if not np.isnan(fit.get("CBA_inf", np.nan)) else "--"
        r2_s = f"{fit['R2']:.3f}" if not np.isnan(fit.get("R2", np.nan)) else "--"
        cba05_idx = np.argmin(np.abs(w - 0.5))
        cba05 = f"{cba[cba05_idx]:.2f}" if len(cba) > 0 else "--"
        print(f"  {paradigm:<14s} {meta['M']:>4d} {int(penv['n_methods'].max()):>5d} {tau_s:>7s} {eta_s:>7s} "
              f"{w[np.argmax(itr)]:>7.2f}s {cba05:>7s} {itr.max():>9.1f} {r2_s:>7s}")
    print("=" * 110)
    print(f"\nOutputs: {OUTPUT}")


if __name__ == "__main__":
    main()
