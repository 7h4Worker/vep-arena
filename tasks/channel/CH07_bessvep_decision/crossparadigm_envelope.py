"""Cross-paradigm decision channel analysis: multi-receiver envelope + convergence.

Two analyses:
1. Single-paradigm multi-receiver: all methods per paradigm, shows convergence to channel limit
2. Cross-paradigm envelope: max(C_BA) across all receivers per paradigm, compares channels

Key insight: when multiple receivers converge to the same C_BA ceiling, that ceiling
approximates the true channel capacity (not just a receiver limitation).
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
OUTPUT = TASKS / "ssvep_jbhi_decision_channel" / "figures_envelope_v20260804b"
TABLES_OUT = TASKS / "ssvep_jbhi_decision_channel" / "tables"

CSV_BENCHMARK = TASKS / "benchmark_decision_channel_capacity" / "results" / "extended" / "combined" / "analysis" / "capacity_by_method_window_aggregate.csv"
CSV_BINOCULAR = TASKS / "ssvep_binocular_ar_trca" / "analysis" / "decision_channel" / "tables" / "capacity_by_condition_method_window.csv"
CSV_DUAL_ALPHA = TASKS / "ssvep_dual_alpha_baselines" / "analysis" / "decision_channel" / "tables" / "capacity_by_paradigm_method_window.csv"
CSV_JFPM = TASKS / "cvep_nbrs_jfpm_tsinghua_2024_baselines" / "analysis" / "decision_channel_coding" / "tables" / "capacity_by_paradigm_method_window.csv"
CSV_JBHI = TASKS / "ssvep_jbhi_decision_channel" / "combined_aggregate_capacity.csv"


PARADIGM_META = {
    "Benchmark": {"C0": np.log2(40), "M": 40, "color": "#1f77b4", "label": "Benchmark (40cls, single-freq SSVEP)"},
    "BinoAR_DFDP": {"C0": np.log2(8), "M": 8, "color": "#d62728", "label": "BinoAR DFDP (8cls, dual-freq)"},
    "DualAlpha_CA": {"C0": np.log2(40), "M": 40, "color": "#9467bd", "label": "DualAlpha CA (40cls, alpha-band)"},
    "JFPM8": {"C0": np.log2(40), "M": 40, "color": "#8c564b", "label": "JFPM-8 (40cls, code-VEP)"},
    "JBHI35": {"C0": np.log2(35), "M": 35, "color": "#ff7f0e", "label": "JBHI35 (35cls, dual-freq SSVEP)"},
    "JBHI16": {"C0": np.log2(16), "M": 16, "color": "#2ca02c", "label": "JBHI16 (16cls, SSVEP)"},
}

PARADIGM_ORDER = ["Benchmark", "JBHI35", "DualAlpha_CA", "JBHI16", "BinoAR_DFDP", "JFPM8"]


def load_all_methods():
    """Load all (paradigm, method, window) triples with C_BA and MI_uniform."""
    records = []

    # Benchmark: columns end with method,window,subject; C_BA is 'c_ba', MI is 'i_uniform'
    df = pd.read_csv(CSV_BENCHMARK)
    for _, row in df[df.iloc[:, -1] == "all"].iterrows():
        records.append({"paradigm": "Benchmark", "method": row.iloc[-3], "window": float(row.iloc[-2]),
                        "C_BA": float(row.iloc[8]), "MI": float(row.iloc[12]), "accuracy": float(row.iloc[1])})

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

    # JBHI combined
    df = pd.read_csv(CSV_JBHI)
    for _, row in df.iterrows():
        paradigm = "JBHI35" if row["dataset"] == "JBHI35" else "JBHI16"
        records.append({"paradigm": paradigm, "method": row["method"], "window": float(row["window"]),
                        "C_BA": float(row["c_ba"]), "MI": float(row["i_uniform"]), "accuracy": float(row["accuracy"])})

    return pd.DataFrame(records)


def compute_envelope(all_data):
    """For each (paradigm, window), take max C_BA across all methods."""
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
        popt, _ = curve_fit(_exp_sat, w, y, p0=[y.max() * 1.3, 0.4],
                            bounds=([0, 0.01], [y.max() * 5, 20.0]), maxfev=10000)
        pred = _exp_sat(w, *popt)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return popt[0], popt[1], r2
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan


# ═══════════════════ Figure 1: Single-paradigm multi-receiver ═══════════════════

def fig01_multi_receiver(all_data, paradigm):
    """All methods for one paradigm: shows convergence to channel ceiling."""
    meta = PARADIGM_META[paradigm]
    pdata = all_data[all_data["paradigm"] == paradigm].copy()
    methods = sorted(pdata["method"].unique())
    if len(methods) < 2:
        return None

    c0 = meta["C0"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 10)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # (a) C_BA(T) per method
    ax = axes[0]
    for idx, method in enumerate(methods):
        mdata = pdata[pdata["method"] == method].sort_values("window")
        ax.plot(mdata["window"], mdata["C_BA"], marker="o", color=colors[idx],
                ms=4, lw=1.3, label=method)
    ax.axhline(c0, color="black", ls="--", lw=0.8, label=f"C0 = {c0:.2f}")
    # envelope
    env = pdata.groupby("window", as_index=False)["C_BA"].max().sort_values("window")
    ax.plot(env["window"], env["C_BA"], "k-", lw=2.5, alpha=0.4, label="Envelope (max)")
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("C_BA (bits/symbol)")
    ax.set_title("(a) Channel capacity by receiver")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, c0 + 0.3)

    # (b) Spread: max - min C_BA at each window (convergence indicator)
    ax = axes[1]
    spread = pdata.groupby("window").agg(
        cba_max=("C_BA", "max"), cba_min=("C_BA", "min"),
        cba_mean=("C_BA", "mean"), cba_std=("C_BA", "std")
    ).reset_index().sort_values("window")
    ax.fill_between(spread["window"], spread["cba_min"], spread["cba_max"],
                    alpha=0.3, color=meta["color"], label="Range (min-max)")
    ax.plot(spread["window"], spread["cba_mean"], "-o", color=meta["color"],
            ms=5, lw=1.5, label="Mean C_BA")
    ax.axhline(c0, color="black", ls="--", lw=0.8)
    gap = spread["cba_max"] - spread["cba_min"]
    ax2 = ax.twinx()
    bar_w = np.diff(spread["window"].values).min() * 0.7 if len(spread) > 1 else 0.08
    ax2.bar(spread["window"], gap, width=bar_w, alpha=0.3, color="gray", label="Spread")
    ax2.set_ylabel("Spread (bits)", color="gray")
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("C_BA (bits/symbol)")
    ax.set_title("(b) Receiver convergence (spread -> 0 = channel limit)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)

    # (c) Utilization per method at longest window
    ax = axes[2]
    max_w = pdata["window"].max()
    at_max = pdata[np.isclose(pdata["window"], max_w)].sort_values("C_BA", ascending=True)
    y_pos = np.arange(len(at_max))
    ax.barh(y_pos, at_max["C_BA"].values / c0 * 100,
            color=[colors[methods.index(m)] for m in at_max["method"]],
            alpha=0.8, edgecolor="black", lw=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(at_max["method"].values, fontsize=8)
    ax.axvline(100, color="black", ls="--", lw=0.8)
    ax.set_xlabel(f"C_BA / C0 (%) at T={max_w}s")
    ax.set_title(f"(c) Receiver efficiency at T={max_w}s")
    ax.set_xlim(0, 110)
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle(f"{meta['label']}: Multi-Receiver Capacity Analysis", fontsize=12)
    fig.savefig(OUTPUT / f"fig01_{paradigm}_multi_receiver.png", dpi=200)
    plt.close(fig)
    print(f"  fig01_{paradigm}_multi_receiver.png ({len(methods)} methods)")
    return spread


# ═══════════════════ Figure 2: Cross-paradigm envelope ═══════════════════

def fig02_envelope_comparison(envelope):
    """Max C_BA envelope per paradigm — compares channels, not receivers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # (a) Absolute C_BA envelope
    ax = axes[0]
    for paradigm in PARADIGM_ORDER:
        meta = PARADIGM_META[paradigm]
        penv = envelope[envelope["paradigm"] == paradigm].sort_values("window")
        if penv.empty:
            continue
        ms = 3 if len(penv) > 20 else 5
        ax.plot(penv["window"], penv["C_BA_max"], "o-", color=meta["color"],
                ms=ms, lw=1.5, label=meta["label"])
        # C0 reference line
        ax.axhline(meta["C0"], color=meta["color"], ls=":", lw=0.5, alpha=0.5)
    ax.set_xlabel("Window (s)", fontsize=11)
    ax.set_ylabel("max C_BA (bits/symbol)", fontsize=11)
    ax.set_title("(a) Channel capacity envelope (best receiver at each T)")
    ax.set_xlim(0, 5.2)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)

    # (b) Normalized: max(C_BA) / C0
    ax = axes[1]
    for paradigm in PARADIGM_ORDER:
        meta = PARADIGM_META[paradigm]
        penv = envelope[envelope["paradigm"] == paradigm].sort_values("window")
        if penv.empty:
            continue
        util = penv["C_BA_max"].values / meta["C0"] * 100
        ms = 3 if len(penv) > 20 else 5
        ax.plot(penv["window"], util, "o-", color=meta["color"],
                ms=ms, lw=1.5, label=meta["label"])
    ax.axhline(100, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Window (s)", fontsize=11)
    ax.set_ylabel("max(C_BA) / C0 (%)", fontsize=11)
    ax.set_title("(b) Channel utilization envelope (receiver-independent)")
    ax.set_xlim(0, 5.2)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)

    fig.suptitle("Cross-Paradigm: Channel Capacity Envelope (max over all receivers)", fontsize=12)
    fig.savefig(OUTPUT / "fig02_envelope_comparison.png", dpi=200)
    plt.close(fig)
    print("  fig02_envelope_comparison.png")


# ═══════════════════ Figure 3: Envelope CT tradeoff ═══════════════════

def fig03_envelope_ct(envelope):
    """ITR from envelope C_BA — the channel's achievable throughput."""
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for paradigm in PARADIGM_ORDER:
        meta = PARADIGM_META[paradigm]
        penv = envelope[envelope["paradigm"] == paradigm].sort_values("window")
        if penv.empty:
            continue
        w = penv["window"].values
        itr = 60.0 / (w + 0.5) * penv["C_BA_max"].values
        ms = 3 if len(penv) > 20 else 5
        ax.plot(w, itr, "o-", color=meta["color"], ms=ms, lw=1.5, label=meta["label"])
        best_idx = np.argmax(itr)
        ax.annotate(f"{itr[best_idx]:.0f}", xy=(w[best_idx], itr[best_idx]),
                    xytext=(4, 6), textcoords="offset points", fontsize=7, color=meta["color"])
    ax.set_xlabel("Window (s)", fontsize=11)
    ax.set_ylabel("60/(T+0.5) x max(C_BA) (bits/min)", fontsize=11)
    ax.set_title("Cross-Paradigm: Channel Throughput Envelope (CT Tradeoff)", fontsize=12)
    ax.set_xlim(0, 5.2)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT / "fig03_envelope_ct_tradeoff.png", dpi=200)
    plt.close(fig)
    print("  fig03_envelope_ct_tradeoff.png")


# ═══════════════════ Figure 4: Exponential fit on envelope ═══════════════════

def fig04_envelope_fit(envelope):
    """Exponential saturation on envelope — channel time constant."""
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
        ax.scatter(w, cba / c0 * 100, color=meta["color"], s=25, zorder=3)
        mi_inf, tau, r2 = fit_exponential(w, cba)
        fit_results[paradigm] = {"CBA_inf": mi_inf, "tau": tau, "R2": r2, "C0": c0}
        if not np.isnan(tau):
            t_fine = np.linspace(0, max(w.max(), 2.0), 100)
            fit_line = _exp_sat(t_fine, mi_inf, tau) / c0 * 100
            ax.plot(t_fine, fit_line, color=meta["color"], ls="-", lw=1.0, alpha=0.6)
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
        ax.scatter(fit["tau"], eta, color=meta["color"], s=120, marker="o",
                   edgecolor="black", lw=0.5, zorder=3)
        short = meta["label"].split("(")[0].strip()
        ax.annotate(short, (fit["tau"], eta), fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Channel time constant tau (s)")
    ax.set_ylabel("Asymptotic capacity CBA_inf / C0 (%)")
    ax.set_title("(b) Channel speed vs. achievable capacity")
    ax.grid(alpha=0.25)

    fig.suptitle("Cross-Paradigm: Channel Envelope Exponential Fit", fontsize=12)
    fig.savefig(OUTPUT / "fig04_envelope_fit.png", dpi=200)
    plt.close(fig)
    print("  fig04_envelope_fit.png")
    return fit_results


# ═══════════════════ Figure 5: Convergence summary ═══════════════════

def fig05_convergence_summary(all_data, envelope):
    """How many receivers approach the envelope — indicates channel limit confidence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # (a) Number of methods within 5% of envelope at each window
    ax = axes[0]
    for paradigm in PARADIGM_ORDER:
        meta = PARADIGM_META[paradigm]
        pdata = all_data[all_data["paradigm"] == paradigm]
        penv = envelope[envelope["paradigm"] == paradigm]
        if penv.empty:
            continue
        windows = sorted(penv["window"].unique())
        fracs = []
        for w in windows:
            env_val = penv[penv["window"] == w]["C_BA_max"].values[0]
            at_w = pdata[np.isclose(pdata["window"], w)]
            n_total = len(at_w)
            n_close = len(at_w[at_w["C_BA"] >= 0.95 * env_val])
            fracs.append(n_close / n_total * 100 if n_total > 0 else 0)
        ax.plot(windows, fracs, "o-", color=meta["color"], ms=4, lw=1.3,
                label=meta["label"].split("(")[0].strip())
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("% of receivers within 5% of envelope")
    ax.set_title("(a) Receiver convergence to channel limit")
    ax.set_xlim(0, 5.2)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)

    # (b) Gap between best and 2nd-best receiver
    ax = axes[1]
    for paradigm in PARADIGM_ORDER:
        meta = PARADIGM_META[paradigm]
        pdata = all_data[all_data["paradigm"] == paradigm]
        methods = pdata["method"].unique()
        if len(methods) < 2:
            continue
        windows = sorted(pdata["window"].unique())
        gaps = []
        ws_valid = []
        for w in windows:
            at_w = pdata[np.isclose(pdata["window"], w)].sort_values("C_BA", ascending=False)
            if len(at_w) >= 2:
                gap = at_w.iloc[0]["C_BA"] - at_w.iloc[1]["C_BA"]
                gaps.append(gap)
                ws_valid.append(w)
        if ws_valid:
            ax.plot(ws_valid, gaps, "o-", color=meta["color"], ms=4, lw=1.3,
                    label=meta["label"].split("(")[0].strip())
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("C_BA gap: 1st - 2nd best (bits)")
    ax.set_title("(b) Gap between top receivers (small = confident limit)")
    ax.set_xlim(0, 5.2)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)

    fig.suptitle("Channel Limit Confidence: Multi-Receiver Convergence", fontsize=12)
    fig.savefig(OUTPUT / "fig05_convergence.png", dpi=200)
    plt.close(fig)
    print("  fig05_convergence.png")


# ═══════════════════ Summary table ═══════════════════

def make_summary(envelope, fit_results):
    rows = []
    for paradigm in PARADIGM_ORDER:
        meta = PARADIGM_META[paradigm]
        penv = envelope[envelope["paradigm"] == paradigm].sort_values("window")
        if penv.empty:
            continue
        w = penv["window"].values
        cba = penv["C_BA_max"].values
        itr = 60.0 / (w + 0.5) * cba
        fit = fit_results.get(paradigm, {})
        rows.append({
            "paradigm": paradigm,
            "label": meta["label"],
            "M": meta["M"],
            "C0": meta["C0"],
            "n_methods": int(penv["n_methods"].max()),
            "CBA_inf": fit.get("CBA_inf", np.nan),
            "tau": fit.get("tau", np.nan),
            "R2": fit.get("R2", np.nan),
            "eta_inf": fit.get("CBA_inf", np.nan) / meta["C0"] if not np.isnan(fit.get("CBA_inf", np.nan)) else np.nan,
            "peak_ITR_envelope": float(itr.max()),
            "T_star": float(w[np.argmax(itr)]),
            "CBA_at_2s": float(cba[np.argmin(np.abs(w - 2.0))]) if len(w) > 0 else np.nan,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES_OUT / "crossparadigm_envelope_summary.csv", index=False)
    return summary


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TABLES_OUT.mkdir(parents=True, exist_ok=True)

    print("Loading all methods across 7 paradigms...")
    all_data = load_all_methods()
    print(f"  Total records: {len(all_data)}")
    for p in PARADIGM_ORDER:
        sub = all_data[all_data["paradigm"] == p]
        methods = sub["method"].unique()
        print(f"  {p}: {len(methods)} methods ({', '.join(methods[:4])}{'...' if len(methods) > 4 else ''})")

    print("\nComputing envelope (max C_BA per paradigm per window)...")
    envelope = compute_envelope(all_data)

    print("\nFigure 1: Single-paradigm multi-receiver analysis...")
    for paradigm in PARADIGM_ORDER:
        fig01_multi_receiver(all_data, paradigm)

    print("\nFigure 2: Cross-paradigm envelope comparison...")
    fig02_envelope_comparison(envelope)

    print("\nFigure 3: Envelope CT tradeoff...")
    fig03_envelope_ct(envelope)

    print("\nFigure 4: Envelope exponential fit...")
    fit_results = fig04_envelope_fit(envelope)

    print("\nFigure 5: Convergence summary...")
    fig05_convergence_summary(all_data, envelope)

    print("\nSummary table...")
    summary = make_summary(envelope, fit_results)

    print("\n" + "=" * 90)
    print("Cross-Paradigm Envelope Summary (channel-level, receiver-independent)")
    print("=" * 90)
    fmt = "{:<20s} {:>4s} {:>5s} {:>7s} {:>7s} {:>8s} {:>8s} {:>10s}"
    print(fmt.format("Paradigm", "M", "#Rx", "tau", "eta%", "T*", "CBA@2s", "Peak ITR"))
    print("-" * 90)
    for _, r in summary.iterrows():
        short = r["paradigm"][:19]
        tau_s = f"{r['tau']:.3f}" if not np.isnan(r["tau"]) else "--"
        eta_s = f"{r['eta_inf']*100:.1f}" if not np.isnan(r["eta_inf"]) else "--"
        cba2 = f"{r['CBA_at_2s']:.2f}" if not np.isnan(r["CBA_at_2s"]) else "--"
        print(f"  {short:<18s} {r['M']:>4d} {r['n_methods']:>5d} {tau_s:>7s} {eta_s:>7s} "
              f"{r['T_star']:>7.1f}s {cba2:>7s} {r['peak_ITR_envelope']:>9.1f}")
    print("=" * 90)
    print(f"\nOutputs: {OUTPUT}")


if __name__ == "__main__":
    main()
