"""Cross-paradigm information accumulation comparison - extended with JBHI16/35 and WN-BCI.

Adds 3 new conditions to the existing 4-dataset framework:
  - JBHI35 Spectral-TDCA (35 targets, dual-freq SSVEP)
  - JBHI16 EFUSIONCA (16 targets, SSVEP)
  - WN-BCI TDCA (160 targets, broadband c-VEP)

Produces unified cross-paradigm figures comparable to information_accumulation_rate task.
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
TASK_DIR = Path(__file__).resolve().parent
OUTPUT = TASK_DIR / "figures_crossparadigm"
TABLES_OUT = TASK_DIR / "tables"

CSV_BENCHMARK = TASKS / "channel" / "CH01_dmc_benchmark" / "results" / "analysis" / "capacity_by_method_window_aggregate.csv"
CSV_BINOCULAR = TASKS / "baselines" / "BL06_ssvep_binocular_ar" / "analysis" / "decision_channel" / "tables" / "capacity_by_condition_method_window.csv"
CSV_DUAL_ALPHA = TASKS / "baselines" / "BL08_ssvep_dual_alpha" / "analysis" / "decision_channel" / "tables" / "capacity_by_paradigm_method_window.csv"
CSV_JFPM = TASKS / "baselines" / "BL12_cvep_nbrs_jfpm" / "analysis" / "decision_channel_coding" / "tables" / "capacity_by_paradigm_method_window.csv"
CSV_JBHI = TASK_DIR / "combined_aggregate_capacity.csv"
CSV_WNBCI = TASKS / "baselines" / "BL11_cvep_wn_tdca" / "analysis" / "decision_channel" / "tables" / "capacity_for_crosstask.csv"


def load_all():
    data = {}
    df = pd.read_csv(CSV_BENCHMARK)
    df = df[df["subject"] == "all"]
    sub = df[df["method"] == "ETRCA"].sort_values("window")
    data["Benchmark_ETRCA"] = pd.DataFrame({"window": sub["window"].values.astype(float), "MI": sub["i_uniform"].values.astype(float), "C0": np.log2(40)})

    df = pd.read_csv(CSV_BINOCULAR)
    sub = df[(df["method"] == "ETRCA") & (df["task"] == "DFDP")].sort_values("window")
    if not sub.empty:
        data["BinoAR_DFDP_ETRCA"] = pd.DataFrame({"window": sub["window"].values.astype(float), "MI": sub["MI_uniform"].values.astype(float), "C0": np.log2(8)})

    df = pd.read_csv(CSV_DUAL_ALPHA)
    sub = df[(df["method"] == "ETRCA") & (df["paradigm"] == "Checkerboard_Arrangment")].sort_values("window")
    if not sub.empty:
        data["DualAlpha_CA_ETRCA"] = pd.DataFrame({"window": sub["window"].values.astype(float), "MI": sub["MI_uniform"].values.astype(float), "C0": np.log2(40)})

    df = pd.read_csv(CSV_JFPM)
    sub = df[(df["paradigm"] == "JFPM-8") & (df["method"] == "TRCA")].sort_values("window")
    if not sub.empty:
        data["JFPM8_TRCA"] = pd.DataFrame({"window": sub["window"].values.astype(float), "MI": sub["I_uniform"].values.astype(float), "C0": np.log2(40)})

    df = pd.read_csv(CSV_JBHI)
    sub = df[(df["dataset"] == "JBHI35") & (df["method"] == "SPECTRAL_TDCA")].sort_values("window")
    if not sub.empty:
        data["JBHI35_SpectralTDCA"] = pd.DataFrame({"window": sub["window"].values.astype(float), "MI": sub["i_uniform"].values.astype(float), "C0": np.log2(35)})

    sub = df[(df["dataset"] == "JBHI16") & (df["method"] == "EFUSIONCA")].sort_values("window")
    if not sub.empty:
        data["JBHI16_EFUSIONCA"] = pd.DataFrame({"window": sub["window"].values.astype(float), "MI": sub["i_uniform"].values.astype(float), "C0": np.log2(16)})

    if CSV_WNBCI.exists():
        df = pd.read_csv(CSV_WNBCI)
        sub = df.sort_values("window")
        data["WNBCI_TDCA"] = pd.DataFrame({"window": sub["window"].values.astype(float), "MI": sub["MI_uniform"].values.astype(float), "C0": np.log2(160)})

    return data


STYLES = {
    "Benchmark_ETRCA": {"color": "#1f77b4", "marker": "o", "ls": "-", "label": "Benchmark ETRCA (40cls, single-freq)"},
    "BinoAR_DFDP_ETRCA": {"color": "#d62728", "marker": "^", "ls": "-", "label": "BinoAR DFDP ETRCA (8cls, dual-freq)"},
    "DualAlpha_CA_ETRCA": {"color": "#9467bd", "marker": "D", "ls": "-", "label": "DualAlpha CA ETRCA (40cls, alpha-band)"},
    "JFPM8_TRCA": {"color": "#8c564b", "marker": "v", "ls": "-", "label": "JFPM-8 TRCA (40cls, code-VEP)"},
    "JBHI35_SpectralTDCA": {"color": "#ff7f0e", "marker": "s", "ls": "-", "label": "JBHI35 Spectral-TDCA (35cls, dual-freq)"},
    "JBHI16_EFUSIONCA": {"color": "#2ca02c", "marker": "p", "ls": "--", "label": "JBHI16 E-FusionCA (16cls, SSVEP)"},
    "WNBCI_TDCA": {"color": "#e377c2", "marker": "*", "ls": "-", "label": "WN-BCI TDCA (160cls, broadband c-VEP)"},
}

MAIN_ORDER = ["WNBCI_TDCA", "Benchmark_ETRCA", "JBHI35_SpectralTDCA", "DualAlpha_CA_ETRCA", "JFPM8_TRCA", "JBHI16_EFUSIONCA", "BinoAR_DFDP_ETRCA"]


def _style(key):
    return STYLES.get(key, {"color": "gray", "marker": "o", "ls": "-", "label": key})


def _exp_sat(t, mi_inf, tau):
    return mi_inf * (1.0 - np.exp(-t / tau))


def fit_exponential(w, mi):
    try:
        if len(w) < 3 or mi.max() <= 0:
            return {"MI_inf": np.nan, "tau": np.nan, "R2": np.nan}
        popt, _ = curve_fit(_exp_sat, w, mi, p0=[mi.max() * 1.3, 0.4], bounds=([0, 0.01], [mi.max() * 5, 20.0]), maxfev=10000)
        pred = _exp_sat(w, *popt)
        ss_res = np.sum((mi - pred) ** 2)
        ss_tot = np.sum((mi - mi.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"MI_inf": popt[0], "tau": popt[1], "R2": r2}
    except (RuntimeError, ValueError):
        return {"MI_inf": np.nan, "tau": np.nan, "R2": np.nan}


def half_life(w, mi):
    target = 0.5 * mi[-1]
    idx = np.searchsorted(mi, target)
    if idx == 0:
        return w[0]
    if idx >= len(mi):
        return w[-1]
    t0, t1 = w[idx - 1], w[idx]
    m0, m1 = mi[idx - 1], mi[idx]
    return t0 + (t1 - t0) * (target - m0) / (m1 - m0) if m1 != m0 else t0


def fig01_utilization(data):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for key in MAIN_ORDER:
        if key not in data:
            continue
        d = data[key]
        s = _style(key)
        util = d["MI"].values / d["C0"].iloc[0] * 100
        ax.plot(d["window"], util, color=s["color"], ls=s["ls"], marker=s["marker"], ms=5, lw=1.5, label=s["label"])
    ax.set_xlabel("Window (s)", fontsize=11)
    ax.set_ylabel("Channel Utilization MI/C0 (%)", fontsize=11)
    ax.set_title("Cross-Paradigm Channel Utilization (7 conditions)", fontsize=12)
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 105)
    ax.axhline(100, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT / "fig01_utilization.png", dpi=200)
    plt.close(fig)
    print("  fig01_utilization.png")


def fig02_dmi_dt(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    ax = axes[0]
    for key in MAIN_ORDER:
        if key not in data:
            continue
        d = data[key]
        s = _style(key)
        w = d["window"].values
        mi = d["MI"].values
        if len(w) >= 3:
            rate = np.gradient(mi, w)
            ax.plot(w, rate, color=s["color"], ls=s["ls"], marker=s["marker"], ms=4, lw=1.3, label=s["label"])
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("dMI/dT (bits/s)")
    ax.set_title("(a) Instantaneous information rate")
    ax.set_xlim(0, 2.2)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for key in MAIN_ORDER:
        if key not in data:
            continue
        d = data[key]
        s = _style(key)
        w = d["window"].values
        mi = d["MI"].values
        c0 = d["C0"].iloc[0]
        if len(w) >= 3:
            rate = np.gradient(mi, w) / c0
            ax.plot(w, rate, color=s["color"], ls=s["ls"], marker=s["marker"], ms=4, lw=1.3, label=s["label"])
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("(dMI/dT) / C0 (1/s)")
    ax.set_title("(b) Normalized information rate")
    ax.set_xlim(0, 2.2)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT / "fig02_dmi_dt.png", dpi=200)
    plt.close(fig)
    print("  fig02_dmi_dt.png")


def fig03_ct_tradeoff(data):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for key in MAIN_ORDER:
        if key not in data:
            continue
        d = data[key]
        s = _style(key)
        w = d["window"].values
        mi = d["MI"].values
        itr = 60.0 / (w + 0.5) * mi
        ax.plot(w, itr, color=s["color"], ls=s["ls"], marker=s["marker"], ms=5, lw=1.5, label=s["label"])
        best_idx = np.argmax(itr)
        ax.annotate(f"{itr[best_idx]:.0f}", xy=(w[best_idx], itr[best_idx]), xytext=(3, 5), textcoords="offset points", fontsize=7, color=s["color"])
    ax.set_xlabel("Window (s)", fontsize=11)
    ax.set_ylabel("60/(T+0.5) x MI (bits/min)", fontsize=11)
    ax.set_title("Cross-Paradigm Capacity-Time Tradeoff", fontsize=12)
    ax.set_xlim(0, 2.2)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT / "fig03_ct_tradeoff.png", dpi=200)
    plt.close(fig)
    print("  fig03_ct_tradeoff.png")


def fig04_exponential_fit(data):
    keys = [k for k in MAIN_ORDER if k in data]
    fit_results = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    ax = axes[0]
    for key in keys:
        d = data[key]
        s = _style(key)
        w = d["window"].values
        mi = d["MI"].values
        c0 = d["C0"].iloc[0]
        ax.scatter(w, mi / c0 * 100, color=s["color"], s=20, alpha=0.7, zorder=3)
        fit = fit_exponential(w, mi)
        fit_results[key] = fit
        if not np.isnan(fit["tau"]):
            t_fine = np.linspace(0, w.max() * 1.1, 100)
            mi_fit = _exp_sat(t_fine, fit["MI_inf"], fit["tau"])
            ax.plot(t_fine, mi_fit / c0 * 100, color=s["color"], ls="-", lw=1.2, alpha=0.7)
    ax.axhline(100, color="gray", ls="--", lw=0.7)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("MI / C0 (%)")
    ax.set_title("(a) Exponential saturation fit")
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for key in keys:
        fit = fit_results[key]
        if np.isnan(fit["tau"]):
            continue
        s = _style(key)
        c0 = data[key]["C0"].iloc[0]
        eta = fit["MI_inf"] / c0 * 100
        ax.scatter(fit["tau"], eta, color=s["color"], s=100, marker=s["marker"], edgecolor="black", lw=0.5, zorder=3)
        short = s["label"].split("(")[0].strip()
        ax.annotate(short, (fit["tau"], eta), fontsize=6.5, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Time constant tau (s)")
    ax.set_ylabel("Asymptotic efficiency MI_inf/C0 (%)")
    ax.set_title("(b) Speed-efficiency trade-off across paradigms")
    ax.grid(alpha=0.25)
    fig.savefig(OUTPUT / "fig04_exponential_fit.png", dpi=200)
    plt.close(fig)
    print("  fig04_exponential_fit.png")
    return fit_results


def fig05_summary_bars(data, fit_results):
    keys = [k for k in MAIN_ORDER if k in data]
    rows = []
    for key in keys:
        d = data[key]
        s = _style(key)
        w = d["window"].values
        mi = d["MI"].values
        c0 = d["C0"].iloc[0]
        itr = 60.0 / (w + 0.5) * mi
        fit = fit_results.get(key, {})
        rows.append({"key": key, "label": s["label"], "n_classes": int(round(2 ** c0)), "C0": c0, "tau": fit.get("tau", np.nan), "MI_inf": fit.get("MI_inf", np.nan), "R2": fit.get("R2", np.nan), "eta_inf": fit.get("MI_inf", np.nan) / c0, "T_50": half_life(w, mi), "peak_ITR": itr.max(), "T_star": w[np.argmax(itr)], "MI_max": mi[-1], "util_max": mi[-1] / c0})
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES_OUT / "crossparadigm_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    labels = [r["label"].split("(")[0].strip() for r in rows]
    colors = [_style(r["key"])["color"] for r in rows]
    x = np.arange(len(rows))

    ax = axes[0, 0]
    vals = summary["peak_ITR"].values
    ax.barh(x, vals, color=colors, alpha=0.8, edgecolor="black", lw=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 5, i, f"{v:.0f} bpm @ T*={summary['T_star'].iloc[i]:.1f}s", va="center", fontsize=7)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Peak ITR (bits/min)")
    ax.set_title("(a) Peak information transfer rate")
    ax.invert_yaxis()
    ax.grid(alpha=0.25, axis="x")

    ax = axes[0, 1]
    vals = summary["tau"].values
    valid = ~np.isnan(vals)
    ax.barh(x[valid], vals[valid], color=[colors[i] for i in range(len(colors)) if valid[i]], alpha=0.8, edgecolor="black", lw=0.5)
    for i, (vi, v) in enumerate(zip(np.where(valid)[0], vals[valid])):
        ax.text(v + 0.01, vi, f"{v:.3f}s (R2={summary['R2'].iloc[vi]:.3f})", va="center", fontsize=7)
    ax.set_yticks(x[valid])
    ax.set_yticklabels([labels[i] for i in range(len(labels)) if valid[i]], fontsize=7.5)
    ax.set_xlabel("Time constant tau (s)")
    ax.set_title("(b) Information accumulation speed")
    ax.invert_yaxis()
    ax.grid(alpha=0.25, axis="x")

    ax = axes[1, 0]
    vals = summary["eta_inf"].values * 100
    valid = ~np.isnan(vals)
    ax.barh(x[valid], vals[valid], color=[colors[i] for i in range(len(colors)) if valid[i]], alpha=0.8, edgecolor="black", lw=0.5)
    for i, (vi, v) in enumerate(zip(np.where(valid)[0], vals[valid])):
        ax.text(v + 0.5, vi, f"{v:.1f}%", va="center", fontsize=7)
    ax.set_yticks(x[valid])
    ax.set_yticklabels([labels[i] for i in range(len(labels)) if valid[i]], fontsize=7.5)
    ax.set_xlabel("MI_inf / C0 (%)")
    ax.set_title("(c) Asymptotic channel utilization")
    ax.set_xlim(0, 110)
    ax.axvline(100, color="gray", ls="--", lw=0.7)
    ax.invert_yaxis()
    ax.grid(alpha=0.25, axis="x")

    ax = axes[1, 1]
    for i, row in summary.iterrows():
        ax.scatter(row["n_classes"], row["peak_ITR"], color=colors[i], s=100, marker=_style(row["key"])["marker"], edgecolor="black", lw=0.5, zorder=3)
        ax.annotate(labels[i], (row["n_classes"], row["peak_ITR"]), fontsize=6.5, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Codebook size M (targets)")
    ax.set_ylabel("Peak ITR (bits/min)")
    ax.set_xscale("log", base=2)
    ax.set_title("(d) Codebook size vs. peak throughput")
    ax.grid(alpha=0.25)

    fig.suptitle("Cross-Paradigm Decision Channel Summary (7 conditions)", fontsize=13)
    fig.savefig(OUTPUT / "fig05_summary.png", dpi=200)
    plt.close(fig)
    print("  fig05_summary.png")
    return summary


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    print("Loading 7 cross-paradigm conditions...")
    data = load_all()
    for key, d in data.items():
        s = _style(key)
        print(f"  {key}: {len(d)} windows, T=[{d['window'].min():.1f}, {d['window'].max():.1f}]s, C0={d['C0'].iloc[0]:.2f}, MI_max={d['MI'].max():.3f}")
    print(f"\nGenerating cross-paradigm figures...")
    fig01_utilization(data)
    fig02_dmi_dt(data)
    fig03_ct_tradeoff(data)
    fit_results = fig04_exponential_fit(data)
    summary = fig05_summary_bars(data, fit_results)
    print("\n" + "=" * 80)
    print("Cross-Paradigm Summary")
    print("=" * 80)
    fmt = "{:<28s} {:>4s} {:>7s} {:>7s} {:>8s} {:>8s} {:>10s}"
    print(fmt.format("Condition", "M", "tau", "eta%", "T*", "T_50", "Peak ITR"))
    print("-" * 80)
    for _, r in summary.iterrows():
        short = r["label"].split("(")[0].strip()[:27]
        tau_s = f"{r['tau']:.3f}" if not np.isnan(r["tau"]) else "--"
        eta_s = f"{r['eta_inf']*100:.1f}" if not np.isnan(r["eta_inf"]) else "--"
        print(f"{short:<28s} {r['n_classes']:>4d} {tau_s:>7s} {eta_s:>7s} {r['T_star']:>7.1f}s {r['T_50']:>7.2f}s {r['peak_ITR']:>9.1f}")
    print("=" * 80)
    print(f"\nOutputs: {OUTPUT}")


if __name__ == "__main__":
    main()
