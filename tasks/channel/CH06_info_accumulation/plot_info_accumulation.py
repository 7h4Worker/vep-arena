"""Information accumulation rate analysis across SSVEP paradigms.

Reads capacity tables from four completed decision-channel analyses,
computes dMI/dT, cumulative fraction, exponential saturation fits,
and tests the first-cycle hypothesis across encoding schemes.

Cross-dataset references: see data_sources.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────
TASK_DIR = Path(__file__).resolve().parent
TASKS = TASK_DIR.parent
FIG_DIR = TASK_DIR / "figures"
TABLE_DIR = TASK_DIR / "tables"

CSV_BENCHMARK = (
    TASKS / "CH01_dmc_benchmark"
    / "results" / "extended" / "combined" / "analysis"
    / "capacity_by_method_window_aggregate.csv"
)
CSV_BINOCULAR = (
    TASKS / "ssvep_binocular_ar_trca"
    / "analysis" / "decision_channel" / "tables"
    / "capacity_by_condition_method_window.csv"
)
CSV_DUAL_ALPHA = (
    TASKS / "ssvep_dual_alpha_baselines"
    / "analysis" / "decision_channel" / "tables"
    / "capacity_by_paradigm_method_window.csv"
)
CSV_JFPM = (
    TASKS / "cvep_nbrs_jfpm_tsinghua_2024_baselines"
    / "analysis" / "decision_channel_coding" / "tables"
    / "capacity_by_paradigm_method_window.csv"
)

# ── constants ────────────────────────────────────────────────────────────
C0_40 = np.log2(40)  # 5.322 bits
C0_8 = np.log2(8)    # 3.0 bits
FULL_AXIS_MAX = 5.2


# ── data loading ─────────────────────────────────────────────────────────

def load_benchmark() -> dict[str, pd.DataFrame]:
    """Load Benchmark aggregate MI(T) for selected methods."""
    df = pd.read_csv(CSV_BENCHMARK)
    df = df[df["subject"] == "all"].copy()
    out = {}
    for method in ["ETRCA", "TRCA", "ECCA", "FBCCA", "CCA"]:
        sub = df[df["method"] == method].sort_values("window")
        if sub.empty:
            continue
        out[f"Benchmark_{method}"] = pd.DataFrame({
            "window": sub["window"].values.astype(float),
            "MI": sub["i_uniform"].values.astype(float),
            "C0": C0_40,
            "accuracy": sub["accuracy"].values.astype(float),
        })
    return out


def load_binocular() -> dict[str, pd.DataFrame]:
    """Load Binocular AR MI(T) for selected conditions."""
    df = pd.read_csv(CSV_BINOCULAR)
    out = {}
    selections = [
        ("ETRCA", "SFSP", "BinoAR_SFSP_ETRCA"),
        ("ETRCA", "DFDP", "BinoAR_DFDP_ETRCA"),
        ("ETRCA", "DFDP3", "BinoAR_DFDP3_ETRCA"),
        ("CCA", "SFSP", "BinoAR_SFSP_CCA"),
        ("CCA", "DFDP", "BinoAR_DFDP_CCA"),
    ]
    for method, task, label in selections:
        sub = df[(df["method"] == method) & (df["task"] == task)].sort_values("window")
        if sub.empty:
            continue
        out[label] = pd.DataFrame({
            "window": sub["window"].values.astype(float),
            "MI": sub["MI_uniform"].values.astype(float),
            "C0": C0_8,
            "accuracy": sub["accuracy"].values.astype(float),
        })
    return out


def load_dual_alpha() -> dict[str, pd.DataFrame]:
    """Load Dual Alpha MI(T) for CA paradigm."""
    df = pd.read_csv(CSV_DUAL_ALPHA)
    out = {}
    for method in ["ETRCA", "FBDCCA"]:
        sub = df[
            (df["method"] == method)
            & (df["paradigm"] == "Checkerboard_Arrangment")
        ].sort_values("window")
        if sub.empty:
            continue
        out[f"DualAlpha_CA_{method}"] = pd.DataFrame({
            "window": sub["window"].values.astype(float),
            "MI": sub["MI_uniform"].values.astype(float),
            "C0": C0_40,
            "accuracy": sub["accuracy"].values.astype(float),
        })
    return out


def load_jfpm() -> dict[str, pd.DataFrame]:
    """Load JFPM MI(T) for JFPM-8 paradigm."""
    df = pd.read_csv(CSV_JFPM)
    out = {}
    selections = [
        ("JFPM-8", "TRCA", "JFPM8_TRCA"),
        ("JFPM-8", "FBCCA-CODE", "JFPM8_FBCCA-CODE"),
        ("NBRS-8", "TRCA", "NBRS8_TRCA"),
    ]
    for paradigm, method, label in selections:
        sub = df[
            (df["paradigm"] == paradigm) & (df["method"] == method)
        ].sort_values("window")
        if sub.empty:
            continue
        c0 = C0_8 if paradigm == "NBRS-8" else C0_40
        out[label] = pd.DataFrame({
            "window": sub["window"].values.astype(float),
            "MI": sub["I_uniform"].values.astype(float),
            "C0": c0,
            "accuracy": sub["accuracy"].values.astype(float),
        })
    return out


def load_all() -> dict[str, pd.DataFrame]:
    """Load all datasets into a unified dict."""
    all_data = {}
    for loader in [load_benchmark, load_binocular, load_dual_alpha, load_jfpm]:
        all_data.update(loader())
    return all_data


# ── analysis functions ───────────────────────────────────────────────────

def exp_saturation(t, mi_inf, tau):
    """MI(T) ≈ MI_inf * (1 - exp(-T/tau))"""
    return mi_inf * (1.0 - np.exp(-t / tau))


def fit_exponential(windows: np.ndarray, mi: np.ndarray) -> dict:
    """Fit exponential saturation model via grid search (avoids scipy segfault)."""
    try:
        mi_max = mi.max()
        if mi_max <= 0 or len(windows) < 3:
            raise ValueError("insufficient data")
        best_r2, best_params = -np.inf, (np.nan, np.nan)
        ss_tot = np.sum((mi - mi.mean()) ** 2)
        if ss_tot == 0:
            raise ValueError("constant MI")
        for mi_inf in np.linspace(mi_max * 0.9, mi_max * 3.0, 80):
            for tau in np.linspace(0.05, windows.max() * 2, 80):
                pred = exp_saturation(windows, mi_inf, tau)
                ss_res = np.sum((mi - pred) ** 2)
                r2 = 1 - ss_res / ss_tot
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = (mi_inf, tau)
        mi_inf, tau = best_params
        # refine around best
        for mi_inf_r in np.linspace(mi_inf * 0.95, mi_inf * 1.05, 40):
            for tau_r in np.linspace(tau * 0.9, tau * 1.1, 40):
                pred = exp_saturation(windows, mi_inf_r, tau_r)
                ss_res = np.sum((mi - pred) ** 2)
                r2 = 1 - ss_res / ss_tot
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = (mi_inf_r, tau_r)
        mi_inf, tau = best_params
        return {"MI_inf": mi_inf, "tau": tau, "R2": best_r2, "converged": True}
    except (RuntimeError, ValueError):
        return {"MI_inf": np.nan, "tau": np.nan, "R2": np.nan, "converged": False}


def compute_dmi_dt(windows: np.ndarray, mi: np.ndarray) -> np.ndarray:
    """Numerical derivative dMI/dT using central differences."""
    return np.gradient(mi, windows)


def cumulative_fraction(mi: np.ndarray) -> np.ndarray:
    """F(T) = MI(T) / MI(T_max)."""
    mi_max = mi[-1]
    return mi / mi_max if mi_max > 0 else np.zeros_like(mi)


def first_cycle_fraction(windows: np.ndarray, mi: np.ndarray,
                         t_cycle: float = 0.2) -> float:
    """Fraction of MI reached by t_cycle seconds."""
    mi_max = mi[-1]
    if mi_max <= 0:
        return 0.0
    idx = np.searchsorted(windows, t_cycle, side="right") - 1
    if idx < 0:
        return 0.0
    if idx < len(windows) - 1:
        t0, t1 = windows[idx], windows[idx + 1]
        m0, m1 = mi[idx], mi[idx + 1]
        mi_at_tc = m0 + (m1 - m0) * (t_cycle - t0) / (t1 - t0)
    else:
        mi_at_tc = mi[idx]
    return mi_at_tc / mi_max


def half_life(windows: np.ndarray, mi: np.ndarray) -> float:
    """Time to reach 50% of MI(T_max)."""
    mi_max = mi[-1]
    target = 0.5 * mi_max
    idx = np.searchsorted(mi, target)
    if idx == 0:
        return windows[0]
    if idx >= len(mi):
        return windows[-1]
    t0, t1 = windows[idx - 1], windows[idx]
    m0, m1 = mi[idx - 1], mi[idx]
    if m1 == m0:
        return t0
    return t0 + (t1 - t0) * (target - m0) / (m1 - m0)


# ── display config ───────────────────────────────────────────────────────

DATASET_GROUPS = {
    "Benchmark": {
        "Benchmark_ETRCA": {"color": "#1f77b4", "ls": "-", "marker": "o",
                            "label": "Benchmark ETRCA (40cls)"},
        "Benchmark_TRCA": {"color": "#17becf", "ls": "-", "marker": "^",
                           "label": "Benchmark TRCA (40cls)"},
        "Benchmark_ECCA": {"color": "#9467bd", "ls": "-.", "marker": "D",
                           "label": "Benchmark ECCA (40cls)"},
        "Benchmark_FBCCA": {"color": "#ff7f0e", "ls": ":", "marker": "v",
                            "label": "Benchmark FBCCA (40cls)"},
        "Benchmark_CCA": {"color": "#1f77b4", "ls": "--", "marker": "s",
                          "label": "Benchmark CCA (40cls)"},
    },
    "Binocular AR": {
        "BinoAR_SFSP_ETRCA": {"color": "#2ca02c", "ls": "-", "marker": "o",
                               "label": "BinoAR SFSP·ETRCA (8cls)"},
        "BinoAR_DFDP_ETRCA": {"color": "#d62728", "ls": "-", "marker": "^",
                               "label": "BinoAR DFDP·ETRCA (8cls)"},
        "BinoAR_DFDP3_ETRCA": {"color": "#ff7f0e", "ls": "-", "marker": "D",
                                "label": "BinoAR DFDP3·ETRCA (8cls)"},
        "BinoAR_SFSP_CCA": {"color": "#2ca02c", "ls": "--", "marker": "s",
                             "label": "BinoAR SFSP·CCA (8cls)"},
        "BinoAR_DFDP_CCA": {"color": "#d62728", "ls": "--", "marker": "v",
                             "label": "BinoAR DFDP·CCA (8cls)"},
    },
    "Dual Alpha": {
        "DualAlpha_CA_ETRCA": {"color": "#9467bd", "ls": "-", "marker": "o",
                                "label": "DualAlpha CA·ETRCA (40cls)"},
        "DualAlpha_CA_FBDCCA": {"color": "#9467bd", "ls": "--", "marker": "s",
                                 "label": "DualAlpha CA·FBDCCA (40cls)"},
    },
    "JFPM/NBRS": {
        "JFPM8_TRCA": {"color": "#8c564b", "ls": "-", "marker": "o",
                        "label": "JFPM-8 TRCA (40cls)"},
        "JFPM8_FBCCA-CODE": {"color": "#8c564b", "ls": "--", "marker": "s",
                              "label": "JFPM-8 FBCCA-CODE (40cls)"},
        "NBRS8_TRCA": {"color": "#e377c2", "ls": "-", "marker": "D",
                        "label": "NBRS-8 TRCA (8cls)"},
    },
}

# Subset for main cross-paradigm comparison (one "best method" per paradigm)
MAIN_KEYS = [
    "Benchmark_ETRCA",
    "BinoAR_SFSP_ETRCA",
    "BinoAR_DFDP_ETRCA",
    "DualAlpha_CA_ETRCA",
    "JFPM8_TRCA",
    "NBRS8_TRCA",
]


def _style(key: str) -> dict:
    for group in DATASET_GROUPS.values():
        if key in group:
            return group[key]
    return {"color": "gray", "ls": "-", "marker": "o", "label": key}


# ── figures ──────────────────────────────────────────────────────────────

def fig01_utilization_curves(data: dict[str, pd.DataFrame]) -> None:
    """MI/C0 utilization vs window — cross-paradigm comparison."""
    print("Fig 01: Utilization curves...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (a) All conditions
    ax = axes[0]
    for key in MAIN_KEYS:
        if key not in data:
            continue
        d = data[key]
        s = _style(key)
        util = d["MI"].values / d["C0"].iloc[0]
        ax.plot(d["window"], util * 100, color=s["color"], ls=s["ls"],
                marker=s["marker"], ms=4, lw=1.5, label=s["label"])
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("Utilization MI/C₀ (%)")
    ax.set_title("(a) Channel utilization across paradigms")
    ax.set_xlim(0, FULL_AXIS_MAX)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # (b) Binocular AR: single vs dual freq (controlled comparison)
    ax = axes[1]
    for key in ["BinoAR_SFSP_ETRCA", "BinoAR_DFDP_ETRCA", "BinoAR_DFDP3_ETRCA",
                "BinoAR_SFSP_CCA", "BinoAR_DFDP_CCA"]:
        if key not in data:
            continue
        d = data[key]
        s = _style(key)
        util = d["MI"].values / d["C0"].iloc[0]
        ax.plot(d["window"], util * 100, color=s["color"], ls=s["ls"],
                marker=s["marker"], ms=3, lw=1.5,
                label=s["label"].replace("BinoAR ", ""))
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("Utilization MI/C₀ (%)")
    ax.set_title("(b) Binocular AR: single vs dual frequency")
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_utilization_curves.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("  -> fig01_utilization_curves.png")


def fig02_dmi_dt(data: dict[str, pd.DataFrame]) -> None:
    """Information accumulation rate dMI/dT (bits/s) vs window."""
    print("Fig 02: dMI/dT curves...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (a) dMI/dT in bits/s
    ax = axes[0]
    for key in MAIN_KEYS:
        if key not in data:
            continue
        d = data[key]
        s = _style(key)
        w = d["window"].values
        mi = d["MI"].values
        rate = compute_dmi_dt(w, mi)
        ax.plot(w, rate, color=s["color"], ls=s["ls"], marker=s["marker"],
                ms=3, lw=1.5, label=s["label"])
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("dMI/dT (bits/s)")
    ax.set_title("(a) Instantaneous information rate")
    ax.set_xlim(0, FULL_AXIS_MAX)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)

    # (b) Normalized rate: (dMI/dT) / C0
    ax = axes[1]
    for key in MAIN_KEYS:
        if key not in data:
            continue
        d = data[key]
        s = _style(key)
        w = d["window"].values
        mi = d["MI"].values
        c0 = d["C0"].iloc[0]
        rate = compute_dmi_dt(w, mi) / c0
        ax.plot(w, rate, color=s["color"], ls=s["ls"], marker=s["marker"],
                ms=3, lw=1.5, label=s["label"])
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("(dMI/dT) / C₀ (s⁻¹)")
    ax.set_title("(b) Normalized information rate")
    ax.set_xlim(0, FULL_AXIS_MAX)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_dmi_dt.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  -> fig02_dmi_dt.png")


def fig03_cumulative_fraction(data: dict[str, pd.DataFrame]) -> None:
    """Cumulative fraction F(T) = MI(T)/MI(T_max) — first-cycle test."""
    print("Fig 03: Cumulative fraction...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (a) Cumulative fraction curves (template methods only)
    ax = axes[0]
    template_keys = [k for k in MAIN_KEYS if k in data]
    for key in template_keys:
        d = data[key]
        s = _style(key)
        w = d["window"].values
        mi = d["MI"].values
        frac = cumulative_fraction(mi)
        ax.plot(w, frac * 100, color=s["color"], ls=s["ls"],
                marker=s["marker"], ms=3, lw=1.5, label=s["label"])

    ax.axhline(50, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.axhline(79, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.axhline(90, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.axvline(0.2, color="red", ls=":", lw=0.8, alpha=0.5)
    ax.text(0.22, 82, "T=0.2s", fontsize=7, color="red", alpha=0.7)
    ax.text(0.02, 80, "79%", fontsize=7, color="gray")
    ax.text(0.02, 51, "50%", fontsize=7, color="gray")
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("MI(T) / MI(T_max) (%)")
    ax.set_title("(a) Cumulative information fraction")
    ax.set_xlim(0, FULL_AXIS_MAX)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.3)

    # (b) First-cycle fraction bar chart
    ax = axes[1]
    labels, fracs_02, fracs_05 = [], [], []
    for key in template_keys:
        d = data[key]
        w = d["window"].values
        mi = d["MI"].values
        f02 = first_cycle_fraction(w, mi, 0.2)
        f05 = first_cycle_fraction(w, mi, 0.5)
        short = _style(key)["label"]
        short = short.split("(")[0].strip()
        labels.append(short)
        fracs_02.append(f02 * 100)
        fracs_05.append(f05 * 100)

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width / 2, fracs_02, width, label="F(0.2s)",
                   color="#e41a1c", alpha=0.8, edgecolor="black", lw=0.5)
    bars2 = ax.bar(x + width / 2, fracs_05, width, label="F(0.5s)",
                   color="#377eb8", alpha=0.8, edgecolor="black", lw=0.5)
    for bar, v in zip(bars1, fracs_02):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    f"{v:.0f}%", ha="center", fontsize=6.5)
    for bar, v in zip(bars2, fracs_05):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    f"{v:.0f}%", ha="center", fontsize=6.5)
    ax.axhline(79, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(len(labels) - 0.5, 80, "79% (first-cycle hypothesis)",
            fontsize=7, color="gray", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Fraction of max MI (%)")
    ax.set_title("(b) First-cycle (0.2s) and half-second fractions")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_cumulative_fraction.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("  -> fig03_cumulative_fraction.png")


def fig04_exponential_fit(data: dict[str, pd.DataFrame]) -> None:
    """Exponential saturation fit MI(T) = MI_inf*(1 - exp(-T/tau))."""
    print("Fig 04: Exponential fit...")

    keys_to_fit = [k for k in MAIN_KEYS if k in data]
    n = len(keys_to_fit)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    fit_results = []
    for idx, key in enumerate(keys_to_fit):
        ax = axes[idx]
        d = data[key]
        s = _style(key)
        w = d["window"].values
        mi = d["MI"].values
        c0 = d["C0"].iloc[0]

        ax.scatter(w, mi, color=s["color"], s=20, zorder=3, alpha=0.8)

        fit = fit_exponential(w, mi)
        fit_results.append({"key": key, "label": s["label"], **fit, "C0": c0})

        if fit["converged"]:
            t_fine = np.linspace(0, w.max() * 1.2, 200)
            mi_fit = exp_saturation(t_fine, fit["MI_inf"], fit["tau"])
            ax.plot(t_fine, mi_fit, color=s["color"], ls="-", lw=1.5)
            ax.axhline(fit["MI_inf"], color="gray", ls=":", lw=0.7)
            ax.axhline(c0, color="black", ls="--", lw=0.7, alpha=0.5)
            ax.text(0.95, 0.95,
                    f"MI∞={fit['MI_inf']:.2f}\nτ={fit['tau']:.3f}s\n"
                    f"R²={fit['R2']:.4f}\nη={fit['MI_inf']/c0:.1%}",
                    transform=ax.transAxes, fontsize=7, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_xlabel("Window (s)")
        ax.set_ylabel("MI (bits)")
        ax.set_title(s["label"], fontsize=8)
        ax.set_xlim(0, w.max() * 1.15)
        ax.set_ylim(0, c0 * 1.1)
        ax.grid(alpha=0.3)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Exponential saturation fit: MI(T) = MI∞·(1 − e^{−T/τ})",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_exponential_fit.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(fit_results).to_csv(TABLE_DIR / "exponential_fit.csv",
                                     index=False)
    print("  -> fig04_exponential_fit.png + exponential_fit.csv")
    return fit_results


def fig05_time_constants(fit_results: list[dict]) -> None:
    """Compare time constants tau across paradigms."""
    print("Fig 05: Time constant comparison...")
    df = pd.DataFrame(fit_results)
    df = df[df["converged"]].copy()
    if df.empty:
        print("  No converged fits, skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # (a) tau bar chart
    ax = axes[0]
    labels = [r["label"].split("(")[0].strip() for _, r in df.iterrows()]
    taus = df["tau"].values
    colors = []
    for _, r in df.iterrows():
        s = _style(r["key"])
        colors.append(s["color"])
    bars = ax.barh(range(len(labels)), taus, color=colors, alpha=0.8,
                   edgecolor="black", lw=0.5)
    for i, (bar, v) in enumerate(zip(bars, taus)):
        ax.text(v + 0.01, i, f"{v:.3f}s", va="center", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Time constant τ (s)")
    ax.set_title("(a) Saturation time constant")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # (b) MI_inf / C0 (asymptotic efficiency)
    ax = axes[1]
    eta = df["MI_inf"].values / df["C0"].values
    bars = ax.barh(range(len(labels)), eta * 100, color=colors, alpha=0.8,
                   edgecolor="black", lw=0.5)
    for i, (bar, v) in enumerate(zip(bars, eta * 100)):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Asymptotic efficiency MI∞/C₀ (%)")
    ax.set_title("(b) Asymptotic utilization")
    ax.set_xlim(0, 110)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # (c) tau vs MI_inf/C0 scatter
    ax = axes[2]
    for i, (_, r) in enumerate(df.iterrows()):
        s = _style(r["key"])
        ax.scatter(r["tau"], r["MI_inf"] / r["C0"] * 100,
                   c=s["color"], s=80, marker=s["marker"],
                   edgecolor="black", lw=0.5, zorder=3)
        short = s["label"].split("·")[0] if "·" in s["label"] else s["label"].split(" ")[0]
        ax.annotate(short, (r["tau"], r["MI_inf"] / r["C0"] * 100),
                    fontsize=6.5, textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Time constant τ (s)")
    ax.set_ylabel("Asymptotic efficiency MI∞/C₀ (%)")
    ax.set_title("(c) Speed–efficiency trade-off")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_time_constants.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("  -> fig05_time_constants.png")


def fig06_encoding_effect_on_rate(data: dict[str, pd.DataFrame]) -> None:
    """Within Binocular AR: how dual-freq encoding changes dMI/dT dynamics."""
    print("Fig 06: Encoding effect on accumulation rate...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    keys_bino = ["BinoAR_SFSP_ETRCA", "BinoAR_DFDP_ETRCA", "BinoAR_DFDP3_ETRCA"]
    available = [k for k in keys_bino if k in data]
    if not available:
        print("  No Binocular AR data, skipping.")
        plt.close(fig)
        return

    # (a) MI(T) comparison
    ax = axes[0]
    for key in available:
        d = data[key]
        s = _style(key)
        ax.plot(d["window"], d["MI"], color=s["color"], ls=s["ls"],
                marker=s["marker"], ms=3, lw=1.5,
                label=s["label"].replace("BinoAR ", "").replace(" (8cls)", ""))
    ax.axhline(C0_8, color="gray", ls="--", lw=0.7, label="C₀ = 3.0")
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("MI (bits)")
    ax.set_title("(a) MI accumulation")
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, C0_8 + 0.3)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) dMI/dT
    ax = axes[1]
    for key in available:
        d = data[key]
        s = _style(key)
        w = d["window"].values
        mi = d["MI"].values
        rate = compute_dmi_dt(w, mi)
        ax.plot(w, rate, color=s["color"], ls=s["ls"],
                marker=s["marker"], ms=3, lw=1.5,
                label=s["label"].replace("BinoAR ", "").replace(" (8cls)", ""))
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("dMI/dT (bits/s)")
    ax.set_title("(b) Instantaneous rate")
    ax.set_xlim(0, 3.2)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) Dual-freq MI gain over SFSP vs T
    ax = axes[2]
    if "BinoAR_SFSP_ETRCA" in data:
        d_sfsp = data["BinoAR_SFSP_ETRCA"]
        for key in ["BinoAR_DFDP_ETRCA", "BinoAR_DFDP3_ETRCA"]:
            if key not in data:
                continue
            d = data[key]
            s = _style(key)
            w_common = np.intersect1d(d_sfsp["window"].values,
                                      d["window"].values)
            if len(w_common) == 0:
                continue
            mi_sfsp = np.interp(w_common, d_sfsp["window"].values,
                                d_sfsp["MI"].values)
            mi_df = np.interp(w_common, d["window"].values, d["MI"].values)
            gain = mi_df - mi_sfsp
            ax.plot(w_common, gain, color=s["color"], ls=s["ls"],
                    marker=s["marker"], ms=3, lw=1.5,
                    label=s["label"].replace("BinoAR ", "").replace(" (8cls)", "")
                    + " − SFSP")
    ax.axhline(0, color="gray", ls=":", lw=0.7)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("ΔMI (bits)")
    ax.set_title("(c) Dual-freq gain over SFSP vs time")
    ax.set_xlim(0, 3.2)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_encoding_effect.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("  -> fig06_encoding_effect.png")


def fig07_marginal_itr(data: dict[str, pd.DataFrame]) -> None:
    """Marginal ITR: bits/min at each window, not cumulative ITR."""
    print("Fig 07: Marginal vs cumulative ITR...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Cumulative ITR = MI(T) * 60/T
    ax = axes[0]
    for key in MAIN_KEYS:
        if key not in data:
            continue
        d = data[key]
        s = _style(key)
        w = d["window"].values
        mi = d["MI"].values
        itr = mi * 60 / w
        ax.plot(w, itr, color=s["color"], ls=s["ls"],
                marker=s["marker"], ms=3, lw=1.5, label=s["label"])
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("Cumulative ITR (bits/min)")
    ax.set_title("(a) Cumulative ITR = MI(T)·60/T")
    ax.set_xlim(0, FULL_AXIS_MAX)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)

    # (b) Marginal ITR = dMI/dT * 60
    ax = axes[1]
    for key in MAIN_KEYS:
        if key not in data:
            continue
        d = data[key]
        s = _style(key)
        w = d["window"].values
        mi = d["MI"].values
        rate = compute_dmi_dt(w, mi) * 60
        ax.plot(w, rate, color=s["color"], ls=s["ls"],
                marker=s["marker"], ms=3, lw=1.5, label=s["label"])
    ax.axhline(0, color="gray", ls=":", lw=0.7)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("Marginal ITR (bits/min)")
    ax.set_title("(b) Marginal ITR = dMI/dT·60")
    ax.set_xlim(0, FULL_AXIS_MAX)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_marginal_itr.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("  -> fig07_marginal_itr.png")


# ── summary table ────────────────────────────────────────────────────────

def build_summary_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build comprehensive summary for all conditions."""
    rows = []
    for key, d in data.items():
        s = _style(key)
        w = d["window"].values
        mi = d["MI"].values
        c0 = d["C0"].iloc[0]
        fit = fit_exponential(w, mi)
        t50 = half_life(w, mi)

        row = {
            "key": key,
            "label": s["label"],
            "C0": c0,
            "n_classes": int(2 ** c0),
            "window_min": w.min(),
            "window_max": w.max(),
            "MI_max": mi[-1],
            "utilization_max": mi[-1] / c0,
            "MI_inf": fit.get("MI_inf", np.nan),
            "tau": fit.get("tau", np.nan),
            "R2": fit.get("R2", np.nan),
            "eta_inf": fit.get("MI_inf", np.nan) / c0,
            "T_50": t50,
            "F_0.2s": first_cycle_fraction(w, mi, 0.2),
            "F_0.5s": first_cycle_fraction(w, mi, 0.5),
            "F_1.0s": first_cycle_fraction(w, mi, 1.0),
            "peak_dMI_dT": compute_dmi_dt(w, mi).max(),
            "peak_dMI_dT_window": w[np.argmax(compute_dmi_dt(w, mi))],
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data from 4 datasets...")
    data = load_all()
    print(f"  Loaded {len(data)} conditions:")
    for key, d in data.items():
        print(f"    {key}: {len(d)} windows, "
              f"T=[{d['window'].min():.1f}, {d['window'].max():.1f}]s, "
              f"C0={d['C0'].iloc[0]:.2f}")

    print("\nBuilding summary table...")
    summary = build_summary_table(data)
    summary.to_csv(TABLE_DIR / "summary.csv", index=False)
    print(f"  {len(summary)} rows saved")

    fig01_utilization_curves(data)
    fig02_dmi_dt(data)
    fig03_cumulative_fraction(data)
    fit_results = fig04_exponential_fit(data)
    fig05_time_constants(fit_results)
    fig06_encoding_effect_on_rate(data)
    fig07_marginal_itr(data)

    # Print summary
    print("\n" + "=" * 85)
    print("Information Accumulation Summary")
    print("=" * 85)
    fmt = "{:<28s} {:>5s} {:>6s} {:>6s} {:>8s} {:>6s} {:>6s} {:>6s}"
    print(fmt.format("Condition", "K", "tau", "R2", "MI_inf", "F0.2", "F0.5", "T50"))
    print("-" * 85)
    for _, r in summary.iterrows():
        short = r["label"].split("(")[0].strip()[:27]
        tau_s = f"{r['tau']:.3f}" if not np.isnan(r["tau"]) else "--"
        r2_s = f"{r['R2']:.3f}" if not np.isnan(r["R2"]) else "--"
        eta_s = f"{r['eta_inf']:.1%}" if not np.isnan(r["eta_inf"]) else "--"
        print(f"{short:<28s} {r['n_classes']:>5d} {tau_s:>6s} {r2_s:>6s} "
              f"{eta_s:>8s} {r['F_0.2s']:>5.1%} {r['F_0.5s']:>5.1%} "
              f"{r['T_50']:>5.2f}s")
    print("=" * 85)
    print("\nDone.")


if __name__ == "__main__":
    main()
