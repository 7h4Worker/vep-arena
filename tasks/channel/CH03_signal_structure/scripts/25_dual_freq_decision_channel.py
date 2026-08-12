"""Dual-frequency paradigm decision channel analysis.

Part A: Binocular AR (Ke 2025) — 9 conditions with controlled dual-freq variations
Part B: JFPM (Zheng 2024) — 3 paradigms (JFPM-8, NBRS-15, NBRS-8) comparison

Uses existing predictions.csv — no new experiments.

Usage
-----
    .venv/Scripts/python.exe scripts/25_dual_freq_decision_channel.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import csv

TASK_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = TASK_DIR / "results" / "figures"

# ── Binocular AR paths ──
BINO_ROOT = Path("D:/ProjData/proj_python/vep_arena/tasks/baselines/BL06_ssvep_binocular_ar/results")
BINO_EXP1 = BINO_ROOT / "experiment1_trca" / "predictions.csv"
BINO_EXP2 = BINO_ROOT / "experiment2_trca" / "predictions.csv"
BINO_EXP3 = BINO_ROOT / "experiment3_trca" / "predictions.csv"

# ── JFPM paths ──
JFPM_ROOT = Path("D:/ProjData/proj_python/vep_arena/tasks/baselines/BL12_cvep_nbrs_jfpm/results")
JFPM_FULL = JFPM_ROOT / "full_occipital9_fbcca_trca_w04_40_20260706" / "predictions.csv"
JFPM_MSTRCA = JFPM_ROOT / "full_occipital9_mstrca_v2_paper_preproc_w04_40_20260706" / "predictions.csv"

# ── Codebooks ──
# Binocular AR Experiment 2 (8 targets each)
BINO_EXP2_CODEBOOK = {
    "SFSP": {
        "left_freq":  [8, 9, 10, 11, 12, 13, 14, 15],
        "right_freq": [8, 9, 10, 11, 12, 13, 14, 15],
        "left_phase":  [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        "right_phase": [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    },
    "SFDP": {
        "left_freq":  [8, 9, 10, 11, 12, 13, 14, 15],
        "right_freq": [8, 9, 10, 11, 12, 13, 14, 15],
        "left_phase":  [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        "right_phase": [1.0, 1.25, 1.5, 1.75, 0, 0.25, 0.5, 0.75],
    },
    "DFSP": {
        "left_freq":  [8, 9, 10, 11, 12, 13, 14, 15],
        "right_freq": [10, 11, 12, 13, 14, 15, 12, 13],
        "left_phase":  [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        "right_phase": [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    },
    "DFDP": {
        "left_freq":  [8, 9, 10, 11, 12, 13, 14, 15],
        "right_freq": [10, 11, 12, 13, 14, 15, 12, 13],
        "left_phase":  [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        "right_phase": [1.0, 1.25, 1.5, 1.75, 0, 0.25, 0.5, 0.75],
    },
}

# Experiment 3 — varying frequency gap
BINO_EXP3_LEFT_FREQ = [7.0, 7.5, 8.0, 8.5, 9.5, 10.5, 11.5, 12.5]
BINO_EXP3_LEFT_PHASE = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
BINO_EXP3_CODEBOOK = {}
for gap in [1, 3, 5]:
    cond = f"DFDP{gap}"
    BINO_EXP3_CODEBOOK[cond] = {
        "left_freq": BINO_EXP3_LEFT_FREQ,
        "right_freq": [f + gap for f in BINO_EXP3_LEFT_FREQ],
        "left_phase": BINO_EXP3_LEFT_PHASE,
        "right_phase": [(p + 1.0) % 2.0 for p in BINO_EXP3_LEFT_PHASE],
    }

N_TARGETS_BINO = 8
N_TARGETS_JFPM = 40
CAPACITY_BINO = np.log2(N_TARGETS_BINO)   # 3.0 bits
CAPACITY_JFPM = np.log2(N_TARGETS_JFPM)   # 5.322 bits

METHOD_COLORS = {
    "CCA": "#6c757d", "FBCCA": "#2a9d8f", "TRCA": "#e9c46a",
    "ETRCA": "#e63946", "ECCA": "#264653",
    "FBCCA-CODE": "#2a9d8f", "MSTRCA": "#e63946",
}
COND_COLORS = {
    "LF": "#264653", "MF": "#2a9d8f",
    "SFSP": "#457b9d", "SFDP": "#a8dadc",
    "DFSP": "#e9c46a", "DFDP": "#e76f51",
    "DFDP1": "#f4a261", "DFDP3": "#e76f51", "DFDP5": "#e63946",
    "JFPM-8": "#e63946", "NBRS-15": "#2a9d8f", "NBRS-8": "#264653",
}


# ═══════════════════════════════════════════════════════════
#  Core functions (aligned with vep_arena.channel interface)
# ═══════════════════════════════════════════════════════════

def load_predictions_bino(csv_path, task, method, window):
    pairs = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            if (row["task"] == task and row["method"] == method
                    and abs(float(row["window"]) - window) < 0.01):
                pairs.append((int(row["true"]), int(row["pred"])))
    return pairs


def load_predictions_jfpm(csv_path, paradigm, method, window):
    pairs = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            if (row["paradigm"] == paradigm and row["method"] == method
                    and abs(float(row["window"]) - window) < 0.01):
                pairs.append((int(row["target"]), int(row["pred"])))
    return pairs


def confusion_counts(pairs, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in pairs:
        if 0 <= true < n_classes and 0 <= pred < n_classes:
            cm[true, pred] += 1
    return cm


def normalize_transition(cm):
    P = cm.astype(np.float64)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return P / row_sums


def mutual_info_uniform(P):
    """MI under uniform input from row-normalized transition matrix P."""
    M = P.shape[0]
    q = np.ones(M) / M
    p_y = P.T @ q
    mi = 0.0
    for i in range(M):
        for j in range(M):
            if P[i, j] > 0 and p_y[j] > 0:
                mi += q[i] * P[i, j] * np.log2(P[i, j] / p_y[j])
    return max(0.0, mi)


def accuracy_from_cm(cm):
    total = cm.sum()
    return np.diag(cm).sum() / max(total, 1) * 100


def per_subject_mi(csv_path, task, method, window, n_classes, loader="bino"):
    subjects = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            key_task = "task" if loader == "bino" else "paradigm"
            key_true = "true" if loader == "bino" else "target"
            if (row[key_task] == task and row["method"] == method
                    and abs(float(row["window"]) - window) < 0.01):
                s = int(row["subject"])
                if s not in subjects:
                    subjects[s] = []
                subjects[s].append((int(row[key_true]), int(row["pred"])))

    mis, accs = [], []
    for s in sorted(subjects):
        cm = confusion_counts(subjects[s], n_classes)
        P = normalize_transition(cm)
        mis.append(mutual_info_uniform(P))
        accs.append(accuracy_from_cm(cm))
    return np.array(mis), np.array(accs), sorted(subjects.keys())


def get_available_windows(csv_path, task, method, key_task="task"):
    windows = set()
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            if row[key_task] == task and row["method"] == method:
                windows.add(float(row["window"]))
    return sorted(windows)


# ═══════════════════════════════════════════════════════════
#  Figure 1: Stimulus waveform comparison (single vs dual freq)
# ═══════════════════════════════════════════════════════════

def fig_stimulus_waveforms():
    """Visualize stimulus waveforms and spectra for representative targets."""
    fig, axes = plt.subplots(3, 4, figsize=(24, 14))
    fs = 1024
    t = np.arange(0, 2.0, 1 / fs)
    n_fft = len(t)
    freqs_fft = np.fft.rfftfreq(n_fft, 1 / fs)

    examples = [
        ("LF — Target 1\n单频 8 Hz", 8, None, 0, None),
        ("SFDP — Target 1\n同频异相 8 Hz", 8, 8, 0, 1.0),
        ("DFSP — Target 1\n异频同相 8+10 Hz", 8, 10, 0, 0),
        ("DFDP — Target 1\n异频异相 8+10 Hz", 8, 10, 0, 1.0),
        ("DFDP1 — Target 1\n7+8 Hz (gap=1)", 7, 8, 0, 1.0),
        ("DFDP3 — Target 1\n7+10 Hz (gap=3)", 7, 10, 0, 1.0),
    ]

    for idx, (title, f_left, f_right, ph_left, ph_right) in enumerate(examples):
        row = idx // 2 if idx < 4 else 2
        col_base = (idx % 2) * 2 if idx < 4 else (idx - 4) * 2

        s_left = np.sin(2 * np.pi * f_left * t + ph_left * np.pi)
        if f_right is not None and f_right != f_left:
            s_right = np.sin(2 * np.pi * f_right * t + ph_right * np.pi)
            composite = s_left + s_right
        elif f_right is not None and f_right == f_left and ph_right is not None:
            s_right = np.sin(2 * np.pi * f_right * t + ph_right * np.pi)
            composite = s_left + s_right
        else:
            composite = s_left

        # Time domain (first 0.5s)
        ax_t = axes[row, col_base]
        mask = t <= 0.5
        ax_t.plot(t[mask] * 1000, s_left[mask], "steelblue", linewidth=1.5,
                  alpha=0.6, label="左眼")
        if f_right is not None:
            s_r = np.sin(2 * np.pi * f_right * t + (ph_right if ph_right else 0) * np.pi)
            ax_t.plot(t[mask] * 1000, s_r[mask], "darkorange", linewidth=1.5,
                      alpha=0.6, label="右眼")
        ax_t.plot(t[mask] * 1000, composite[mask], "k", linewidth=1.2,
                  alpha=0.8, label="合成")
        ax_t.set_xlim(0, 500)
        ax_t.set_xlabel("时间 (ms)", fontsize=9)
        ax_t.set_ylabel("幅度", fontsize=9)
        ax_t.set_title(title, fontsize=10, fontweight="bold")
        if idx == 0:
            ax_t.legend(fontsize=8, loc="upper right")
        ax_t.grid(alpha=0.15)

        # Frequency domain
        ax_f = axes[row, col_base + 1]
        spectrum = np.abs(np.fft.rfft(composite)) / n_fft * 2
        f_mask = freqs_fft <= 40
        ax_f.plot(freqs_fft[f_mask], spectrum[f_mask], "k", linewidth=1.2)
        ax_f.fill_between(freqs_fft[f_mask], 0, spectrum[f_mask], alpha=0.2,
                          color="steelblue")

        # Mark fundamental frequencies
        ax_f.axvline(f_left, color="steelblue", linestyle="--", alpha=0.7,
                     linewidth=1)
        ax_f.text(f_left, spectrum.max() * 0.9, f"f₁={f_left}",
                  fontsize=8, color="steelblue", ha="center")
        if f_right is not None and f_right != f_left:
            ax_f.axvline(f_right, color="darkorange", linestyle="--", alpha=0.7,
                         linewidth=1)
            ax_f.text(f_right, spectrum.max() * 0.75, f"f₂={f_right}",
                      fontsize=8, color="darkorange", ha="center")
            # Mark intermodulation
            f_diff = abs(f_right - f_left)
            f_sum = f_left + f_right
            if f_diff > 0 and f_diff <= 40:
                ax_f.axvline(f_diff, color="red", linestyle=":", alpha=0.4)
                ax_f.text(f_diff, spectrum.max() * 0.5, f"Δf={f_diff:.0f}",
                          fontsize=7, color="red", ha="center")
            if f_sum <= 40:
                ax_f.axvline(f_sum, color="red", linestyle=":", alpha=0.4)

        ax_f.set_xlim(0, 35)
        ax_f.set_xlabel("频率 (Hz)", fontsize=9)
        ax_f.set_ylabel("|FFT|", fontsize=9)
        ax_f.grid(alpha=0.15)

    fig.suptitle("双频编码 SSVEP 刺激波形与频谱 (Binocular AR)\n"
                 "左眼=蓝, 右眼=橙, 合成信号=黑",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  Figure 2: Confusion matrices — Experiment 2 conditions
# ═══════════════════════════════════════════════════════════

def fig_binocular_confusion(csv_path, method, window):
    """8x8 confusion matrices for 4 Experiment 2 conditions."""
    conditions = ["SFSP", "SFDP", "DFSP", "DFDP"]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    for idx, cond in enumerate(conditions):
        ax = axes[idx]
        pairs = load_predictions_bino(csv_path, cond, method, window)
        cm = confusion_counts(pairs, N_TARGETS_BINO)
        P = normalize_transition(cm)
        mi = mutual_info_uniform(P)
        acc = accuracy_from_cm(cm)

        im = ax.imshow(P, aspect="equal", cmap="Blues", vmin=0,
                       vmax=max(0.5, P.max()), interpolation="nearest")

        for i in range(N_TARGETS_BINO):
            for j in range(N_TARGETS_BINO):
                if P[i, j] > 0.005:
                    color = "white" if P[i, j] > 0.4 else "black"
                    ax.text(j, i, f"{P[i,j]:.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)

        cb = BINO_EXP2_CODEBOOK[cond]
        labels = [f"{cb['left_freq'][i]}" +
                  (f"+{cb['right_freq'][i]}" if cb['right_freq'][i] != cb['left_freq'][i] else "")
                  for i in range(8)]

        ax.set_xticks(range(8))
        ax.set_xticklabels(labels, fontsize=8, rotation=45)
        ax.set_yticks(range(8))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("预测", fontsize=10)
        if idx == 0:
            ax.set_ylabel("真实", fontsize=10)
        ax.set_title(f"{cond}\nAcc={acc:.1f}%, MI={mi:.3f}/{CAPACITY_BINO:.2f} bits",
                     fontsize=12, fontweight="bold",
                     color=COND_COLORS.get(cond, "black"))
        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(f"Binocular AR Experiment 2 — 决策信道混淆矩阵\n"
                 f"{method}, {window}s, 14 subjects, 8 targets, 信道容量上限 {CAPACITY_BINO:.2f} bits",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  Figure 3: MI vs window for all conditions
# ═══════════════════════════════════════════════════════════

def fig_mi_curves_binocular():
    """MI vs window for all 9 binocular conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    exp_configs = [
        (BINO_EXP1, ["LF", "MF"], "Exp 1: 单频基线", "task"),
        (BINO_EXP2, ["SFSP", "SFDP", "DFSP", "DFDP"], "Exp 2: 编码方式比较", "task"),
        (BINO_EXP3, ["DFDP1", "DFDP3", "DFDP5"], "Exp 3: 频率间距", "task"),
    ]

    best_method = "ETRCA"

    for ax_idx, (csv_path, conditions, title, key_task) in enumerate(exp_configs):
        ax = axes[ax_idx]

        for cond in conditions:
            windows = get_available_windows(csv_path, cond, best_method)
            if not windows:
                continue
            mis = []
            for w in windows:
                pairs = load_predictions_bino(csv_path, cond, best_method, w)
                if not pairs:
                    mis.append(0)
                    continue
                cm = confusion_counts(pairs, N_TARGETS_BINO)
                P = normalize_transition(cm)
                mis.append(mutual_info_uniform(P))

            ax.plot(windows, mis, "o-", color=COND_COLORS.get(cond, "gray"),
                    linewidth=2, markersize=4, label=cond)

        ax.axhline(CAPACITY_BINO, color="red", linestyle=":", linewidth=1.5,
                   label=f"上限 {CAPACITY_BINO:.2f} bits")
        ax.set_xlabel("窗口长度 (s)", fontsize=12)
        ax.set_ylabel("MI (bits)", fontsize=12)
        ax.set_title(f"({chr(97+ax_idx)}) {title}\n{best_method}",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_ylim(0, CAPACITY_BINO * 1.15)

    fig.suptitle("Binocular AR 决策信道容量 vs 窗口长度\n"
                 "8 targets, 10ch, ETRCA — 单频 vs 同频异相 vs 异频同相 vs 异频异相",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  Figure 4: Frequency gap vs MI (coherence bandwidth)
# ═══════════════════════════════════════════════════════════

def fig_freq_gap_analysis():
    """MI vs frequency gap for DFDP1/3/5 — coherence bandwidth analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    methods = ["CCA", "FBCCA", "TRCA", "ETRCA"]
    gaps = [1, 3, 5]
    gap_conditions = ["DFDP1", "DFDP3", "DFDP5"]

    # Panel (a): MI vs gap at fixed windows
    ax = axes[0]
    for method in methods:
        for window, ls in [(0.5, "--"), (1.0, "-"), (2.0, ":")]:
            mis = []
            for cond in gap_conditions:
                pairs = load_predictions_bino(BINO_EXP3, cond, method, window)
                if not pairs:
                    mis.append(0)
                    continue
                cm = confusion_counts(pairs, N_TARGETS_BINO)
                P = normalize_transition(cm)
                mis.append(mutual_info_uniform(P))
            if any(m > 0 for m in mis):
                label = f"{method} {window}s" if ls == "-" else None
                ax.plot(gaps, mis, f"{ls}o", color=METHOD_COLORS[method],
                        linewidth=2, markersize=8, label=label)

    ax.axhline(CAPACITY_BINO, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel("双眼频率间距 Δf (Hz)", fontsize=12)
    ax.set_ylabel("MI (bits)", fontsize=12)
    ax.set_title("(a) MI vs 频率间距\n实线=1.0s, 虚线=0.5s, 点线=2.0s",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_xticks(gaps)
    ax.set_ylim(0, CAPACITY_BINO * 1.15)

    # Panel (b): Per-subject MI scatter for DFDP1 vs DFDP5
    ax = axes[1]
    window = 1.0
    mi_1, _, subs_1 = per_subject_mi(BINO_EXP3, "DFDP1", "ETRCA", window,
                                      N_TARGETS_BINO, "bino")
    mi_5, _, subs_5 = per_subject_mi(BINO_EXP3, "DFDP5", "ETRCA", window,
                                      N_TARGETS_BINO, "bino")
    mi_3, _, subs_3 = per_subject_mi(BINO_EXP3, "DFDP3", "ETRCA", window,
                                      N_TARGETS_BINO, "bino")
    n = min(len(mi_1), len(mi_5))
    ax.scatter(mi_1[:n], mi_5[:n], c=COND_COLORS["DFDP5"], s=40, alpha=0.7,
               edgecolors="0.3", linewidth=0.3, label="DFDP1 vs DFDP5")
    lim = max(mi_1.max(), mi_5.max()) * 1.1 if n > 0 else 3
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.4)
    ax.set_xlabel("DFDP1 MI (bits) — gap=1 Hz", fontsize=11)
    ax.set_ylabel("DFDP5 MI (bits) — gap=5 Hz", fontsize=11)
    ax.set_title(f"(b) 被试级 MI: DFDP1 vs DFDP5\nETRCA, {window}s",
                 fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    pct = np.mean(mi_1[:n] > mi_5[:n]) * 100 if n > 0 else 0
    ax.text(0.05, 0.92, f"{pct:.0f}% 被试 gap=1 > gap=5",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    # Panel (c): Confusion structure difference DFDP1 vs DFDP5
    ax = axes[2]
    cm_1 = confusion_counts(
        load_predictions_bino(BINO_EXP3, "DFDP1", "ETRCA", 1.0), N_TARGETS_BINO)
    cm_5 = confusion_counts(
        load_predictions_bino(BINO_EXP3, "DFDP5", "ETRCA", 1.0), N_TARGETS_BINO)
    P_1 = normalize_transition(cm_1)
    P_5 = normalize_transition(cm_5)
    diff = P_1 - P_5

    im = ax.imshow(diff, aspect="equal", cmap="RdBu_r", vmin=-0.3, vmax=0.3,
                   interpolation="nearest")
    for i in range(N_TARGETS_BINO):
        for j in range(N_TARGETS_BINO):
            if abs(diff[i, j]) > 0.02:
                ax.text(j, i, f"{diff[i,j]:+.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if abs(diff[i, j]) > 0.15 else "black")

    cb1 = BINO_EXP3_CODEBOOK["DFDP1"]
    labels = [f"{cb1['left_freq'][i]:.0f}+{cb1['right_freq'][i]:.0f}"
              for i in range(8)]
    ax.set_xticks(range(8))
    ax.set_xticklabels(labels, fontsize=8, rotation=45)
    ax.set_yticks(range(8))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("(c) P(Y|X) 差异: DFDP1 - DFDP5\n红=gap1更强, 蓝=gap5更强",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.7)

    mi_1_all = mutual_info_uniform(P_1)
    mi_5_all = mutual_info_uniform(P_5)
    fig.suptitle(f"双频相干带宽分析 — Binocular AR Experiment 3\n"
                 f"DFDP1 (gap=1Hz) MI={mi_1_all:.3f} vs "
                 f"DFDP3 MI={mutual_info_uniform(normalize_transition(confusion_counts(load_predictions_bino(BINO_EXP3, 'DFDP3', 'ETRCA', 1.0), N_TARGETS_BINO))):.3f} vs "
                 f"DFDP5 (gap=5Hz) MI={mi_5_all:.3f}",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  Figure 5: Condition comparison bar chart + method ranking
# ═══════════════════════════════════════════════════════════

def fig_condition_comparison():
    """Bar chart comparing MI across all 9 conditions and 4 methods."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    all_conditions = ["LF", "MF", "SFSP", "SFDP", "DFSP", "DFDP",
                      "DFDP1", "DFDP3", "DFDP5"]
    csv_map = {
        "LF": BINO_EXP1, "MF": BINO_EXP1,
        "SFSP": BINO_EXP2, "SFDP": BINO_EXP2,
        "DFSP": BINO_EXP2, "DFDP": BINO_EXP2,
        "DFDP1": BINO_EXP3, "DFDP3": BINO_EXP3, "DFDP5": BINO_EXP3,
    }

    methods = ["CCA", "FBCCA", "TRCA", "ETRCA"]
    window = 1.0

    # Panel (a): Grouped bar chart
    ax = axes[0]
    x = np.arange(len(all_conditions))
    width = 0.2

    for m_idx, method in enumerate(methods):
        mis = []
        for cond in all_conditions:
            pairs = load_predictions_bino(csv_map[cond], cond, method, window)
            if not pairs:
                mis.append(0)
                continue
            cm = confusion_counts(pairs, N_TARGETS_BINO)
            P = normalize_transition(cm)
            mis.append(mutual_info_uniform(P))
        ax.bar(x + m_idx * width - 1.5 * width, mis, width,
               color=METHOD_COLORS[method], alpha=0.8, label=method)

    ax.axhline(CAPACITY_BINO, color="red", linestyle=":", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(all_conditions, fontsize=10, rotation=30)
    ax.set_ylabel("MI (bits)", fontsize=12)
    ax.set_title(f"(a) 各条件 × 方法 MI 对比 ({window}s)\n"
                 f"信道容量上限 = {CAPACITY_BINO:.2f} bits",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.15, axis="y")
    ax.set_ylim(0, CAPACITY_BINO * 1.15)

    # Vertical separators between experiments
    ax.axvline(1.5, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(5.5, color="gray", linestyle="--", alpha=0.3)
    ax.text(0.5, CAPACITY_BINO * 1.08, "Exp1", ha="center", fontsize=9, color="gray")
    ax.text(3.5, CAPACITY_BINO * 1.08, "Exp2", ha="center", fontsize=9, color="gray")
    ax.text(7, CAPACITY_BINO * 1.08, "Exp3", ha="center", fontsize=9, color="gray")

    # Panel (b): Summary text
    ax = axes[1]
    ax.axis("off")

    lines = ["条件编码方式总结与信道度量\n" + "=" * 50 + "\n"]
    for cond in all_conditions:
        pairs = load_predictions_bino(csv_map[cond], cond, "ETRCA", window)
        cm = confusion_counts(pairs, N_TARGETS_BINO)
        P = normalize_transition(cm)
        mi = mutual_info_uniform(P)
        acc = accuracy_from_cm(cm)

        if cond in BINO_EXP2_CODEBOOK:
            cb = BINO_EXP2_CODEBOOK[cond]
            f_desc = (f"L={cb['left_freq'][0]}-{cb['left_freq'][-1]}, "
                      f"R={cb['right_freq'][0]}-{cb['right_freq'][-1]}")
        elif cond in BINO_EXP3_CODEBOOK:
            cb = BINO_EXP3_CODEBOOK[cond]
            f_desc = (f"L={cb['left_freq'][0]}-{cb['left_freq'][-1]}, "
                      f"R={cb['right_freq'][0]:.0f}-{cb['right_freq'][-1]:.0f}")
        elif cond == "LF":
            f_desc = "8-15 Hz, 双眼同频同相"
        elif cond == "MF":
            f_desc = "23-30 Hz, 双眼同频同相"
        else:
            f_desc = ""

        lines.append(f"{cond:6s}  MI={mi:.3f} ({mi/CAPACITY_BINO*100:.0f}%)  "
                     f"Acc={acc:.1f}%  {f_desc}")

    lines.append("\n" + "=" * 50)
    lines.append("\n编码维度分析:")
    lines.append("• 频率分离 (SFSP→DFSP): +Δf 提供频率维度信息")
    lines.append("• 相位分离 (SFSP→SFDP): +Δφ 提供相位维度信息")
    lines.append("• 联合分离 (SFSP→DFDP): 频率+相位联合编码")
    lines.append("• 间距效应 (DFDP1→3→5): 双频间距影响信道质量")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, fontfamily="Microsoft YaHei", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.suptitle("Binocular AR 全条件决策信道分析汇总\n"
                 "9 条件 × 4 方法, 1.0s 窗口, ETRCA 为最佳方法",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  Figure 6: JFPM vs NBRS channel comparison
# ═══════════════════════════════════════════════════════════

def fig_jfpm_channel():
    """Decision channel analysis for JFPM-8 vs NBRS paradigms."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    paradigms = ["JFPM-8", "NBRS-15", "NBRS-8"]

    # Panel (a): MI vs window
    ax = axes[0]
    for paradigm in paradigms:
        for method, csv_path in [("FBCCA-CODE", JFPM_FULL), ("TRCA", JFPM_FULL),
                                  ("MSTRCA", JFPM_MSTRCA)]:
            windows = get_available_windows(csv_path, paradigm, method, "paradigm")
            if not windows:
                continue
            mis = []
            for w in windows:
                pairs = load_predictions_jfpm(csv_path, paradigm, method, w)
                cm = confusion_counts(pairs, N_TARGETS_JFPM)
                P = normalize_transition(cm)
                mis.append(mutual_info_uniform(P))

            ls = "-" if method == "TRCA" else ("--" if method == "FBCCA-CODE" else ":")
            ax.plot(windows, mis, f"{ls}o",
                    color=COND_COLORS.get(paradigm, "gray"),
                    linewidth=2, markersize=4,
                    label=f"{paradigm} {method}")

    ax.axhline(CAPACITY_JFPM, color="red", linestyle=":", linewidth=1.5,
               label=f"上限 {CAPACITY_JFPM:.2f} bits")
    ax.set_xlabel("窗口长度 (s)", fontsize=12)
    ax.set_ylabel("MI (bits)", fontsize=12)
    ax.set_title("(a) MI vs 窗口 — 三范式比较\n100 subjects, 40 targets",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    ax.set_ylim(0, CAPACITY_JFPM * 1.1)

    # Panel (b): Confusion matrix for JFPM-8 TRCA at best window
    ax = axes[1]
    best_window = 4.0
    pairs = load_predictions_jfpm(JFPM_FULL, "JFPM-8", "TRCA", best_window)
    if pairs:
        cm = confusion_counts(pairs, N_TARGETS_JFPM)
        P = normalize_transition(cm)
        mi = mutual_info_uniform(P)
        acc = accuracy_from_cm(cm)

        im = ax.imshow(P, aspect="equal", cmap="Blues", vmin=0,
                       vmax=max(0.3, P.max()), interpolation="nearest")
        ax.set_title(f"(b) JFPM-8 混淆矩阵\nTRCA {best_window}s, Acc={acc:.1f}%, "
                     f"MI={mi:.3f}/{CAPACITY_JFPM:.2f}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("预测目标", fontsize=11)
        ax.set_ylabel("真实目标", fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.7)
    else:
        ax.text(0.5, 0.5, "无数据", transform=ax.transAxes, ha="center")

    # Panel (c): Per-subject MI comparison JFPM vs NBRS
    ax = axes[2]
    window = 4.0
    mi_jfpm, _, _ = per_subject_mi(JFPM_FULL, "JFPM-8", "TRCA", window,
                                    N_TARGETS_JFPM, "jfpm")
    mi_nbrs15, _, _ = per_subject_mi(JFPM_FULL, "NBRS-15", "TRCA", window,
                                      N_TARGETS_JFPM, "jfpm")
    mi_nbrs8, _, _ = per_subject_mi(JFPM_FULL, "NBRS-8", "TRCA", window,
                                     N_TARGETS_JFPM, "jfpm")

    n = min(len(mi_jfpm), len(mi_nbrs15), len(mi_nbrs8))
    if n > 0:
        ax.scatter(mi_jfpm[:n], mi_nbrs15[:n], c=COND_COLORS["NBRS-15"],
                   s=30, alpha=0.6, label="JFPM vs NBRS-15")
        ax.scatter(mi_jfpm[:n], mi_nbrs8[:n], c=COND_COLORS["NBRS-8"],
                   s=30, alpha=0.6, marker="s", label="JFPM vs NBRS-8")
        lim = max(mi_jfpm.max(), mi_nbrs15.max(), mi_nbrs8.max()) * 1.1
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.4)
        ax.set_xlabel("JFPM-8 MI (bits)", fontsize=11)
        ax.set_ylabel("NBRS MI (bits)", fontsize=11)
        ax.set_title(f"(c) 被试级 MI: JFPM vs NBRS\nTRCA {window}s, N={n}",
                     fontsize=13, fontweight="bold")
        ax.set_aspect("equal")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2)

    fig.suptitle("JFPM 联合频率-相位编码 vs NBRS 码序列 — 决策信道对比\n"
                 "Zheng 2024, 100 subjects, 40 targets, 9ch occipital",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Fig 1: Stimulus waveforms ──
    print("Fig 1: Stimulus waveforms...")
    fig = fig_stimulus_waveforms()
    fig.savefig(FIG_DIR / "fig_dual_freq_stimulus.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> fig_dual_freq_stimulus.png")

    # ── Fig 2: Confusion matrices (Exp2) ──
    print("Fig 2: Confusion matrices (Exp2)...")
    fig = fig_binocular_confusion(BINO_EXP2, "ETRCA", 1.0)
    fig.savefig(FIG_DIR / "fig_dual_freq_confusion_exp2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> fig_dual_freq_confusion_exp2.png")

    # ── Fig 3: MI curves all conditions ──
    print("Fig 3: MI curves all conditions...")
    fig = fig_mi_curves_binocular()
    fig.savefig(FIG_DIR / "fig_dual_freq_mi_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> fig_dual_freq_mi_curves.png")

    # ── Fig 4: Frequency gap analysis ──
    print("Fig 4: Frequency gap analysis...")
    fig = fig_freq_gap_analysis()
    fig.savefig(FIG_DIR / "fig_dual_freq_gap_analysis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> fig_dual_freq_gap_analysis.png")

    # ── Fig 5: Condition comparison ──
    print("Fig 5: Condition comparison...")
    fig = fig_condition_comparison()
    fig.savefig(FIG_DIR / "fig_dual_freq_condition_comparison.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print("-> fig_dual_freq_condition_comparison.png")

    # ── Fig 6: JFPM channel ──
    print("Fig 6: JFPM channel...")
    fig = fig_jfpm_channel()
    fig.savefig(FIG_DIR / "fig_jfpm_channel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("-> fig_jfpm_channel.png")

    # ── Summary table ──
    print(f"\n{'='*80}")
    print("双频决策信道分析汇总 (ETRCA, 1.0s)")
    print(f"{'='*80}")
    print(f"{'条件':<8} {'类型':<16} {'Acc%':>6} {'MI':>7} {'利用率':>6} {'目标数':>5}")

    all_conds = [
        ("LF", BINO_EXP1, "单频基线"),
        ("MF", BINO_EXP1, "中频基线"),
        ("SFSP", BINO_EXP2, "同频同相"),
        ("SFDP", BINO_EXP2, "同频异相"),
        ("DFSP", BINO_EXP2, "异频同相"),
        ("DFDP", BINO_EXP2, "异频异相"),
        ("DFDP1", BINO_EXP3, "异频gap=1Hz"),
        ("DFDP3", BINO_EXP3, "异频gap=3Hz"),
        ("DFDP5", BINO_EXP3, "异频gap=5Hz"),
    ]

    for cond, csv_path, desc in all_conds:
        pairs = load_predictions_bino(csv_path, cond, "ETRCA", 1.0)
        cm = confusion_counts(pairs, N_TARGETS_BINO)
        P = normalize_transition(cm)
        mi = mutual_info_uniform(P)
        acc = accuracy_from_cm(cm)
        print(f"{cond:<8} {desc:<16} {acc:>5.1f}% {mi:>6.3f} {mi/CAPACITY_BINO*100:>5.1f}% {8:>5}")

    print()

    # JFPM summary
    print(f"{'='*80}")
    print("JFPM / NBRS 决策信道分析汇总 (TRCA, 4.0s)")
    print(f"{'='*80}")
    for paradigm in ["JFPM-8", "NBRS-15", "NBRS-8"]:
        pairs = load_predictions_jfpm(JFPM_FULL, paradigm, "TRCA", 4.0)
        cm = confusion_counts(pairs, N_TARGETS_JFPM)
        P = normalize_transition(cm)
        mi = mutual_info_uniform(P)
        acc = accuracy_from_cm(cm)
        print(f"{paradigm:<10} Acc={acc:.1f}%  MI={mi:.3f}/{CAPACITY_JFPM:.2f}  "
              f"利用率={mi/CAPACITY_JFPM*100:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
