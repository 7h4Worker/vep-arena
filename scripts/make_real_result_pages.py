from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


BASE = Path("D:/ProjData/proj_python/vep_arena/results/benchmark_9ch")
SUMMARY = BASE / "final_compare" / "summary.csv"
SUBJECT = BASE / "final_compare" / "subject.csv"
OUT = BASE / "figures"

METHOD_ORDER = ["CCA", "FBCCA", "FBTRCA", "TDCA", "DNN", "TRCA-Net", "SSVEPFormer"]
COLORS = {
    "CCA": "#6b7280",
    "FBCCA": "#0891b2",
    "FBTRCA": "#2f6f5f",
    "TDCA": "#c46a1a",
    "DNN": "#315aa3",
    "TRCA-Net": "#bf3f55",
    "SSVEPFormer": "#7b59b6",
}


def choose_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]:
        if name in available:
            return name
    return "DejaVu Sans"


def setup_style() -> None:
    font = choose_font()
    plt.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "figure.facecolor": "#f7f8fb",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d0d5dd",
            "axes.labelcolor": "#101828",
            "xtick.color": "#344054",
            "ytick.color": "#344054",
            "text.color": "#101828",
            "savefig.facecolor": "#f7f8fb",
        }
    )


def style_ax(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#e4e7ec", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d0d5dd")
    ax.spines["bottom"].set_color("#d0d5dd")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY)
    subject = pd.read_csv(SUBJECT)
    summary["window"] = summary["window"].astype(float).round(1)
    subject["window"] = subject["window"].astype(float).round(1)
    summary = summary[summary["method"].isin(METHOD_ORDER)].copy()
    subject = subject[subject["method"].isin(METHOD_ORDER)].copy()
    return summary, subject


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.035, 0.965, title, fontsize=26, fontweight="bold", ha="left", va="top")
    fig.text(0.035, 0.925, subtitle, fontsize=13.5, ha="left", va="top", color="#475467")


def plot_curves_page(summary: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(16, 9), dpi=240)
    add_header(
        fig,
        "Benchmark 9ch 实验结果总览：窗口长度曲线",
        "真实结果来自 final_compare/summary.csv；误差线为 subject-level SEM；协议为 35 subjects × 6 blocks leave-one-block-out。",
    )
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.98, top=0.865, bottom=0.075, hspace=0.34, wspace=0.18)
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_itr = fig.add_subplot(gs[0, 1])
    ax_delta = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])

    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("window")
        color = COLORS[method]
        ax_acc.errorbar(
            sub["window"],
            sub["accuracy"],
            yerr=sub["subject_sem"],
            marker="o",
            markersize=4.2,
            linewidth=2.2,
            capsize=3,
            color=color,
            label=method,
        )
        ax_itr.errorbar(
            sub["window"],
            sub["itr"],
            yerr=sub["itr_sem"],
            marker="o",
            markersize=4.2,
            linewidth=2.2,
            capsize=3,
            color=color,
            label=method,
        )

    ax_acc.set_title("A. Accuracy 随时间窗变化", loc="left", fontsize=15, fontweight="bold")
    ax_acc.set_xlabel("Signal window (s)")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.set_xticks(sorted(summary["window"].unique()))
    style_ax(ax_acc)
    ax_acc.legend(frameon=False, ncol=4, fontsize=9, loc="lower right")

    ax_itr.set_title("B. ITR 随时间窗变化", loc="left", fontsize=15, fontweight="bold")
    ax_itr.set_xlabel("Signal window (s)")
    ax_itr.set_ylabel("ITR (bits/min)")
    ax_itr.set_xticks(sorted(summary["window"].unique()))
    style_ax(ax_itr)

    wide = summary.pivot(index="window", columns="method", values="accuracy")
    for method in METHOD_ORDER:
        if method == "DNN":
            continue
        ax_delta.plot(
            wide.index,
            wide[method] - wide["DNN"],
            marker="o",
            linewidth=2.0,
            color=COLORS[method],
            label=method,
        )
    ax_delta.axhline(0, color="#101828", linewidth=1.2)
    ax_delta.set_title("C. Accuracy 相对 DNN 的差值", loc="left", fontsize=15, fontweight="bold")
    ax_delta.set_xlabel("Signal window (s)")
    ax_delta.set_ylabel("Delta accuracy")
    ax_delta.set_xticks(sorted(summary["window"].unique()))
    style_ax(ax_delta)
    ax_delta.legend(frameon=False, ncol=3, fontsize=9, loc="lower left")

    ax_table.axis("off")
    peak = (
        summary.loc[summary.groupby("method")["itr"].idxmax(), ["method", "window", "itr"]]
        .set_index("method")
        .loc[METHOD_ORDER]
        .reset_index()
    )
    mean_rank = summary.groupby("method", as_index=False)["accuracy"].mean().sort_values("accuracy", ascending=False)
    table_rows = []
    for _, row in mean_rank.iterrows():
        p = peak[peak["method"] == row["method"]].iloc[0]
        table_rows.append(
            [
                row["method"],
                f"{row['accuracy']:.3f}",
                f"{p['window']:.1f}s",
                f"{p['itr']:.1f}",
            ]
        )
    tbl = ax_table.table(
        cellText=table_rows,
        colLabels=["Method", "Mean Acc", "Peak ITR win", "Peak ITR"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.06, 0.96, 0.82],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d0d5dd")
        if r == 0:
            cell.set_facecolor("#e6f4f1")
            cell.set_text_props(fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#f2f4f7")
            cell.set_text_props(fontweight="bold")
    ax_table.set_title("D. 方法整体排序与 Peak ITR", loc="left", fontsize=15, fontweight="bold")

    out = OUT / "real_results_page1_curves.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_key_windows_page(summary: pd.DataFrame, subject: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(16, 9), dpi=240)
    add_header(
        fig,
        "Benchmark 9ch 实验结果总览：关键窗口与受试者分布",
        "重点看 0.5s 和 1.0s：既能对照 SSVEP 文献常见窗口，也能观察短窗 ITR 与长窗 accuracy 的取舍。",
    )
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.98, top=0.865, bottom=0.075, hspace=0.34, wspace=0.18)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_box = fig.add_subplot(gs[1, 0])
    ax_itrbar = fig.add_subplot(gs[1, 1])

    windows = [0.5, 1.0]
    x = np.arange(len(METHOD_ORDER))
    width = 0.36
    rows = summary[summary["window"].isin(windows)].copy()
    acc_piv = rows.pivot(index="method", columns="window", values="accuracy").loc[METHOD_ORDER]
    acc_sem = rows.pivot(index="method", columns="window", values="subject_sem").loc[METHOD_ORDER]
    ax_bar.bar(x - width / 2, acc_piv[0.5], width, yerr=acc_sem[0.5], capsize=3, color="#80b5d8", label="0.5s")
    ax_bar.bar(x + width / 2, acc_piv[1.0], width, yerr=acc_sem[1.0], capsize=3, color="#315aa3", label="1.0s")
    ax_bar.set_title("A. Accuracy：0.5s vs 1.0s", loc="left", fontsize=15, fontweight="bold")
    ax_bar.set_xticks(x, METHOD_ORDER, rotation=22, ha="right")
    ax_bar.set_ylabel("Accuracy")
    ax_bar.set_ylim(0.0, 1.02)
    ax_bar.legend(frameon=False)
    style_ax(ax_bar)

    heat = summary.pivot(index="method", columns="window", values="accuracy").loc[METHOD_ORDER]
    im = ax_heat.imshow(heat.values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax_heat.set_title("B. Accuracy heatmap（全部窗口）", loc="left", fontsize=15, fontweight="bold")
    ax_heat.set_yticks(np.arange(len(METHOD_ORDER)), METHOD_ORDER)
    ax_heat.set_xticks(np.arange(len(heat.columns)), [f"{w:.1f}" for w in heat.columns])
    ax_heat.set_xlabel("Signal window (s)")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.8, color="#101828")
    cb = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03)
    cb.set_label("Accuracy")

    data = [
        subject[(subject["method"] == method) & (subject["window"] == 1.0)]["accuracy"].to_numpy()
        for method in METHOD_ORDER
    ]
    bp = ax_box.boxplot(data, tick_labels=METHOD_ORDER, patch_artist=True, showfliers=False)
    for patch, method in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.68)
        patch.set_edgecolor("#344054")
    ax_box.set_title("C. 1.0s subject-level accuracy 分布", loc="left", fontsize=15, fontweight="bold")
    ax_box.set_xticks(x + 1, METHOD_ORDER, rotation=22, ha="right")
    ax_box.set_ylabel("Subject mean accuracy")
    ax_box.set_ylim(0.0, 1.02)
    style_ax(ax_box)

    itr_piv = rows.pivot(index="method", columns="window", values="itr").loc[METHOD_ORDER]
    itr_sem = rows.pivot(index="method", columns="window", values="itr_sem").loc[METHOD_ORDER]
    ax_itrbar.bar(x - width / 2, itr_piv[0.5], width, yerr=itr_sem[0.5], capsize=3, color="#f2b47e", label="0.5s")
    ax_itrbar.bar(x + width / 2, itr_piv[1.0], width, yerr=itr_sem[1.0], capsize=3, color="#c46a1a", label="1.0s")
    ax_itrbar.set_title("D. ITR：0.5s vs 1.0s", loc="left", fontsize=15, fontweight="bold")
    ax_itrbar.set_xticks(x, METHOD_ORDER, rotation=22, ha="right")
    ax_itrbar.set_ylabel("ITR (bits/min)")
    ax_itrbar.legend(frameon=False)
    style_ax(ax_itrbar)

    out = OUT / "real_results_page2_key_windows.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    setup_style()
    summary, subject = load_data()
    p1 = plot_curves_page(summary)
    p2 = plot_key_windows_page(summary, subject)
    print(p1)
    print(p2)


if __name__ == "__main__":
    main()
