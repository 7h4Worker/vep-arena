# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import html
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path("D:/ProjData/proj_python/vep_arena")
SOURCE = PROJECT / "results" / "benchmark_9ch" / "final_compare_plus_sa_mvmd" / "summary.csv"
OUT = Path("D:/ProjData/SSVEP方法对比_总览_20260526")
FIG_DIR = OUT / "图表"
TABLE_DIR = OUT / "数据表"

SHARED_WINDOWS = [0.4, 0.6, 0.8, 1.0]
METHOD_ORDER = [
    "DNN",
    "TRCA-Net",
    "TDCA",
    "SA-MVMD-eTRCA",
    "FBTRCA",
    "SA-MVMD-TRCA",
    "SSVEPFormer",
    "FBCCA",
    "CCA",
]
MAIN_METHODS = [
    "DNN",
    "TRCA-Net",
    "TDCA",
    "SA-MVMD-eTRCA",
    "FBTRCA",
    "SA-MVMD-TRCA",
    "SSVEPFormer",
]
REFERENCE_METHODS = ["FBCCA", "CCA"]
COLORS = {
    "DNN": "#2f5fb3",
    "TRCA-Net": "#b23a48",
    "TDCA": "#b8681d",
    "SA-MVMD-eTRCA": "#139f7f",
    "FBTRCA": "#2f6f5f",
    "SA-MVMD-TRCA": "#d97941",
    "SSVEPFormer": "#7957b8",
    "FBCCA": "#6f8792",
    "CCA": "#9ca3af",
}


def setup_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#303030"
    plt.rcParams["axes.labelcolor"] = "#202020"
    plt.rcParams["xtick.color"] = "#303030"
    plt.rcParams["ytick.color"] = "#303030"


def apply_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#d9dee7", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    df["window"] = df["window"].astype(float).round(1)
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    return df


def shared_summary(df: pd.DataFrame) -> pd.DataFrame:
    shared = df[df["window"].isin(SHARED_WINDOWS)].copy()
    rows = (
        shared.groupby("method", as_index=False)
        .agg(
            mean_acc=("accuracy", "mean"),
            mean_itr=("itr", "mean"),
            min_acc=("accuracy", "min"),
            max_acc=("accuracy", "max"),
        )
        .sort_values("mean_acc", ascending=False)
    )
    rows["method"] = pd.Categorical(rows["method"], METHOD_ORDER, ordered=True)
    return rows.sort_values("mean_acc", ascending=False).reset_index(drop=True)


def window_table(df: pd.DataFrame, window: float) -> pd.DataFrame:
    rows = df[df["window"] == window].copy()
    rows["method"] = pd.Categorical(rows["method"], METHOD_ORDER, ordered=True)
    rows = rows.sort_values("accuracy", ascending=False)
    return rows[["method", "accuracy", "accuracy_sem", "itr", "itr_sem", "subjects"]].reset_index(drop=True)


def peak_table(df: pd.DataFrame) -> pd.DataFrame:
    itr_peak = df.loc[df.groupby("method")["itr"].idxmax(), ["method", "window", "accuracy", "itr"]].rename(
        columns={"window": "itr_peak_window", "accuracy": "acc_at_itr_peak", "itr": "peak_itr"}
    )
    acc_peak = df.loc[df.groupby("method")["accuracy"].idxmax(), ["method", "window", "accuracy", "itr"]].rename(
        columns={"window": "acc_peak_window", "accuracy": "peak_acc", "itr": "itr_at_acc_peak"}
    )
    rows = itr_peak.merge(acc_peak, on="method", how="outer")
    rows["method"] = pd.Categorical(rows["method"], METHOD_ORDER, ordered=True)
    return rows.sort_values("method").reset_index(drop=True)


def plot_shared_curve(df: pd.DataFrame, metric: str, err: str, ylabel: str, title: str, filename: str) -> None:
    shared = df[df["window"].isin(SHARED_WINDOWS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [3.2, 1.1]})
    for method in MAIN_METHODS:
        rows = shared[shared["method"] == method].sort_values("window")
        axes[0].errorbar(
            rows["window"],
            rows[metric],
            yerr=rows[err],
            marker="o",
            linewidth=2.1,
            capsize=3,
            color=COLORS[method],
            label=method,
        )
    axes[0].set_title(title + "：主要方法")
    axes[0].set_xlabel("时间窗（秒）")
    axes[0].set_ylabel(ylabel)
    axes[0].set_xticks(SHARED_WINDOWS)
    if metric == "accuracy":
        axes[0].set_ylim(0.45, 1.0)
    apply_axis(axes[0])
    axes[0].legend(frameon=False, ncol=2, fontsize=9)

    for method in REFERENCE_METHODS:
        rows = shared[shared["method"] == method].sort_values("window")
        axes[1].errorbar(
            rows["window"],
            rows[metric],
            yerr=rows[err],
            marker="o",
            linewidth=2.1,
            capsize=3,
            color=COLORS[method],
            label=method,
        )
    axes[1].set_title("参考基线")
    axes[1].set_xlabel("时间窗（秒）")
    axes[1].set_xticks(SHARED_WINDOWS)
    apply_axis(axes[1])
    axes[1].legend(frameon=False, fontsize=9)
    save(fig, filename)


def plot_window_bars(rows: pd.DataFrame, window: float, filename: str) -> None:
    rows = rows.sort_values("accuracy", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    y = np.arange(len(rows))
    methods = rows["method"].astype(str).to_numpy()
    colors = [COLORS[m] for m in methods]

    axes[0].barh(y, rows["accuracy"], xerr=rows["accuracy_sem"], color=colors, alpha=0.92)
    axes[0].set_yticks(y, methods)
    axes[0].set_xlabel("ACC")
    axes[0].set_xlim(0, 1.0)
    axes[0].set_title(f"{window:.1f}s 准确率")
    apply_axis(axes[0])
    for yy, value in zip(y, rows["accuracy"]):
        axes[0].text(min(value + 0.015, 0.98), yy, fmt_pct(value), va="center", fontsize=9)

    axes[1].barh(y, rows["itr"], xerr=rows["itr_sem"], color=colors, alpha=0.92)
    axes[1].set_xlabel("ITR（bits/min）")
    axes[1].set_title(f"{window:.1f}s 信息传输率")
    apply_axis(axes[1])
    max_itr = max(rows["itr"].max(), 1)
    axes[1].set_xlim(0, max_itr * 1.18)
    for yy, value in zip(y, rows["itr"]):
        axes[1].text(value + max_itr * 0.025, yy, f"{value:.1f}", va="center", fontsize=9)

    fig.suptitle(f"{window:.1f}s 直接横向对比", fontsize=15, fontweight="bold", y=1.02)
    save(fig, filename)


def plot_ranking(rows: pd.DataFrame) -> None:
    rows = rows.sort_values("mean_acc", ascending=True)
    fig, ax = plt.subplots(figsize=(9.8, 5.5))
    y = np.arange(len(rows))
    methods = rows["method"].astype(str).to_numpy()
    colors = [COLORS[m] for m in methods]
    ax.barh(y, rows["mean_acc"], color=colors, alpha=0.92)
    ax.set_yticks(y, methods)
    ax.set_xlabel("0.4-1.0s 平均 ACC")
    ax.set_xlim(0, 1.0)
    ax.set_title("共享窗口平均准确率排行")
    apply_axis(ax)
    for yy, value in zip(y, rows["mean_acc"]):
        ax.text(min(value + 0.015, 0.98), yy, fmt_pct(value), va="center", fontsize=10)
    save(fig, "01_共享窗口平均ACC排行.png")


def plot_peak(rows: pd.DataFrame) -> None:
    itr_rows = rows.sort_values("peak_itr", ascending=True)
    acc_rows = rows.sort_values("peak_acc", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    for ax, data, value_col, window_col, title, xlabel in [
        (axes[0], itr_rows, "peak_itr", "itr_peak_window", "ITR 峰值", "ITR（bits/min）"),
        (axes[1], acc_rows, "peak_acc", "acc_peak_window", "ACC 峰值", "ACC"),
    ]:
        y = np.arange(len(data))
        methods = data["method"].astype(str).to_numpy()
        colors = [COLORS[m] for m in methods]
        ax.barh(y, data[value_col], color=colors, alpha=0.92)
        ax.set_yticks(y, methods)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        apply_axis(ax)
        if value_col == "peak_acc":
            ax.set_xlim(0, 1.0)
        max_value = max(data[value_col].max(), 1)
        for yy, value, win in zip(y, data[value_col], data[window_col]):
            label = f"{value:.1f} @ {win:.1f}s" if value_col == "peak_itr" else f"{fmt_pct(value)} @ {win:.1f}s"
            ax.text(value + max_value * 0.015, yy, label, va="center", fontsize=9)
    save(fig, "07_峰值窗口对比.png")


def plot_sa_mvmd_long(df: pd.DataFrame) -> None:
    rows = df[df["method"].isin(["SA-MVMD-TRCA", "SA-MVMD-eTRCA"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))
    for method in ["SA-MVMD-TRCA", "SA-MVMD-eTRCA"]:
        sub = rows[rows["method"] == method].sort_values("window")
        axes[0].errorbar(sub["window"], sub["accuracy"], yerr=sub["accuracy_sem"], marker="o", linewidth=2.2, capsize=3, color=COLORS[method], label=method)
        axes[1].errorbar(sub["window"], sub["itr"], yerr=sub["itr_sem"], marker="o", linewidth=2.2, capsize=3, color=COLORS[method], label=method)
    axes[0].set_title("SA-MVMD 准确率：长窗继续上升")
    axes[0].set_xlabel("时间窗（秒）")
    axes[0].set_ylabel("ACC")
    axes[0].set_ylim(0.5, 1.0)
    axes[1].set_title("SA-MVMD ITR：短窗更占优")
    axes[1].set_xlabel("时间窗（秒）")
    axes[1].set_ylabel("ITR（bits/min）")
    for ax in axes:
        ax.set_xticks(sorted(rows["window"].unique()))
        ax.legend(frameon=False)
        apply_axis(ax)
    save(fig, "06_SA-MVMD长窗走势.png")


def to_cn_table(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "ranking":
        out = df[["method", "mean_acc", "mean_itr", "min_acc", "max_acc"]].copy()
        out.columns = ["方法", "0.4-1.0s平均ACC", "0.4-1.0s平均ITR", "最低ACC", "最高ACC"]
        return out
    if kind == "window":
        out = df[["method", "accuracy", "accuracy_sem", "itr", "itr_sem", "subjects"]].copy()
        out.columns = ["方法", "ACC", "ACC_SEM", "ITR", "ITR_SEM", "受试者数"]
        return out
    if kind == "peak":
        out = df[["method", "itr_peak_window", "peak_itr", "acc_peak_window", "peak_acc"]].copy()
        out.columns = ["方法", "ITR峰值窗口", "ITR峰值", "ACC峰值窗口", "ACC峰值"]
        return out
    return df


def html_table(df: pd.DataFrame, percent_cols: set[str] | None = None) -> str:
    percent_cols = percent_cols or set()
    lines = ["<table>", "<thead><tr>"]
    for col in df.columns:
        lines.append(f"<th>{html.escape(str(col))}</th>")
    lines.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        lines.append("<tr>")
        for col, value in row.items():
            if isinstance(value, (float, np.floating)):
                text = fmt_pct(float(value)) if col in percent_cols else f"{float(value):.3f}"
            else:
                text = str(value)
            lines.append(f"<td>{html.escape(text)}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def write_html(
    ranking: pd.DataFrame,
    w04: pd.DataFrame,
    w10: pd.DataFrame,
    peaks: pd.DataFrame,
) -> None:
    top04 = w04.iloc[0]
    top10 = w10.iloc[0]
    etrca_10 = w10[w10["method"].astype(str) == "SA-MVMD-eTRCA"].iloc[0]
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>SSVEP 方法对比总览</title>
  <style>
    body {{ margin: 0; background: #f5f6f8; color: #1f2933; font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif; }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 34px 28px 60px; }}
    h1 {{ font-size: 30px; margin: 0 0 10px; }}
    h2 {{ margin: 34px 0 14px; font-size: 22px; }}
    p {{ line-height: 1.75; }}
    .sub {{ color: #5b6472; margin-bottom: 22px; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 20px 0 28px; }}
    .card {{ background: #fff; border: 1px solid #dfe3ea; border-radius: 8px; padding: 18px; }}
    .card b {{ display: block; font-size: 16px; margin-bottom: 8px; }}
    .num {{ font-size: 24px; font-weight: 700; color: #123b7a; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    figure {{ background: #fff; border: 1px solid #dfe3ea; border-radius: 8px; margin: 0; padding: 14px; }}
    figure img {{ width: 100%; display: block; border-radius: 4px; }}
    figcaption {{ color: #5b6472; font-size: 14px; margin-top: 10px; line-height: 1.6; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 14px; }}
    th, td {{ border: 1px solid #dde3ea; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f6; }}
    .note {{ background: #fff; border-left: 4px solid #139f7f; padding: 14px 18px; line-height: 1.75; }}
    .path {{ font-family: Consolas, monospace; background: #eef2f6; padding: 2px 5px; border-radius: 4px; }}
    @media (max-width: 900px) {{ .cards {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<div class="page">
  <h1>SSVEP Benchmark 9ch 方法对比总览</h1>
  <p class="sub">整理时间：2026-05-26。这个页面是主入口，优先看这里，不需要再翻一堆 Markdown。</p>

  <div class="cards">
    <div class="card">
      <b>0.4s 直接对比第一</b>
      <div class="num">{html.escape(str(top04.method))}</div>
      <p>ACC {fmt_pct(float(top04.accuracy))}，ITR {float(top04.itr):.1f} bits/min。</p>
    </div>
    <div class="card">
      <b>1.0s 直接对比第一</b>
      <div class="num">{html.escape(str(top10.method))}</div>
      <p>ACC {fmt_pct(float(top10.accuracy))}，和 SA-MVMD-eTRCA 的 {fmt_pct(float(etrca_10.accuracy))} 几乎贴住。</p>
    </div>
    <div class="card">
      <b>SA-MVMD 的特点</b>
      <div class="num">长窗稳，计算重</div>
      <p>eTRCA 版在 1.0s 已经到 {fmt_pct(float(etrca_10.accuracy))}，但峰值 ITR 仍偏短窗。</p>
    </div>
  </div>

  <h2>先看这几张图</h2>
  <div class="grid">
    <figure>
      <img src="图表/01_共享窗口平均ACC排行.png" alt="共享窗口平均ACC排行">
      <figcaption>0.4-1.0s 的公平排行。这里比的是所有方法共有的窗口，不把 SA-MVMD 的长窗优势混进去。</figcaption>
    </figure>
    <figure>
      <img src="图表/02_ACC曲线_共享窗口.png" alt="共享窗口ACC曲线">
      <figcaption>共享窗口 ACC 曲线。左边是主要方法，右边单独放 CCA/FBCCA，避免低基线把主图压扁。</figcaption>
    </figure>
    <figure>
      <img src="图表/03_ITR曲线_共享窗口.png" alt="共享窗口ITR曲线">
      <figcaption>共享窗口 ITR 曲线。可以看到 ITR 不等于 ACC，短窗方法会因为时间更短而有优势。</figcaption>
    </figure>
    <figure>
      <img src="图表/04_0.4秒直接对比.png" alt="0.4秒直接对比">
      <figcaption>0.4s 横向对比。这个图回答“短窗直接比谁强”。</figcaption>
    </figure>
    <figure>
      <img src="图表/05_1.0秒直接对比.png" alt="1.0秒直接对比">
      <figcaption>1.0s 横向对比。TDCA、SA-MVMD-eTRCA、DNN 三者非常接近。</figcaption>
    </figure>
    <figure>
      <img src="图表/06_SA-MVMD长窗走势.png" alt="SA-MVMD长窗走势">
      <figcaption>SA-MVMD 单独看：ACC 随窗口变长继续上升，ITR 峰值则在更短窗口。</figcaption>
    </figure>
    <figure>
      <img src="图表/07_峰值窗口对比.png" alt="峰值窗口对比">
      <figcaption>每个方法自己的 ACC 峰值和 ITR 峰值。注意：ACC 峰值和 ITR 峰值通常不是同一个窗口。</figcaption>
    </figure>
  </div>

  <h2>核心表格</h2>
  <h3>共享窗口平均 ACC 排行</h3>
  {html_table(to_cn_table(ranking, "ranking"), {"0.4-1.0s平均ACC", "最低ACC", "最高ACC"})}

  <h3>0.4s 直接对比</h3>
  {html_table(to_cn_table(w04, "window"), {"ACC", "ACC_SEM"})}

  <h3>1.0s 直接对比</h3>
  {html_table(to_cn_table(w10, "window"), {"ACC", "ACC_SEM"})}

  <h3>峰值窗口</h3>
  {html_table(to_cn_table(peaks, "peak"), {"ACC峰值"})}

  <h2>我把文件收在哪里了</h2>
  <div class="note">
    <p>主入口：<span class="path">{OUT / "index.html"}</span></p>
    <p>图都在：<span class="path">{FIG_DIR}</span></p>
    <p>表都在：<span class="path">{TABLE_DIR}</span></p>
    <p>你之后只需要打开这个 HTML。CSV 只是给后续继续分析或者写文章用。</p>
  </div>
</div>
</body>
</html>
"""
    (OUT / "index.html").write_text(html_text, encoding="utf-8-sig")
    (OUT / "打开这个_总览.html").write_text(html_text, encoding="utf-8-sig")


def main() -> None:
    setup_style()
    if OUT.exists():
        for old in [FIG_DIR, TABLE_DIR]:
            if old.exists():
                shutil.rmtree(old)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_summary()
    ranking = shared_summary(summary)
    w04 = window_table(summary, 0.4)
    w10 = window_table(summary, 1.0)
    peaks = peak_table(summary)

    summary.to_csv(TABLE_DIR / "完整结果_summary.csv", index=False, encoding="utf-8-sig")
    to_cn_table(ranking, "ranking").to_csv(TABLE_DIR / "共享窗口平均ACC排行.csv", index=False, encoding="utf-8-sig")
    to_cn_table(w04, "window").to_csv(TABLE_DIR / "0.4秒直接对比.csv", index=False, encoding="utf-8-sig")
    to_cn_table(w10, "window").to_csv(TABLE_DIR / "1.0秒直接对比.csv", index=False, encoding="utf-8-sig")
    to_cn_table(peaks, "peak").to_csv(TABLE_DIR / "峰值窗口.csv", index=False, encoding="utf-8-sig")

    plot_ranking(ranking)
    plot_shared_curve(summary, "accuracy", "accuracy_sem", "ACC", "共享窗口准确率", "02_ACC曲线_共享窗口.png")
    plot_shared_curve(summary, "itr", "itr_sem", "ITR（bits/min）", "共享窗口信息传输率", "03_ITR曲线_共享窗口.png")
    plot_window_bars(w04, 0.4, "04_0.4秒直接对比.png")
    plot_window_bars(w10, 1.0, "05_1.0秒直接对比.png")
    plot_sa_mvmd_long(summary)
    plot_peak(peaks)
    write_html(ranking, w04, w10, peaks)
    print(OUT)


if __name__ == "__main__":
    main()
