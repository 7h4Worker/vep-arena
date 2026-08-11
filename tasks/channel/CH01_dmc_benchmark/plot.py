from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vep_arena.channel.confusion import normalize_confusion
from vep_arena.config import BENCHMARK_FREQS


TASK = Path(__file__).resolve().parent
RESULTS = TASK / "results"
ANALYSIS = RESULTS / "analysis"
FIGURES = RESULTS / "figures"


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def mean_sem(frame: pd.DataFrame, group: list[str], value: str, out_name: str = "mean") -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(group):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = part[value].to_numpy(dtype=float)
        rows.append({**dict(zip(group, keys)), out_name: float(np.mean(vals)), "sem": sem(vals), "n": int(vals.size)})
    return pd.DataFrame(rows)


def best_method_window(capacity: pd.DataFrame) -> tuple[str, float]:
    agg = capacity.groupby(["method", "window"], as_index=False)["c_ba"].mean()
    row = agg.sort_values("c_ba").iloc[-1]
    return str(row["method"]), float(row["window"])


def key(method: str, window: float) -> str:
    w = f"{float(window):.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{method.lower()}_w{w}_sall"


def parse_float_list(text: object) -> list[float]:
    if pd.isna(text):
        return []
    value = str(text).strip()
    if not value:
        return []
    return [float(part) for part in value.split()]


def plot_capacity_ladder(capacity: pd.DataFrame) -> None:
    methods = sorted(capacity["method"].unique())
    fig, axes = plt.subplots(len(methods), 1, figsize=(9, max(4, 3.1 * len(methods))), sharex=True)
    if len(methods) == 1:
        axes = [axes]
    metrics = [("c0", "C0"), ("c1", "C1"), ("c_ba", "C_BA"), ("i_uniform", "I_uniform")]
    for ax, method in zip(axes, methods):
        part = capacity[capacity["method"] == method]
        for metric, label in metrics:
            agg = mean_sem(part, ["window"], metric, "value").sort_values("window")
            ax.errorbar(agg["window"], agg["value"], yerr=agg["sem"], marker="o", capsize=3, label=label)
        ax.set_title(method)
        ax.set_ylabel("bits/symbol")
        ax.grid(alpha=0.25)
        ax.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Window (s)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01_capacity_ladder.png", dpi=190)
    plt.close(fig)


def plot_method_capacity(capacity: pd.DataFrame) -> None:
    plt.figure(figsize=(8.6, 5.2))
    for method in sorted(capacity["method"].unique()):
        agg = mean_sem(capacity[capacity["method"] == method], ["window"], "c_ba", "c_ba").sort_values("window")
        plt.errorbar(agg["window"], agg["c_ba"], yerr=agg["sem"], marker="o", capsize=3, label=method)
    plt.xlabel("Window (s)")
    plt.ylabel("C_BA (bits/symbol)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fig02_method_capacity_compare.png", dpi=190)
    plt.close()


def plot_confusion_and_error(best_method: str, best_window: float, counts_npz: np.lib.npyio.NpzFile) -> None:
    counts = counts_npz[key(best_method, best_window)]
    P = normalize_confusion(counts)
    plt.figure(figsize=(7.4, 6.3))
    plt.imshow(np.log10(P + 1e-4), cmap="viridis", aspect="equal")
    plt.colorbar(label="log10 P(pred | true)")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"Mean confusion: {best_method}, {best_window:.1f}s")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig03_mean_confusion_heatmap.png", dpi=190)
    plt.close()

    freqs = np.asarray(BENCHMARK_FREQS, dtype=float)
    rows = []
    for i in range(P.shape[0]):
        denom = max(1.0 - P[i, i], 1e-12)
        for j in range(P.shape[1]):
            if i == j:
                continue
            rows.append({"df": abs(freqs[i] - freqs[j]), "p_error": P[i, j], "p_error_cond": P[i, j] / denom})
    err = pd.DataFrame(rows)
    binned = err.groupby("df", as_index=False).agg(p_error=("p_error", "mean"), p_error_cond=("p_error_cond", "mean"))
    plt.figure(figsize=(8.4, 4.8))
    plt.scatter(err["df"], err["p_error"], s=8, alpha=0.18, label="class pairs")
    plt.plot(binned["df"], binned["p_error"], marker="o", color="black", label="mean by distance")
    plt.xlabel("Frequency distance |df| (Hz)")
    plt.ylabel("Mean error probability")
    plt.title(f"Error destination vs frequency distance: {best_method}, {best_window:.1f}s")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fig04_error_vs_frequency_distance.png", dpi=190)
    plt.close()


def plot_qstar(qstar: pd.DataFrame, best_method: str) -> None:
    freqs = np.asarray(BENCHMARK_FREQS, dtype=float)
    plt.figure(figsize=(9.2, 5.2))
    part = qstar[qstar["method"] == best_method]
    for window in sorted(part["window"].unique()):
        rows = part[part["window"] == window].groupby("class", as_index=False)["q_star"].mean().sort_values("class")
        plt.plot(freqs[rows["class"].to_numpy(dtype=int)], rows["q_star"], marker="o", markersize=3, label=f"{window:.1f}s")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("BA optimal input probability q*")
    plt.title(f"Optimal input distribution by window: {best_method}")
    plt.grid(alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig05_optimal_input_distribution.png", dpi=190)
    plt.close()


def plot_scatter_and_distributions(capacity: pd.DataFrame) -> None:
    methods = sorted(capacity["method"].unique())
    plt.figure(figsize=(8.2, 5.2))
    for method in methods:
        part = capacity[capacity["method"] == method]
        plt.scatter(part["delta_asm"], part["c_ba_minus_c1"], s=28, alpha=0.7, label=method)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Delta_asm")
    plt.ylabel("C_BA - C1 (bits/symbol)")
    plt.title("Asymmetry vs BA capacity gain")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fig06_asymmetry_vs_capacity_gain.png", dpi=190)
    plt.close()

    valid = capacity[capacity["c2_valid"] == True].copy()
    plt.figure(figsize=(6.5, 5.8))
    if not valid.empty:
        colors = np.log10(valid["c2_condition"].clip(lower=1.0))
        plt.scatter(valid["c_ba"], valid["c2_closed"], c=colors, cmap="viridis", s=28, alpha=0.75)
        plt.colorbar(label="log10 cond(P)")
        lo = min(valid["c_ba"].min(), valid["c2_closed"].min())
        hi = max(valid["c_ba"].max(), valid["c2_closed"].max())
        plt.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    plt.xlabel("C_BA")
    plt.ylabel("C2 closed")
    plt.title("Closed-form C2 vs BA capacity")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig07_c2_vs_ba_consistency.png", dpi=190)
    plt.close()

    data = [capacity[capacity["method"] == method]["c_ba_minus_c1"].to_numpy(dtype=float) for method in methods]
    plt.figure(figsize=(7.8, 4.8))
    plt.violinplot(data, showmeans=True)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(range(1, len(methods) + 1), methods)
    plt.ylabel("C_BA - C1 (bits/symbol)")
    plt.title("When traditional C1 under/over-estimates BA capacity")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig08_cba_minus_c1_distribution.png", dpi=190)
    plt.close()


def plot_time_tradeoff(capacity: pd.DataFrame) -> None:
    plt.figure(figsize=(8.5, 5.1))
    tmp = capacity.copy()
    tmp["cba_bits_per_min"] = 60.0 * tmp["c_ba"] / tmp["window"]
    for method in sorted(tmp["method"].unique()):
        agg = mean_sem(tmp[tmp["method"] == method], ["window"], "cba_bits_per_min", "rate").sort_values("window")
        plt.errorbar(agg["window"], agg["rate"], yerr=agg["sem"], marker="o", capsize=3, label=method)
    plt.xlabel("Window (s)")
    plt.ylabel("60/T * C_BA (bits/min)")
    plt.title("Capacity-time tradeoff")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fig09_itr_time_tradeoff.png", dpi=190)
    plt.close()


def plot_pruning(pruning: pd.DataFrame) -> None:
    if pruning.empty:
        return
    methods = sorted(pruning["method"].unique())
    best_rows = []
    for method in methods:
        row = pruning[pruning["method"] == method].sort_values("best_pruned_ba").iloc[-1]
        best_rows.append(row)
    best = pd.DataFrame(best_rows)
    x = np.arange(len(best))
    plt.figure(figsize=(8.2, 5.0))
    plt.bar(x, best["ba_gain"], label="BA input gain")
    plt.bar(x, best["pruning_gain"], bottom=best["ba_gain"], label="codebook pruning gain")
    plt.axhline(0, color="black", linewidth=0.8)
    labels = [f"{m}\n{w:.1f}s\nK={k}" for m, w, k in zip(best["method"], best["window"], best["best_size"])]
    plt.xticks(x, labels)
    plt.ylabel("Capacity gain vs C1 (bits/symbol)")
    plt.title("Gain decomposition for best pruned codebooks")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig10_codebook_pruning_gain.png", dpi=190)
    plt.close()


def plot_costa_selection_path(path: pd.DataFrame, best: pd.DataFrame) -> None:
    if path.empty or best.empty:
        return
    rows = []
    for method in sorted(best["method"].unique()):
        row = best[best["method"] == method].sort_values("best_capacity_ba").iloc[-1]
        rows.append(row)
    plt.figure(figsize=(8.8, 5.4))
    for row in rows:
        method = str(row["method"])
        window = float(row["window"])
        seed_class = int(row["seed_class"])
        part = path[
            (path["method"] == method)
            & (np.isclose(path["window"].astype(float), window))
            & (path["seed_class"].astype(int) == seed_class)
        ].sort_values("size")
        if part.empty:
            continue
        label = f"{method} {window:.1f}s"
        plt.plot(part["size"], part["capacity_ba"], marker="o", markersize=3.5, label=label)
        best_idx = part["capacity_ba"].astype(float).idxmax()
        plt.scatter(
            [part.loc[best_idx, "size"]],
            [part.loc[best_idx, "capacity_ba"]],
            s=70,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
    plt.xlabel("Codebook size K")
    plt.ylabel("C_BA after cropped-channel pruning (bits/symbol)")
    plt.title("Costa-style forward codebook selection path")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fig11_costa_selection_path.png", dpi=190)
    plt.close()


def plot_costa_selected_frequencies(best: pd.DataFrame) -> None:
    if best.empty:
        return
    rows = []
    for method in sorted(best["method"].unique()):
        rows.append(best[best["method"] == method].sort_values("best_capacity_ba").iloc[-1])
    labels = [f"{row['method']} {float(row['window']):.1f}s\nK={int(row['best_size'])}" for row in rows]
    y = np.arange(len(rows))
    all_freqs = np.asarray(BENCHMARK_FREQS, dtype=float)
    plt.figure(figsize=(9.2, 4.4))
    for yi, row in zip(y, rows):
        selected_freqs = parse_float_list(row["best_frequencies_hz"])
        plt.scatter(all_freqs, np.full_like(all_freqs, yi, dtype=float), s=14, color="#d0d0d0", label=None)
        plt.scatter(selected_freqs, [yi] * len(selected_freqs), s=42, label=str(row["method"]))
    plt.yticks(y, labels)
    plt.xlabel("Benchmark stimulus frequency (Hz)")
    plt.title("Frequencies retained by best Costa-style codebooks")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig12_costa_selected_frequencies.png", dpi=190)
    plt.close()


def plot_costa_gain_heatmap(pruning: pd.DataFrame) -> None:
    if pruning.empty:
        return
    pivot = pruning.pivot(index="method", columns="window", values="pruning_gain").sort_index()
    fig, ax = plt.subplots(figsize=(8.7, 3.9))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{float(x):.1f}" for x in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Window (s)")
    ax.set_title("Codebook pruning gain over full 40-class BA capacity")
    fig.colorbar(im, ax=ax, label="Pruned C_BA - full C_BA (bits/symbol)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = float(pivot.iloc[i, j])
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_costa_pruning_gain_heatmap.png", dpi=190)
    plt.close(fig)


def write_report(
    capacity: pd.DataFrame,
    aggregate: pd.DataFrame,
    best_method: str,
    best_window: float,
    pruning: pd.DataFrame,
    costa_best: pd.DataFrame,
) -> None:
    methods = ", ".join(sorted(capacity["method"].unique()))
    windows = ", ".join(f"{x:.1f}s" for x in sorted(capacity["window"].unique()))
    subjects = capacity["subject"].nunique()
    best = aggregate.sort_values("c_ba").iloc[-1]
    c1_delta = capacity["c_ba_minus_c1"]
    pruning_note = ""
    if not pruning.empty:
        best_pruned = pruning.sort_values("best_pruned_ba").iloc[-1]
        pruning_note = (
            f"- 最佳裁剪码本：{best_pruned['method']} {float(best_pruned['window']):.1f}s，"
            f"K={int(best_pruned['best_size'])}，"
            f"裁剪后 C_BA={float(best_pruned['best_pruned_ba']):.4f} bit/symbol，"
            f"相对全 40 类 BA 增益={float(best_pruned['pruning_gain']):.4f}。"
        )
    else:
        pruning_note = "- 未生成 Costa-style 码本裁剪结果。"
    selected_note = ""
    if not costa_best.empty:
        rows = []
        for method in sorted(costa_best["method"].unique()):
            row = costa_best[costa_best["method"] == method].sort_values("best_capacity_ba").iloc[-1]
            rows.append(
                f"{row['method']} {float(row['window']):.1f}s: K={int(row['best_size'])}, "
                f"C_BA={float(row['best_capacity_ba']):.3f}"
            )
        selected_note = "；".join(rows)
    lines = [
        "# Benchmark 9ch 决策信道容量复现与分析报告",
        "",
        "## 1. 研究问题",
        "",
        "传统 BCI ITR 使用平均准确率和对称错误假设，把分类器输出压缩成一个标量。",
        "本任务把 Benchmark SSVEP 的分类结果显式建模为离散决策信道，比较 Costa 2020 的闭式容量思路和 Arslan-Sinha 2024 的 Blahut-Arimoto 容量估计。",
        "",
        "## 2. 数据与协议",
        "",
        f"- 数据集：Tsinghua Benchmark 9ch occipital preset。",
        f"- 被试数：{subjects}。",
        f"- 方法：{methods}。",
        f"- 窗口：{windows}。",
        "- 协议：subject-specific leave-one-block-out；每个 subject/method/window 生成 40x40 混淆矩阵。",
        "- ECCA 定位：使用训练折的个体平均模板与正余弦参考信号，融合标准 CCA、模板 CCA 和模板-参考 spatial filter 相关项，是从无监督 CCA 过渡到 TRCA/eTRCA 这类校准模板方法的经典中间基线。",
        "",
        "## 3. 容量指标",
        "",
        "- `C0 = log2(M)`：理想无噪声容量。",
        "- `C1`：传统对称错误公式，对应常用 ITR 的单次信息量。",
        "- `C2`：Costa 2020 / Muroga 闭式容量；矩阵不可逆、病态或最优分布不在概率单纯形时标记 invalid。",
        "- `C_BA`：Blahut-Arimoto 数值容量，是本报告主结果。",
        "- `I_uniform`：均匀输入分布下的互信息。",
        "- Costa-style 码本裁剪：当前实现从全 40 类混淆矩阵裁剪子矩阵并重新归一化，做前向选择；代码支持多起点，本次默认使用最高对角准确率起点。它用于分析编码空间，不等同于原文每个候选码本都重训分类器的 wrapper。",
        "",
        "## 4. 主要结果图",
        "",
        "![容量阶梯](results/figures/fig01_capacity_ladder.png)",
        "",
        "图 1 展示 C0、C1、C_BA 和 I_uniform 随窗口变化。C_BA 与 I_uniform 的差距表示输入分布优化空间，C0 与 C_BA 的差距表示决策信道噪声代价。",
        "",
        "![方法容量对比](results/figures/fig02_method_capacity_compare.png)",
        "",
        "图 2 比较不同解码器的 BA 容量。若某方法在准确率上提升但 C_BA 提升有限，说明它主要改善了均匀使用下的表现，而不是显著改变容量上限。",
        "",
        "![平均混淆矩阵](results/figures/fig03_mean_confusion_heatmap.png)",
        "",
        f"图 3 使用当前最高平均 C_BA 的组合：{best_method}, {best_window:.1f}s。非对角结构是后续编码空间和频率混淆分析的核心对象。",
        "",
        "![错误与频率距离](results/figures/fig04_error_vs_frequency_distance.png)",
        "",
        "图 4 检查错误是否集中在邻近频率之间，用于判断混淆结构是否主要来自频率分辨率限制。",
        "",
        "![最优输入分布](results/figures/fig05_optimal_input_distribution.png)",
        "",
        "图 5 给出 BA 容量实现分布 q*。若 q* 接近均匀，说明 Benchmark 40 类码本对该解码器已经接近均匀最优；若部分频率权重被压低，则提示个体化或非均匀命令先验可能有收益。",
        "",
        "![非对称度与容量增益](results/figures/fig06_asymmetry_vs_capacity_gain.png)",
        "",
        "图 6 对应 Arslan-Sinha 的核心问题：信道非对称度是否解释 BA 相对传统 C1 的增益。",
        "",
        "![C2 与 BA 一致性](results/figures/fig07_c2_vs_ba_consistency.png)",
        "",
        "图 7 只使用闭式 C2 有效的组合。偏离对角线通常来自病态矩阵、有限样本稀疏性或闭式解有效性边界。",
        "",
        "![CBA-C1 分布](results/figures/fig08_cba_minus_c1_distribution.png)",
        "",
        "图 8 直接检查传统公式何时低估或高估 BA 容量。",
        "",
        "![容量时间权衡](results/figures/fig09_itr_time_tradeoff.png)",
        "",
        "图 9 使用 60/T * C_BA(T) 观察最优工作窗口。该图服务于在线系统中速度与可靠性的折中。",
        "",
        "![码本裁剪增益](results/figures/fig10_codebook_pruning_gain.png)",
        "",
        "图 10 把全 40 类 BA 输入优化增益与基于混淆子矩阵的码本裁剪增益分开。",
        "",
        "![Costa 选择路径](results/figures/fig11_costa_selection_path.png)",
        "",
        "图 11 展示各方法代表最佳窗口下的前向码本选择路径。曲线峰值对应当前固定混淆矩阵近似下的最佳 K，随后若继续加入类别，额外类别引入的混淆会抵消类别数增加带来的容量收益。",
        "",
        "![Costa 保留频率](results/figures/fig12_costa_selected_frequencies.png)",
        "",
        "图 12 把最佳裁剪码本映射回 Benchmark 频率轴。该图用于观察被保留刺激是否覆盖整个频段，还是集中在局部更稳定的频率区间。",
        "",
        "![Costa 增益热图](results/figures/fig13_costa_pruning_gain_heatmap.png)",
        "",
        "图 13 展示 method × window 下裁剪相对全 40 类 BA 容量的增益。它给出后续在线码本大小选择和 HD-200 迁移的直接入口。",
        "",
        "## 5. 数值摘要",
        "",
        f"- 最高 aggregate C_BA：{best['method']} {best['window']:.1f}s，C_BA={best['c_ba']:.4f} bit/symbol，accuracy={best['accuracy']:.4f}。",
        f"- C_BA - C1 平均值：{c1_delta.mean():.4f} bit/symbol；最小值：{c1_delta.min():.4f}；最大值：{c1_delta.max():.4f}。",
        pruning_note,
        f"- 各方法最佳裁剪摘要：{selected_note}。" if selected_note else "- 各方法最佳裁剪摘要：未生成。",
        "",
        "## 6. 限制与下一步",
        "",
        "- 第一轮使用 Benchmark 9ch，不是 Costa 2020 的 64ch 完整复刻。",
        "- SSCOR 模块已经补入代码层，当前报告中的全量结果若未重新运行 SSCOR/ESSCOR，则仍只反映已有 predictions 中的方法。",
        "- 类别选择是从固定 40 类混淆矩阵裁剪子矩阵，不重训分类器；下一版可补 Costa 原文式 wrapper。",
        "- 下一步可以把同一套 channel 模块迁移到 HD-200、BETA 和个体化码本分析。",
    ]
    (TASK / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global FIGURES
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES)
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()
    FIGURES = args.figures_dir
    FIGURES.mkdir(parents=True, exist_ok=True)
    capacity = pd.read_csv(args.analysis_dir / "capacity_by_subject_method_window.csv")
    aggregate = pd.read_csv(args.analysis_dir / "capacity_by_method_window_aggregate.csv")
    qstar = pd.read_csv(args.analysis_dir / "qstar_by_class.csv")
    counts_npz = np.load(args.analysis_dir / "confusion_counts_method_window_aggregate.npz")
    pruning_path = args.analysis_dir / "codebook_pruning_by_method_window.csv"
    pruning = pd.read_csv(pruning_path) if pruning_path.exists() else pd.DataFrame()
    costa_path = args.analysis_dir / "costa_selection_path.csv"
    costa_best_path = args.analysis_dir / "costa_best_codebooks.csv"
    costa_path_df = pd.read_csv(costa_path) if costa_path.exists() else pd.DataFrame()
    costa_best = pd.read_csv(costa_best_path) if costa_best_path.exists() else pd.DataFrame()
    best_method, best_window = best_method_window(capacity)
    plot_capacity_ladder(capacity)
    plot_method_capacity(capacity)
    plot_confusion_and_error(best_method, best_window, counts_npz)
    plot_qstar(qstar, best_method)
    plot_scatter_and_distributions(capacity)
    plot_time_tradeoff(capacity)
    plot_pruning(pruning)
    plot_costa_selection_path(costa_path_df, costa_best)
    plot_costa_selected_frequencies(costa_best)
    plot_costa_gain_heatmap(pruning)
    if not args.skip_report:
        write_report(capacity, aggregate, best_method, best_window, pruning, costa_best)
    print(FIGURES)


if __name__ == "__main__":
    main()
