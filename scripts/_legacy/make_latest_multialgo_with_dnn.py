from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("D:/ProjData/proj_python/vep_arena")
BASE_SUMMARY = ROOT / "results" / "benchmark_9ch" / "final_compare" / "summary.csv"
DNN_SUMMARY = ROOT / "results" / "benchmark_9ch" / "dnn_w02_2s_arena" / "summary.csv"
OUT = ROOT / "results" / "benchmark_9ch" / "latest_multialgo_with_arena_dnn"

WINDOWS = [round(0.2 + i * 0.1, 1) for i in range(9)]
METHOD_ORDER = ["CCA", "FBCCA", "FBTRCA", "TDCA", "DNN", "TRCA-Net", "SSVEPFormer"]
COLORS = {
    "CCA": "#6b7280",
    "FBCCA": "#0891b2",
    "FBTRCA": "#2f6f5f",
    "TDCA": "#b15f18",
    "DNN": "#315aa3",
    "DNN global": "#7f9ccf",
    "TRCA-Net": "#bf3f55",
    "SSVEPFormer": "#7b59b6",
}


def normalize_dnn(summary: pd.DataFrame, *, finetuned: bool) -> pd.DataFrame:
    if finetuned:
        rows = summary[summary["method"].isin(["DNN-finetuned-pt", "DNN-finetuned-trained"])].copy()
        rows["method"] = "DNN"
    else:
        rows = summary[summary["method"].isin(["DNN-global-pt", "DNN-global-trained"])].copy()
        rows["method"] = "DNN global"
    rows["window"] = rows["window"].round(1)
    rows = rows[rows["window"].isin(WINDOWS)].copy()
    rows = rows.rename(columns={"accuracy_sem": "subject_sem"})
    rows["subject_sd"] = pd.NA
    rows["itr_sd"] = pd.NA
    rows["block_sd"] = pd.NA
    rows["block_sd_sem"] = pd.NA
    return rows[
        [
            "method",
            "window",
            "accuracy",
            "subject_sd",
            "subject_sem",
            "itr",
            "itr_sd",
            "itr_sem",
            "block_sd",
            "block_sd_sem",
            "subjects",
        ]
    ]


def load_latest() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(BASE_SUMMARY)
    base["window"] = base["window"].round(1)
    base = base[(base["window"].isin(WINDOWS)) & (base["method"] != "DNN")].copy()

    dnn = pd.read_csv(DNN_SUMMARY)
    dnn_ft = normalize_dnn(dnn, finetuned=True)
    dnn_global = normalize_dnn(dnn, finetuned=False)

    main = pd.concat([base, dnn_ft], ignore_index=True)
    main["method"] = pd.Categorical(main["method"], METHOD_ORDER, ordered=True)
    main = main.sort_values(["method", "window"]).reset_index(drop=True)
    return main, dnn_global


def plot_metric(summary: pd.DataFrame, metric: str, err: str, ylabel: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for method in METHOD_ORDER:
        sub = summary[summary["method"].astype(str).eq(method)].sort_values("window")
        ax.errorbar(
            sub["window"],
            sub[metric],
            yerr=sub[err],
            marker="o",
            markersize=4.5,
            linewidth=1.9,
            capsize=3,
            color=COLORS[method],
            label=method,
        )
    ax.set_xlabel("Signal window (s)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(WINDOWS)
    if metric == "accuracy":
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="lower right" if metric == "accuracy" else "best")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=220)
    fig.savefig(OUT / "figures" / f"{name}.svg")
    plt.close(fig)


def plot_dnn_global(summary: pd.DataFrame, dnn_global: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    dnn_ft = summary[summary["method"].astype(str).eq("DNN")].copy()
    both = pd.concat([dnn_ft, dnn_global], ignore_index=True)
    for ax, metric, err, ylabel in [
        (axes[0], "accuracy", "subject_sem", "Accuracy"),
        (axes[1], "itr", "itr_sem", "ITR (bits/min)"),
    ]:
        for method in ["DNN", "DNN global"]:
            sub = both[both["method"].eq(method)].sort_values("window")
            ax.errorbar(
                sub["window"],
                sub[metric],
                yerr=sub[err],
                marker="o",
                markersize=4.5,
                linewidth=1.9,
                capsize=3,
                color=COLORS[method],
                label=method,
            )
        ax.set_xlabel("Signal window (s)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(WINDOWS)
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "dnn_finetuned_vs_global.png", dpi=220)
    fig.savefig(OUT / "figures" / "dnn_finetuned_vs_global.svg")
    plt.close(fig)


def write_report(summary: pd.DataFrame, dnn_global: pd.DataFrame) -> None:
    best_acc = summary.loc[summary.groupby("window")["accuracy"].idxmax()].sort_values("window")
    peak_itr = summary.loc[summary.groupby("method", observed=False)["itr"].idxmax(), ["method", "window", "itr"]]
    dnn_sources = pd.read_csv(DNN_SUMMARY)
    source_rows = dnn_sources[dnn_sources["window"].isin(WINDOWS)][["method", "window", "label"]]
    source_rows.to_csv(OUT / "dnn_source_ledger.csv", index=False)

    lines = [
        "# Latest Multi-Algorithm Comparison With Arena-Side DNN",
        "",
        "Scope: 0.2-1.0 s, because the current full multi-algorithm comparison has complete rows for all listed methods only in this window range.",
        "",
        "DNN is shown as one method in the main figures. The DNN source ledger records whether each row came from saved global checkpoints or Arena-trained global models.",
        "",
        "Preprocessing note: DNN was re-run inside Arena from raw Benchmark data, but keeps the DNN model's input contract: 9 channels, 3 Chebyshev-I order-2 subbands, no 50 Hz notch. Classical CCA/TRCA/TDCA-style methods use their Arena/traditional preprocessing contracts.",
        "",
        "## Best Accuracy By Window",
        "",
        "| Window | Best method | Accuracy |",
        "| ---: | --- | ---: |",
    ]
    for row in best_acc.itertuples(index=False):
        lines.append(f"| {row.window:.1f}s | {row.method} | {row.accuracy:.4f} |")

    lines += [
        "",
        "## Peak ITR",
        "",
        "| Method | Peak window | Peak ITR |",
        "| --- | ---: | ---: |",
    ]
    for row in peak_itr.sort_values("itr", ascending=False).itertuples(index=False):
        lines.append(f"| {row.method} | {row.window:.1f}s | {row.itr:.2f} |")

    dnn_best = dnn_global.loc[dnn_global["accuracy"].idxmax()]
    lines += [
        "",
        "## Files",
        "",
        "- `summary.csv`: main leaderboard summary with updated Arena-side fine-tuned DNN.",
        "- `dnn_global_summary.csv`: non-fine-tuned DNN rows for reference.",
        "- `dnn_source_ledger.csv`: DNN row source labels.",
        "- `figures/latest_accuracy_curve.png`",
        "- `figures/latest_itr_curve.png`",
        "- `figures/dnn_finetuned_vs_global.png`",
        "",
        f"DNN global reference best accuracy in this common range: {dnn_best.accuracy:.4f} at {dnn_best.window:.1f}s.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    summary, dnn_global = load_latest()
    summary.to_csv(OUT / "summary.csv", index=False)
    dnn_global.to_csv(OUT / "dnn_global_summary.csv", index=False)
    plot_metric(summary, "accuracy", "subject_sem", "Accuracy", "latest_accuracy_curve")
    plot_metric(summary, "itr", "itr_sem", "ITR (bits/min)", "latest_itr_curve")
    plot_dnn_global(summary, dnn_global)
    write_report(summary, dnn_global)
    print(OUT)


if __name__ == "__main__":
    main()
