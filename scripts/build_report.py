from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import RESULT_ROOT, BenchmarkSpec
from vep_arena.metrics import cohen_dz, itr_bits_per_minute, sem


ROOT = Path("D:/ProjData/proj_python")
SOURCES = {
    "DNN": ROOT / "dnn_ssvep_pytorch" / "results_clean" / "subject_block_results.csv",
    "SSVEPFormer": ROOT / "ssvepformer_benchmark_pytorch" / "outputs" / "sweeps" / "reference_aligned_20260520" / "analysis" / "subject_block_results.csv",
    "FBTRCA": ROOT / "fbtrca_benchmark_python" / "outputs" / "sweeps" / "benchmark_9ch_20260520" / "analysis" / "subject_block_results.csv",
    "TDCA": ROOT / "vep_arena" / "runs" / "tdca" / "subject_block.csv",
    "TRCA-Net": ROOT / "trcanet_benchmark_pytorch" / "outputs" / "sweeps" / "benchmark_9ch_20260521" / "analysis" / "subject_block_results.csv",
}
METHOD_ORDER = ["FBTRCA", "TDCA", "DNN", "TRCA-Net", "SSVEPFormer"]
COLORS = {
    "FBTRCA": "#246b55",
    "TDCA": "#aa5a00",
    "DNN": "#365caa",
    "TRCA-Net": "#b73549",
    "SSVEPFormer": "#8a4ebf",
}


def load_source(method: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "signal_length" in df.columns:
        df = df.rename(columns={"signal_length": "window"})
    df["method"] = method
    keep = ["method", "window", "subject", "block", "accuracy", "samples"]
    if "seconds" in df.columns:
        keep.append("seconds")
    return df[keep].copy()


def method_note(method: str) -> str:
    notes = {
        "DNN": "imported; original DNN-style filterbank, global training plus subject fine-tune",
        "TRCA-Net": "imported; TRCA-projected CNN, global training plus subject fine-tune",
        "SSVEPFormer": "imported; reference-aligned architecture on Benchmark 9ch validation",
        "FBTRCA": "imported; Nakanishi-style ensemble FBTRCA",
        "TDCA": "VEP Arena adapter; paper-style Benchmark settings, 5 filter banks, 5 harmonics, 8 components, 5 delays",
    }
    return notes[method]


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = BenchmarkSpec()
    subject = (
        df.groupby(["method", "window", "subject"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            block_std=("accuracy", lambda x: float(np.std(x, ddof=1))),
            blocks=("block", "nunique"),
            samples=("samples", "sum"),
        )
    )
    subject["itr"] = subject.apply(
        lambda r: itr_bits_per_minute(float(r["accuracy"]), spec.classes, float(r["window"]) + spec.cue_seconds),
        axis=1,
    )
    block = (
        df.groupby(["method", "window", "block"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), subject_std=("accuracy", lambda x: float(np.std(x, ddof=1))), subjects=("subject", "nunique"))
    )
    summary = (
        subject.groupby(["method", "window"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            subject_sd=("accuracy", lambda x: float(np.std(x, ddof=1))),
            subject_sem=("accuracy", lambda x: sem(x.to_numpy())),
            itr=("itr", "mean"),
            itr_sd=("itr", lambda x: float(np.std(x, ddof=1))),
            itr_sem=("itr", lambda x: sem(x.to_numpy())),
            mean_block_std=("block_std", "mean"),
            block_std_sem=("block_std", lambda x: sem(x.to_numpy())),
            subjects=("subject", "nunique"),
        )
    )
    return summary, subject, block


def paired_stats(subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in sorted(subject["window"].unique()):
        wide = subject[subject["window"] == window].pivot(index="subject", columns="method", values="accuracy")
        for a, b in itertools.combinations(METHOD_ORDER, 2):
            if a not in wide or b not in wide:
                continue
            pair = wide[[a, b]].dropna()
            diff = pair[a].to_numpy() - pair[b].to_numpy()
            stat = ttest_rel(pair[a], pair[b])
            rows.append(
                {
                    "window": window,
                    "method_a": a,
                    "method_b": b,
                    "mean_a": float(pair[a].mean()),
                    "mean_b": float(pair[b].mean()),
                    "mean_delta_a_minus_b": float(diff.mean()),
                    "t": float(stat.statistic),
                    "p": float(stat.pvalue),
                    "cohen_dz": cohen_dz(diff),
                    "n_subjects": int(len(pair)),
                }
            )
    return pd.DataFrame(rows)


def plot_metric(summary: pd.DataFrame, metric: str, sem_col: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("window")
        if sub.empty:
            continue
        ax.errorbar(
            sub["window"],
            sub[metric],
            yerr=sub[sem_col],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=method,
            color=COLORS.get(method),
        )
    ax.set_xlabel("Signal window (s)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(summary["window"].unique()))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_markdown_table(summary: pd.DataFrame, path: Path) -> None:
    pivot = summary.pivot(index="window", columns="method", values="accuracy")
    sd = summary.pivot(index="window", columns="method", values="subject_sd")
    lines = ["# Accuracy Table", "", "| Window | " + " | ".join(METHOD_ORDER) + " |"]
    lines.append("| ---: | " + " | ".join(["---:" for _ in METHOD_ORDER]) + " |")
    for window in sorted(pivot.index):
        cells = []
        for method in METHOD_ORDER:
            if method in pivot.columns and not pd.isna(pivot.loc[window, method]):
                cells.append(f"{pivot.loc[window, method]:.4f} +/- {sd.loc[window, method]:.4f}")
            else:
                cells.append("")
        lines.append(f"| {window:.1f} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    frames = []
    missing = []
    for method, path in SOURCES.items():
        if path.exists():
            frames.append(load_source(method, path))
        else:
            missing.append({"method": method, "path": str(path)})
    if not frames:
        raise FileNotFoundError("No benchmark result sources were found.")
    df = pd.concat(frames, ignore_index=True)
    summary, subject, block = aggregate(df)
    stats = paired_stats(subject)

    df.to_csv(RESULT_ROOT / "trials.csv", index=False)
    summary.to_csv(RESULT_ROOT / "summary.csv", index=False)
    subject.to_csv(RESULT_ROOT / "subject.csv", index=False)
    block.to_csv(RESULT_ROOT / "block.csv", index=False)
    stats.to_csv(RESULT_ROOT / "stats.csv", index=False)
    pd.DataFrame(missing).to_csv(RESULT_ROOT / "missing_sources.csv", index=False)

    paper_compare = pd.DataFrame(
        [
            {"window": 0.5, "method": "TDCA", "paper_accuracy": 0.850, "paper_error": 0.027},
            {"window": 1.0, "method": "TDCA", "paper_accuracy": 0.968, "paper_error": 0.010},
            {"window": 0.5, "method": "FBTRCA", "paper_accuracy": 0.794, "paper_error": 0.033},
            {"window": 1.0, "method": "FBTRCA", "paper_accuracy": 0.936, "paper_error": 0.019},
        ]
    )
    ours = summary[["method", "window", "accuracy", "subject_sem"]]
    paper_compare = paper_compare.merge(ours, on=["method", "window"], how="left")
    paper_compare["accuracy_delta_ours_minus_paper"] = paper_compare["accuracy"] - paper_compare["paper_accuracy"]
    paper_compare.to_csv(RESULT_ROOT / "paper_tdca_comparison.csv", index=False)

    write_markdown_table(summary, RESULT_ROOT / "tables" / "accuracy.md")
    plot_metric(summary, "accuracy", "subject_sem", "Accuracy", RESULT_ROOT / "figures" / "accuracy.svg")
    plot_metric(summary, "itr", "itr_sem", "ITR (bits/min)", RESULT_ROOT / "figures" / "itr.svg")
    plot_metric(summary, "mean_block_std", "block_std_sem", "Within-subject block SD", RESULT_ROOT / "figures" / "block_sd.svg")

    best = summary.sort_values(["window", "accuracy"], ascending=[True, False]).groupby("window").head(1)
    notes = "\n".join(f"- {m}: {method_note(m)}" for m in METHOD_ORDER)
    method_count = summary["method"].nunique()
    report = f"""# VEP Arena Benchmark 9ch

This report combines {method_count} methods on the Tsinghua Benchmark SSVEP dataset using
9 occipital/parietal channels and leave-one-block-out testing.

## Methods

{notes}

## Files

- `summary.csv`: group mean, subject SD, SEM, ITR, and block stability by method/window
- `subject.csv`: subject-level mean across blocks plus within-subject block SD
- `block.csv`: block-level group results
- `stats.csv`: paired subject-level tests for method differences at each window
- `paper_tdca_comparison.csv`: quick check against reported TDCA/TRCA values at 0.5s and 1.0s
- `tables/accuracy.md`: compact accuracy table
- `figures/accuracy.svg`: accuracy curves with SEM
- `figures/itr.svg`: ITR curves with SEM
- `figures/block_sd.svg`: within-subject block stability

## Best Method By Window

| Window | Method | Accuracy |
| ---: | --- | ---: |
"""
    for row in best.itertuples(index=False):
        report += f"| {row.window:.1f} | {row.method} | {row.accuracy:.4f} |\n"
    report += "\n## Paper Check\n\n"
    report += "| Method | Window | Ours | Paper | Delta |\n"
    report += "| --- | ---: | ---: | ---: | ---: |\n"
    for row in paper_compare.itertuples(index=False):
        report += (
            f"| {row.method} | {row.window:.1f} | {row.accuracy:.4f} | "
            f"{row.paper_accuracy:.4f} | {row.accuracy_delta_ours_minus_paper:+.4f} |\n"
        )
    report += "\n## Accuracy Table\n\n"
    report += (RESULT_ROOT / "tables" / "accuracy.md").read_text(encoding="utf-8").split("\n", 2)[2]
    (RESULT_ROOT / "report.md").write_text(report, encoding="utf-8")
    print(RESULT_ROOT)


if __name__ == "__main__":
    main()
