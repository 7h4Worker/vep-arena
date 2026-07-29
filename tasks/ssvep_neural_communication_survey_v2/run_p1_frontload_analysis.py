from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vep_arena.channel.capacity import capacity_ba, capacity_c1, mutual_info_uniform
from vep_arena.channel.confusion import confusion_counts, normalize_confusion
from vep_arena.metrics import sem


METHODS = ("CCA", "FBCCA", "TRCA", "ETRCA")
WINDOWS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
FULL_V2_WINDOWS = (0.2, 0.3, 0.5, 1.0)
SHORT_WINDOWS = (0.05, 0.1, 0.15)
CLASSES = 40
SUBJECTS = tuple(range(1, 36))
BLOCKS = tuple(range(1, 7))
METHOD_COLORS = {
    "CCA": "#0072B2",
    "FBCCA": "#E69F00",
    "TRCA": "#009E73",
    "ETRCA": "#D55E00",
}
REQUIRED_COLUMNS = {"method", "window", "subject", "block", "true", "pred"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_windows(frame: pd.DataFrame, windows: tuple[float, ...]) -> pd.DataFrame:
    wanted = np.asarray(windows, dtype=float)
    mask = np.isclose(frame["window"].to_numpy(dtype=float)[:, None], wanted[None, :]).any(axis=1)
    selected = frame.loc[mask & frame["method"].isin(METHODS)].copy()
    for window in windows:
        selected.loc[np.isclose(selected["window"], window), "window"] = window
    return selected


def validate_prediction_grid(frame: pd.DataFrame, windows: tuple[float, ...]) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Prediction grid contains null values in required columns.")
    expected_rows = len(METHODS) * len(windows) * len(SUBJECTS) * len(BLOCKS) * CLASSES
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(frame)}.")
    if set(frame["method"].astype(str)) != set(METHODS):
        raise ValueError(f"Unexpected methods: {sorted(frame['method'].astype(str).unique())}")
    actual_windows = sorted(frame["window"].astype(float).unique())
    if len(actual_windows) != len(windows) or not np.allclose(actual_windows, windows):
        raise ValueError(f"Unexpected windows: {actual_windows}")
    if set(frame["subject"].astype(int)) != set(SUBJECTS):
        raise ValueError("Subject grid is not exactly 1-35.")
    if set(frame["block"].astype(int)) != set(BLOCKS):
        raise ValueError("Block grid is not exactly 1-6.")
    for column in ("true", "pred"):
        values = frame[column].astype(int)
        if values.min() != 0 or values.max() != CLASSES - 1:
            raise ValueError(f"{column} labels are not exactly within 0-{CLASSES - 1}.")
    keys = ["method", "window", "subject", "block", "true"]
    if frame.duplicated(keys).any():
        raise ValueError(f"Prediction grid has duplicate keys: {keys}")
    group_sizes = frame.groupby(["method", "window", "subject", "block"], observed=True).size()
    if not (group_sizes == CLASSES).all():
        raise ValueError("Each method/window/subject/block cell must contain 40 trials.")


def load_predictions(full_v2_path: Path, short_path: Path, w075_path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    sources = {
        "full_v2": {"path": str(full_v2_path), "sha256": sha256_file(full_v2_path)},
        "short": {"path": str(short_path), "sha256": sha256_file(short_path)},
        "w075": {"path": str(w075_path), "sha256": sha256_file(w075_path)},
    }
    full_v2 = select_windows(pd.read_csv(full_v2_path), FULL_V2_WINDOWS)
    short = select_windows(pd.read_csv(short_path), SHORT_WINDOWS)
    w075 = select_windows(pd.read_csv(w075_path), (0.75,))
    validate_prediction_grid(full_v2, FULL_V2_WINDOWS)
    validate_prediction_grid(short, SHORT_WINDOWS)
    validate_prediction_grid(w075, (0.75,))
    combined = pd.concat([short, full_v2, w075], ignore_index=True)
    validate_prediction_grid(combined, WINDOWS)
    return combined, sources


def channel_metrics(true: pd.Series, pred: pd.Series, classes: int) -> dict[str, float | bool | int]:
    accuracy = float((true.to_numpy(dtype=int) == pred.to_numpy(dtype=int)).mean())
    transition = normalize_confusion(confusion_counts(true, pred, classes), alpha=0.0)
    ba = capacity_ba(transition)
    return {
        "accuracy": accuracy,
        "c_ba": float(ba.capacity),
        "mi_uniform": float(mutual_info_uniform(transition)),
        "fano_c1": float(capacity_c1(classes, accuracy)),
        "ba_converged": bool(ba.converged),
        "ba_iterations": int(ba.iterations),
    }


def analyze_frontload(frame: pd.DataFrame, classes: int = CLASSES) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_rows: list[dict[str, object]] = []
    for (method, window, subject), group in frame.groupby(["method", "window", "subject"], sort=True):
        subject_rows.append(
            {
                "method": str(method),
                "window_seconds": float(window),
                "subject": int(subject),
                **channel_metrics(group["true"], group["pred"], classes),
            }
        )
    subject_metrics = pd.DataFrame(subject_rows)

    rows: list[dict[str, object]] = []
    for (method, window), group in frame.groupby(["method", "window"], sort=True):
        per_subject = subject_metrics[
            (subject_metrics["method"] == method) & np.isclose(subject_metrics["window_seconds"], window)
        ]
        pooled = channel_metrics(group["true"], group["pred"], classes)
        row: dict[str, object] = {
            "method": str(method),
            "window_seconds": float(window),
            "subjects": int(per_subject["subject"].nunique()),
            "trials": int(len(group)),
            "pooled_accuracy": pooled["accuracy"],
            "pooled_c_ba": pooled["c_ba"],
            "pooled_mi_uniform": pooled["mi_uniform"],
            "pooled_fano_c1": pooled["fano_c1"],
            "pooled_ba_converged": pooled["ba_converged"],
            "pooled_ba_iterations": pooled["ba_iterations"],
        }
        for metric in ("accuracy", "c_ba", "mi_uniform", "fano_c1"):
            values = per_subject[metric].to_numpy(dtype=float)
            row[f"subject_mean_{metric}"] = float(values.mean())
            row[f"subject_median_{metric}"] = float(np.median(values))
            row[f"subject_sem_{metric}"] = sem(values)
        row["jensen_gap_c_ba"] = float(row["subject_mean_c_ba"] - row["pooled_c_ba"])
        rows.append(row)

    results = pd.DataFrame(rows)
    order = {method: index for index, method in enumerate(METHODS)}
    results["_order"] = results["method"].map(order)
    results = results.sort_values(["_order", "window_seconds"]).drop(columns="_order").reset_index(drop=True)
    for metric in ("c_ba", "mi_uniform", "fano_c1"):
        for scope in ("pooled", "subject_mean", "subject_median"):
            column = f"{scope}_{metric}"
            baseline = results[np.isclose(results["window_seconds"], 1.0)].set_index("method")[column]
            results[f"normalized_{column}"] = results.apply(
                lambda row: float(row[column]) / float(baseline.loc[row["method"]]), axis=1
            )
    for scope in ("pooled", "subject_mean"):
        column = f"{scope}_c_ba"
        rate_column = f"marginal_rate_{scope}_c_ba_bits_per_second"
        results[rate_column] = np.nan
        for method in METHODS:
            indices = results.index[results["method"] == method]
            windows = results.loc[indices, "window_seconds"].to_numpy(dtype=float)
            values = results.loc[indices, column].to_numpy(dtype=float)
            rates = np.r_[values[0] / windows[0], np.diff(values) / np.diff(windows)]
            results.loc[indices, rate_column] = rates

    reference = subject_metrics[
        (subject_metrics["method"] == "ETRCA") & np.isclose(subject_metrics["window_seconds"], 1.0)
    ][["subject", "c_ba"]].sort_values(["c_ba", "subject"])
    reference["stratum"] = pd.qcut(reference["c_ba"], 3, labels=["weak", "middle", "strong"])
    strata = subject_metrics[subject_metrics["method"] == "ETRCA"].merge(
        reference[["subject", "stratum"]], on="subject", how="left", validate="many_to_one"
    )
    stratum_summary = (
        strata.groupby(["stratum", "window_seconds"], observed=True)
        .agg(
            subjects=("subject", "nunique"),
            c_ba_mean=("c_ba", "mean"),
            c_ba_median=("c_ba", "median"),
            accuracy_mean=("accuracy", "mean"),
        )
        .reset_index()
    )
    baselines = stratum_summary[np.isclose(stratum_summary["window_seconds"], 1.0)].set_index("stratum")["c_ba_mean"]
    stratum_summary["normalized_c_ba_mean"] = stratum_summary.apply(
        lambda row: float(row["c_ba_mean"]) / float(baselines.loc[row["stratum"]]), axis=1
    )
    return results, subject_metrics, stratum_summary


def plot_f4(results: pd.DataFrame, output_base: Path, style_path: Path) -> None:
    plt.style.use(style_path)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.4), sharex=True)
    for method in METHODS:
        rows = results[results["method"] == method].sort_values("window_seconds")
        color = METHOD_COLORS[method]
        axes[0, 0].plot(rows["window_seconds"], rows["subject_mean_c_ba"], "o-", color=color, label=method)
        axes[0, 0].plot(rows["window_seconds"], rows["pooled_c_ba"], "--", color=color, alpha=0.55)
        axes[0, 1].plot(rows["window_seconds"], rows["normalized_subject_mean_c_ba"], "o-", color=color, label=method)
        axes[0, 1].plot(rows["window_seconds"], rows["normalized_pooled_c_ba"], "--", color=color, alpha=0.55)
        axes[1, 0].plot(
            rows["window_seconds"], rows["marginal_rate_subject_mean_c_ba_bits_per_second"], "o-", color=color
        )
        axes[1, 0].plot(
            rows["window_seconds"],
            rows["marginal_rate_pooled_c_ba_bits_per_second"],
            "--",
            color=color,
            alpha=0.55,
        )
        axes[1, 1].plot(rows["window_seconds"], rows["jensen_gap_c_ba"], "o-", color=color)
    axes[0, 0].set_ylabel("C_BA (bits/trial)")
    axes[0, 0].set_title("Absolute capacity")
    axes[0, 1].set_ylabel("C(T) / C(1.0 s)")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Normalized accumulation")
    axes[1, 0].set_ylabel("Marginal dC/dT (bits/s)")
    axes[1, 0].set_title("Marginal information rate")
    axes[1, 1].set_ylabel("Subject mean - pooled (bits/trial)")
    axes[1, 1].set_title("Capacity scope gap")
    for axis in axes.flat:
        axis.set_xlim(0, 1.2)
        axis.set_xlabel("Observation window (s)")
        axis.axvline(0.2, color="#999999", linestyle=":", linewidth=1.0)
    axes[0, 0].legend(ncol=2, loc="lower right")
    fig.text(0.5, 0.01, "Solid: subject-wise mean; dashed: pooled confusion", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def git_head(project: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()


def build_report(results: pd.DataFrame, sources: dict[str, dict[str, str]], head: str, figure_name: str) -> str:
    index = results.set_index(["method", "window_seconds"])
    rows = []
    for method in METHODS:
        w01 = index.loc[(method, 0.1)]
        w02 = index.loc[(method, 0.2)]
        w10 = index.loc[(method, 1.0)]
        rows.append(
            f"| {method} | {w10['pooled_c_ba']:.4f} | {w10['subject_mean_c_ba']:.4f} | "
            f"{100*w01['normalized_subject_mean_c_ba']:.2f}% | {100*w02['normalized_subject_mean_c_ba']:.2f}% | "
            f"{w02['jensen_gap_c_ba']:.4f} |"
        )
    source_text = "; ".join(f"{name}={item['sha256'][:12]}" for name, item in sources.items())
    return "\n".join(
        [
            "# P1 information-frontloading validation",
            "",
            "① 问题：实际单码元信息量在 0.2 s 内累计多少，历史 79% 是否依赖估计口径与解码器？",
            "② 数据：Tsinghua Benchmark 35 subjects × 6 blocks × 40 targets，所有公开结果仅使用数值被试索引。",
            "③ 方法：CCA、FBCCA、TRCA、ETRCA；同时计算合并混淆与逐被试后平均的 C_BA，并附 MI_uniform、Fano C1。",
            f"④ 管线与参数：commit `{head}`；window=0.05/0.1/0.15/0.2/0.3/0.5/0.75/1.0 s；0.14 s latency；1.0 s filter cache；sources {source_text}。",
            "⑤ 关键数字：",
            "",
            "| Method | pooled C_BA @1.0 | subject-mean C_BA @1.0 | subject-mean ratio @0.1 | subject-mean ratio @0.2 | Jensen gap @0.2 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            f"⑥ 关键图：`{figure_name}`。Absolute/normalized C(T), marginal dC/dT, and the capacity-scope gap under one protocol.",
            "⑦ 结论：历史 79% 只在逐被试平均 C_BA 的 ETRCA@0.2 s 成立；它不是四解码器一致的 pooled-confusion 结论。逐被试矩阵每类仅 6 次观测，因此 scope gap 同时包含被试差异与 plug-in 有限样本上偏，不能全部解释为生理异质性（证据等级：实测）。",
            "⑧ 进入正文：综述 §4(2)，正文候选图 F4。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen P1 information-frontloading deliverables.")
    parser.add_argument("--full-v2-predictions", type=Path, required=True)
    parser.add_argument("--short-predictions", type=Path, required=True)
    parser.add_argument("--w075-predictions", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame, sources = load_predictions(args.full_v2_predictions, args.short_predictions, args.w075_predictions)
    results, subjects, strata = analyze_frontload(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "frontload_short_windows.csv"
    figure_base = args.output_dir / "F4_frontload_CofT_P1_2026-07-29"
    results.to_csv(csv_path, index=False, float_format="%.12g")
    results.to_csv(figure_base.with_suffix(".csv"), index=False, float_format="%.12g")
    subjects.to_csv(args.output_dir / "frontload_subject_metrics.csv", index=False, float_format="%.12g")
    strata.to_csv(args.output_dir / "frontload_etraca_subject_strata.csv", index=False, float_format="%.12g")
    plot_f4(results, figure_base, args.style)
    project = Path(__file__).resolve().parents[2]
    head = git_head(project)
    report = build_report(results, sources, head, figure_base.with_suffix(".png").name)
    (args.output_dir / "P1_frontload_report_2026-07-29.md").write_text(report, encoding="utf-8")
    provenance = {
        "status": "complete",
        "git_head": head,
        "methods": list(METHODS),
        "windows_seconds": list(WINDOWS),
        "classes": CLASSES,
        "subjects": list(SUBJECTS),
        "blocks": list(BLOCKS),
        "preprocessing": {"visual_latency_seconds": 0.14, "epoch_cache_window_seconds": 1.0},
        "capacity_scopes": ["pooled_confusion", "mean_of_subject_confusion_capacities"],
        "sources": sources,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(csv_path)


if __name__ == "__main__":
    main()
