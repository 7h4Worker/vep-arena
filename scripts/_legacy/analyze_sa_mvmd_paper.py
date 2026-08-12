# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import confusion_matrix
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import RUN_ROOT


METHOD_ORDER = ["SA-MVMD-TRCA", "SA-MVMD-eTRCA"]
COLORS = {
    "SA-MVMD-TRCA": "#d97941",
    "SA-MVMD-eTRCA": "#315aa3",
}
PAPER_WINDOW_HINTS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def sem(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


def apply_style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    subject_block = pd.read_csv(run_dir / "subject_block.csv")
    predictions = pd.read_csv(run_dir / "predictions.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    subject_block["source"] = run_dir.name
    predictions["source"] = run_dir.name
    return subject_block, predictions, manifest


def aggregate_trials(subject_block: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject = (
        subject_block.groupby(["method", "window", "subject"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            itr=("itr", "mean"),
            block_sd=("accuracy", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")),
        )
    )
    block = (
        subject_block.groupby(["method", "window", "block"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            itr=("itr", "mean"),
            subject_sd=("accuracy", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")),
            subjects=("subject", "nunique"),
        )
    )
    summary = (
        subject.groupby(["method", "window"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_sem=("accuracy", sem),
            itr=("itr", "mean"),
            itr_sem=("itr", sem),
            block_sd=("block_sd", "mean"),
            block_sd_sem=("block_sd", sem),
            subjects=("subject", "nunique"),
        )
    )
    return summary, subject, block


def paired_stats(subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if len(METHOD_ORDER) < 2:
        return pd.DataFrame(rows)
    ref = METHOD_ORDER[0]
    other = METHOD_ORDER[1]
    for window in sorted(subject["window"].unique()):
        wide = subject[subject["window"] == window].pivot(index="subject", columns="method", values="accuracy")
        if ref not in wide.columns or other not in wide.columns:
            continue
        pair = wide[[ref, other]].dropna()
        if len(pair) < 2:
            continue
        diff = pair[other] - pair[ref]
        stat = ttest_rel(pair[other], pair[ref])
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
        rows.append(
            {
                "window": float(window),
                "method_a": ref,
                "method_b": other,
                "mean_a": float(pair[ref].mean()),
                "mean_b": float(pair[other].mean()),
                "delta_b_minus_a": float(diff.mean()),
                "t": float(stat.statistic),
                "p": float(stat.pvalue),
                "cohen_dz": float(diff.mean() / sd) if sd and sd > 1e-12 else float("nan"),
                "subjects": int(len(pair)),
            }
        )
    return pd.DataFrame(rows)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=220)
    fig.savefig(out_dir / f"{stem}.svg")
    plt.close(fig)


def draw_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, *, face: str, edge: str = "#2f2f2f", fontsize: int = 10) -> None:
    box = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.15,
    )


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = "#444", connectionstyle: str = "arc3,rad=0.0") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.2, color=color, connectionstyle=connectionstyle),
    )


def plot_method_profile(out_dir: Path, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(14.0, 6.5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(0.03, 0.94, "SA-MVMD pipeline", fontsize=14, weight="bold", ha="left", va="center")
    ax.text(0.03, 0.47, "Spatial-filter baseline", fontsize=14, weight="bold", ha="left", va="center")

    # Top lane: SA-MVMD.
    y = 0.66
    h = 0.14
    w = 0.12
    xs = [0.08, 0.23, 0.38, 0.53, 0.69, 0.84]
    top_text = [
        "EEG trial\n(9 ch)",
        "40 candidate\nfrequencies",
        "Add sinusoidal\nassistance",
        "MVMD\n(iterative)",
        "Keep IMF2-6\n+ reconstruction",
        "TRCA / eTRCA\nscore",
    ]
    top_faces = ["#f2f2f2", "#efe3d6", "#d8e7f2", "#e6edf7", "#dceadf", "#f2ead8"]
    for x, text, face in zip(xs, top_text, top_faces):
        draw_box(ax, (x, y), (w, h), text, face=face)
    for idx in range(len(xs) - 1):
        draw_arrow(ax, (xs[idx] + w, y + h / 2), (xs[idx + 1], y + h / 2))
    ax.text(0.91, 0.72, "Argmax", fontsize=11, ha="center", va="center")
    draw_arrow(ax, (xs[-1] + w, y + h / 2), (0.93, y + h / 2))

    ax.text(
        0.50,
        0.57,
        "Main cost: the MVMD box is repeated for every candidate frequency,\nso one test trial triggers 40 decompositions before scoring.",
        ha="center",
        va="center",
        fontsize=10,
        color="#555",
    )

    # Bottom lane: spatial-filter style baseline.
    y2 = 0.22
    h2 = 0.14
    w2 = 0.16
    xs2 = [0.10, 0.34, 0.58]
    bottom_text = [
        "EEG trial",
        "Band bank / CCA refs",
        "CCA / FBCCA / TRCA",
    ]
    bottom_faces = ["#f2f2f2", "#e7eef5", "#dceadf"]
    for x, text, face in zip(xs2, bottom_text, bottom_faces):
        draw_box(ax, (x, y2), (w2, h2), text, face=face)
    for idx in range(len(xs2) - 1):
        draw_arrow(ax, (xs2[idx] + w2, y2 + h2 / 2), (xs2[idx + 1], y2 + h2 / 2))
    ax.text(0.87, 0.29, "Argmax", fontsize=11, ha="center", va="center")
    draw_arrow(ax, (xs2[-1] + w2, y2 + h2 / 2), (0.90, y2 + h2 / 2))

    ax.text(
        0.69,
        0.08,
        "No per-candidate iterative decomposition loop.",
        ha="center",
        va="center",
        fontsize=10,
        color="#555",
    )

    save_figure(fig, out_dir, stem)


def plot_runtime_by_window(manifests: list[dict[str, object]], out_dir: Path, stem: str) -> pd.DataFrame:
    rows = []
    for manifest in manifests:
        windows = manifest.get("windows") or []
        if not windows:
            continue
        seconds = manifest.get("seconds")
        if seconds is None:
            continue
        rows.append(
            {
                "window": float(windows[0]),
                "seconds": float(seconds),
                "minutes": float(seconds) / 60.0,
            }
        )
    runtime = pd.DataFrame(rows).sort_values("window") if rows else pd.DataFrame(columns=["window", "seconds", "minutes"])
    if runtime.empty:
        return runtime
    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    ax.bar(runtime["window"].astype(str), runtime["minutes"], color="#5d7fbf", width=0.6)
    ax.set_xlabel("Signal window (s)")
    ax.set_ylabel("Runtime per window (min)")
    ax.set_title("Observed wall-clock pressure")
    apply_style(ax)
    save_figure(fig, out_dir, stem)
    return runtime


def complexity_table(n_subjects: int, n_windows: int, n_classes: int, n_blocks: int, n_components: int) -> pd.DataFrame:
    per_subject_window = [
        {
            "stage": "Cached MVMD runs",
            "per_subject_window": n_classes * n_blocks,
            "benchmark_total": n_classes * n_blocks * n_subjects * n_windows,
            "note": "One assisted decomposition for each target and each block; cached once per subject-window.",
        },
        {
            "stage": "Test MVMD runs",
            "per_subject_window": n_classes * n_blocks * n_classes,
            "benchmark_total": n_classes * n_blocks * n_classes * n_subjects * n_windows,
            "note": "40 candidate decompositions for every held-out trial.",
        },
        {
            "stage": "TRCA/eTRCA filter fits",
            "per_subject_window": n_classes * n_components * n_blocks,
            "benchmark_total": n_classes * n_components * n_blocks * n_subjects * n_windows,
            "note": "One filter per target and component, refit once for each held-out block.",
        },
        {
            "stage": "Final target scores",
            "per_subject_window": n_classes * n_blocks * n_classes,
            "benchmark_total": n_classes * n_blocks * n_classes * n_subjects * n_windows,
            "note": "One final score per candidate target, before the weighted sum over components.",
        },
    ]
    df = pd.DataFrame(per_subject_window)
    return df


def plot_curve(summary: pd.DataFrame, metric: str, err: str, ylabel: str, out_dir: Path, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("window")
        if sub.empty:
            continue
        ax.errorbar(
            sub["window"],
            sub[metric],
            yerr=sub[err],
            marker="o",
            linewidth=1.9,
            capsize=3,
            color=COLORS[method],
            label=method,
        )
    ax.set_xlabel("Signal window (s)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(summary["window"].unique()))
    apply_style(ax)
    ax.legend(frameon=False)
    save_figure(fig, out_dir, stem)


def plot_bars(summary: pd.DataFrame, windows: list[float], out_dir: Path, stem: str) -> None:
    existing = [w for w in windows if w in set(summary["window"])]
    if len(existing) < 1:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(existing))
    width = 0.36 if len(METHOD_ORDER) == 2 else 0.22
    offsets = np.linspace(-width / 2, width / 2, num=len(METHOD_ORDER))
    for offset, method in zip(offsets, METHOD_ORDER):
        rows = summary[summary["method"] == method].set_index("window")
        vals = [rows.loc[w, "accuracy"] for w in existing if w in rows.index]
        errs = [rows.loc[w, "accuracy_sem"] for w in existing if w in rows.index]
        ax.bar(x + offset, vals, width=width / max(1, len(METHOD_ORDER) / 2), yerr=errs, capsize=3, color=COLORS[method], label=method)
    ax.set_xticks(x, [f"{w:.1f}s" for w in existing])
    ax.set_ylabel("Accuracy")
    apply_style(ax)
    ax.legend(frameon=False)
    save_figure(fig, out_dir, stem)


def plot_subject_box(subject: pd.DataFrame, windows: list[float], out_dir: Path, stem: str) -> None:
    existing = [w for w in windows if w in set(subject["window"])]
    if len(existing) == 0:
        return
    fig, axes = plt.subplots(1, len(existing), figsize=(5.2 * len(existing), 4.5), sharey=True)
    if len(existing) == 1:
        axes = [axes]
    for ax, window in zip(axes, existing):
        data = [
            subject[(subject["method"] == method) & (subject["window"] == window)]["accuracy"].to_numpy()
            for method in METHOD_ORDER
        ]
        bp = ax.boxplot(data, tick_labels=METHOD_ORDER, patch_artist=True, showfliers=False)
        for patch, method in zip(bp["boxes"], METHOD_ORDER):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.72)
        ax.set_title(f"{window:.1f}s")
        ax.tick_params(axis="x", labelrotation=25)
        apply_style(ax)
    axes[0].set_ylabel("Subject mean accuracy")
    save_figure(fig, out_dir, stem)


def plot_block_profile(block: pd.DataFrame, window: float, out_dir: Path, stem: str) -> None:
    sub = block[block["window"] == window]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for method in METHOD_ORDER:
        rows = sub[sub["method"] == method].sort_values("block")
        if rows.empty:
            continue
        ax.plot(rows["block"], rows["accuracy"], marker="o", linewidth=1.9, color=COLORS[method], label=method)
    ax.set_xlabel("Block")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(sorted(sub["block"].unique()))
    ax.set_title(f"Block profile at {window:.1f}s")
    apply_style(ax)
    ax.legend(frameon=False)
    save_figure(fig, out_dir, stem)


def plot_confusion(predictions: pd.DataFrame, method: str, window: float, out_dir: Path, stem: str) -> None:
    sub = predictions[(predictions["method"] == method) & (predictions["window"] == window)]
    if sub.empty:
        return
    labels = np.arange(40)
    cm = confusion_matrix(sub["true"], sub["pred"], labels=labels, normalize="true")
    np.save(out_dir / f"{stem}.npy", cm)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"{method} {window:.1f}s")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(0, 40, 5))
    ax.set_yticks(range(0, 40, 5))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, out_dir, stem)


def find_available_windows(summary: pd.DataFrame) -> list[float]:
    windows = sorted(float(x) for x in summary["window"].unique())
    if windows:
        return windows
    return PAPER_WINDOW_HINTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT / "sa_mvmd_paper_full")
    parser.add_argument("--out-dir", type=Path, default=Path("D:/ProjData/proj_python/vep_arena/results/benchmark_9ch/sa_mvmd_paper"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    run_dirs = sorted([path for path in args.run_root.iterdir() if path.is_dir() and path.name.startswith("w")])
    if not run_dirs:
        raise FileNotFoundError(f"No window run dirs found under {args.run_root}")

    subject_parts = []
    prediction_parts = []
    manifests = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        subject_path = run_dir / "subject_block.csv"
        prediction_path = run_dir / "predictions.csv"
        if not (manifest_path.exists() and subject_path.exists() and prediction_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" and not args.allow_partial:
            continue
        subject, pred, _ = read_run(run_dir)
        subject_parts.append(subject)
        prediction_parts.append(pred)
        manifests.append({"run_dir": run_dir.name, **manifest})

    if not subject_parts:
        raise RuntimeError("No complete SA-MVMD paper runs found. Re-run the experiment or use --allow-partial.")

    subject_block = pd.concat(subject_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    summary, subject, block = aggregate_trials(subject_block)
    stats = paired_stats(subject)
    windows = find_available_windows(summary)

    out = args.out_dir
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    first_manifest = manifests[0] if manifests else {}
    n_subjects = int(subject_block["subject"].nunique())
    n_windows = int(subject_block["window"].nunique())
    n_blocks = int(subject_block["block"].nunique())
    n_classes = int(predictions["true"].nunique()) if "true" in predictions.columns else 40
    n_components = int(first_manifest.get("selected_harmonics", 5)) + 1 if first_manifest else 6
    runtime = plot_runtime_by_window(manifests, fig_dir, "runtime_by_window")
    complexity = complexity_table(n_subjects, n_windows, n_classes, n_blocks, n_components)
    method_profile_stem = "method_profile"
    plot_method_profile(fig_dir, method_profile_stem)

    write_csv(out / "summary.csv", summary)
    write_csv(out / "subject.csv", subject)
    write_csv(out / "block.csv", block)
    write_csv(out / "paired_stats.csv", stats)
    write_csv(out / "complexity.csv", complexity)
    if not runtime.empty:
        write_csv(out / "runtime_by_window.csv", runtime)

    plot_curve(summary, "accuracy", "accuracy_sem", "Accuracy", fig_dir, "accuracy_curve")
    plot_curve(summary, "itr", "itr_sem", "ITR (bits/min)", fig_dir, "itr_curve")
    plot_curve(summary, "block_sd", "block_sd_sem", "Within-subject block SD", fig_dir, "block_stability")

    plot_bars(summary, [0.6, 0.8, 1.0], fig_dir, "accuracy_bars_06_08_10")
    plot_subject_box(subject, [0.6, 0.8, 1.0], fig_dir, "subject_box_06_08_10")
    plot_block_profile(block, 0.8 if 0.8 in set(block["window"]) else windows[len(windows) // 2], fig_dir, "block_profile")

    for method in METHOD_ORDER:
        if 0.8 in set(summary["window"]):
            plot_confusion(predictions, method, 0.8, fig_dir, f"confusion_{method.lower().replace('-', '_')}_0p8")
        if 1.0 in set(summary["window"]):
            plot_confusion(predictions, method, 1.0, fig_dir, f"confusion_{method.lower().replace('-', '_')}_1p0")

    best_acc = summary.loc[summary.groupby("window")["accuracy"].idxmax()].sort_values("window")
    best_itr = summary.loc[summary.groupby("method")["itr"].idxmax(), ["method", "window", "itr"]]
    mean_by_method = summary.groupby("method", as_index=False)["accuracy"].mean().sort_values("accuracy", ascending=False)
    method_profile_rows = [
        {
            "aspect": "Input",
            "SA-MVMD": "EEG trial plus target-specific assistance",
            "Spatial-filter route": "EEG trial plus fixed band bank / references",
            "comment": "SA-MVMD changes the signal itself before classification.",
        },
        {
            "aspect": "Core operator",
            "SA-MVMD": "Iterative MVMD on each candidate frequency",
            "Spatial-filter route": "Closed-form projection / correlation",
            "comment": "The heavy part is repeated decomposition, not one projection.",
        },
        {
            "aspect": "Per-target behavior",
            "SA-MVMD": "One decomposition per candidate target per test trial",
            "Spatial-filter route": "Usually one pass through a bank of filters or refs",
            "comment": "This is why the wall-clock cost grows fast.",
        },
        {
            "aspect": "Learning object",
            "SA-MVMD": "TRCA / eTRCA filters on decomposed components",
            "Spatial-filter route": "TRCA / CCA family directly on filtered signal",
            "comment": "The classifier sits after the decomposition loop.",
        },
        {
            "aspect": "Typical bottleneck",
            "SA-MVMD": "Candidate-wise MVMD iterations",
            "Spatial-filter route": "Matrix products and correlation",
            "comment": "So the compute profile is qualitatively different.",
        },
    ]
    method_profile = pd.DataFrame(method_profile_rows)

    paper_notes = [
        "Paper comparison notes:",
        "- Main paper Fig. 11 reports SA-MVMD-TRCA superior to all controls across 0.4-2.0 s, with clear separation above 0.8 s.",
        "- Main paper Fig. 12 reports SA-MVMD-eTRCA superior to ensemble controls across 0.4-2.0 s, with the best ITR at 0.6 s.",
        "- Supplementary Table IV/V were not locally available during this run, so exact row-wise numeric verification is still pending.",
    ]

    lines = [
        "# SA-MVMD Paper Results",
        "",
        f"- Run root: `{args.run_root}`",
        f"- Subjects: {sorted(subject_block['subject'].unique().tolist())[:3]} ... {sorted(subject_block['subject'].unique().tolist())[-3:]}",
        f"- Windows: {sorted(subject_block['window'].unique().tolist())}",
        f"- Methods: {METHOD_ORDER}",
        f"- Run parameters: max_iter={first_manifest.get('max_iter')}, tol={first_manifest.get('tol')}, rp={first_manifest.get('rp')}",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Method Profile",
        "",
        markdown_table(method_profile),
        "",
        "## Complexity And Pressure",
        "",
        f"- Subjects: {n_subjects}; windows: {n_windows}; blocks: {n_blocks}; classes: {n_classes}; components per candidate: {n_components}.",
        "- Rough cost shape: one cached decomposition pass over all blocks, then `O(subjects * windows * blocks * classes^2 * MVMD_iteration_cost)` for the candidate loop, plus repeated TRCA/eTRCA fitting on decomposed components.",
        "- In plain words: the method is slower because every test trial is decomposed 40 times, once for each target frequency, before scoring, and the fold-specific filters are refit each round.",
        "",
        markdown_table(complexity),
        "",
        "## Runtime By Window",
        "",
        markdown_table(runtime) if not runtime.empty else "_runtime not available_",
        "",
        "## Paired Stats",
        "",
        markdown_table(stats) if not stats.empty else "_no paired stats_",
        "",
        "## Method Means",
        "",
        markdown_table(mean_by_method),
        "",
        "## Best Accuracy By Window",
        "",
        markdown_table(best_acc[["window", "method", "accuracy", "accuracy_sem", "itr"]]),
        "",
        "## Peak ITR By Method",
        "",
        markdown_table(best_itr.sort_values("method")),
        "",
        "## Paper Notes",
        "",
        *paper_notes,
        "",
        "## Key Files",
        "",
        "- `figures/accuracy_curve.png`",
        "- `figures/itr_curve.png`",
        "- `figures/block_stability.png`",
        f"- `figures/{method_profile_stem}.png`",
        "- `figures/runtime_by_window.png`",
        "- `figures/accuracy_bars_06_08_10.png`",
        "- `figures/subject_box_06_08_10.png`",
        "- `figures/block_profile.png`",
        "- `block.csv`",
        "- `subject.csv`",
        "- `summary.csv`",
        "- `complexity.csv`",
        "- `runtime_by_window.csv`",
        "- `paired_stats.csv`",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "run_root": str(args.run_root),
                "run_dirs": [m["run_dir"] for m in manifests],
                "methods": METHOD_ORDER,
                "windows": sorted(subject_block["window"].unique().tolist()),
                "subjects": sorted(subject_block["subject"].unique().tolist()),
                "rows_subject_block": int(len(subject_block)),
                "rows_predictions": int(len(predictions)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
