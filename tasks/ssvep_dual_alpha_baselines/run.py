from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.binocular import (  # noqa: E402
    DUAL_ALPHA_BLOCKS,
    DUAL_ALPHA_CLASSES,
    DUAL_ALPHA_ITR_SHIFT_SECONDS,
    DUAL_ALPHA_PARADIGMS,
    DUAL_ALPHA_ROOT,
    DUAL_ALPHA_SAMPLING_RATE,
    available_dual_alpha_subjects,
    dual_alpha_apply_fbdcca_filterbank,
    dual_alpha_apply_trca_filterbank,
    load_dual_alpha_epochs,
)
from vep_arena.metrics import itr_bits_per_minute, sem  # noqa: E402
from vep_arena.methods.fbdcca import FBDCCA  # noqa: E402
from vep_arena.methods.traditional import CCA, FBCCA, TRCA, multi_frequency_reference_signals  # noqa: E402


OFFICIAL_FBDCCA_PARADIGMS = {"Checkerboard_Arrangment", "Binocular_Vision"}


def parse_range(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    return values


def parse_csv(text: str, upper: bool = True) -> list[str]:
    items = [item.strip() for item in text.split(",") if item.strip()]
    return [item.upper() for item in items] if upper else items


def parse_windows(text: str) -> list[float]:
    if ":" in text:
        lo, step, hi = (float(x) for x in text.split(":", 2))
        count = int(round((hi - lo) / step)) + 1
        return [round(lo + idx * step, 10) for idx in range(count)]
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def normalize_method(method: str) -> str:
    method = method.upper()
    if method == "OFFICIAL_TRCA":
        return "ETRCA"
    return method


def make_model(method: str, n_fbs: int, window: float, references: list[np.ndarray], harmonics: int):
    if method == "CCA":
        return CCA(window=window, harmonics=harmonics, references=references)
    if method == "FBCCA":
        return FBCCA(window=window, harmonics=harmonics, n_fbs=n_fbs, references=references, square_scores=True)
    if method == "FBDCCA":
        return FBDCCA(references=references)
    if method == "TRCA":
        return TRCA(n_fbs=n_fbs, ensemble=False)
    if method == "ETRCA":
        return TRCA(n_fbs=n_fbs, ensemble=True)
    raise ValueError(f"Unsupported method for this task: {method}")


def method_filterbank(method: str) -> str:
    if method in {"TRCA", "ETRCA"}:
        return "trca"
    if method in {"FBCCA", "FBDCCA"}:
        return "fbdcca"
    if method == "CCA":
        return "raw"
    raise ValueError(f"Unsupported method for this task: {method}")


def _section_to_paradigm(line: str) -> str | None:
    lower = line.lower()
    if "binocular" in lower and "swap" in lower:
        return "Binocular-Swap_Vision"
    if "binocular vision" in lower:
        return "Binocular_Vision"
    if "checkerboard" in lower:
        return "Checkerboard_Arrangment"
    return None


def load_official_results(root: Path, method: str) -> pd.DataFrame:
    file_name = {
        "ETRCA": "Classification_results_of_TRCA.csv",
        "TRCA": "Classification_results_of_TRCA.csv",
        "FBDCCA": "Classification_results_of_FBDCCA.csv",
    }.get(method)
    if file_name is None:
        return pd.DataFrame()
    path = root / "metadata" / file_name
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    current: str | None = None
    windows: list[float] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if not parts or not parts[0]:
            continue
        section = _section_to_paradigm(parts[0])
        if section is not None:
            current = section
            windows = []
            continue
        if parts[0].startswith("Name"):
            windows = [float(part) for part in parts[1:] if part != ""]
            continue
        if current is None or not parts[0].startswith("Subject") or not windows:
            continue
        subject = int(parts[0].replace("Subject", ""))
        for window, value in zip(windows, parts[1:]):
            if value == "":
                continue
            rows.append(
                {
                    "official_method": "TRCA" if method in {"TRCA", "ETRCA"} else method,
                    "method": method,
                    "paradigm": current,
                    "subject": subject,
                    "window": float(window),
                    "official_accuracy": float(value) / 100.0,
                }
            )
    return pd.DataFrame(rows)


def run_unit(task: dict[str, object]) -> dict[str, object]:
    root = Path(str(task["root"]))
    subject = int(task["subject"])
    paradigm = str(task["paradigm"])
    channel_set = str(task["channel_set"])
    method_windows = {str(method): [float(w) for w in windows] for method, windows in dict(task["method_windows"]).items()}
    all_windows = sorted({window for windows in method_windows.values() for window in windows})
    harmonics = int(task.get("harmonics", 5))
    trca_n_fbs = int(task.get("trca_n_fbs", 7))
    fbdcca_n_fbs = int(task.get("fbdcca_n_fbs", 5))
    fbdcca_filter_backend = str(task.get("fbdcca_filter_backend", "mne-fir"))
    fbdcca_mne_n_jobs = task.get("fbdcca_mne_n_jobs", 1)
    started = time.perf_counter()

    try:
        data = load_dual_alpha_epochs(
            root=root,
            paradigm=paradigm,
            subject=subject,
            window_seconds=max(all_windows),
            channel_set=channel_set,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "reason": repr(exc),
            "trial_rows": [],
            "pred_rows": [],
            "runtime_rows": [],
            "manifest_rows": [
                {
                    "subject": subject,
                    "paradigm": paradigm,
                    "channel_set": channel_set,
                    "methods": json.dumps(list(method_windows)),
                    "windows": json.dumps(all_windows),
                    "status": "failed",
                    "reason": repr(exc),
                }
            ],
            "logs": [f"FAILED {paradigm} sub-{subject:03d}: {exc!r}"],
        }

    labels = np.arange(DUAL_ALPHA_CLASSES, dtype=np.int64)
    blocks = list(range(DUAL_ALPHA_BLOCKS))
    trial_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    logs: list[str] = []

    for window in all_windows:
        samples = int(round(window * data.sampling_rate))
        raw_epochs = data.x[:, :, :, :samples]
        needed_banks = {method_filterbank(method) for method, windows in method_windows.items() if window in windows}
        bank_cache: dict[str, np.ndarray] = {}
        if "raw" in needed_banks:
            bank_cache["raw"] = raw_epochs[:, :, None]
        if "trca" in needed_banks:
            bank_cache["trca"] = dual_alpha_apply_trca_filterbank(raw_epochs, n_fbs=trca_n_fbs, fs=data.sampling_rate)
        if "fbdcca" in needed_banks:
            bank_cache["fbdcca"] = dual_alpha_apply_fbdcca_filterbank(
                raw_epochs,
                n_fbs=fbdcca_n_fbs,
                fs=data.sampling_rate,
                backend=fbdcca_filter_backend,
                n_jobs=fbdcca_mne_n_jobs,
            )
        references = multi_frequency_reference_signals(
            target_frequencies=data.target_freqs,
            samples=samples,
            fs=data.sampling_rate,
            harmonics=harmonics,
        )

        for method, windows in method_windows.items():
            if window not in windows:
                continue
            epochs = bank_cache[method_filterbank(method)]
            n_fbs = epochs.shape[2]
            for block_idx in blocks:
                train_blocks = [idx for idx in blocks if idx != block_idx]
                train_x = epochs[:, train_blocks].reshape(-1, epochs.shape[2], epochs.shape[3], epochs.shape[4])
                train_y = np.repeat(labels, len(train_blocks))
                test_x = epochs[:, block_idx]
                model = make_model(method, n_fbs=n_fbs, window=window, references=references, harmonics=harmonics)
                t0 = time.perf_counter()
                model.fit(train_x, train_y)
                fit_done = time.perf_counter()
                pred, scores = model.predict(test_x)
                predict_done = time.perf_counter()
                accuracy = float(np.mean(pred == labels))
                itr = itr_bits_per_minute(accuracy, DUAL_ALPHA_CLASSES, window + DUAL_ALPHA_ITR_SHIFT_SECONDS)
                block_key = data.block_keys[block_idx]
                trial_rows.append(
                    {
                        "method": method,
                        "paradigm": paradigm,
                        "channel_set": channel_set,
                        "window": window,
                        "subject": subject,
                        "block": block_idx + 1,
                        "block_key": block_key,
                        "fbdcca_filter_backend": fbdcca_filter_backend,
                        "accuracy": accuracy,
                        "itr": float(itr),
                        "samples": DUAL_ALPHA_CLASSES,
                        "seconds": predict_done - t0,
                    }
                )
                for true_label, pred_label in zip(labels, pred):
                    pred_rows.append(
                        {
                            "method": method,
                            "paradigm": paradigm,
                            "channel_set": channel_set,
                            "window": window,
                            "subject": subject,
                            "block": block_idx + 1,
                            "block_key": block_key,
                            "fbdcca_filter_backend": fbdcca_filter_backend,
                            "true": int(true_label),
                            "pred": int(pred_label),
                            "score_true": float(scores[int(true_label), int(true_label)]),
                            "score_pred": float(scores[int(true_label), int(pred_label)]),
                        }
                    )
                runtime_base = {
                    "method": method,
                    "paradigm": paradigm,
                    "channel_set": channel_set,
                    "window": window,
                    "subject": subject,
                    "block": block_idx + 1,
                    "worker_pid": os.getpid(),
                    "fbdcca_filter_backend": fbdcca_filter_backend,
                }
                runtime_rows.extend(
                    [
                        {**runtime_base, "stage": "fit", "seconds": fit_done - t0},
                        {**runtime_base, "stage": "predict", "seconds": predict_done - fit_done},
                        {**runtime_base, "stage": "fit_predict", "seconds": predict_done - t0},
                    ]
                )
            method_rows = [row for row in trial_rows if row["method"] == method and row["window"] == window]
            mean_acc = float(np.mean([row["accuracy"] for row in method_rows[-DUAL_ALPHA_BLOCKS:]]))
            logs.append(f"{method} {paradigm} sub-{subject:03d} w={window:g} acc={mean_acc:.3f}")

    runtime_rows.append(
        {
            "method": "__all__",
            "paradigm": paradigm,
            "channel_set": channel_set,
            "window": "__all__",
            "subject": subject,
            "block": "__all__",
            "stage": "unit_total",
            "seconds": time.perf_counter() - started,
            "worker_pid": os.getpid(),
        }
    )
    return {
        "status": "complete",
        "trial_rows": trial_rows,
        "pred_rows": pred_rows,
        "runtime_rows": runtime_rows,
        "logs": logs,
        "manifest_rows": [
            {
                "subject": subject,
                "paradigm": paradigm,
                "channel_set": channel_set,
                "methods": json.dumps(list(method_windows)),
                "method_windows": json.dumps(method_windows),
                "windows": json.dumps(all_windows),
                "status": "complete",
                "blocks": DUAL_ALPHA_BLOCKS,
                "channels": len(data.channels),
                "samples_max": int(max(all_windows) * data.sampling_rate),
                "harmonics": harmonics,
                "trca_n_fbs": trca_n_fbs,
                "fbdcca_n_fbs": fbdcca_n_fbs,
                "fbdcca_filter_backend": fbdcca_filter_backend,
                "fbdcca_mne_n_jobs": fbdcca_mne_n_jobs,
                "target_freqs": json.dumps(data.target_freqs),
            }
        ],
    }


def load_existing(result_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    def read(name: str) -> list[dict[str, object]]:
        path = result_dir / name
        return pd.read_csv(path).to_dict("records") if path.exists() and path.stat().st_size > 0 else []

    return read("trials.csv"), read("predictions.csv"), read("runtime.csv"), read("unit_manifest.csv")


def completed_windows_from_trials(
    trial_rows: list[dict[str, object]],
) -> dict[tuple[int, str, str, str, str], set[float]]:
    if not trial_rows:
        return {}
    df = pd.DataFrame(trial_rows)
    required = {"subject", "paradigm", "channel_set", "method", "window", "block"}
    if not required.issubset(df.columns):
        return {}
    if "fbdcca_filter_backend" not in df.columns:
        df["fbdcca_filter_backend"] = "__unknown__"
    done: dict[tuple[int, str, str, str, str], set[float]] = {}
    for (subject, paradigm, channel_set, method, window), group in df.groupby(
        ["subject", "paradigm", "channel_set", "method", "window"]
    ):
        if int(group["block"].nunique()) >= DUAL_ALPHA_BLOCKS:
            backend = str(group["fbdcca_filter_backend"].iloc[0])
            key = (int(subject), str(paradigm), str(channel_set), str(method), backend)
            done.setdefault(key, set()).add(float(window))
    return done


def _dedupe(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    existing = [col for col in subset if col in df.columns]
    return df.drop_duplicates(subset=existing, keep="last") if existing else df


def summarize(trials: pd.DataFrame, official: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trials.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "fbdcca_filter_backend" not in trials.columns:
        trials = trials.copy()
        trials["fbdcca_filter_backend"] = "__unknown__"
    subject = (
        trials.groupby(["method", "paradigm", "channel_set", "fbdcca_filter_backend", "window", "subject"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), itr=("itr", "mean"), blocks=("block", "nunique"), seconds=("seconds", "sum"))
        .sort_values(["paradigm", "method", "window", "subject"])
    )
    rows = []
    for (method, paradigm, channel_set, backend, window), group in subject.groupby(
        ["method", "paradigm", "channel_set", "fbdcca_filter_backend", "window"]
    ):
        row = {
            "method": method,
            "paradigm": paradigm,
            "channel_set": channel_set,
            "fbdcca_filter_backend": backend,
            "window": float(window),
            "accuracy": float(group["accuracy"].mean()),
            "accuracy_sem": sem(group["accuracy"].to_numpy()),
            "itr": float(group["itr"].mean()),
            "itr_sem": sem(group["itr"].to_numpy()),
            "subjects": int(group["subject"].nunique()),
            "blocks": int(group["blocks"].sum()),
            "seconds": float(group["seconds"].sum()),
        }
        if not official.empty:
            off = official[
                (official["method"] == method)
                & (official["paradigm"] == paradigm)
                & (np.isclose(official["window"].astype(float), float(window)))
                & (official["subject"].isin(group["subject"].astype(int)))
            ]
            if not off.empty:
                row["official_accuracy"] = float(off["official_accuracy"].mean())
                row["accuracy_delta_vs_official"] = row["accuracy"] - row["official_accuracy"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["paradigm", "method", "window"]), subject


def confusion_counts(preds: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((DUAL_ALPHA_CLASSES, DUAL_ALPHA_CLASSES), dtype=np.int64)
    for row in preds.itertuples(index=False):
        matrix[int(row.true), int(row.pred)] += 1
    return matrix


def plot_outputs(summary: pd.DataFrame, subject: pd.DataFrame, official: pd.DataFrame, result_dir: Path) -> None:
    if summary.empty:
        return
    figdir = result_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, fname in (
        ("accuracy", "Accuracy", "accuracy_curve.png"),
        ("itr", "ITR (bits/min)", "itr_curve.png"),
    ):
        paradigms = list(summary["paradigm"].drop_duplicates())
        fig, axes = plt.subplots(len(paradigms), 1, figsize=(8.8, max(3.2, 2.8 * len(paradigms))), sharex=True)
        if len(paradigms) == 1:
            axes = [axes]
        for ax, paradigm in zip(axes, paradigms):
            rows_paradigm = summary[summary["paradigm"] == paradigm]
            for method in rows_paradigm["method"].drop_duplicates():
                rows = rows_paradigm[rows_paradigm["method"] == method]
                ax.errorbar(
                    rows["window"],
                    rows[metric],
                    yerr=rows[f"{metric}_sem"],
                    marker="o",
                    capsize=3,
                    label=method,
                )
                if metric == "accuracy" and not official.empty:
                    off = official[(official["method"] == method) & (official["paradigm"] == paradigm)]
                    if not off.empty:
                        off_mean = off.groupby("window", as_index=False)["official_accuracy"].mean()
                        ax.plot(off_mean["window"], off_mean["official_accuracy"], "--", alpha=0.55, label=f"{method} official")
            ax.set_title(paradigm)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            if metric == "accuracy":
                ax.set_ylim(0, 1.02)
        axes[-1].set_xlabel("Window (s)")
        fig.tight_layout()
        fig.savefig(figdir / fname, dpi=180)
        plt.close(fig)

    piv = summary.pivot_table(index=["paradigm", "method"], columns="window", values="accuracy")
    fig, ax = plt.subplots(figsize=(9.5, max(3.0, 0.44 * len(piv) + 1.4)))
    im = ax.imshow(piv.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Accuracy")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{x:g}" for x in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{idx[0]} / {idx[1]}" for idx in piv.index])
    ax.set_xlabel("Window (s)")
    fig.tight_layout()
    fig.savefig(figdir / "accuracy_heatmap.png", dpi=180)
    plt.close(fig)

    if not subject.empty:
        best = summary.sort_values("accuracy").groupby(["paradigm", "method"], as_index=False).tail(1)
        rows = []
        labels = []
        for row in best.itertuples(index=False):
            values = subject[
                (subject["paradigm"] == row.paradigm)
                & (subject["method"] == row.method)
                & (subject["window"] == row.window)
            ]["accuracy"].to_numpy()
            if values.size:
                rows.append(values)
                labels.append(f"{row.paradigm}\n{row.method}")
        if rows:
            fig, ax = plt.subplots(figsize=(max(7.5, 1.25 * len(rows)), 4.8))
            ax.boxplot(rows, tick_labels=labels, showmeans=True)
            ax.set_ylim(0, 1.02)
            ax.set_ylabel("Subject Accuracy at Best Window")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figdir / "subject_box_best.png", dpi=180)
            plt.close(fig)


def write_report(summary: pd.DataFrame, result_dir: Path, manifest: dict[str, object]) -> None:
    lines = [
        "# Dual-Alpha 基线运行报告",
        "",
        "协议：GigaDB 102557 Dual-Alpha epoch CSV，受试者内 5-block leave-one-block-out。",
        "公开 TRCA 使用 ensemble 模式；Arena 将该对齐路径记为 ETRCA。",
        "ITR 使用 40 类与 `window + 0.5 s`，与公开示例脚本一致。",
        "预处理：先从 epoch CSV 首样点裁剪，再应用方法对应 filter bank。",
        "",
        "## 运行配置",
        "",
        f"- 受试者：`{manifest['subjects']}`",
        f"- 范式：`{manifest['paradigms']}`",
        f"- 时间窗：`{manifest['windows']}`",
        f"- 方法：`{manifest['methods']}`",
        f"- 通道集：`{manifest['channel_set']}`",
        f"- workers：`{manifest['workers']}`",
        "",
    ]
    if not summary.empty:
        lines.extend(
            [
                "## 各范式最优准确率",
                "",
                "| 范式 | 方法 | 时间窗 | 准确率 | ITR | 受试者数 | 公开表准确率 | 差值 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        best = summary.sort_values("accuracy").groupby(["paradigm", "method"], as_index=False).tail(1)
        for row in best.sort_values(["paradigm", "method"]).itertuples(index=False):
            official_acc = getattr(row, "official_accuracy", np.nan)
            delta = getattr(row, "accuracy_delta_vs_official", np.nan)
            lines.append(
                f"| {row.paradigm} | {row.method} | {row.window:g}s | {row.accuracy:.4f} | {row.itr:.2f} | "
                f"{row.subjects} | {official_acc:.4f} | {delta:+.4f} |"
            )
        lines.extend(
            [
                "",
                "## 输出文件",
                "",
                "- `trials.csv`",
                "- `predictions.csv`",
                "- `summary.csv`",
                "- `subject.csv`",
                "- `runtime.csv`",
                "- `unit_manifest.csv`",
                "- `official_comparison.csv`",
                "- `figures/accuracy_curve.png`",
                "- `figures/itr_curve.png`",
                "- `figures/accuracy_heatmap.png`",
            ]
        )
    (result_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    result_dir: Path,
    trial_rows: list[dict[str, object]],
    pred_rows: list[dict[str, object]],
    runtime_rows: list[dict[str, object]],
    unit_rows: list[dict[str, object]],
    official: pd.DataFrame,
    manifest: dict[str, object],
    complete: bool,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    trials = _dedupe(
        pd.DataFrame(trial_rows),
        ["method", "paradigm", "channel_set", "fbdcca_filter_backend", "window", "subject", "block"],
    )
    preds = _dedupe(
        pd.DataFrame(pred_rows),
        ["method", "paradigm", "channel_set", "fbdcca_filter_backend", "window", "subject", "block", "true"],
    )
    runtime = pd.DataFrame(runtime_rows)
    units = _dedupe(pd.DataFrame(unit_rows), ["subject", "paradigm", "channel_set", "fbdcca_filter_backend", "methods", "windows"])
    trials.to_csv(result_dir / "trials.csv", index=False)
    preds.to_csv(result_dir / "predictions.csv", index=False)
    runtime.to_csv(result_dir / "runtime.csv", index=False)
    units.to_csv(result_dir / "unit_manifest.csv", index=False)
    summary, subject = summarize(trials, official)
    summary.to_csv(result_dir / "summary.csv", index=False)
    subject.to_csv(result_dir / "subject.csv", index=False)
    official.to_csv(result_dir / "official_comparison.csv", index=False)
    if not preds.empty:
        conf_dir = result_dir / "confusions"
        conf_dir.mkdir(exist_ok=True)
        for (paradigm, method, window), group in preds.groupby(["paradigm", "method", "window"]):
            np.save(conf_dir / f"{paradigm}_{method}_w{float(window):g}.npy", confusion_counts(group))
    manifest["status"] = "complete" if complete else "partial"
    manifest["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["rows_written"] = int(len(trials))
    (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_outputs(summary, subject, official, result_dir)
    write_report(summary, result_dir, manifest)


def build_work(
    root: Path,
    subjects: list[int],
    paradigms: list[str],
    windows: list[float],
    methods: list[str],
    channel_set: str,
    done: dict[tuple[int, str, str, str, str], set[float]],
    include_fbdcca_bsv: bool,
    harmonics: int,
    trca_n_fbs: int,
    fbdcca_n_fbs: int,
    fbdcca_filter_backend: str,
    fbdcca_mne_n_jobs: int,
) -> list[dict[str, object]]:
    work = []
    for paradigm in paradigms:
        available = set(available_dual_alpha_subjects(root, paradigm))
        missing = [subject for subject in subjects if subject not in available]
        if missing:
            raise ValueError(f"Missing Dual-Alpha subjects for {paradigm}: {missing}")
        for subject in subjects:
            method_windows: dict[str, list[float]] = {}
            for method in methods:
                if method == "FBDCCA" and paradigm not in OFFICIAL_FBDCCA_PARADIGMS and not include_fbdcca_bsv:
                    continue
                key = (subject, paradigm, channel_set, method, fbdcca_filter_backend)
                missing_windows = [window for window in windows if window not in done.get(key, set())]
                if missing_windows:
                    method_windows[method] = missing_windows
            if method_windows:
                work.append(
                    {
                        "root": str(root),
                        "subject": subject,
                        "paradigm": paradigm,
                        "channel_set": channel_set,
                        "method_windows": method_windows,
                        "harmonics": harmonics,
                        "trca_n_fbs": trca_n_fbs,
                        "fbdcca_n_fbs": fbdcca_n_fbs,
                        "fbdcca_filter_backend": fbdcca_filter_backend,
                        "fbdcca_mne_n_jobs": fbdcca_mne_n_jobs,
                    }
                )
    return work


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dual-Alpha TRCA/ETRCA/FBDCCA baselines.")
    parser.add_argument("--root", type=Path, default=DUAL_ALPHA_ROOT)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "latest")
    parser.add_argument("--subjects", default="1")
    parser.add_argument("--paradigms", default="Checkerboard_Arrangment,Binocular_Vision,Binocular-Swap_Vision")
    parser.add_argument("--windows", default="0.2,0.4")
    parser.add_argument("--methods", default="ETRCA,FBDCCA")
    parser.add_argument("--channel-set", default="official", choices=["official", "all", "occipital9", "9ch"])
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--trca-n-fbs", type=int, default=7)
    parser.add_argument("--fbdcca-n-fbs", type=int, default=5)
    parser.add_argument("--fbdcca-filter-backend", default="mne-fir", choices=["mne-fir", "scipy-fir-legacy"])
    parser.add_argument("--fbdcca-mne-n-jobs", type=int, default=1)
    parser.add_argument("--include-fbdcca-bsv", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    subjects = parse_range(args.subjects)
    paradigms = parse_csv(args.paradigms, upper=False)
    unknown = [paradigm for paradigm in paradigms if paradigm not in DUAL_ALPHA_PARADIGMS]
    if unknown:
        raise ValueError(f"Unknown Dual-Alpha paradigms: {unknown}")
    windows = parse_windows(args.windows)
    methods = [normalize_method(method) for method in parse_csv(args.methods)]
    result_dir = args.out
    if "official" in result_dir.name.lower() and args.fbdcca_filter_backend != "mne-fir":
        raise ValueError(
            "Refusing to write a non-MNE FBDCCA backend into an official result directory. "
            "Use backend='mne-fir' for official runs or choose a legacy/diagnostic output directory."
        )

    if args.resume:
        trial_rows, pred_rows, runtime_rows, unit_rows = load_existing(result_dir)
        done = completed_windows_from_trials(trial_rows)
    else:
        trial_rows, pred_rows, runtime_rows, unit_rows, done = [], [], [], [], {}

    official = pd.concat([load_official_results(args.root, method) for method in sorted(set(methods))], ignore_index=True)

    manifest = {
        "dataset": "Dual-Alpha GigaDB 102557",
        "dataset_doi": "10.5524/102557",
        "paper_doi": "10.1093/gigascience/giae041",
        "method_source": "public Dual-Alpha source_code directory",
        "implementation_identity": {
            "ETRCA": "Arena adapter aligned to the public meegkit ensemble-TRCA call",
            "FBDCCA": "direct score/weight port; filtering backend recorded separately",
        },
        "evidence_role": "paper-and-public-code reproduction",
        "root": str(args.root),
        "subjects": subjects,
        "paradigms": paradigms,
        "windows": windows,
        "methods": methods,
        "channel_set": args.channel_set,
        "classes": DUAL_ALPHA_CLASSES,
        "blocks": DUAL_ALPHA_BLOCKS,
        "sampling_rate": DUAL_ALPHA_SAMPLING_RATE,
        "harmonics": args.harmonics,
        "trca_n_fbs": args.trca_n_fbs,
        "fbdcca_n_fbs": args.fbdcca_n_fbs,
        "fbdcca_filter_backend": args.fbdcca_filter_backend,
        "fbdcca_mne_n_jobs": args.fbdcca_mne_n_jobs,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "official_trca_ensemble": True,
        "protocol": "subject-specific 5-block leave-one-block-out",
        "preprocessing": "epoch CSV crop first samples; method-specific filterbank",
        "known_limitations": [
            (
                "The public FBDCCA code crops each epoch before MNE FIR filtering. "
                "At short windows MNE may warn that the designed FIR is longer than "
                "the cropped signal; Arena preserves that public-code order."
            )
        ],
        "itr_trial_seconds": f"window + {DUAL_ALPHA_ITR_SHIFT_SECONDS}",
        "workers": args.workers,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    work = build_work(
        root=args.root,
        subjects=subjects,
        paradigms=paradigms,
        windows=windows,
        methods=methods,
        channel_set=args.channel_set,
        done=done,
        include_fbdcca_bsv=args.include_fbdcca_bsv,
        harmonics=args.harmonics,
        trca_n_fbs=args.trca_n_fbs,
        fbdcca_n_fbs=args.fbdcca_n_fbs,
        fbdcca_filter_backend=args.fbdcca_filter_backend,
        fbdcca_mne_n_jobs=args.fbdcca_mne_n_jobs,
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    if args.workers > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_unit, item) for item in work]
            for future in as_completed(futures):
                result = future.result()
                trial_rows.extend(result["trial_rows"])
                pred_rows.extend(result["pred_rows"])
                runtime_rows.extend(result["runtime_rows"])
                unit_rows.extend(result["manifest_rows"])
                if not args.quiet:
                    for line in result.get("logs", []):
                        print(line, flush=True)
                completed += 1
                if completed % max(1, args.checkpoint_every) == 0:
                    write_outputs(result_dir, trial_rows, pred_rows, runtime_rows, unit_rows, official, manifest, complete=False)
                    print(f"checkpoint {completed}/{len(work)} -> {result_dir}", flush=True)
    else:
        for item in work:
            result = run_unit(item)
            trial_rows.extend(result["trial_rows"])
            pred_rows.extend(result["pred_rows"])
            runtime_rows.extend(result["runtime_rows"])
            unit_rows.extend(result["manifest_rows"])
            if not args.quiet:
                for line in result.get("logs", []):
                    print(line, flush=True)
            completed += 1
            if completed % max(1, args.checkpoint_every) == 0:
                write_outputs(result_dir, trial_rows, pred_rows, runtime_rows, unit_rows, official, manifest, complete=False)
                print(f"checkpoint {completed}/{len(work)} -> {result_dir}", flush=True)

    manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_outputs(result_dir, trial_rows, pred_rows, runtime_rows, unit_rows, official, manifest, complete=True)
    print(result_dir)


if __name__ == "__main__":
    main()
