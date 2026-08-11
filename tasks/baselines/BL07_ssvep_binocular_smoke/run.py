from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.binocular import (
    BINOCULAR_AR_ROOT,
    DUAL_ALPHA_OCCIPITAL9,
    DUAL_ALPHA_PARADIGMS,
    DUAL_ALPHA_ROOT,
    EpochSample,
    ar_epoch_file_status,
    available_ar_subjects,
    describe_epoch_sample,
    dual_alpha_sample,
    list_ar_tasks,
    list_dual_alpha_files,
    load_ar_epoch_sample,
    parse_dual_alpha_codebooks,
    preprocess_for_smoke,
    write_csv_rows,
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def plot_time_series(sample: EpochSample, filtered: np.ndarray, path: Path) -> None:
    raw = sample.x[0].astype(np.float64)
    filt = filtered[0].astype(np.float64)
    channels = sample.channels[: min(6, len(sample.channels))]
    idx = np.arange(len(channels))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for row, label, ax in ((raw, "raw", axes[0]), (filt, "notch + bandpass", axes[1])):
        span = np.nanpercentile(np.abs(row[: len(channels)]), 95)
        offset = max(span * 3.0, 1.0)
        for i, ch_idx in enumerate(idx):
            ax.plot(sample.time, row[ch_idx] + i * offset, lw=0.8, label=channels[i])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time from sample window start (s)")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    fig.suptitle(f"{sample.dataset} | {sample.paradigm} | sub-{sample.subject:03d} time series")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_psd(sample: EpochSample, filtered: np.ndarray, path: Path) -> dict[str, object]:
    raw_freqs, raw_psd = signal.welch(
        sample.x,
        fs=sample.sampling_rate,
        nperseg=min(512, sample.x.shape[-1]),
        axis=-1,
    )
    filt_freqs, filt_psd = signal.welch(
        filtered,
        fs=sample.sampling_rate,
        nperseg=min(512, filtered.shape[-1]),
        axis=-1,
    )
    raw_1d = np.mean(raw_psd, axis=tuple(range(raw_psd.ndim - 1)))
    filt_1d = np.mean(filt_psd, axis=tuple(range(filt_psd.ndim - 1)))
    freq_mask = (raw_freqs >= 1.0) & (raw_freqs <= min(90.0, sample.sampling_rate / 2))
    target_freqs = tuple(float(freq) for freq in sample.target_freqs[0])

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(raw_freqs[freq_mask], 10 * np.log10(raw_1d[freq_mask] + 1e-18), lw=1.0, label="raw")
    ax.plot(filt_freqs[freq_mask], 10 * np.log10(filt_1d[freq_mask] + 1e-18), lw=1.0, label="notch + bandpass")
    for freq in target_freqs:
        ax.axvline(freq, color="#d62728", lw=1.0, alpha=0.9)
        if 2 * freq < sample.sampling_rate / 2:
            ax.axvline(2 * freq, color="#ff7f0e", lw=0.8, alpha=0.45, ls="--")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title(f"{sample.dataset} | {sample.paradigm} | target freqs: {', '.join(f'{f:g}' for f in target_freqs)} Hz")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    peaks: list[dict[str, object]] = []
    for freq in target_freqs:
        idx = int(np.argmin(np.abs(filt_freqs - freq)))
        peaks.append({"target_freq": freq, "nearest_bin": float(filt_freqs[idx]), "filtered_psd_db": float(10 * np.log10(filt_1d[idx] + 1e-18))})
    return {"target_peaks": peaks}


def plot_dual_alpha_codebook(root: Path, out_path: Path) -> None:
    codebooks = parse_dual_alpha_codebooks(root)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    styles = {
        "Checkerboard_Arrangment": {"marker": "o", "facecolors": "none", "edgecolors": "#2ca02c", "linewidths": 1.2},
        "Binocular_Vision": {"marker": "s", "facecolors": "none", "edgecolors": "#ff7f0e", "linewidths": 1.2},
        "Binocular-Swap_Vision": {"marker": "^", "color": "#1f77b4"},
    }
    for paradigm, codebook in codebooks.items():
        ax.scatter(
            codebook.freq1,
            codebook.freq2,
            s=42,
            label=DUAL_ALPHA_PARADIGMS[paradigm]["display"],
            alpha=0.9,
            **styles.get(paradigm, {}),
        )
    ax.set_xlabel("Freq1 (Hz)")
    ax.set_ylabel("Freq2 (Hz)")
    ax.set_title("Dual-Alpha target frequency pairs")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_dual_alpha(args: argparse.Namespace, out_dir: Path) -> list[dict[str, object]]:
    root = Path(args.dual_alpha_root)
    figure_dir = ensure_dir(out_dir / "figures")
    summaries: list[dict[str, object]] = []

    write_csv_rows(out_dir / "dual_alpha_file_status.csv", list_dual_alpha_files(root))
    code_rows = []
    for paradigm, codebook in parse_dual_alpha_codebooks(root).items():
        code_rows.append(
            {
                "dataset": "dual_alpha",
                "paradigm": paradigm,
                "targets": codebook.n_targets,
                "freq1_min": min(codebook.freq1),
                "freq1_max": max(codebook.freq1),
                "freq2_min": min(codebook.freq2),
                "freq2_max": max(codebook.freq2),
                "unique_pairs": len(set(zip(codebook.freq1, codebook.freq2))),
            }
        )
    write_csv_rows(out_dir / "dual_alpha_codebook_summary.csv", code_rows)
    plot_dual_alpha_codebook(root, figure_dir / "dual_alpha_codebook_pairs.png")

    for paradigm in parse_csv(args.dual_alpha_paradigms):
        sample = dual_alpha_sample(
            root=root,
            paradigm=paradigm,
            subject=args.dual_alpha_subject,
            channels=DUAL_ALPHA_OCCIPITAL9,
            max_epochs=args.max_epochs,
            window_seconds=args.window,
        )
        filtered = preprocess_for_smoke(sample.x, sample.sampling_rate)
        base = f"dual_alpha_{paradigm}_sub{sample.subject:03d}"
        plot_time_series(sample, filtered, figure_dir / f"{base}_timeseries.png")
        peak_info = plot_psd(sample, filtered, figure_dir / f"{base}_psd.png")
        summary = describe_epoch_sample(sample)
        summary.update(peak_info)
        summaries.append(summary)
    return summaries


def run_binocular_ar(args: argparse.Namespace, out_dir: Path) -> list[dict[str, object]]:
    root = Path(args.binocular_ar_root)
    figure_dir = ensure_dir(out_dir / "figures")
    status_rows = [status.__dict__ for status in ar_epoch_file_status(root)]
    write_csv_rows(out_dir / "binocular_ar_file_status.csv", status_rows)

    available = available_ar_subjects(root)
    subject = args.ar_subject if args.ar_subject > 0 else (available[0] if available else 0)
    if subject <= 0:
        raise RuntimeError("No complete binocular AR subject zip is available for smoke.")

    task_rows = list_ar_tasks(root, subject)
    write_csv_rows(out_dir / f"binocular_ar_sub{subject:03d}_task_inventory.csv", task_rows)
    summaries: list[dict[str, object]] = []
    for task in parse_csv(args.ar_tasks):
        sample = load_ar_epoch_sample(
            root=root,
            subject=subject,
            task=task,
            session=args.ar_session,
            event_index=args.ar_event_index,
            window_seconds=args.ar_window,
        )
        filtered = preprocess_for_smoke(sample.x, sample.sampling_rate)
        base = f"binocular_ar_{task}_sub{sample.subject:03d}_{args.ar_session}"
        plot_time_series(sample, filtered, figure_dir / f"{base}_timeseries.png")
        peak_info = plot_psd(sample, filtered, figure_dir / f"{base}_psd.png")
        summary = describe_epoch_sample(sample)
        summary.update(peak_info)
        summaries.append(summary)
    return summaries


def write_report(out_dir: Path, summaries: list[dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# 双目 SSVEP 数据集 smoke 探查报告",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 输出目录：`{out_dir}`",
        f"- Dual-Alpha root：`{args.dual_alpha_root}`",
        f"- Binocular AR root：`{args.binocular_ar_root}`",
        "",
        "## 结论",
        "",
        "- Dual-Alpha 是 epoch 化 CSV，适合直接进入 40 类 block-CV 的 TRCA/ETRCA；双频参考算法需要读取 Freq1/Freq2。",
        "- Binocular AR 是 BIDS-like 连续 EEG + events；当前本地 epoch zip 已补齐，可进入正式 epoching 与分类任务。",
        "- smoke 图中红线是当前 trial 的目标频率，虚线为二次谐波；该图用于确认目标频段是否在 PSD 中可见。",
        "",
        "## 样本摘要",
        "",
    ]
    for item in summaries:
        lines.extend(
            [
                f"### {item['dataset']} / {item['paradigm']} / sub-{item['subject']:03d}",
                "",
                f"- shape：{item['trials']} trials × {item['channels']} channels × {item['samples']} samples",
                f"- fs：{item['sampling_rate']} Hz；window：{item['seconds']:.3f} s",
                f"- first target frequencies：{item['target_freqs_first']}",
                f"- mean/std/rms：{item['mean']:.4g} / {item['std']:.4g} / {item['rms']:.4g}",
                "",
            ]
        )
    (out_dir / "smoke_report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-probe binocular SSVEP datasets.")
    parser.add_argument("--dataset", choices=["both", "dual-alpha", "binocular-ar"], default="both")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "latest")
    parser.add_argument("--dual-alpha-root", default=str(DUAL_ALPHA_ROOT))
    parser.add_argument("--dual-alpha-subject", type=int, default=1)
    parser.add_argument("--dual-alpha-paradigms", default="Checkerboard_Arrangment,Binocular_Vision,Binocular-Swap_Vision")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--binocular-ar-root", default=str(BINOCULAR_AR_ROOT))
    parser.add_argument("--ar-subject", type=int, default=0, help="0 selects the first complete local subject.")
    parser.add_argument("--ar-session", default="ses-01")
    parser.add_argument("--ar-tasks", default="LF,DFDP,DFDP1")
    parser.add_argument("--ar-event-index", type=int, default=0)
    parser.add_argument("--ar-window", type=float, default=3.0)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    summaries: list[dict[str, object]] = []
    if args.dataset in {"both", "dual-alpha"}:
        summaries.extend(run_dual_alpha(args, out_dir))
    if args.dataset in {"both", "binocular-ar"}:
        summaries.extend(run_binocular_ar(args, out_dir))
    write_csv_rows(out_dir / "sample_summary.csv", [{k: v for k, v in item.items() if k != "metadata"} for item in summaries])
    save_json(out_dir / "sample_summary.json", summaries)
    save_json(out_dir / "run_manifest.json", {"args": vars(args), "created_at": datetime.now().isoformat(timespec="seconds")})
    write_report(out_dir, summaries, args)
    print(f"Wrote binocular smoke outputs to {out_dir}")


if __name__ == "__main__":
    main()
