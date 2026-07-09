from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run import (
    BASE_FREQS,
    DEFAULT_DATASET,
    FB_NUM,
    FS,
    LAG,
    LATENCY,
    RAW_REL,
    fit_tdca_single,
    matlab_data250hz,
    projection_matrices,
)
from run_offline_tdca_grid import CHANNEL_SETS, TARGET_SETS, load_or_create_filtered


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = TASK_DIR / "results" / "offline_feature_analysis_S1"
SPATIAL_LABELS = ["right", "down", "left", "up", "center"]
HARMONICS = (1, 2, 3)


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def stratified_target_conditions(freq_bins_per_slot: int) -> np.ndarray:
    if freq_bins_per_slot <= 0 or freq_bins_per_slot >= 40:
        return np.arange(200, dtype=np.int64)
    freq_bins = np.linspace(0, 39, freq_bins_per_slot, dtype=np.int64)
    conditions = []
    for freq_bin in freq_bins:
        for slot in range(5):
            conditions.append(freq_bin * 5 + slot)
    return np.asarray(conditions, dtype=np.int64)


def target_slot(original_target: int) -> int:
    return (original_target - 1) % 5


def target_freq_bin(original_target: int) -> int:
    return (original_target - 1) // 5


def final_tdca_scores(epoch: np.ndarray, model: dict[str, np.ndarray], time_samples: int, p_cond: np.ndarray) -> np.ndarray:
    n_chan, _, n_band = epoch.shape
    target_num = p_cond.shape[0]
    weights = np.asarray([(i + 1) ** (-1.25) + 0.25 for i in range(10)], dtype=np.float32)
    rr = np.zeros((n_band, target_num), dtype=np.float32)
    templates = model["templates"]
    filters = model["filters"]

    for fb in range(n_band):
        test_parts = []
        bpdata = epoch[:, : time_samples + LATENCY, fb]
        for lag_idx in range(1, LAG + 1):
            start = LATENCY + lag_idx - 1
            stop = time_samples + LATENCY
            part = bpdata[:, start:stop]
            if part.shape[1] < time_samples:
                part = np.pad(part, ((0, 0), (0, time_samples - part.shape[1])))
            test_parts.append(part)
        testdatah = np.concatenate(test_parts, axis=0)
        xh = testdatah.T
        filt = filters[:, :n_chan, fb]
        base_u = xh @ filt
        projected_x = np.einsum("cts,td->csd", p_cond, xh, optimize=True)
        projected_u = np.einsum("csd,dr->csr", projected_x, filt, optimize=True)
        u1 = np.concatenate(
            [np.broadcast_to(base_u, (target_num, base_u.shape[0], base_u.shape[1])), projected_u],
            axis=1,
        )
        tmpl_t = templates[..., fb].transpose(2, 1, 0)
        v1 = np.einsum("ctd,dr->ctr", tmpl_t, filt, optimize=True)
        numerator = np.sum(u1 * v1, axis=(1, 2))
        denominator = np.sqrt(np.sum(u1 * u1, axis=(1, 2)) * np.sum(v1 * v1, axis=(1, 2)))
        rr[fb] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)

    return weights[:n_band] @ (np.sign(rr) * np.abs(rr) ** 2)


def snr_at_frequency(power: np.ndarray, freqs: np.ndarray, freq: float) -> np.ndarray:
    peak_idx = int(np.argmin(np.abs(freqs - freq)))
    side = ((np.abs(freqs - freq) >= 0.5) & (np.abs(freqs - freq) <= 1.5))
    if not np.any(side):
        return np.full(power.shape[0], np.nan, dtype=np.float64)
    signal = power[:, peak_idx]
    noise = np.mean(power[:, side], axis=1)
    return 10 * np.log10(np.divide(signal, noise, out=np.full_like(signal, np.nan), where=noise > 0))


def compute_snr_tables(raw: np.ndarray, windows_ms: list[int], out_dir: Path, max_harmonic: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    nfft = 4096
    fft_freqs = np.fft.rfftfreq(nfft, 1 / FS)
    target_rows = []
    channel_rows = []
    channels = CHANNEL_SETS["66"]
    target_indices = TARGET_SETS["200"]["indices"]
    raw = raw[np.ix_(channels, np.arange(raw.shape[1]), target_indices, np.arange(raw.shape[3]))]

    for window_ms in windows_ms:
        time_samples = window_ms // 4
        segment = raw[:, LATENCY : LATENCY + time_samples, :, :]
        taper = np.hanning(time_samples).astype(np.float32)
        channel_accum = []
        for cond, original_idx in enumerate(target_indices):
            freq = BASE_FREQS[target_freq_bin(int(original_idx) + 1)]
            trial = segment[:, :, cond, :].astype(np.float32, copy=False)
            trial = trial - trial.mean(axis=1, keepdims=True)
            spec = np.fft.rfft(trial * taper[None, :, None], n=nfft, axis=1)
            power = np.mean(np.abs(spec) ** 2, axis=2)
            harmonic_snr = []
            used_harmonics = []
            for harmonic in HARMONICS[:max_harmonic]:
                harmonic_freq = float(freq * harmonic)
                if harmonic_freq >= FS / 2:
                    continue
                harmonic_snr.append(snr_at_frequency(power, fft_freqs, harmonic_freq))
                used_harmonics.append(harmonic)
            snr_by_channel = np.nanmean(np.vstack(harmonic_snr), axis=0)
            channel_accum.append(snr_by_channel)
            target_rows.append(
                {
                    "window_ms": window_ms,
                    "original_target": int(original_idx) + 1,
                    "freq_bin": target_freq_bin(int(original_idx) + 1) + 1,
                    "spatial_slot": SPATIAL_LABELS[target_slot(int(original_idx) + 1)],
                    "frequency_hz": float(freq),
                    "harmonics": "+".join(str(x) for x in used_harmonics),
                    "snr_db_mean66": float(np.nanmean(snr_by_channel)),
                    "snr_db_median66": float(np.nanmedian(snr_by_channel)),
                    "snr_db_best_channel": float(np.nanmax(snr_by_channel)),
                    "best_channel_1based": int(np.nanargmax(snr_by_channel)) + 1,
                }
            )
        channel_mat = np.vstack(channel_accum)
        for channel_idx in range(channel_mat.shape[1]):
            channel_rows.append(
                {
                    "window_ms": window_ms,
                    "channel_1based": channel_idx + 1,
                    "snr_db_mean_targets": float(np.nanmean(channel_mat[:, channel_idx])),
                    "snr_db_median_targets": float(np.nanmedian(channel_mat[:, channel_idx])),
                }
            )

    target_df = pd.DataFrame(target_rows)
    channel_df = pd.DataFrame(channel_rows)
    target_df.to_csv(out_dir / "snr_by_target_window.csv", index=False)
    channel_df.to_csv(out_dir / "snr_by_channel_window.csv", index=False)
    return target_df, channel_df


def compute_tdca_margin_table(
    dataset_root: Path,
    subject: str,
    windows_ms: list[int],
    heldout_blocks: list[int],
    out_dir: Path,
    freq_bins_per_slot: int,
) -> pd.DataFrame:
    filtered = load_or_create_filtered(dataset_root, subject)
    channels = CHANNEL_SETS["66"]
    target_indices = TARGET_SETS["200"]["indices"]
    data = np.asarray(
        filtered[np.ix_(channels, np.arange(filtered.shape[1]), target_indices, np.arange(18), np.arange(FB_NUM))],
        dtype=np.float32,
    )
    rows = []
    selected_conditions = stratified_target_conditions(freq_bins_per_slot)
    out_path = out_dir / "tdca_score_margins.csv"
    if out_path.exists():
        out_path.unlink()
    fields = [
        "subject",
        "window_ms",
        "heldout_block",
        "true",
        "pred",
        "original_true",
        "original_pred",
        "correct",
        "true_score",
        "best_wrong_score",
        "margin",
        "true_rank",
        "same_frequency_pred",
        "same_spatial_slot_pred",
    ]
    for window_ms in windows_ms:
        time_samples = window_ms // 4
        p_cond = projection_matrices(time_samples, np.repeat(BASE_FREQS, 5))
        for block_1based in heldout_blocks:
            cv = block_1based - 1
            train_blocks = [idx for idx in range(18) if idx != cv]
            model = fit_tdca_single(data[:, :, :, train_blocks, :], time_samples, p_cond)
            datatest = data[:, :, :, cv, :]
            batch_rows = []
            for cond in selected_conditions:
                original_idx = target_indices[cond]
                scores = final_tdca_scores(datatest[:, :, cond, :], model, time_samples, p_cond)
                order = np.argsort(scores)[::-1]
                pred = int(order[0])
                true_score = float(scores[cond])
                wrong_scores = np.delete(scores, cond)
                best_wrong = float(np.max(wrong_scores))
                true_rank = int(np.where(order == cond)[0][0]) + 1
                batch_rows.append(
                    {
                        "subject": subject,
                        "window_ms": window_ms,
                        "heldout_block": block_1based,
                        "true": cond + 1,
                        "pred": pred + 1,
                        "original_true": int(original_idx) + 1,
                        "original_pred": int(target_indices[pred]) + 1,
                        "correct": int(pred == cond),
                        "true_score": true_score,
                        "best_wrong_score": best_wrong,
                        "margin": true_score - best_wrong,
                        "true_rank": true_rank,
                        "same_frequency_pred": int(target_freq_bin(int(original_idx) + 1) == target_freq_bin(int(target_indices[pred]) + 1)),
                        "same_spatial_slot_pred": int(target_slot(int(original_idx) + 1) == target_slot(int(target_indices[pred]) + 1)),
                    }
                )
            rows.extend(batch_rows)
            with out_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerows(batch_rows)
            print(f"{subject} {window_ms}ms heldout_block={block_1based} margins done", flush=True)
    df = pd.DataFrame(rows)
    return df


def plot_snr_summary(target_df: pd.DataFrame, channel_df: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_window = target_df.groupby("window_ms")["snr_db_mean66"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.errorbar(by_window["window_ms"], by_window["mean"], yerr=by_window["sem"], marker="o", linewidth=2)
    ax.set_xlabel("window (ms)")
    ax.set_ylabel("mean target SNR (dB)")
    ax.set_title("S1 200-target SNR vs Window")
    ax.grid(alpha=0.25)
    fig.savefig(fig_dir / "snr_vs_window.png", dpi=180)
    plt.close(fig)

    pivot = target_df.pivot_table(index="freq_bin", columns="spatial_slot", values="snr_db_mean66", aggfunc="mean")
    pivot = pivot.reindex(columns=SPATIAL_LABELS)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(SPATIAL_LABELS)), SPATIAL_LABELS, rotation=30, ha="right")
    ax.set_yticks(np.arange(0, 40, 5), np.arange(1, 41, 5))
    ax.set_xlabel("spatial slot")
    ax.set_ylabel("frequency bin")
    ax.set_title("Mean SNR by Frequency Bin and Spatial Slot")
    fig.colorbar(im, ax=ax, label="SNR (dB)")
    fig.savefig(fig_dir / "snr_frequency_space_heatmap.png", dpi=180)
    plt.close(fig)

    best_window = int(by_window.sort_values("mean").iloc[-1]["window_ms"])
    chan = channel_df[channel_df["window_ms"] == best_window].sort_values("channel_1based")
    values = chan["snr_db_mean_targets"].to_numpy()
    side = int(math.ceil(math.sqrt(len(values))))
    grid = np.full((side, side), np.nan)
    grid.ravel()[: len(values)] = values
    fig, ax = plt.subplots(figsize=(6.2, 5.5), constrained_layout=True)
    im = ax.imshow(grid, cmap="magma")
    for idx, value in enumerate(values):
        y, x = divmod(idx, side)
        ax.text(x, y, str(idx + 1), ha="center", va="center", fontsize=6, color="white" if value < np.nanmean(values) else "black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Channel SNR Map, {best_window} ms")
    fig.colorbar(im, ax=ax, label="mean SNR (dB)")
    fig.savefig(fig_dir / "snr_channel_index_map.png", dpi=180)
    plt.close(fig)


def plot_margin_summary(margins: pd.DataFrame, snr_targets: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    by_window = margins.groupby("window_ms").agg(accuracy=("correct", "mean"), margin=("margin", "mean"), rank=("true_rank", "mean")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].plot(by_window["window_ms"], by_window["accuracy"], marker="o", linewidth=2)
    axes[0].set_ylabel("accuracy")
    axes[0].set_xlabel("window (ms)")
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Subset Accuracy")
    axes[1].plot(by_window["window_ms"], by_window["margin"], marker="o", linewidth=2, color="#7768ae")
    axes[1].set_ylabel("true - best wrong score")
    axes[1].set_xlabel("window (ms)")
    axes[1].grid(alpha=0.25)
    axes[1].set_title("TDCA Score Margin")
    axes[2].plot(by_window["window_ms"], by_window["rank"], marker="o", linewidth=2, color="#40798c")
    axes[2].set_ylabel("mean true rank")
    axes[2].set_xlabel("window (ms)")
    axes[2].grid(alpha=0.25)
    axes[2].set_title("True-Class Rank")
    fig.savefig(fig_dir / "tdca_margin_vs_window.png", dpi=180)
    plt.close(fig)

    merged = margins.merge(
        snr_targets[["window_ms", "original_target", "snr_db_mean66"]],
        left_on=["window_ms", "original_true"],
        right_on=["window_ms", "original_target"],
        how="left",
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    ax.scatter(merged["snr_db_mean66"], merged["margin"], s=12, alpha=0.45, c=merged["correct"], cmap="coolwarm")
    corr = merged[["snr_db_mean66", "margin"]].corr().iloc[0, 1]
    ax.set_xlabel("target SNR (dB)")
    ax.set_ylabel("TDCA score margin")
    ax.set_title(f"SNR vs TDCA Margin (r={corr:.2f})")
    ax.grid(alpha=0.25)
    fig.savefig(fig_dir / "snr_vs_tdca_margin.png", dpi=180)
    plt.close(fig)

    err = margins[margins["correct"] == 0].copy()
    err_summary = []
    for window_ms, group in err.groupby("window_ms"):
        err_summary.append(
            {
                "window_ms": int(window_ms),
                "errors": int(len(group)),
                "same_frequency_error_rate": float(group["same_frequency_pred"].mean()) if len(group) else 0.0,
                "same_spatial_slot_error_rate": float(group["same_spatial_slot_pred"].mean()) if len(group) else 0.0,
            }
        )
    err_df = pd.DataFrame(err_summary)
    err_df.to_csv(out_dir / "tdca_margin_error_types.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
    if not err_df.empty:
        ax.plot(err_df["window_ms"], err_df["same_frequency_error_rate"], marker="o", label="same frequency")
        ax.plot(err_df["window_ms"], err_df["same_spatial_slot_error_rate"], marker="o", label="same spatial slot")
    ax.set_xlabel("window (ms)")
    ax.set_ylabel("share of errors")
    ax.set_title("TDCA Error Type by Window")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "tdca_error_type_by_window.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--windows-ms", default="100,200,300,400,500")
    parser.add_argument("--heldout-blocks", default="1")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--max-harmonic", type=int, default=3)
    parser.add_argument("--margin-freq-bins-per-slot", type=int, default=8)
    parser.add_argument("--skip-tdca-margins", action="store_true")
    args = parser.parse_args()

    windows_ms = parse_int_csv(args.windows_ms)
    heldout_blocks = parse_int_csv(args.heldout_blocks)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = args.dataset_root / RAW_REL / "data" / "offline" / f"{args.subject}.mat"
    raw = matlab_data250hz(raw_path).astype(np.float32, copy=False)
    snr_targets, snr_channels = compute_snr_tables(raw, windows_ms, args.out_dir, args.max_harmonic)
    plot_snr_summary(snr_targets, snr_channels, args.out_dir)

    manifest = {
        "subject": args.subject,
        "dataset_root": str(args.dataset_root),
        "raw_path": str(raw_path),
        "target_set": 200,
        "channel_set": 66,
        "windows_ms": windows_ms,
        "heldout_blocks": heldout_blocks,
        "margin_freq_bins_per_slot": args.margin_freq_bins_per_slot,
        "snr": "zero-padded FFT after 140 ms latency; harmonic peak divided by local sideband mean",
        "tdca_margins": "final 5-filter-bank weighted TDCA score; true score minus best wrong score",
    }

    if not args.skip_tdca_margins:
        margins = compute_tdca_margin_table(
            args.dataset_root,
            args.subject,
            windows_ms,
            heldout_blocks,
            args.out_dir,
            args.margin_freq_bins_per_slot,
        )
        plot_margin_summary(margins, snr_targets, args.out_dir)
        manifest["tdca_margin_rows"] = int(len(margins))

    manifest["snr_target_rows"] = int(len(snr_targets))
    manifest["snr_channel_rows"] = int(len(snr_channels))
    (args.out_dir / "feature_analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.out_dir)


if __name__ == "__main__":
    main()
