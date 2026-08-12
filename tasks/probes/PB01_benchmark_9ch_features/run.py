from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vep_arena.config import BENCHMARK_CHANNELS_9, BENCHMARK_FREQS, DATA_ROOT, PROJECT_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import load_subject_filterbank, load_subject_trials
from vep_arena.methods.traditional import FBCCA


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = TASK_DIR / "results" / "feature_probe"
CHANNEL_NAMES_9 = ("Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2")
CHANNEL_GRID = np.array(
    [
        [None, "Pz", None],
        ["PO5", "PO3", "POz"],
        ["PO6", "PO4", "Oz"],
        ["O1", None, "O2"],
    ],
    dtype=object,
)


@dataclass(frozen=True)
class ProbeConfig:
    subjects: tuple[int, ...]
    targets: tuple[int, ...]
    window: float
    psd_min_hz: float
    psd_max_hz: float
    peak_band_hz: float
    noise_inner_hz: float
    noise_outer_hz: float
    harmonics: tuple[int, ...]
    cwt_min_hz: float
    cwt_max_hz: float
    cwt_points: int
    morlet_w: float
    fbcca_harmonics: int
    fbcca_bands: int


def parse_int_set(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return tuple(dict.fromkeys(values))


def ensure_dirs(out_dir: Path) -> Path:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save(fig: plt.Figure, fig_dir: Path, name: str) -> None:
    fig.savefig(fig_dir / f"{name}.png", dpi=190)
    fig.savefig(fig_dir / f"{name}.svg")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def selected_epochs(subjects: tuple[int, ...], targets: tuple[int, ...], window: float, spec: BenchmarkSpec) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    target_idx = np.asarray([t - 1 for t in targets], dtype=np.int64)
    for subject in subjects:
        epochs = load_subject_trials(DATA_ROOT, subject, window, BENCHMARK_CHANNELS_9, spec)
        out[subject] = epochs[target_idx]
    return out


def selected_filterbank_epochs(
    subjects: tuple[int, ...],
    targets: tuple[int, ...],
    window: float,
    n_fbs: int,
    spec: BenchmarkSpec,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    target_idx = np.asarray([t - 1 for t in targets], dtype=np.int64)
    for subject in subjects:
        epochs = load_subject_filterbank(DATA_ROOT, subject, window, n_fbs, BENCHMARK_CHANNELS_9, spec)
        out[subject] = epochs[target_idx]
    return out


def demean_trials(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def welch_epochs(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    x = demean_trials(x)
    nperseg = min(x.shape[-1], 512)
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd


def peak_power(freqs: np.ndarray, psd: np.ndarray, freq: float, band: float) -> float:
    mask = np.abs(freqs - freq) <= band
    if not np.any(mask):
        return float("nan")
    return float(np.nanmax(psd[..., mask]))


def harmonic_snr_db(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq: float,
    peak_band: float,
    noise_inner: float,
    noise_outer: float,
) -> float:
    peak_mask = np.abs(freqs - freq) <= peak_band
    noise_mask = (np.abs(freqs - freq) >= noise_inner) & (np.abs(freqs - freq) <= noise_outer)
    if not np.any(peak_mask) or not np.any(noise_mask):
        return float("nan")
    peak = float(np.nanmax(psd[..., peak_mask]))
    noise = float(np.nanmean(psd[..., noise_mask]))
    if noise <= 0 or not np.isfinite(noise):
        return float("nan")
    return float(10 * np.log10(max(peak, 1e-30) / noise))


def compute_feature_tables(
    epochs_by_subject: dict[int, np.ndarray],
    targets: tuple[int, ...],
    cfg: ProbeConfig,
    spec: BenchmarkSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]]:
    snr_rows: list[dict[str, object]] = []
    harmonic_rows: list[dict[str, object]] = []
    psd_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    fs = spec.sampling_rate
    for subject, epochs in epochs_by_subject.items():
        # targets x blocks x channels x samples
        freqs, psd = welch_epochs(epochs, fs)
        for target_pos, target in enumerate(targets):
            f0 = float(BENCHMARK_FREQS[target - 1])
            psd_target = psd[target_pos]
            psd_cache[(subject, target)] = (freqs, psd_target)
            for channel_pos, channel_name in enumerate(CHANNEL_NAMES_9):
                psd_tc = psd_target[:, channel_pos, :]
                harmonic_snrs: list[float] = []
                for harmonic in cfg.harmonics:
                    hf = f0 * harmonic
                    if hf >= fs / 2:
                        continue
                    h_power = peak_power(freqs, psd_tc, hf, cfg.peak_band_hz)
                    h_snr = harmonic_snr_db(freqs, psd_tc, hf, cfg.peak_band_hz, cfg.noise_inner_hz, cfg.noise_outer_hz)
                    harmonic_rows.append(
                        {
                            "subject": subject,
                            "target": target,
                            "frequency_hz": f0,
                            "channel": channel_name,
                            "harmonic": harmonic,
                            "harmonic_frequency_hz": hf,
                            "peak_power": h_power,
                            "snr_db": h_snr,
                        }
                    )
                    if np.isfinite(h_snr):
                        harmonic_snrs.append(h_snr)
                snr_rows.append(
                    {
                        "subject": subject,
                        "target": target,
                        "frequency_hz": f0,
                        "channel": channel_name,
                        "snr_db_mean_harmonics": float(np.nanmean(harmonic_snrs)) if harmonic_snrs else float("nan"),
                        "snr_db_best_harmonic": float(np.nanmax(harmonic_snrs)) if harmonic_snrs else float("nan"),
                    }
                )
    snr_channel = pd.DataFrame(snr_rows)
    harmonic = pd.DataFrame(harmonic_rows)
    snr_target = (
        snr_channel.groupby(["subject", "target", "frequency_hz"], as_index=False)
        .agg(
            snr_db_mean_channels=("snr_db_mean_harmonics", "mean"),
            snr_db_median_channels=("snr_db_mean_harmonics", "median"),
            snr_db_best_channel=("snr_db_mean_harmonics", "max"),
            best_channel=("channel", lambda s: str(s.iloc[int(np.nanargmax(snr_channel.loc[s.index, "snr_db_mean_harmonics"].to_numpy()))])),
        )
    )
    return snr_channel, snr_target, harmonic, psd_cache


def plot_time_evoked_grid(
    epochs_by_subject: dict[int, np.ndarray],
    targets: tuple[int, ...],
    fig_dir: Path,
    spec: BenchmarkSpec,
    channel_name: str = "Oz",
) -> None:
    channel_idx = CHANNEL_NAMES_9.index(channel_name)
    max_targets = min(12, len(targets))
    cols = 4
    rows = int(math.ceil(max_targets / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.0 * rows), sharex=True, sharey=True, constrained_layout=True)
    axes = np.asarray(axes).ravel()
    t = np.arange(round(spec.sampling_rate * next(iter(epochs_by_subject.values())).shape[-1] / spec.sampling_rate)) / spec.sampling_rate
    for ax, target_pos in zip(axes, range(max_targets)):
        target = targets[target_pos]
        chunks = []
        for epochs in epochs_by_subject.values():
            chunks.append(epochs[target_pos, :, channel_idx, :])
        trials = demean_trials(np.concatenate(chunks, axis=0))
        for row in trials:
            ax.plot(t, row, color="#9aa4b2", alpha=0.35, linewidth=0.7)
        ax.plot(t, trials.mean(axis=0), color="#174a7c", linewidth=1.9)
        ax.set_title(f"T{target:02d} {BENCHMARK_FREQS[target - 1]:.1f} Hz", fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes[max_targets:]:
        ax.axis("off")
    for ax in axes[-cols:]:
        ax.set_xlabel("time after crop (s)")
    for ax in axes[::cols]:
        ax.set_ylabel(f"{channel_name}, demeaned")
    save(fig, fig_dir, "target_time_evoked_grid")


def plot_psd_grid(
    psd_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    targets: tuple[int, ...],
    subjects: tuple[int, ...],
    cfg: ProbeConfig,
    fig_dir: Path,
    channel_name: str = "Oz",
) -> None:
    channel_idx = CHANNEL_NAMES_9.index(channel_name)
    max_targets = min(12, len(targets))
    cols = 4
    rows = int(math.ceil(max_targets / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.0 * rows), sharex=True, constrained_layout=True)
    axes = np.asarray(axes).ravel()
    for ax, target in zip(axes, targets[:max_targets]):
        subject_psds = []
        freqs = None
        for subject in subjects:
            freqs, psd_target = psd_cache[(subject, target)]
            subject_psds.append(psd_target[:, channel_idx, :].mean(axis=0))
        mean_psd = np.mean(subject_psds, axis=0)
        mask = (freqs >= cfg.psd_min_hz) & (freqs <= cfg.psd_max_hz)
        ax.semilogy(freqs[mask], mean_psd[mask], color="#174a7c", linewidth=1.8)
        f0 = BENCHMARK_FREQS[target - 1]
        for harmonic in cfg.harmonics:
            hf = f0 * harmonic
            if cfg.psd_min_hz <= hf <= cfg.psd_max_hz:
                ax.axvline(hf, color="#b45309", linestyle="--", alpha=0.65, linewidth=0.9)
        ax.set_title(f"T{target:02d} {f0:.1f} Hz", fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes[max_targets:]:
        ax.axis("off")
    for ax in axes[-cols:]:
        ax.set_xlabel("frequency (Hz)")
    for ax in axes[::cols]:
        ax.set_ylabel("PSD")
    save(fig, fig_dir, "target_psd_grid")


def plot_snr_summary(snr_channel: pd.DataFrame, snr_target: pd.DataFrame, targets: tuple[int, ...], fig_dir: Path) -> None:
    pivot = snr_target.pivot_table(index="subject", columns="target", values="snr_db_mean_channels", aggfunc="mean")
    pivot = pivot.reindex(columns=list(targets))
    fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), [str(x) for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [f"S{x}" for x in pivot.index])
    ax.set_xlabel("target")
    ax.set_ylabel("subject")
    ax.set_title("Mean Harmonic SNR by Subject and Target")
    fig.colorbar(im, ax=ax, label="SNR (dB)")
    save(fig, fig_dir, "snr_target_heatmap")

    chan = snr_channel.groupby("channel", as_index=False)["snr_db_mean_harmonics"].agg(["mean", "sem"]).reset_index()
    chan["channel"] = pd.Categorical(chan["channel"], categories=list(CHANNEL_NAMES_9), ordered=True)
    chan = chan.sort_values("channel")
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    ax.bar(chan["channel"].astype(str), chan["mean"], yerr=chan["sem"], color="#3f6f8f", edgecolor="#1f2937", linewidth=0.6)
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Channel Profile, Mean Harmonic SNR")
    ax.grid(axis="y", alpha=0.25)
    save(fig, fig_dir, "snr_channel_profile")

    channel_mean = snr_channel.groupby("channel")["snr_db_mean_harmonics"].mean()
    grid = np.full(CHANNEL_GRID.shape, np.nan, dtype=float)
    labels = np.empty(CHANNEL_GRID.shape, dtype=object)
    for y in range(CHANNEL_GRID.shape[0]):
        for x in range(CHANNEL_GRID.shape[1]):
            name = CHANNEL_GRID[y, x]
            labels[y, x] = "" if name is None else name
            if name is not None and name in channel_mean:
                grid[y, x] = float(channel_mean[name])
    fig, ax = plt.subplots(figsize=(5.6, 6.2), constrained_layout=True)
    im = ax.imshow(grid, cmap="magma")
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if labels[y, x]:
                ax.text(x, y, labels[y, x], ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Occipital 9ch SNR Grid")
    fig.colorbar(im, ax=ax, label="SNR (dB)")
    save(fig, fig_dir, "occipital_grid_snr")


def morlet2_kernel(length: int, width_samples: float, w: float) -> np.ndarray:
    x = np.arange(length, dtype=np.float64) - (length - 1) / 2
    scaled = x / width_samples
    return (np.pi ** -0.25) * np.sqrt(1 / width_samples) * np.exp(1j * w * scaled) * np.exp(-0.5 * scaled**2)


def morlet_cwt(x: np.ndarray, freqs: np.ndarray, fs: int, w: float) -> np.ndarray:
    rows = []
    for freq in freqs:
        width = w * fs / (2 * np.pi * freq)
        length = int(max(31, min(len(x), math.ceil(width * 10))))
        if length % 2 == 0:
            length += 1
        kernel = morlet2_kernel(length, width, w)
        conv = signal.fftconvolve(x, np.conj(kernel[::-1]), mode="same")
        rows.append(conv)
    return np.asarray(rows)


def morlet_cwt_average(epochs_by_subject: dict[int, np.ndarray], cfg: ProbeConfig, spec: BenchmarkSpec, target_pos: int, channel_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    channel_idx = CHANNEL_NAMES_9.index(channel_name)
    trials = []
    for epochs in epochs_by_subject.values():
        trials.append(epochs[target_pos, :, channel_idx, :])
    x = demean_trials(np.concatenate(trials, axis=0)).mean(axis=0)
    freqs = np.linspace(cfg.cwt_min_hz, cfg.cwt_max_hz, cfg.cwt_points)
    cwt = morlet_cwt(x, freqs, spec.sampling_rate, cfg.morlet_w)
    power_db = 20 * np.log10(np.abs(cwt) + 1e-12)
    t = np.arange(x.shape[-1]) / spec.sampling_rate
    return t, freqs, power_db


def plot_cwt(epochs_by_subject: dict[int, np.ndarray], targets: tuple[int, ...], cfg: ProbeConfig, spec: BenchmarkSpec, fig_dir: Path) -> None:
    t, freqs, power_db = morlet_cwt_average(epochs_by_subject, cfg, spec, 0, "Oz")
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    im = ax.pcolormesh(t, freqs, power_db, shading="auto", cmap="magma")
    f0 = BENCHMARK_FREQS[targets[0] - 1]
    for harmonic in cfg.harmonics:
        hf = f0 * harmonic
        if freqs[0] <= hf <= freqs[-1]:
            ax.axhline(hf, color="cyan", linestyle="--", linewidth=0.9, alpha=0.75)
    ax.set_xlabel("time after crop (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"Morlet CWT, Oz, T{targets[0]:02d} ({f0:.1f} Hz)")
    fig.colorbar(im, ax=ax, label="amplitude (dB)")
    save(fig, fig_dir, "cwt_target01_oz")


def run_fbcca_subset(
    fb_epochs_by_subject: dict[int, np.ndarray],
    subjects: tuple[int, ...],
    targets: tuple[int, ...],
    cfg: ProbeConfig,
    spec: BenchmarkSpec,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = np.arange(len(targets), dtype=np.int64)
    target_zero = np.asarray([t - 1 for t in targets], dtype=np.int64)
    frequencies = tuple(float(BENCHMARK_FREQS[idx]) for idx in target_zero)
    phases = tuple(0.0 for _ in target_zero)
    for subject in subjects:
        fb_epochs = fb_epochs_by_subject[subject]
        for block in range(spec.blocks):
            train_blocks = [idx for idx in range(spec.blocks) if idx != block]
            train_x = fb_epochs[:, train_blocks].reshape(-1, fb_epochs.shape[2], fb_epochs.shape[3], fb_epochs.shape[4])
            train_y = np.repeat(labels, len(train_blocks))
            test_x = fb_epochs[:, block]
            model = FBCCA(
                window=cfg.window,
                harmonics=cfg.fbcca_harmonics,
                n_fbs=cfg.fbcca_bands,
                spec=spec,
                frequencies=frequencies,
                phases_pi=phases,
            )
            model.fit(train_x, train_y)
            pred, score = model.predict(test_x)
            for pos, target in enumerate(targets):
                true = int(labels[pos])
                rows.append(
                    {
                        "subject": subject,
                        "target": target,
                        "block": block + 1,
                        "true_index": true,
                        "pred_index": int(pred[pos]),
                        "pred_target": int(targets[int(pred[pos])]),
                        "correct": int(pred[pos] == true),
                        "score_true": float(score[pos, true]),
                        "score_best": float(np.max(score[pos])),
                        "score_margin": float(score[pos, true] - np.max(np.delete(score[pos], true))),
                    }
                )
    return pd.DataFrame(rows)


def plot_feature_accuracy(snr_target: pd.DataFrame, preds: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    acc = preds.groupby(["subject", "target"], as_index=False).agg(
        fbcca_accuracy=("correct", "mean"),
        fbcca_margin=("score_margin", "mean"),
    )
    merged = acc.merge(snr_target, on=["subject", "target"], how="left")
    merged.to_csv(fig_dir.parent / "feature_vs_accuracy.csv", index=False)
    corr_acc = merged[["snr_db_mean_channels", "fbcca_accuracy"]].corr().iloc[0, 1]
    corr_margin = merged[["snr_db_mean_channels", "fbcca_margin"]].corr().iloc[0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    axes[0].scatter(merged["snr_db_mean_channels"], merged["fbcca_accuracy"], c=merged["frequency_hz"], cmap="viridis", s=48, alpha=0.85)
    axes[0].set_xlabel("mean harmonic SNR (dB)")
    axes[0].set_ylabel("FBCCA subset accuracy")
    axes[0].set_title(f"SNR vs Accuracy (r={corr_acc:.2f})")
    axes[0].grid(alpha=0.25)
    sc = axes[1].scatter(merged["snr_db_mean_channels"], merged["fbcca_margin"], c=merged["frequency_hz"], cmap="viridis", s=48, alpha=0.85)
    axes[1].set_xlabel("mean harmonic SNR (dB)")
    axes[1].set_ylabel("FBCCA score margin")
    axes[1].set_title(f"SNR vs Margin (r={corr_margin:.2f})")
    axes[1].grid(alpha=0.25)
    fig.colorbar(sc, ax=axes, label="target frequency (Hz)", shrink=0.85)
    save(fig, fig_dir, "feature_vs_accuracy")
    return merged


def try_mne_topomap(epochs_by_subject: dict[int, np.ndarray], targets: tuple[int, ...], fig_dir: Path, spec: BenchmarkSpec) -> dict[str, object]:
    status: dict[str, object] = {"enabled": False, "figures": []}
    try:
        import mne
    except Exception as exc:
        status["error"] = f"MNE import failed: {exc}"
        return status

    try:
        target_pos = 0
        data = []
        for subject, epochs in epochs_by_subject.items():
            evoked = epochs[target_pos].mean(axis=0)
            data.append(evoked)
        mean_evoked = np.mean(np.stack(data, axis=0), axis=0)
        info = mne.create_info(list(CHANNEL_NAMES_9), sfreq=spec.sampling_rate, ch_types="eeg")
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing="ignore")
        evoked = mne.EvokedArray(mean_evoked * 1e-6, info, tmin=0.0)
        times = [0.1, 0.3, 0.5, 1.0]
        fig = evoked.plot_topomap(times=times, show=False, colorbar=True, scalings=1)
        fig.savefig(fig_dir / "mne_topomap_target01_evoked.png", dpi=190)
        plt.close(fig)
        status["enabled"] = True
        status["target"] = int(targets[target_pos])
        status["figures"] = ["mne_topomap_target01_evoked.png"]
    except Exception as exc:
        status["error"] = f"MNE topomap failed: {exc}"
    return status


def write_report(out_dir: Path, cfg: ProbeConfig, feature_acc: pd.DataFrame, mne_status: dict[str, object]) -> None:
    best = feature_acc.sort_values("snr_db_mean_channels", ascending=False).head(5)
    worst = feature_acc.sort_values("snr_db_mean_channels", ascending=True).head(5)
    lines = [
        "# Feature Probe Report",
        "",
        "This exploratory run summarizes a small Benchmark 9ch subset. It is intended to guide which feature views should become part of a larger VEP benchmark ledger.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(cfg), indent=2),
        "```",
        "",
        "## MNE Status",
        "",
        "```json",
        json.dumps(mne_status, indent=2),
        "```",
        "",
        "## Highest Mean Harmonic SNR Rows",
        "",
        markdown_table(best[["subject", "target", "frequency_hz", "snr_db_mean_channels", "fbcca_accuracy", "fbcca_margin"]]),
        "",
        "## Lowest Mean Harmonic SNR Rows",
        "",
        markdown_table(worst[["subject", "target", "frequency_hz", "snr_db_mean_channels", "fbcca_accuracy", "fbcca_margin"]]),
        "",
        "## Figures",
        "",
        "- `figures/target_time_evoked_grid.png`",
        "- `figures/target_psd_grid.png`",
        "- `figures/snr_target_heatmap.png`",
        "- `figures/snr_channel_profile.png`",
        "- `figures/occipital_grid_snr.png`",
        "- `figures/cwt_target01_oz.png`",
        "- `figures/feature_vs_accuracy.png`",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1,2,3")
    parser.add_argument("--targets", default="1-8,17-20")
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-mne", action="store_true")
    args = parser.parse_args()

    cfg = ProbeConfig(
        subjects=parse_int_set(args.subjects),
        targets=parse_int_set(args.targets),
        window=args.window,
        psd_min_hz=4.0,
        psd_max_hz=60.0,
        peak_band_hz=0.25,
        noise_inner_hz=0.75,
        noise_outer_hz=1.75,
        harmonics=(1, 2, 3),
        cwt_min_hz=4.0,
        cwt_max_hz=60.0,
        cwt_points=96,
        morlet_w=6.0,
        fbcca_harmonics=5,
        fbcca_bands=5,
    )
    spec = BenchmarkSpec()
    fig_dir = ensure_dirs(args.output_dir)

    epochs = selected_epochs(cfg.subjects, cfg.targets, cfg.window, spec)
    snr_channel, snr_target, harmonic, psd_cache = compute_feature_tables(epochs, cfg.targets, cfg, spec)
    snr_channel.to_csv(args.output_dir / "snr_by_subject_target_channel.csv", index=False)
    snr_target.to_csv(args.output_dir / "snr_by_subject_target.csv", index=False)
    harmonic.to_csv(args.output_dir / "harmonic_power.csv", index=False)

    plot_time_evoked_grid(epochs, cfg.targets, fig_dir, spec)
    plot_psd_grid(psd_cache, cfg.targets, cfg.subjects, cfg, fig_dir)
    plot_snr_summary(snr_channel, snr_target, cfg.targets, fig_dir)
    plot_cwt(epochs, cfg.targets, cfg, spec, fig_dir)

    fb_epochs = selected_filterbank_epochs(cfg.subjects, cfg.targets, cfg.window, cfg.fbcca_bands, spec)
    preds = run_fbcca_subset(fb_epochs, cfg.subjects, cfg.targets, cfg, spec)
    preds.to_csv(args.output_dir / "fbcca_selected_predictions.csv", index=False)
    feature_acc = plot_feature_accuracy(snr_target, preds, fig_dir)

    mne_status = {"enabled": False, "skipped": True}
    if not args.skip_mne:
        mne_status = try_mne_topomap(epochs, cfg.targets, fig_dir, spec)

    manifest = {
        "dataset": "Tsinghua Benchmark SSVEP",
        "task": "ssvep_benchmark_9ch_feature_analysis",
        "config": asdict(cfg),
        "data_root": str(DATA_ROOT),
        "output_dir": str(args.output_dir),
        "channels": list(CHANNEL_NAMES_9),
        "channel_indices_1based": list(BENCHMARK_CHANNELS_9),
        "sampling_rate": spec.sampling_rate,
        "crop_start_seconds": spec.cue_seconds + spec.latency_seconds,
        "mne": mne_status,
        "artifacts": [
            "snr_by_subject_target_channel.csv",
            "snr_by_subject_target.csv",
            "harmonic_power.csv",
            "fbcca_selected_predictions.csv",
            "feature_vs_accuracy.csv",
            "report.md",
            "figures/",
        ],
    }
    (args.output_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(args.output_dir, cfg, feature_acc, mne_status)
    print(args.output_dir)


if __name__ == "__main__":
    main()
