"""Phase 1: Signal-level characterisation of Benchmark 9ch SSVEP data.

For every subject × target frequency, computes:
- narrowband SNR (fundamental only)
- harmonic SNR (fundamental + harmonics)
- FFT-based ITPC (phase locking value)
- spectral concentration (sparsity indicator)

Also computes the deterministic 40×40 harmonic interference matrix.

Usage
-----
    python analyze_signal.py                       # full 35 subjects, 1.0 s window
    python analyze_signal.py --subjects 1-5        # subset
    python analyze_signal.py --window 0.5          # shorter window
    python analyze_signal.py --window 1.0 --nfft-mult 4   # zero-padded FFT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vep_arena.config import (
    BENCHMARK_CHANNELS_9,
    BENCHMARK_FREQS,
    DATA_ROOT,
    BenchmarkSpec,
)
from vep_arena.data.benchmark import load_subject_trials
from vep_arena.signal.spectrum import compute_psd
from vep_arena.signal.snr import snr_narrowband, snr_harmonic
from vep_arena.signal.plv import itpc_fft
from vep_arena.signal.utils import spectral_concentration, harmonic_interference_matrix


TASK_DIR = Path(__file__).resolve().parent
CHANNEL_NAMES_9 = ("Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2")
FREQS = np.asarray(BENCHMARK_FREQS, dtype=np.float64)
FREQ_SORTED_IDX = np.argsort(FREQS)
FREQS_SORTED = FREQS[FREQ_SORTED_IDX]

N_HARMONICS = 5
N_NEIGHBORS = 10


# ── CLI ─────────────────────────────────────────────────────────────

def parse_int_set(text: str) -> tuple[int, ...]:
    vals: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            vals.extend(range(int(a), int(b) + 1))
        else:
            vals.append(int(part))
    return tuple(dict.fromkeys(vals))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark signal profile analysis")
    p.add_argument("--subjects", type=str, default="1-35")
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    p.add_argument("--nfft-mult", type=int, default=4,
                   help="Zero-pad FFT to nfft = nfft_mult × n_samples for finer interpolation")
    p.add_argument("--psd-method", type=str, default="fft", choices=["fft", "welch"])
    return p.parse_args()


# ── per-subject computation ─────────────────────────────────────────

def analyse_subject(
    data_root: Path,
    subject: int,
    window: float,
    nfft_mult: int,
    psd_method: str,
    spec: BenchmarkSpec,
) -> list[dict]:
    """Return one row per target frequency for this subject."""
    epochs = load_subject_trials(data_root, subject, window, BENCHMARK_CHANNELS_9, spec)
    # epochs: (classes=40, blocks=6, channels=9, samples)
    n_samples = epochs.shape[-1]
    fs = spec.sampling_rate
    nfft = n_samples * nfft_mult

    rows: list[dict] = []
    for cls_idx in range(spec.classes):
        f0 = float(FREQS[cls_idx])
        trials = epochs[cls_idx]  # (blocks=6, channels=9, samples)

        # PSD: average across blocks, keep per-channel → (9, n_freqs)
        freqs_psd, psd = compute_psd(trials, fs, method=psd_method, nfft=nfft)
        psd_mean_blocks = psd.mean(axis=0)  # (9, n_freqs)

        # per-channel SNR at Oz (index 7), and mean across channels
        oz_idx = CHANNEL_NAMES_9.index("Oz")
        snr_nb_oz = snr_narrowband(
            psd_mean_blocks[oz_idx], freqs_psd, f0,
            n_neighbors=N_NEIGHBORS, exclude_harmonics_of=f0, db=True,
        )
        snr_hm_oz = snr_harmonic(
            psd_mean_blocks[oz_idx], freqs_psd, f0,
            n_harmonics=N_HARMONICS, n_neighbors=N_NEIGHBORS, db=True,
        )
        snr_nb_mean = snr_narrowband(
            psd_mean_blocks, freqs_psd, f0,
            n_neighbors=N_NEIGHBORS, exclude_harmonics_of=f0, db=True,
        )
        snr_hm_mean = snr_harmonic(
            psd_mean_blocks, freqs_psd, f0,
            n_harmonics=N_HARMONICS, n_neighbors=N_NEIGHBORS, db=True,
        )

        # spectral concentration (sparsity)
        sc = spectral_concentration(
            psd_mean_blocks, freqs_psd, f0,
            n_harmonics=N_HARMONICS, bandwidth=0.5,
        )

        # PLV: per-channel, then average; use Oz as primary
        plv_oz = itpc_fft(trials[:, oz_idx, :], fs, f0, nfft=nfft)
        plv_vals = [itpc_fft(trials[:, ch, :], fs, f0, nfft=nfft) for ch in range(trials.shape[1])]
        plv_mean = float(np.mean(plv_vals))

        # per-channel SNR vector for full profile
        snr_per_ch = []
        for ch in range(trials.shape[1]):
            snr_per_ch.append(snr_harmonic(
                psd_mean_blocks[ch], freqs_psd, f0,
                n_harmonics=N_HARMONICS, n_neighbors=N_NEIGHBORS, db=True,
            ))

        best_ch_idx = int(np.nanargmax(snr_per_ch))

        rows.append({
            "subject": subject,
            "target_idx": cls_idx + 1,
            "frequency_hz": f0,
            "window_s": window,
            "snr_narrow_oz_db": snr_nb_oz,
            "snr_harm_oz_db": snr_hm_oz,
            "snr_narrow_mean_db": snr_nb_mean,
            "snr_harm_mean_db": snr_hm_mean,
            "plv_oz": plv_oz,
            "plv_mean": plv_mean,
            "spectral_concentration": sc,
            "best_snr_channel": CHANNEL_NAMES_9[best_ch_idx],
            "best_snr_channel_db": snr_per_ch[best_ch_idx],
        })
    return rows


# ── aggregation ─────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_subject = df.groupby("subject", as_index=False).agg(
        mean_snr_harm_oz=("snr_harm_oz_db", "mean"),
        median_snr_harm_oz=("snr_harm_oz_db", "median"),
        std_snr_harm_oz=("snr_harm_oz_db", "std"),
        mean_plv_oz=("plv_oz", "mean"),
        median_plv_oz=("plv_oz", "median"),
        mean_spectral_conc=("spectral_concentration", "mean"),
    )
    by_target = df.groupby(["target_idx", "frequency_hz"], as_index=False).agg(
        mean_snr_harm_oz=("snr_harm_oz_db", "mean"),
        std_snr_harm_oz=("snr_harm_oz_db", "std"),
        mean_plv_oz=("plv_oz", "mean"),
        std_plv_oz=("plv_oz", "std"),
        mean_spectral_conc=("spectral_concentration", "mean"),
    )
    return by_subject, by_target


# ── plotting ────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_snr_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    subjects = sorted(df["subject"].unique())
    mat = np.full((len(subjects), 40), np.nan)
    for i, s in enumerate(subjects):
        sub = df[df["subject"] == s].sort_values("target_idx")
        mat[i, :] = sub["snr_harm_oz_db"].values[FREQ_SORTED_IDX]

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_xlabel("Target frequency (Hz)")
    ax.set_ylabel("Subject")
    ax.set_xticks(range(40))
    ax.set_xticklabels([f"{f:.1f}" for f in FREQS_SORTED], rotation=90, fontsize=6)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=7)
    ax.set_title("Harmonic SNR (dB) — Oz channel, by subject × frequency")
    fig.colorbar(im, ax=ax, label="SNR (dB)", shrink=0.8)
    _save(fig, fig_dir / "fig_snr_heatmap.png")


def plot_plv_heatmap(df: pd.DataFrame, fig_dir: Path) -> None:
    subjects = sorted(df["subject"].unique())
    mat = np.full((len(subjects), 40), np.nan)
    for i, s in enumerate(subjects):
        sub = df[df["subject"] == s].sort_values("target_idx")
        mat[i, :] = sub["plv_oz"].values[FREQ_SORTED_IDX]

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("Target frequency (Hz)")
    ax.set_ylabel("Subject")
    ax.set_xticks(range(40))
    ax.set_xticklabels([f"{f:.1f}" for f in FREQS_SORTED], rotation=90, fontsize=6)
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=7)
    ax.set_title("ITPC (PLV) — Oz channel, by subject × frequency")
    fig.colorbar(im, ax=ax, label="PLV", shrink=0.8)
    _save(fig, fig_dir / "fig_plv_heatmap.png")


def plot_snr_vs_plv_scatter(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(df["snr_harm_oz_db"], df["plv_oz"], s=6, alpha=0.3, edgecolors="none")
    ax.set_xlabel("Harmonic SNR (dB)")
    ax.set_ylabel("ITPC (PLV)")
    ax.set_title("SNR vs PLV — all subjects × frequencies")
    ax.grid(alpha=0.2)
    _save(fig, fig_dir / "fig_snr_vs_plv.png")


def plot_interference_matrix(fig_dir: Path) -> np.ndarray:
    mat = harmonic_interference_matrix(FREQS, n_harmonics=N_HARMONICS, resolution_hz=0.25)
    mat_sorted = mat[np.ix_(FREQ_SORTED_IDX, FREQ_SORTED_IDX)]

    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    im = ax.imshow(mat_sorted, aspect="equal", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(40))
    ax.set_xticklabels([f"{f:.1f}" for f in FREQS_SORTED], rotation=90, fontsize=5)
    ax.set_yticks(range(40))
    ax.set_yticklabels([f"{f:.1f}" for f in FREQS_SORTED], fontsize=5)
    ax.set_title(f"Harmonic interference matrix (N_h={N_HARMONICS}, res={0.25} Hz)")
    fig.colorbar(im, ax=ax, label="# colliding harmonic pairs", shrink=0.8)
    _save(fig, fig_dir / "fig_harmonic_interference.png")
    return mat


def plot_frequency_profile(df_target: pd.DataFrame, fig_dir: Path) -> None:
    df_s = df_target.sort_values("frequency_hz")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

    axes[0].bar(range(40), df_s["mean_snr_harm_oz"].values, color="#3b82f6", width=0.7)
    axes[0].errorbar(range(40), df_s["mean_snr_harm_oz"], yerr=df_s["std_snr_harm_oz"],
                     fmt="none", color="#1e3a5f", capsize=2, linewidth=0.8)
    axes[0].set_ylabel("Harmonic SNR (dB)")
    axes[0].set_title("Signal quality profile across 40 target frequencies (mean ± SD, n=35)")
    axes[0].grid(alpha=0.15)

    axes[1].bar(range(40), df_s["mean_plv_oz"].values, color="#10b981", width=0.7)
    axes[1].errorbar(range(40), df_s["mean_plv_oz"], yerr=df_s["std_plv_oz"],
                     fmt="none", color="#064e3b", capsize=2, linewidth=0.8)
    axes[1].set_ylabel("PLV")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.15)

    axes[2].bar(range(40), df_s["mean_spectral_conc"].values, color="#f59e0b", width=0.7)
    axes[2].set_ylabel("Spectral concentration")
    axes[2].set_xlabel("Target frequency (Hz)")
    axes[2].grid(alpha=0.15)

    for ax in axes:
        ax.set_xticks(range(40))
        ax.set_xticklabels([f"{f:.1f}" for f in df_s["frequency_hz"]], rotation=90, fontsize=6)

    _save(fig, fig_dir / "fig_frequency_profile.png")


# ── main ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    subjects = parse_int_set(args.subjects)
    data_root = Path(args.data_root)
    spec = BenchmarkSpec()

    out_dir = TASK_DIR / "results"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Signal profile analysis ===")
    print(f"  subjects:   {subjects}")
    print(f"  window:     {args.window} s")
    print(f"  nfft_mult:  {args.nfft_mult}")
    print(f"  psd_method: {args.psd_method}")
    print(f"  data_root:  {data_root}")
    print()

    all_rows: list[dict] = []
    t0 = time.time()
    for i, subj in enumerate(subjects, 1):
        ts = time.time()
        rows = analyse_subject(data_root, subj, args.window, args.nfft_mult, args.psd_method, spec)
        all_rows.extend(rows)
        elapsed = time.time() - ts
        print(f"  [{i}/{len(subjects)}] Subject {subj:2d}  ({elapsed:.1f}s)")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "signal_profile.csv", index=False)
    print(f"\n  Wrote signal_profile.csv  ({len(df)} rows)")

    by_subject, by_target = aggregate(df)
    by_subject.to_csv(out_dir / "signal_summary_by_subject.csv", index=False)
    by_target.to_csv(out_dir / "signal_summary_by_target.csv", index=False)
    print(f"  Wrote signal_summary_by_subject.csv  ({len(by_subject)} rows)")
    print(f"  Wrote signal_summary_by_target.csv  ({len(by_target)} rows)")

    # harmonic interference matrix (deterministic)
    imat = plot_interference_matrix(fig_dir)
    imat_sorted = imat[np.ix_(FREQ_SORTED_IDX, FREQ_SORTED_IDX)]
    pd.DataFrame(imat_sorted, index=FREQS_SORTED, columns=FREQS_SORTED).to_csv(
        out_dir / "harmonic_interference.csv",
    )
    print(f"  Wrote harmonic_interference.csv")

    # plots
    plot_snr_heatmap(df, fig_dir)
    plot_plv_heatmap(df, fig_dir)
    plot_snr_vs_plv_scatter(df, fig_dir)
    plot_frequency_profile(by_target, fig_dir)
    print(f"  Wrote 5 figures to {fig_dir}")

    total = time.time() - t0
    manifest = {
        "subjects": list(subjects),
        "window_s": args.window,
        "nfft_mult": args.nfft_mult,
        "psd_method": args.psd_method,
        "n_harmonics": N_HARMONICS,
        "n_neighbors": N_NEIGHBORS,
        "total_rows": len(df),
        "elapsed_s": round(total, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  Done in {total:.1f}s")


if __name__ == "__main__":
    main()
