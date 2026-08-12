from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.embc_jbhi import (  # noqa: E402
    available_subject_ids,
    load_subject,
    load_target_frequency_pairs,
    resolve_dataset_root,
)


DATASET_LABELS = {
    "embc9": "EMBC 9-target",
    "jbhi16": "JBHI 16-target",
    "jbhi35": "JBHI 35-target",
}

BASE_FREQUENCIES = {
    "embc9": (8.5, 9.5),
    "jbhi16": (11.0, 12.0, 13.0),
    "jbhi35": (11.0, 12.0, 13.0, 14.0, 15.0),
}

def _oz_epochs(dataset: str, subject_id: str) -> tuple[np.ndarray, int, str]:
    subject = load_subject(dataset, subject_id)
    oz_index = subject.channels.index("Oz")
    epochs = subject.x[:, :, oz_index]
    if dataset == "jbhi16":
        epochs = epochs[..., :4000]
    profile = "full paper-preprocessed epoch at native 1000 Hz; no additional receiver filtering"
    return epochs, subject.sampling_rate, profile


def _mean_trial_power_db(
    epochs: np.ndarray,
    sampling_rate: int,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray]:
    centered = epochs - np.mean(epochs, axis=-1, keepdims=True)
    taper = signal.windows.hann(centered.shape[-1], sym=False)
    spectra = np.fft.rfft(centered * taper, n=n_fft, axis=-1)
    power = np.mean(np.abs(spectra) ** 2, axis=1)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)
    power_db = 10.0 * np.log10(power + np.finfo(float).tiny)
    display_mask = (frequencies >= 5.0) & (frequencies <= 40.0)
    row_baseline = np.median(power_db[:, display_mask], axis=1, keepdims=True)
    return frequencies, power_db - row_baseline


def _mapping(dataset: str) -> tuple[dict[int, tuple[float, float]], str]:
    pairs = load_target_frequency_pairs(dataset)
    provenance = (
        "file-verified 35-target codebook"
        if dataset == "jbhi35"
        else "paper interface and experiment-code verified codebook"
    )
    return {index: pair for index, pair in enumerate(pairs, start=1)}, provenance


def _grid_layout(dataset: str) -> tuple[int, int, dict[int, tuple[int, int]], str]:
    if dataset == "embc9":
        positions = {target: divmod(target - 1, 3) for target in range(1, 10)}
        return 3, 3, positions, "3x3 paper-interface placement"
    if dataset == "jbhi16":
        positions = {
            column * 4 + row + 1: (row, column)
            for column in range(4)
            for row in range(4)
        }
        return 4, 4, positions, "4x4 paper-interface placement; labels follow column-major trial order"
    root = resolve_dataset_root("jbhi35")
    with (root / "target_mapping_35.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    positions = {
        int(row["target_label_1based"]): (
            int(row["grid_row_1based"]) - 1,
            int(row["grid_col_1based"]) - 1,
        )
        for row in rows
    }
    if len(positions) != 35 or len(set(positions.values())) != 35:
        raise ValueError("JBHI35 target grid must contain 35 unique cells.")
    return 5, 7, positions, "5x7 file-verified target grid"


def _target_frequency_snr_db(
    frequencies: np.ndarray,
    relative_db: np.ndarray,
    mapping: dict[int, tuple[float, float]],
) -> float:
    scores = []
    for target, pair in mapping.items():
        active = sorted({frequency for frequency in pair if frequency > 0})
        for frequency in active:
            peak_mask = np.abs(frequencies - frequency) <= 0.2
            noise_mask = (
                (np.abs(frequencies - frequency) >= 0.6)
                & (np.abs(frequencies - frequency) <= 1.6)
            )
            peak = float(np.max(relative_db[target - 1, peak_mask]))
            noise = float(np.median(relative_db[target - 1, noise_mask]))
            scores.append(peak - noise)
    return float(np.median(scores))


def _select_subject(dataset: str, n_fft: int) -> tuple[str, float, list[dict[str, object]]]:
    mapping, _ = _mapping(dataset)
    scores = []
    for subject_id in available_subject_ids(dataset):
        epochs, sampling_rate, _ = _oz_epochs(dataset, subject_id)
        frequencies, relative_db = _mean_trial_power_db(epochs, sampling_rate, n_fft)
        score = _target_frequency_snr_db(frequencies, relative_db, mapping)
        scores.append({"subject": subject_id, "median_target_snr_db": score})
    scores.sort(key=lambda row: float(row["median_target_snr_db"]), reverse=True)
    return str(scores[0]["subject"]), float(scores[0]["median_target_snr_db"]), scores


def plot_dataset(
    dataset: str,
    subject_id: str,
    output_dir: Path,
    n_fft: int,
    selection_score_db: float | None = None,
) -> dict[str, object]:
    epochs, sampling_rate, profile = _oz_epochs(dataset, subject_id)
    frequencies, relative_db = _mean_trial_power_db(epochs, sampling_rate, n_fft)
    display_mask = (frequencies >= 5.0) & (frequencies <= 40.0)
    display_frequencies = frequencies[display_mask]
    display_values = relative_db[:, display_mask]
    mapping, mapping_status = _mapping(dataset)
    grid_rows, grid_columns, positions, placement_status = _grid_layout(dataset)
    fig_width = 11.0 if dataset == "embc9" else 13.0 if dataset == "jbhi16" else 18.0
    fig_height = 8.0 if dataset == "embc9" else 10.0 if dataset == "jbhi16" else 12.0
    fig, axes = plt.subplots(grid_rows, grid_columns, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    lower = float(np.percentile(display_values, 0.5))
    upper = float(np.max(display_values) + 1.0)
    for target in range(1, epochs.shape[0] + 1):
        row, column = positions[target]
        ax = axes[row, column]
        ax.plot(display_frequencies, display_values[target - 1], color="#3366aa", lw=0.85)
        for frequency in BASE_FREQUENCIES[dataset]:
            ax.axvline(frequency, color="#999999", lw=0.55, ls=":", alpha=0.7)
        title = f"T{target:02d}"
        if target in mapping:
            left, right = mapping[target]
            title += f"\nL={left:g}  R={right:g}"
            if left > 0 and right > 0 and np.isclose(left, right):
                ax.axvline(left, color="#1b9e77", lw=1.25)
            else:
                if left > 0:
                    ax.axvline(left, color="#1f9ed4", lw=1.25)
                if right > 0:
                    ax.axvline(right, color="#e6a700", lw=1.25)
        ax.set_title(title, fontsize=8.0 if dataset == "jbhi35" else 9.0, pad=3)
        ax.set_xlim(5.0, 40.0)
        ax.set_ylim(lower, upper)
        ax.grid(alpha=0.16, lw=0.4)
        if row == grid_rows - 1:
            ax.set_xlabel("Hz", fontsize=8)
        if column == 0:
            ax.set_ylabel("rel. dB", fontsize=8)
        ax.tick_params(labelsize=7)

    duration = epochs.shape[-1] / sampling_rate
    trials_per_target = epochs.shape[1]
    fig.suptitle(
        f"{DATASET_LABELS[dataset]} | {subject_id} | Oz | mean of {trials_per_target} trials per label | {duration:g} s\n"
        f"mean per-trial power spectrum | {n_fft}-point FFT | {placement_status}\n{profile}",
        fontsize=12,
    )
    legend_items = [
        Line2D([0], [0], color="#3366aa", lw=1.2, label="mean trial power"),
        Line2D([0], [0], color="#999999", lw=0.8, ls=":", label="frequency pool"),
    ]
    if mapping:
        legend_items.extend(
            [
                Line2D([0], [0], color="#1f9ed4", lw=1.4, label="left-eye frequency"),
                Line2D([0], [0], color="#e6a700", lw=1.4, label="right-eye frequency"),
                Line2D([0], [0], color="#1b9e77", lw=1.4, label="same frequency L=R"),
            ]
        )
    fig.legend(handles=legend_items, loc="lower center", ncol=len(legend_items), fontsize=8, frameon=False)
    fig.tight_layout(rect=[0.02, 0.055, 1.0, 0.91])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_{subject_id.lower()}_oz_array_fft.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "dataset": dataset,
        "subject": subject_id,
        "channel": "Oz",
        "targets": epochs.shape[0],
        "trials_averaged_per_target": trials_per_target,
        "window_seconds": duration,
        "sampling_rate": sampling_rate,
        "fft_points": n_fft,
        "frequency_bin_hz": sampling_rate / n_fft,
        "subject_selection_median_target_snr_db": selection_score_db,
        "mapping_status": mapping_status,
        "placement_status": placement_status,
        "grid_shape": [grid_rows, grid_columns],
        "mapped_targets": sorted(mapping),
        "output": output_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot anonymous Oz multi-trial FFT label checks for EMBC/JBHI datasets.")
    parser.add_argument(
        "--subject",
        default="auto",
        help="Anonymous subject ID used for all datasets, or 'auto' to select the clearest Oz target peaks per dataset.",
    )
    parser.add_argument("--n-fft", type=int, default=16384, help="Zero-padded FFT length.")
    parser.add_argument(
        "--output",
        type=Path,
        default=TASK / "results" / "BS05_fft",
    )
    args = parser.parse_args()
    if args.n_fft < 4001:
        raise ValueError("--n-fft must be at least 4001 so no source epoch is truncated.")
    rows = []
    subject_ranking: dict[str, list[dict[str, object]]] = {}
    for dataset in DATASET_LABELS:
        selection_score = None
        if args.subject.lower() == "auto":
            subject_id, selection_score, ranking = _select_subject(dataset, args.n_fft)
            subject_ranking[dataset] = ranking
        else:
            subject_id = args.subject.upper()
        rows.append(plot_dataset(dataset, subject_id, args.output.resolve(), args.n_fft, selection_score))
    manifest = {
        "status": "complete",
        "identity_policy": "anonymous subject IDs only; source filenames are not exposed",
        "fft_order": "compute a tapered spectrum for every full-length trial, then average power across trials",
        "subject_selection": "highest median target-frequency Oz SNR" if subject_ranking else "explicit subject",
        "subject_rankings": subject_ranking,
        "rows": rows,
    }
    (args.output.resolve() / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
