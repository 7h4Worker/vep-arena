from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy import signal
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.dual_frequency import (  # noqa: E402
    LIANG2020_METHOD1_DUAL_FREQUENCY_PHASE_CODEBOOK,
    LIANG2020_SAMPLING_RATE,
    iter_liang_mat_files,
    load_liang_mat,
)
from vep_arena.methods.tdca import TDCA  # noqa: E402
from vep_arena.methods.traditional import filterbank_weights, multi_frequency_reference_signals  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TASK = Path(__file__).resolve().parent
DEFAULT_BANDS = ((8.0, 80.0), (16.0, 80.0), (24.0, 80.0), (32.0, 80.0), (40.0, 80.0))
CODEBOOK_SOURCE = "Liang et al. 2020, Figure 1(a), row-major targets 1-40"


def parse_windows(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def filterband(data: np.ndarray, low: float, high: float) -> np.ndarray:
    high = min(high, LIANG2020_SAMPLING_RATE / 2.0 - 1.0)
    sos = signal.butter(4, (low, high), btype="bandpass", fs=LIANG2020_SAMPLING_RATE, output="sos")
    return signal.sosfiltfilt(sos, data, axis=1)


def method1_references(samples: int, harmonics: int, ordered_codebook) -> list[np.ndarray]:
    frequencies = [(entry[0], entry[1]) for entry in ordered_codebook]
    phases = [(entry[2], entry[2]) for entry in ordered_codebook]
    return multi_frequency_reference_signals(
        frequencies,
        samples=samples,
        fs=LIANG2020_SAMPLING_RATE,
        harmonics=harmonics,
        target_phases_pi=phases,
    )


def load_group_epochs(records, window: float, onset_shift: float, n_bands: int, n_delay: int):
    start = int(round(onset_shift * LIANG2020_SAMPLING_RATE))
    samples = int(round(window * LIANG2020_SAMPLING_RATE))
    needed = samples + n_delay
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    blocks: list[np.ndarray] = []
    for record in sorted(records, key=lambda item: item.run):
        epoch = load_liang_mat(record.path)
        if start + needed > epoch.n_samples:
            raise ValueError(f"{record.path} is too short for {window}s TDCA with n_delay={n_delay}")
        result = np.empty((epoch.n_trials, n_bands, epoch.n_channels, needed), dtype=np.float64)
        for band_index, (low, high) in enumerate(DEFAULT_BANDS[:n_bands]):
            filtered = filterband(epoch.x, low, high)
            result[:, band_index] = np.transpose(filtered[:, start : start + needed, :], (2, 0, 1))
        labels = np.asarray(epoch.labels, dtype=np.int64).ravel() - 1
        xs.append(result)
        ys.append(labels)
        blocks.append(np.full(labels.shape, record.run, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(blocks)


def splits(experiment: str, blocks: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if experiment == "exp4":
        return [("paper_train1-6_test7-9", blocks <= 6, (blocks >= 7) & (blocks <= 9))]
    return [(f"leave_block_{int(block)}", blocks != block, blocks == block) for block in sorted(np.unique(blocks))]


def write_codebook(path: Path, label_to_figure_index: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_id", "frequency_1_hz", "frequency_2_hz", "phase_pi", "source"])
        writer.writeheader()
        for target_id, figure_index in enumerate(label_to_figure_index, start=1):
            freq1, freq2, phase = LIANG2020_METHOD1_DUAL_FREQUENCY_PHASE_CODEBOOK[int(figure_index)]
            writer.writerow(
                {
                    "target_id": target_id,
                    "frequency_1_hz": freq1,
                    "frequency_2_hz": freq2,
                    "phase_pi": phase,
                    "source": f"{CODEBOOK_SOURCE}; label alignment inferred from Exp3-C3 S1 spectral amplitudes",
                }
            )


def read_label_map(path: Path) -> np.ndarray:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) != 40:
        raise ValueError(f"Expected 40 rows in label map, got {len(rows)}: {path}")
    figure_lookup = {tuple(float(value) for value in entry): index for index, entry in enumerate(LIANG2020_METHOD1_DUAL_FREQUENCY_PHASE_CODEBOOK)}
    mapping = np.empty(40, dtype=np.int64)
    for row in rows:
        target = int(row["target_id"]) - 1
        entry = (float(row["frequency_1_hz"]), float(row["frequency_2_hz"]), float(row["phase_pi"]))
        if entry not in figure_lookup:
            raise ValueError(f"Label map entry is not in Figure 1 codebook: {entry}")
        mapping[target] = figure_lookup[entry]
    return mapping


def write_spectral_validation(
    records,
    onset_shift: float,
    output: Path,
    subject: int,
    fixed_label_to_figure_index: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    start = int(round(onset_shift * LIANG2020_SAMPLING_RATE))
    samples = 2000
    class_trials: dict[int, list[np.ndarray]] = defaultdict(list)
    for record in sorted(records, key=lambda item: item.run):
        epoch = load_liang_mat(record.path)
        if start + samples > epoch.n_samples:
            raise ValueError(f"{record.path} is too short for 2.0s spectral validation")
        for trial_index, label in enumerate(np.asarray(epoch.labels, dtype=np.int64).ravel() - 1):
            class_trials[int(label)].append(epoch.x[:, start : start + samples, trial_index])

    candidates = np.asarray(sorted({freq for entry in LIANG2020_METHOD1_DUAL_FREQUENCY_PHASE_CODEBOOK for freq in entry[:2]}))
    time = np.arange(samples, dtype=np.float64) / LIANG2020_SAMPLING_RATE
    basis = np.exp(-2j * np.pi * candidates[:, None] * time[None, :])
    powers = np.zeros((len(LIANG2020_METHOD1_DUAL_FREQUENCY_PHASE_CODEBOOK), candidates.size), dtype=np.float64)
    for target in range(40):
        signal_mean = np.mean(np.stack(class_trials[target], axis=0), axis=(0, 1))
        amplitude = np.abs(basis @ signal_mean) / samples
        powers[target] = amplitude

    pair_scores = np.zeros((40, 40), dtype=np.float64)
    for label in range(40):
        for figure_index, (freq1, freq2, _) in enumerate(LIANG2020_METHOD1_DUAL_FREQUENCY_PHASE_CODEBOOK):
            idx1 = int(np.where(np.isclose(candidates, freq1))[0][0])
            idx2 = int(np.where(np.isclose(candidates, freq2))[0][0])
            pair_scores[label, figure_index] = powers[label, idx1] + powers[label, idx2]
    if fixed_label_to_figure_index is None:
        labels, figure_indices = linear_sum_assignment(-pair_scores)
        label_to_figure_index = np.empty(40, dtype=np.int64)
        label_to_figure_index[labels] = figure_indices
        alignment_source = "inferred_from_this_subject"
    else:
        label_to_figure_index = fixed_label_to_figure_index
        alignment_source = "loaded_from_external_subject_map"

    selected: list[float] = []
    unselected: list[float] = []
    exact_top_two = 0
    for target, figure_index in enumerate(label_to_figure_index):
        freq1, freq2, _ = LIANG2020_METHOD1_DUAL_FREQUENCY_PHASE_CODEBOOK[int(figure_index)]
        expected = {int(np.where(np.isclose(candidates, freq1))[0][0]), int(np.where(np.isclose(candidates, freq2))[0][0])}
        amplitude = powers[target]
        selected.extend(amplitude[list(expected)].tolist())
        unselected.extend(amplitude[index] for index in range(candidates.size) if index not in expected)
        if expected.issubset(set(np.argsort(amplitude)[-2:])):
            exact_top_two += 1

    normalized = powers / np.maximum(np.max(powers, axis=1, keepdims=True), 1e-12)
    figure, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    image = axis.imshow(normalized, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0.0, vmax=1.0)
    for target, figure_index in enumerate(label_to_figure_index):
        freq1, freq2, _ = LIANG2020_METHOD1_DUAL_FREQUENCY_PHASE_CODEBOOK[int(figure_index)]
        for freq in (freq1, freq2):
            index = int(np.where(np.isclose(candidates, freq))[0][0])
            axis.scatter(index, target, marker="x", s=24, linewidths=0.9, color="white")
    axis.set_title(f"Liang2020 Method 1 spectral label alignment (S{subject}, Exp3 C3)")
    axis.set_xlabel("Candidate stimulation frequency (Hz); white x = assigned Figure 1 pair")
    axis.set_ylabel("Target ID")
    axis.set_xticks(np.arange(candidates.size), [f"{freq:g}" for freq in candidates])
    axis.set_yticks(np.arange(0, 40, 5), np.arange(1, 41, 5))
    figure.colorbar(image, ax=axis, label="Per-target normalized amplitude")
    figure.savefig(output / "codebook_spectral_validation.png", dpi=180)
    plt.close(figure)

    metrics = {
        "selected_frequency_mean_amplitude": float(np.mean(selected)),
        "unselected_frequency_mean_amplitude": float(np.mean(unselected)),
        "selected_to_unselected_ratio": float(np.mean(selected) / max(np.mean(unselected), 1e-12)),
        "both_expected_frequencies_top2_targets": float(exact_top_two),
        "identity_order_matches": float(np.sum(label_to_figure_index == np.arange(40))),
        "alignment_source": alignment_source,
        "targets": 40.0,
    }
    with (output / "codebook_spectral_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics, label_to_figure_index


def write_confusion(path: Path, true: list[int], predicted: list[int], title: str) -> None:
    matrix = confusion_matrix(true, predicted, labels=list(range(40)))
    figure, axis = plt.subplots(figsize=(7, 6), dpi=160)
    image = axis.imshow(matrix, cmap="viridis", interpolation="nearest")
    axis.set_title(title)
    axis.set_xlabel("Predicted target")
    axis.set_ylabel("True target")
    axis.set_xticks(np.arange(0, 40, 5))
    axis.set_yticks(np.arange(0, 40, 5))
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def itr_bits_per_min(classes: int, accuracy: float, seconds: float) -> float:
    accuracy = min(max(float(accuracy), 1e-12), 1.0 - 1e-12)
    bits = math.log2(classes) + accuracy * math.log2(accuracy) + (1.0 - accuracy) * math.log2((1.0 - accuracy) / (classes - 1))
    return bits * 60.0 / seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="TDCA smoke validation for Liang2020 Method 1 dual-frequency phase codebook.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--experiments", default="exp3,exp4")
    parser.add_argument("--windows", default="0.4,0.6")
    parser.add_argument("--n-bands", type=int, default=5, choices=range(1, len(DEFAULT_BANDS) + 1))
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-delay", type=int, default=5)
    parser.add_argument("--onset-shift", type=float, default=0.14)
    parser.add_argument("--label-map", type=Path, default=None, help="CSV map from a previous subject; disables same-subject label alignment.")
    parser.add_argument("--output", type=Path, default=TASK / "results" / "smoke_tdca_method1_spectral_alignment_20260720")
    args = parser.parse_args()

    experiments = [value.strip() for value in args.experiments.split(",") if value.strip()]
    windows = parse_windows(args.windows)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    all_records = iter_liang_mat_files(experiments=experiments, subjects=[args.subject])
    expected_conditions = {"exp3": 3, "exp4": 1}
    records = [record for record in all_records if expected_conditions.get(record.experiment) == record.condition]
    if not records:
        raise RuntimeError("No Method 1 Exp3-C3 or Exp4-C1 records found for TDCA smoke.")

    grouped: dict[str, list[object]] = defaultdict(list)
    for record in records:
        grouped[record.experiment].append(record)
    if "exp3" not in grouped:
        raise RuntimeError("Exp3-C3 is required to establish the data-label to Figure 1 codebook alignment.")
    fixed_label_to_figure_index = None
    if args.label_map is not None:
        label_map_path = args.label_map if args.label_map.is_absolute() else PROJECT_ROOT / args.label_map
        fixed_label_to_figure_index = read_label_map(label_map_path)
    spectral_metrics, label_to_figure_index = write_spectral_validation(
        grouped["exp3"],
        args.onset_shift,
        figures,
        args.subject,
        fixed_label_to_figure_index=fixed_label_to_figure_index,
    )
    ordered_codebook = [LIANG2020_METHOD1_DUAL_FREQUENCY_PHASE_CODEBOOK[int(index)] for index in label_to_figure_index]
    write_codebook(output / "digitized_method1_codebook.csv", label_to_figure_index)

    prediction_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for experiment in sorted(grouped):
        for window in windows:
            print(f"[Liang2020 TDCA smoke] {experiment} S{args.subject} window={window:.1f}s load/filter", flush=True)
            x, y, blocks = load_group_epochs(
                grouped[experiment],
                window=window,
                onset_shift=args.onset_shift,
                n_bands=args.n_bands,
                n_delay=args.n_delay,
            )
            refs = method1_references(int(round(window * LIANG2020_SAMPLING_RATE)), args.harmonics, ordered_codebook)
            true_all: list[int] = []
            pred_all: list[int] = []
            train_total = 0
            test_total = 0
            used_splits = 0
            for split_name, train_mask, test_mask in splits(experiment, blocks):
                if not set(np.unique(y[test_mask])).issubset(set(np.unique(y[train_mask]))):
                    raise RuntimeError(f"Invalid class coverage in {experiment} {split_name}")
                model = TDCA(
                    n_components=args.n_components,
                    n_delay=args.n_delay,
                    fb_weights=filterbank_weights(args.n_bands),
                )
                model.fit(x[train_mask], y[train_mask], refs)
                predicted, scores_by_band = model.predict(x[test_mask])
                scores = np.einsum("f,tfc->tc", model.fb_weights, scores_by_band)
                true = y[test_mask]
                test_indices = np.flatnonzero(test_mask)
                for local_index, index in enumerate(test_indices):
                    prediction_rows.append(
                        {
                            "experiment": experiment,
                            "subject": args.subject,
                            "window": window,
                            "method": "TDCA",
                            "split": split_name,
                            "run": int(blocks[index]),
                            "target_id": int(true[local_index] + 1),
                            "predicted_id": int(predicted[local_index] + 1),
                            "correct": int(predicted[local_index] == true[local_index]),
                            "score_true": float(scores[local_index, true[local_index]]),
                            "score_predicted": float(scores[local_index, predicted[local_index]]),
                        }
                    )
                true_all.extend(true.tolist())
                pred_all.extend(predicted.tolist())
                train_total += int(np.sum(train_mask))
                test_total += int(np.sum(test_mask))
                used_splits += 1
            accuracy = float(np.mean(np.asarray(true_all) == np.asarray(pred_all)))
            summary_rows.append(
                {
                    "experiment": experiment,
                    "subject": args.subject,
                    "window": window,
                    "method": "TDCA",
                    "classes": 40,
                    "splits": used_splits,
                    "train_trials_total": train_total,
                    "test_trials": test_total,
                    "accuracy": accuracy,
                    "itr_bits_per_min": itr_bits_per_min(40, accuracy, window + 0.5),
                }
            )
            write_confusion(
                figures / f"confusion_{experiment}_tdca_{window:.1f}s.png",
                true_all,
                pred_all,
                f"Liang2020 {experiment} Method 1 TDCA S{args.subject}, {window:.1f}s",
            )
            print(f"[Liang2020 TDCA smoke] {experiment} S{args.subject} window={window:.1f}s acc={accuracy:.4f}", flush=True)

    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest = {
        "task": "ssvep_dual_frequency_phase_liang2020",
        "run_type": "diagnostic_smoke_tdca_spectral_aligned_method1_codebook",
        "subject": args.subject,
        "experiments": sorted(grouped),
        "conditions": {"exp3": 3, "exp4": 1},
        "windows": windows,
        "sampling_rate": LIANG2020_SAMPLING_RATE,
        "onset_shift_seconds": args.onset_shift,
        "n_bands": args.n_bands,
        "bands": DEFAULT_BANDS[: args.n_bands],
        "harmonics": args.harmonics,
        "n_components": args.n_components,
        "n_delay_samples": args.n_delay,
        "codebook_source": CODEBOOK_SOURCE,
        "codebook_status": "Figure 1 values digitized; MAT-label order inferred from Exp3-C3 S1 spectral amplitudes",
        "label_map_input": None if args.label_map is None else str(args.label_map),
        "spectral_metrics": spectral_metrics,
        "exp3_protocol": "leave-one-block-out",
        "exp4_protocol": "runs 1-6 train, runs 7-9 test",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
