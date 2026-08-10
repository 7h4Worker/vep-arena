from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
from scipy import signal
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.dual_frequency import (  # noqa: E402
    iter_sun_cnt_files,
    sun_annotation_rows,
    sun_stimulus_codebook,
    sun_target_repeat_trials,
)
from vep_arena.methods.traditional import TRCA  # noqa: E402
from vep_arena.methods.trca_core import corr_rows, trca_filter  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FS = 250.0
ONSET_SHIFT_SECONDS = 0.14
BASELINE_BANDS = ((8.0, 90.0), (16.0, 90.0))
METHODS = (
    "baseline_2band_trca",
    "global_soft_comb_trca",
    "pair_soft_comb_trca",
    "pair_phase_aligned_comb_trca",
    "pair_periodic_comb_trca",
)


def parse_subjects(text: str, available: list[int]) -> list[int]:
    if text.strip().lower() == "all":
        return available
    return sorted({int(item.strip()) for item in text.split(",") if item.strip()})


def parse_windows(text: str) -> list[float]:
    return sorted({float(item.strip()) for item in text.split(",") if item.strip()})


def itr_bits_per_min(accuracy: float, window: float) -> float:
    p = min(max(float(accuracy), 1e-12), 1.0 - 1e-12)
    bits = math.log2(40) + p * math.log2(p) + (1.0 - p) * math.log2((1.0 - p) / 39.0)
    return bits * 60.0 / (window + 0.5)


def build_pair_specs(codebook: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    lookup = {
        (round(float(row["freq_for_left_eye"]), 6), round(float(row["freq_for_right_eye"]), 6)): row
        for row in codebook
    }
    specs: dict[int, dict[str, object]] = {}
    for row in codebook:
        left = float(row["freq_for_left_eye"])
        right = float(row["freq_for_right_eye"])
        target = int(row["trigger_num"])
        partner_row = lookup.get((round(right, 6), round(left, 6)))
        if partner_row is None:
            raise ValueError(f"Target {target} has no swapped-frequency partner.")
        partner = int(partner_row["trigger_num"])
        pair_id = min(target, partner)
        if target != pair_id:
            continue
        phase_left = float(row["phase_pi_for_left_eye"])
        phase_right = float(row["phase_pi_for_right_eye"])
        if not np.isclose(float(partner_row["phase_pi_for_left_eye"]), phase_right):
            raise ValueError(f"Target pair {target}/{partner} has inconsistent swapped left phase.")
        if not np.isclose(float(partner_row["phase_pi_for_right_eye"]), phase_left):
            raise ValueError(f"Target pair {target}/{partner} has inconsistent swapped right phase.")
        components = (
            (left, phase_left),
            (right, phase_right),
            (2.0 * left, 2.0 * phase_left),
            (2.0 * right, 2.0 * phase_right),
            (left + right, phase_left + phase_right),
        )
        specs[pair_id] = {
            "classes": (target - 1, partner - 1),
            "fundamentals": (left, right),
            "components": tuple((freq, phase) for freq, phase in components if 1.0 <= freq < 90.0),
        }
    if len(specs) != 20:
        raise ValueError(f"Expected 20 swap pairs, found {len(specs)}.")
    return specs


def load_subject(
    subject: int,
    repetitions: int,
    max_window: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import mne

    records = {record.subject: record for record in iter_sun_cnt_files(splits=["forty_targets"])}
    if subject not in records:
        raise ValueError(f"Missing Sun2024 subject {subject}.")
    record = records[subject]
    all_trials = sun_target_repeat_trials(sun_annotation_rows(record.path), n_targets=40)
    trials = [trial for trial in all_trials if trial.repetition <= repetitions]
    labels = np.asarray([trial.target_id - 1 for trial in trials], dtype=np.int64)
    folds = np.asarray([trial.repetition for trial in trials], dtype=np.int64)
    if Counter(labels.tolist()) != Counter({target: repetitions for target in range(40)}):
        raise ValueError(f"Subject {subject}: incomplete target repetition coverage.")

    raw = mne.io.read_raw_cnt(record.path, preload=False, verbose="ERROR")
    raw.pick(mne.pick_types(raw.info, eeg=True, meg=False, stim=False, exclude=[]))
    first = min(trial.start_seconds + ONSET_SHIFT_SECONDS for trial in trials)
    last = max(trial.start_seconds + ONSET_SHIFT_SECONDS + max_window for trial in trials)
    crop_tmin = max(0.0, first - 3.0)
    raw.crop(tmin=crop_tmin, tmax=min(float(raw.times[-1]), last + 3.0))
    raw.load_data(verbose="ERROR")
    raw.resample(FS, npad="auto", verbose="ERROR")
    data = raw.get_data().astype(np.float64, copy=False)
    starts = np.asarray(
        [int(round((trial.start_seconds + ONSET_SHIFT_SECONDS - crop_tmin) * FS)) for trial in trials],
        dtype=np.int64,
    )
    max_samples = int(round(max_window * FS))
    if np.max(starts + max_samples) > data.shape[-1]:
        raise ValueError(f"Subject {subject}: epoch exceeds cropped recording.")
    raw_epochs = np.stack([data[:, start : start + max_samples] for start in starts])
    baseline_epochs = np.empty(
        (raw_epochs.shape[0], len(BASELINE_BANDS), raw_epochs.shape[1], max_samples),
        dtype=np.float64,
    )
    for band_index, (low, high) in enumerate(BASELINE_BANDS):
        sos = signal.butter(4, (low, high), btype="bandpass", fs=FS, output="sos")
        filtered = signal.sosfiltfilt(sos, data, axis=-1)
        baseline_epochs[:, band_index] = np.stack(
            [filtered[:, start : start + max_samples] for start in starts]
        )
    return raw_epochs, baseline_epochs, labels, folds


def padded_fft(epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    centered = epochs - np.mean(epochs, axis=-1, keepdims=True)
    pad = epochs.shape[-1]
    padded = np.pad(centered, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
    spectrum = np.fft.rfft(padded, axis=-1)
    frequencies = np.fft.rfftfreq(padded.shape[-1], d=1.0 / FS)
    return spectrum, frequencies, pad


def apply_response(spectrum: np.ndarray, response: np.ndarray, samples: int, pad: int) -> np.ndarray:
    filtered = np.fft.irfft(spectrum * response[None, None, :], n=spectrum.shape[-1] * 2 - 2, axis=-1)
    return filtered[..., pad : pad + samples]


def soft_response(frequencies: np.ndarray, centers: tuple[float, ...], sigma_hz: float) -> np.ndarray:
    distances = np.asarray([(frequencies - center) / sigma_hz for center in centers])
    response = np.max(np.exp(-0.5 * distances * distances), axis=0)
    response[(frequencies < 6.0) | (frequencies > 90.0)] = 0.0
    return response


def phase_aligned_response(
    frequencies: np.ndarray,
    components: tuple[tuple[float, float], ...],
    sigma_hz: float,
) -> np.ndarray:
    weights = np.asarray([np.exp(-0.5 * ((frequencies - freq) / sigma_hz) ** 2) for freq, _ in components])
    phases = np.asarray([np.exp(-1j * phase_pi * np.pi) for _, phase_pi in components])[:, None]
    total = np.sum(weights, axis=0)
    envelope = np.max(weights, axis=0)
    response = envelope * np.sum(weights * phases, axis=0) / np.maximum(total, 1e-12)
    response[(frequencies < 6.0) | (frequencies > 90.0)] = 0.0
    response[0] = 0.0
    return response


def periodic_response(
    frequencies: np.ndarray,
    fundamentals: tuple[float, float],
    power: int = 8,
) -> np.ndarray:
    branches = np.asarray([np.abs(np.cos(np.pi * frequencies / frequency)) ** power for frequency in fundamentals])
    response = np.max(branches, axis=0)
    response[(frequencies < 6.0) | (frequencies > 90.0)] = 0.0
    return response


def class_scores(
    epochs: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    classes: tuple[int, ...] | range,
    repetitions: int,
    scores: np.ndarray,
) -> None:
    for class_id in classes:
        for fold in range(1, repetitions + 1):
            train = (labels == class_id) & (folds != fold)
            test = folds == fold
            spatial = trca_filter(epochs[train])
            template = np.mean(epochs[train], axis=0)
            projected_test = epochs[test].transpose(0, 2, 1) @ spatial
            projected_template = template.T @ spatial
            scores[test, class_id] = corr_rows(projected_test, projected_template)


def candidate_predictions(
    raw_epochs: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    repetitions: int,
    pair_specs: dict[int, dict[str, object]],
    mode: str,
    sigma_hz: float,
) -> np.ndarray:
    samples = raw_epochs.shape[-1]
    spectrum, frequencies, pad = padded_fft(raw_epochs)
    scores = np.full((labels.size, 40), -np.inf, dtype=np.float64)
    if mode == "global_soft":
        centers = tuple(
            sorted({float(freq) for spec in pair_specs.values() for freq, _ in spec["components"]})
        )
        response = soft_response(frequencies, centers, sigma_hz)
        filtered = apply_response(spectrum, response, samples, pad)
        class_scores(filtered, labels, folds, range(40), repetitions, scores)
    else:
        for spec in pair_specs.values():
            components = spec["components"]
            if mode == "pair_soft":
                response = soft_response(frequencies, tuple(float(freq) for freq, _ in components), sigma_hz)
            elif mode == "pair_phase":
                response = phase_aligned_response(frequencies, components, sigma_hz)
            elif mode == "pair_periodic":
                response = periodic_response(frequencies, spec["fundamentals"])
            else:
                raise ValueError(f"Unknown comb mode: {mode}")
            filtered = apply_response(spectrum, response, samples, pad)
            class_scores(filtered, labels, folds, spec["classes"], repetitions, scores)
    return np.argmax(scores, axis=1)


def baseline_predictions(
    baseline_epochs: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    repetitions: int,
) -> np.ndarray:
    predicted = np.empty(labels.size, dtype=np.int64)
    for fold in range(1, repetitions + 1):
        train = folds != fold
        test = folds == fold
        model = TRCA(n_fbs=len(BASELINE_BANDS), ensemble=False)
        model.fit(baseline_epochs[train], labels[train])
        predicted[test], _ = model.predict(baseline_epochs[test])
    return predicted


def run_subject(subject: int, windows: list[float], repetitions: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    codebook = sun_stimulus_codebook(kind="forty_targets")
    pair_specs = build_pair_specs(codebook)
    raw_epochs, baseline_epochs, labels, folds = load_subject(subject, repetitions, max(windows))
    summary_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    cohort = "development" if subject in {11, 13} else "confirmation"
    for window in windows:
        samples = int(round(window * FS))
        epochs = raw_epochs[..., :samples]
        sigma_hz = 0.5 / window
        predictions = {
            "baseline_2band_trca": baseline_predictions(
                baseline_epochs[..., :samples], labels, folds, repetitions
            ),
            "global_soft_comb_trca": candidate_predictions(
                epochs, labels, folds, repetitions, pair_specs, "global_soft", sigma_hz
            ),
            "pair_soft_comb_trca": candidate_predictions(
                epochs, labels, folds, repetitions, pair_specs, "pair_soft", sigma_hz
            ),
            "pair_phase_aligned_comb_trca": candidate_predictions(
                epochs, labels, folds, repetitions, pair_specs, "pair_phase", sigma_hz
            ),
            "pair_periodic_comb_trca": candidate_predictions(
                epochs, labels, folds, repetitions, pair_specs, "pair_periodic", sigma_hz
            ),
        }
        for method in METHODS:
            predicted = predictions[method]
            accuracy = float(np.mean(predicted == labels))
            summary_rows.append(
                {
                    "subject": subject,
                    "cohort": cohort,
                    "window": window,
                    "method": method,
                    "repetitions": repetitions,
                    "trials": labels.size,
                    "channels": raw_epochs.shape[1],
                    "fixed_onset_shift_seconds": ONSET_SHIFT_SECONDS,
                    "soft_sigma_hz": sigma_hz if "soft" in method or "phase" in method else "",
                    "accuracy": accuracy,
                    "itr_bits_per_min": itr_bits_per_min(accuracy, window),
                }
            )
            for index in range(labels.size):
                trial_rows.append(
                    {
                        "subject": subject,
                        "cohort": cohort,
                        "window": window,
                        "method": method,
                        "fold": int(folds[index]),
                        "target_id": int(labels[index] + 1),
                        "predicted_id": int(predicted[index] + 1),
                        "correct": int(predicted[index] == labels[index]),
                    }
                )
            print(
                f"[Sun2024 comb candidates] S{subject:02d} {window:.1f}s {method} acc={accuracy:.4f}",
                flush=True,
            )
    return summary_rows, trial_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    cohorts = ("all", "development", "confirmation")
    for cohort in cohorts:
        for window in sorted({float(row["window"]) for row in rows}):
            for method in METHODS:
                values = np.asarray(
                    [
                        float(row["accuracy"])
                        for row in rows
                        if float(row["window"]) == window
                        and row["method"] == method
                        and (cohort == "all" or row["cohort"] == cohort)
                    ],
                    dtype=np.float64,
                )
                if values.size == 0:
                    continue
                output.append(
                    {
                        "cohort": cohort,
                        "window": window,
                        "method": method,
                        "subjects": values.size,
                        "accuracy": float(np.mean(values)),
                        "accuracy_sem": float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0,
                        "itr_bits_per_min": itr_bits_per_min(float(np.mean(values)), window),
                    }
                )
    return output


def plot_curves(path: Path, aggregate: list[dict[str, object]]) -> None:
    rows = [row for row in aggregate if row["cohort"] == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), dpi=160)
    labels = {
        "baseline_2band_trca": "2-band TRCA",
        "global_soft_comb_trca": "global soft comb",
        "pair_soft_comb_trca": "pair soft comb",
        "pair_phase_aligned_comb_trca": "phase-aligned pair comb",
        "pair_periodic_comb_trca": "periodic pair comb",
    }
    for method in METHODS:
        selected = sorted((row for row in rows if row["method"] == method), key=lambda row: float(row["window"]))
        windows = np.asarray([float(row["window"]) for row in selected])
        accuracy = np.asarray([float(row["accuracy"]) for row in selected])
        sem = np.asarray([float(row["accuracy_sem"]) for row in selected])
        itr = np.asarray([float(row["itr_bits_per_min"]) for row in selected])
        axes[0].errorbar(windows, accuracy, yerr=sem, marker="o", capsize=3, label=labels[method])
        axes[1].plot(windows, itr, marker="o", label=labels[method])
    axes[0].set(xlabel="Analysis window (s)", ylabel="Accuracy", title="Sun2024 comb candidate accuracy")
    axes[1].set(xlabel="Analysis window (s)", ylabel="ITR (bits/min)", title="Sun2024 comb candidate ITR")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusions(path: Path, trials: list[dict[str, object]], window: float) -> None:
    fig, axes = plt.subplots(1, len(METHODS), figsize=(19, 3.8), dpi=150)
    for axis, method in zip(axes, METHODS):
        selected = [row for row in trials if float(row["window"]) == window and row["method"] == method]
        true = [int(row["target_id"]) - 1 for row in selected]
        predicted = [int(row["predicted_id"]) - 1 for row in selected]
        matrix = confusion_matrix(true, predicted, labels=list(range(40)))
        axis.imshow(matrix, cmap="viridis", interpolation="nearest")
        axis.set_title(method.replace("_trca", "").replace("_", "\n"), fontsize=8)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    fig.suptitle(f"Sun2024 comb candidates, all sample subjects, {window:.1f} s")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare plausible Sun2024 comb-filter candidates on sample subjects.")
    parser.add_argument("--subjects", default="11,13,12,14")
    parser.add_argument("--windows", default="0.6,1.0")
    parser.add_argument("--repetitions", type=int, default=5, choices=[5, 8])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "comb_candidates_smoke_20260720",
    )
    args = parser.parse_args()

    available = sorted(record.subject for record in iter_sun_cnt_files(splits=["forty_targets"]))
    subjects = parse_subjects(args.subjects, available)
    missing = sorted(set(subjects) - set(available))
    if missing:
        raise ValueError(f"Missing subjects: {missing}")
    windows = parse_windows(args.windows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(subjects))) as pool:
        futures = {pool.submit(run_subject, subject, windows, args.repetitions): subject for subject in subjects}
        for future in as_completed(futures):
            subject = futures[future]
            subject_summary, subject_trials = future.result()
            summary_rows.extend(subject_summary)
            trial_rows.extend(subject_trials)
            print(f"[Sun2024 comb candidates] S{subject:02d} complete", flush=True)

    summary_rows.sort(key=lambda row: (int(row["subject"]), float(row["window"]), METHODS.index(str(row["method"]))))
    trial_rows.sort(
        key=lambda row: (
            int(row["subject"]),
            float(row["window"]),
            METHODS.index(str(row["method"])),
            int(row["fold"]),
            int(row["target_id"]),
        )
    )
    aggregate = aggregate_summary(summary_rows)
    write_csv(args.output / "summary.csv", summary_rows)
    write_csv(args.output / "trials.csv", trial_rows)
    write_csv(args.output / "aggregate_summary.csv", aggregate)
    plot_curves(args.output / "figures" / "comb_candidate_acc_itr.png", aggregate)
    plot_confusions(args.output / "figures" / "comb_candidate_confusions_0.6s.png", trial_rows, window=0.6)
    manifest = {
        "task": "ssvep_efficient_dual_frequency_sun2024",
        "run_type": "diagnostic_comb_candidates_smoke",
        "subjects": subjects,
        "cohorts": {"development": [subject for subject in subjects if subject in {11, 13}], "confirmation": [subject for subject in subjects if subject not in {11, 13}]},
        "windows": windows,
        "repetitions": args.repetitions,
        "channels": "all EEG",
        "fixed_onset_shift_seconds": ONSET_SHIFT_SECONDS,
        "methods": list(METHODS),
        "phase_codebook_source": "Sun et al. (2024) Figure 2(III), manually digitized",
        "status": "diagnostic sensitivity comparison; not an author comb-filter reproduction",
        "run_state": "complete",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
