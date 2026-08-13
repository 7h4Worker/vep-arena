from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.binocular import (  # noqa: E402
    DUAL_ALPHA_BLOCKS,
    DUAL_ALPHA_CLASSES,
    DUAL_ALPHA_ITR_SHIFT_SECONDS,
    DUAL_ALPHA_PARADIGMS,
    DUAL_ALPHA_SAMPLING_RATE,
    dual_alpha_apply_trca_filterbank,
    load_dual_alpha_epochs,
    parse_dual_alpha_codebooks,
)
from vep_arena.methods.tdca import TDCA  # noqa: E402
from vep_arena.methods.traditional import TRCA, filterbank_weights, multi_frequency_reference_signals  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METHODS = ("TDCA", "ETRCA")
DEFAULT_PARADIGMS = tuple(DUAL_ALPHA_PARADIGMS)
PARADIGM_SHORT = {
    "Checkerboard_Arrangment": "CA",
    "Binocular_Vision": "BV",
    "Binocular-Swap_Vision": "BsV",
}


def parse_subjects(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, stop = (int(value) for value in item.split("-", 1))
            values.extend(range(start, stop + 1))
        else:
            values.append(int(item))
    return sorted(set(values))


def parse_windows(text: str) -> list[float]:
    return sorted({float(item.strip()) for item in text.split(",") if item.strip()})


def parse_paradigms(text: str) -> list[str]:
    paradigms = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(paradigms) - set(DEFAULT_PARADIGMS))
    if unknown:
        raise ValueError(f"Unknown paradigms: {unknown}")
    return paradigms


def itr_bits_per_min(accuracy: float, window: float) -> float:
    p = min(max(float(accuracy), 1e-12), 1.0 - 1e-12)
    bits = math.log2(DUAL_ALPHA_CLASSES) + p * math.log2(p) + (1.0 - p) * math.log2(
        (1.0 - p) / (DUAL_ALPHA_CLASSES - 1)
    )
    return bits * 60.0 / (window + DUAL_ALPHA_ITR_SHIFT_SECONDS)


def swapped_target_map(target_freqs: tuple[tuple[float, ...], ...]) -> dict[int, int]:
    lookup = {tuple(round(float(value), 6) for value in frequencies): target for target, frequencies in enumerate(target_freqs)}
    output: dict[int, int] = {}
    for target, frequencies in enumerate(target_freqs):
        partner = lookup.get(tuple(round(float(value), 6) for value in reversed(frequencies)))
        if partner is not None and partner != target:
            output[target] = partner
    return output


def run_unit(
    subject: int,
    paradigm: str,
    windows: list[float],
    n_fbs: int,
    harmonics: int,
    n_delay: int,
    n_components: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    max_seconds = max(windows)
    data = load_dual_alpha_epochs(
        paradigm=paradigm,
        subject=subject,
        window_seconds=max_seconds,
        channel_set="occipital9",
    )
    labels = np.arange(DUAL_ALPHA_CLASSES, dtype=np.int64)
    swap_map = swapped_target_map(data.target_freqs)
    summary_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []

    for window in windows:
        samples = int(round(window * data.sampling_rate))
        needed = samples + n_delay
        available = min(needed, data.x.shape[-1])
        raw_epochs = data.x[..., :available]
        tail_padding_samples = needed - available
        if tail_padding_samples:
            raw_epochs = np.pad(raw_epochs, ((0, 0), (0, 0), (0, 0), (0, tail_padding_samples)))
        filtered = dual_alpha_apply_trca_filterbank(raw_epochs, n_fbs=n_fbs, fs=data.sampling_rate)
        references = multi_frequency_reference_signals(
            target_frequencies=data.target_freqs,
            samples=samples,
            fs=data.sampling_rate,
            harmonics=harmonics,
        )
        method_true = {method: [] for method in METHODS}
        method_pred = {method: [] for method in METHODS}

        for block in range(DUAL_ALPHA_BLOCKS):
            train_blocks = [index for index in range(DUAL_ALPHA_BLOCKS) if index != block]
            train_x = filtered[:, train_blocks].reshape(
                -1,
                filtered.shape[2],
                filtered.shape[3],
                filtered.shape[4],
            )
            train_y = np.repeat(labels, len(train_blocks))
            test_x = filtered[:, block]

            tdca = TDCA(
                n_components=n_components,
                n_delay=n_delay,
                fb_weights=filterbank_weights(n_fbs),
            )
            tdca.fit(train_x, train_y, references)
            tdca_pred, tdca_band_scores = tdca.predict(test_x)
            tdca_scores = np.einsum("f,tfc->tc", tdca.fb_weights, tdca_band_scores)

            etrca = TRCA(n_fbs=n_fbs, ensemble=True)
            etrca.fit(train_x[..., :samples], train_y)
            etrca_pred, etrca_scores = etrca.predict(test_x[..., :samples])

            for method, predicted, scores in (
                ("TDCA", tdca_pred, tdca_scores),
                ("ETRCA", etrca_pred, etrca_scores),
            ):
                method_true[method].extend(labels.tolist())
                method_pred[method].extend(predicted.tolist())
                for target in range(DUAL_ALPHA_CLASSES):
                    trial_rows.append(
                        {
                            "subject": subject,
                            "paradigm": paradigm,
                            "channel_set": "occipital9",
                            "window": window,
                            "method": method,
                            "block": block + 1,
                            "target_id": target + 1,
                            "predicted_id": int(predicted[target] + 1),
                            "correct": int(predicted[target] == target),
                            "swap_partner_error": int(
                                target in swap_map and predicted[target] == swap_map[target]
                            ),
                            "score_true": float(scores[target, target]),
                            "score_predicted": float(scores[target, predicted[target]]),
                        }
                    )

        for method in METHODS:
            true = np.asarray(method_true[method], dtype=np.int64)
            predicted = np.asarray(method_pred[method], dtype=np.int64)
            errors = predicted != true
            swap_errors = np.asarray(
                [
                    int(target in swap_map and prediction == swap_map[target])
                    for target, prediction in zip(true, predicted)
                ],
                dtype=np.int64,
            )
            accuracy = float(np.mean(predicted == true))
            summary_rows.append(
                {
                    "subject": subject,
                    "paradigm": paradigm,
                    "channel_set": "occipital9",
                    "window": window,
                    "method": method,
                    "classes": DUAL_ALPHA_CLASSES,
                    "blocks": DUAL_ALPHA_BLOCKS,
                    "test_trials": true.size,
                    "n_fbs": n_fbs,
                    "harmonics": harmonics,
                    "n_delay": n_delay if method == "TDCA" else 0,
                    "n_components": n_components if method == "TDCA" else "",
                    "tail_padding_samples": tail_padding_samples if method == "TDCA" else 0,
                    "accuracy": accuracy,
                    "itr_bits_per_min": itr_bits_per_min(accuracy, window),
                    "errors": int(np.sum(errors)),
                    "swap_partner_errors": int(np.sum(swap_errors)),
                    "swap_error_fraction": float(np.sum(swap_errors) / np.sum(errors)) if np.any(errors) else 0.0,
                }
            )
    return summary_rows, trial_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for paradigm in DEFAULT_PARADIGMS:
        for window in sorted({float(row["window"]) for row in rows}):
            for method in METHODS:
                selected = [
                    row
                    for row in rows
                    if row["paradigm"] == paradigm
                    and float(row["window"]) == window
                    and row["method"] == method
                ]
                if not selected:
                    continue
                values = np.asarray([float(row["accuracy"]) for row in selected])
                errors = sum(int(row["errors"]) for row in selected)
                swap_errors = sum(int(row["swap_partner_errors"]) for row in selected)
                mean_accuracy = float(np.mean(values))
                output.append(
                    {
                        "paradigm": paradigm,
                        "channel_set": "occipital9",
                        "window": window,
                        "method": method,
                        "subjects": values.size,
                        "accuracy": mean_accuracy,
                        "accuracy_sem": float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0,
                        "itr_bits_per_min": itr_bits_per_min(mean_accuracy, window),
                        "errors": errors,
                        "swap_partner_errors": swap_errors,
                        "swap_error_fraction": swap_errors / errors if errors else 0.0,
                    }
                )
    return output


def plot_accuracy_curves(path: Path, aggregate: list[dict[str, object]]) -> None:
    all_windows = sorted({float(row["window"]) for row in aggregate})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), dpi=160, sharey=True)
    for axis, paradigm in zip(axes, DEFAULT_PARADIGMS):
        for method in METHODS:
            selected = sorted(
                [row for row in aggregate if row["paradigm"] == paradigm and row["method"] == method],
                key=lambda row: float(row["window"]),
            )
            windows = np.asarray([float(row["window"]) for row in selected])
            accuracy = np.asarray([float(row["accuracy"]) for row in selected])
            sem = np.asarray([float(row["accuracy_sem"]) for row in selected])
            axis.errorbar(windows, accuracy, yerr=sem, marker="o", capsize=3, label=method)
        axis.set_title(paradigm.replace("_", " "))
        axis.set_xlabel("Analysis window (s)")
        axis.set_xticks(all_windows)
        axis.set_xticklabels([f"{window:.1f}" for window in all_windows], rotation=45, ha="right")
        axis.set_xlim(all_windows[0] - 0.04, all_windows[-1] + 0.04)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend()
    fig.suptitle("Dual-Alpha native Arena TDCA: paradigm comparison")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_subject_heatmap(path: Path, rows: list[dict[str, object]], subjects: list[int], windows: list[float]) -> None:
    columns = [(paradigm, window) for paradigm in DEFAULT_PARADIGMS for window in windows]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(14, len(columns) * 0.7), max(5.8, len(subjects) * 0.55)),
        dpi=160,
        constrained_layout=True,
    )
    for axis, method in zip(axes, METHODS):
        matrix = np.full((len(subjects), len(columns)), np.nan)
        for row in rows:
            if row["method"] != method:
                continue
            row_index = subjects.index(int(row["subject"]))
            column_index = columns.index((str(row["paradigm"]), float(row["window"])))
            matrix[row_index, column_index] = float(row["accuracy"])
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        axis.set_title(method)
        axis.set_yticks(range(len(subjects)), [f"S{subject}" for subject in subjects])
        axis.set_xticks(
            range(len(columns)),
            [f"{PARADIGM_SHORT[paradigm]}\n{window:.1f}s" for paradigm, window in columns],
        )
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] < 0.55 else "black",
                    fontsize=8,
                )
        fig.colorbar(image, ax=axis, fraction=0.018, pad=0.015)
    fig.suptitle("Dual-Alpha subject x paradigm x window accuracy")
    fig.savefig(path)
    plt.close(fig)


def plot_method_delta(path: Path, rows: list[dict[str, object]], subjects: list[int], windows: list[float]) -> None:
    columns = [(paradigm, window) for paradigm in DEFAULT_PARADIGMS for window in windows]
    matrix = np.full((len(subjects), len(columns)), np.nan)
    for row_index, subject in enumerate(subjects):
        for column_index, (paradigm, window) in enumerate(columns):
            values = {
                str(row["method"]): float(row["accuracy"])
                for row in rows
                if int(row["subject"]) == subject
                and row["paradigm"] == paradigm
                and float(row["window"]) == window
            }
            matrix[row_index, column_index] = values["TDCA"] - values["ETRCA"]
    limit = max(0.2, float(np.nanmax(np.abs(matrix))))
    fig, axis = plt.subplots(
        figsize=(max(14, len(columns) * 0.7), max(3.2, len(subjects) * 0.3)),
        dpi=160,
        constrained_layout=True,
    )
    image = axis.imshow(matrix, vmin=-limit, vmax=limit, cmap="coolwarm", aspect="auto")
    axis.set_title("Dual-Alpha TDCA minus ETRCA accuracy")
    axis.set_yticks(range(len(subjects)), [f"S{subject}" for subject in subjects])
    axis.set_xticks(
        range(len(columns)),
        [f"{PARADIGM_SHORT[paradigm]}\n{window:.1f}s" for paradigm, window in columns],
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:+.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.015, label="Accuracy difference")
    fig.savefig(path)
    plt.close(fig)


def plot_confusions(path: Path, trials: list[dict[str, object]], window: float) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=150)
    for row_index, method in enumerate(METHODS):
        for column_index, paradigm in enumerate(DEFAULT_PARADIGMS):
            selected = [
                row
                for row in trials
                if row["method"] == method
                and row["paradigm"] == paradigm
                and float(row["window"]) == window
            ]
            true = [int(row["target_id"]) - 1 for row in selected]
            predicted = [int(row["predicted_id"]) - 1 for row in selected]
            matrix = confusion_matrix(true, predicted, labels=list(range(DUAL_ALPHA_CLASSES)))
            axis = axes[row_index, column_index]
            axis.imshow(matrix, cmap="viridis", interpolation="nearest")
            axis.set_title(f"{method} - {paradigm.replace('_', ' ')}", fontsize=9)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
    fig.suptitle(f"Dual-Alpha aggregate confusion matrices at {window:.1f} s")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def reference_summary(paradigms: list[str]) -> list[dict[str, object]]:
    codebooks = parse_dual_alpha_codebooks()
    rows: list[dict[str, object]] = []
    for paradigm in paradigms:
        codebook = codebooks[paradigm]
        ordered = [(round(codebook.freq1[index], 6), round(codebook.freq2[index], 6)) for index in range(codebook.n_targets)]
        unordered = {tuple(sorted(pair)) for pair in ordered}
        swap_map = swapped_target_map(tuple(ordered))
        rows.append(
            {
                "paradigm": paradigm,
                "targets": len(ordered),
                "unique_ordered_frequency_pairs": len(set(ordered)),
                "unique_unordered_reference_subspaces": len(unordered),
                "targets_with_swap_partner": len(swap_map),
                "phase_available": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Arena-native TDCA smoke for the Dual-Alpha dataset.")
    parser.add_argument("--subjects", default="1-3")
    parser.add_argument("--paradigms", default=",".join(DEFAULT_PARADIGMS))
    parser.add_argument("--windows", default="0.4,0.6,1.0")
    parser.add_argument("--n-fbs", type=int, default=5)
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--n-delay", type=int, default=5)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "tdca_smoke_s1_s3_20260720",
    )
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    paradigms = parse_paradigms(args.paradigms)
    windows = parse_windows(args.windows)
    if args.output.exists() and not args.resume and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; pass --resume or choose a new path: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)

    units = [(subject, paradigm) for subject in subjects for paradigm in paradigms]
    manifest_path = args.output / "manifest.json"
    unit_manifest_path = args.output / "unit_manifest.csv"
    summary_path = args.output / "summary.csv"
    trials_path = args.output / "trials.csv"
    unit_rows = read_csv(unit_manifest_path)
    latest_status: dict[tuple[int, str], str] = {}
    for row in unit_rows:
        latest_status[(int(row["subject"]), str(row["paradigm"]))] = str(row["status"])
    pending = [unit for unit in units if latest_status.get(unit) != "complete"]
    run_type = "arena_native_tdca_full" if len(subjects) == 35 else "arena_native_tdca_smoke"
    manifest = {
        "task": "BL08_ssvep_dual_alpha",
        "run_type": run_type,
        "subjects": subjects,
        "paradigms": paradigms,
        "windows": windows,
        "channel_set": "occipital9",
        "methods": list(METHODS),
        "n_fbs": args.n_fbs,
        "harmonics": args.harmonics,
        "n_delay": args.n_delay,
        "n_components": args.n_components,
        "cv": "five-block leave-one-block-out",
        "reference_phase": "zero; public Stimulate_Code.txt contains frequencies only",
        "max_window_tail_padding": "zero-pad up to n_delay samples when the source epoch ends",
        "workers": min(args.workers, len(units)),
        "expected_units": len(units),
        "status": "Arena extension; not an author TDCA reproduction",
        "run_state": "running",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    failed_units: list[tuple[int, str, str]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(units))) as pool:
        futures = {
            pool.submit(
                run_unit,
                subject,
                paradigm,
                windows,
                args.n_fbs,
                args.harmonics,
                args.n_delay,
                args.n_components,
            ): (subject, paradigm)
            for subject, paradigm in pending
        }
        for future in as_completed(futures):
            subject, paradigm = futures[future]
            try:
                unit_summary, unit_trials = future.result()
            except Exception as exc:
                error = repr(exc)
                append_csv(
                    unit_manifest_path,
                    [{"subject": subject, "paradigm": paradigm, "status": "failed", "summary_rows": 0, "trial_rows": 0, "error": error}],
                )
                failed_units.append((subject, paradigm, error))
                print(f"[Dual-Alpha TDCA] S{subject:02d} {paradigm} failed: {error}", flush=True)
                continue
            append_csv(summary_path, unit_summary)
            append_csv(trials_path, unit_trials)
            append_csv(
                unit_manifest_path,
                [{"subject": subject, "paradigm": paradigm, "status": "complete", "summary_rows": len(unit_summary), "trial_rows": len(unit_trials), "error": ""}],
            )
            print(f"[Dual-Alpha TDCA] S{subject:02d} {paradigm} complete", flush=True)

    summary_rows = read_csv(summary_path)
    trial_rows = read_csv(trials_path)
    aggregate = aggregate_summary(summary_rows)
    references = reference_summary(paradigms)
    write_csv(args.output / "aggregate_summary.csv", aggregate)
    write_csv(args.output / "reference_summary.csv", references)
    plot_accuracy_curves(figures / "tdca_etrca_paradigm_curves.png", aggregate)
    plot_subject_heatmap(figures / "subject_paradigm_window_heatmap.png", summary_rows, subjects, windows)
    plot_method_delta(figures / "tdca_minus_etrca_delta_heatmap.png", summary_rows, subjects, windows)
    plot_confusions(figures / "confusions_0.6s.png", trial_rows, window=0.6)

    final_units = read_csv(unit_manifest_path)
    final_status: dict[tuple[int, str], str] = {}
    for row in final_units:
        final_status[(int(row["subject"]), str(row["paradigm"]))] = str(row["status"])
    completed_count = sum(final_status.get(unit) == "complete" for unit in units)
    manifest["completed_units"] = completed_count
    manifest["failed_units"] = [
        {"subject": subject, "paradigm": paradigm, "error": error}
        for subject, paradigm, error in failed_units
    ]
    manifest["run_state"] = "complete" if completed_count == len(units) else "partial"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if completed_count != len(units):
        raise RuntimeError(f"Only {completed_count}/{len(units)} units completed; rerun with --resume.")


if __name__ == "__main__":
    main()
