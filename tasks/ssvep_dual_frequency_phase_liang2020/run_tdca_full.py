from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TASK) not in sys.path:
    sys.path.insert(0, str(TASK))

from run_tdca_smoke import (  # noqa: E402
    DEFAULT_BANDS,
    itr_bits_per_min,
    load_group_epochs,
    method1_references,
    read_label_map,
    splits,
)
from vep_arena.data.dual_frequency import (  # noqa: E402
    LIANG2020_METHOD1_DUAL_FREQUENCY_PHASE_CODEBOOK,
    LIANG2020_SAMPLING_RATE,
    iter_liang_mat_files,
)
from vep_arena.methods.tdca import TDCA  # noqa: E402
from vep_arena.methods.traditional import filterbank_weights  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_WINDOWS = {
    "exp3": (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0),
    "exp4": (0.2, 0.4, 0.6, 0.8, 1.0),
}
CONDITION_LABELS = {
    "exp3": "method1_dual_frequency_phase",
    "exp4": "method1_dual_frequency_phase",
}
CONDITIONS = {"exp3": 3, "exp4": 1}


@dataclass(frozen=True)
class SubjectTask:
    subject: int
    experiments: tuple[str, ...]
    windows: tuple[tuple[str, tuple[float, ...]], ...]
    ordered_codebook: tuple[tuple[float, float, float], ...]
    n_bands: int
    harmonics: int
    n_components: int
    n_delay: int
    onset_shift: float


def parse_subjects(text: str, available: list[int]) -> list[int]:
    if text.strip().lower() == "all":
        return available
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
    unknown = sorted(set(values).difference(available))
    if unknown:
        raise ValueError(f"Subjects unavailable for selected experiments: {unknown}")
    return sorted(set(values))


def parse_experiments(text: str) -> tuple[str, ...]:
    values = tuple(value.strip().lower() for value in text.split(",") if value.strip())
    unknown = sorted(set(values).difference(DEFAULT_WINDOWS))
    if unknown:
        raise ValueError(f"Only Method 1 codebook experiments are supported: {unknown}")
    return values


def parse_windows(text: str, experiments: tuple[str, ...]) -> dict[str, tuple[float, ...]]:
    if text.strip().lower() == "paper":
        return {experiment: DEFAULT_WINDOWS[experiment] for experiment in experiments}
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("At least one analysis window is required")
    return {experiment: values for experiment in experiments}


def run_subject(task: SubjectTask) -> dict[str, object]:
    windows_by_experiment = dict(task.windows)
    records = iter_liang_mat_files(experiments=task.experiments, subjects=[task.subject])
    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    unit_rows: list[dict[str, object]] = []
    for experiment in task.experiments:
        group_records = [record for record in records if record.condition == CONDITIONS[experiment]]
        if not group_records:
            raise RuntimeError(f"Missing {experiment} Method 1 files for subject {task.subject}")
        for window in windows_by_experiment[experiment]:
            x, y, blocks = load_group_epochs(
                group_records,
                window=window,
                onset_shift=task.onset_shift,
                n_bands=task.n_bands,
                n_delay=task.n_delay,
            )
            refs = method1_references(
                int(round(window * LIANG2020_SAMPLING_RATE)),
                task.harmonics,
                task.ordered_codebook,
            )
            true_all: list[int] = []
            predicted_all: list[int] = []
            train_total = 0
            test_total = 0
            used_splits = 0
            for split_name, train_mask, test_mask in splits(experiment, blocks):
                if not set(np.unique(y[test_mask])).issubset(set(np.unique(y[train_mask]))):
                    raise RuntimeError(f"Invalid class coverage for S{task.subject} {experiment} {split_name}")
                model = TDCA(
                    n_components=task.n_components,
                    n_delay=task.n_delay,
                    fb_weights=filterbank_weights(task.n_bands),
                )
                model.fit(x[train_mask], y[train_mask], refs)
                predicted, scores_by_band = model.predict(x[test_mask])
                scores = np.einsum("f,tfc->tc", model.fb_weights, scores_by_band)
                true = y[test_mask]
                indices = np.flatnonzero(test_mask)
                for local_index, index in enumerate(indices):
                    prediction_rows.append(
                        {
                            "experiment": experiment,
                            "subject": task.subject,
                            "condition": CONDITIONS[experiment],
                            "condition_label": CONDITION_LABELS[experiment],
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
                predicted_all.extend(predicted.tolist())
                train_total += int(np.sum(train_mask))
                test_total += int(np.sum(test_mask))
                used_splits += 1
            accuracy = float(np.mean(np.asarray(true_all) == np.asarray(predicted_all)))
            summary_rows.append(
                {
                    "experiment": experiment,
                    "subject": task.subject,
                    "condition": CONDITIONS[experiment],
                    "condition_label": CONDITION_LABELS[experiment],
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
            unit_rows.append(
                {
                    "experiment": experiment,
                    "subject": task.subject,
                    "window": window,
                    "method": "TDCA",
                    "status": "complete",
                    "test_trials": test_total,
                    "accuracy": accuracy,
                }
            )
    return {"summary_rows": summary_rows, "prediction_rows": prediction_rows, "unit_rows": unit_rows}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows were produced for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary(summary: pd.DataFrame, origin_subject: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, source in (("all_subjects", summary), ("external_subjects", summary[summary["subject"] != origin_subject])):
        for keys, group in source.groupby(["experiment", "condition", "condition_label", "window", "method"], sort=True):
            experiment, condition, label, window, method = keys
            rows.append(
                {
                    "scope": scope,
                    "experiment": experiment,
                    "condition": condition,
                    "condition_label": label,
                    "window": window,
                    "method": method,
                    "subjects": int(group["subject"].nunique()),
                    "accuracy": float(group["accuracy"].mean()),
                    "accuracy_sem": float(group["accuracy"].std(ddof=1) / math.sqrt(len(group))) if len(group) > 1 else 0.0,
                    "itr_bits_per_min": float(group["itr_bits_per_min"].mean()),
                    "itr_sem": float(group["itr_bits_per_min"].std(ddof=1) / math.sqrt(len(group))) if len(group) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["scope", "experiment", "window"])


def plot_curves(aggregate: pd.DataFrame, figures: Path) -> None:
    source = aggregate[aggregate["scope"] == "external_subjects"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, metric, sem, ylabel in (
        (axes[0], "accuracy", "accuracy_sem", "Accuracy"),
        (axes[1], "itr_bits_per_min", "itr_sem", "ITR (bits/min)"),
    ):
        for experiment, color in (("exp3", "#6B5CA5"), ("exp4", "#007C91")):
            rows = source[source["experiment"] == experiment].sort_values("window")
            if rows.empty:
                continue
            axis.errorbar(
                rows["window"], rows[metric], yerr=rows[sem], color=color, marker="o", linewidth=2.2,
                capsize=3, label=f"{experiment} Method 1 TDCA (S2-S12)",
            )
        axis.set_xlabel("Analysis window (s)")
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", color="#D8DEE9", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle("Liang2020 diagnostic full TDCA: fixed S1-derived label map", fontsize=14)
    figure.text(
        0.5, -0.02,
        "Curves exclude S1 because its Exp3 data calibrated the MAT-label mapping. "
        "Exp3: leave-one-block-out; Exp4: runs 1-6 train, 7-9 test; ITR denominator: window + 0.5 s.",
        ha="center", va="top", fontsize=8.8, color="#4F5B66",
    )
    figure.savefig(figures / "tdca_window_curves_external_subjects.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_confusions(predictions: pd.DataFrame, figures: Path) -> None:
    for experiment, window in (("exp3", 0.6), ("exp4", 0.6)):
        rows = predictions[(predictions["experiment"] == experiment) & np.isclose(predictions["window"], window)]
        if rows.empty:
            continue
        matrix = confusion_matrix(rows["target_id"], rows["predicted_id"], labels=list(range(1, 41)))
        figure, axis = plt.subplots(figsize=(7, 6), dpi=160)
        image = axis.imshow(matrix, cmap="viridis", interpolation="nearest")
        axis.set_title(f"Liang2020 diagnostic TDCA {experiment}, {window:.1f}s")
        axis.set_xlabel("Predicted target")
        axis.set_ylabel("True target")
        axis.set_xticks(np.arange(0, 40, 5))
        axis.set_yticks(np.arange(0, 40, 5))
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(figures / f"confusion_{experiment}_tdca_0.6s.png")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic full TDCA run for Liang2020 Method 1 coded conditions.")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--experiments", default="exp3,exp4")
    parser.add_argument("--windows", default="paper")
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--n-bands", type=int, default=5, choices=range(1, len(DEFAULT_BANDS) + 1))
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-delay", type=int, default=5)
    parser.add_argument("--onset-shift", type=float, default=0.14)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=TASK / "results" / "diagnostic_full_tdca_method1_20260720")
    args = parser.parse_args()

    experiments = parse_experiments(args.experiments)
    all_records = iter_liang_mat_files(experiments=experiments)
    available = sorted({record.subject for record in all_records})
    subjects = parse_subjects(args.subjects, available)
    windows = parse_windows(args.windows, experiments)
    label_map_path = args.label_map if args.label_map.is_absolute() else PROJECT_ROOT / args.label_map
    label_to_figure_index = read_label_map(label_map_path)
    ordered_codebook = tuple(LIANG2020_METHOD1_DUAL_FREQUENCY_PHASE_CODEBOOK[int(index)] for index in label_to_figure_index)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    subject_tasks = [
        SubjectTask(
            subject=subject,
            experiments=experiments,
            windows=tuple((experiment, windows[experiment]) for experiment in experiments),
            ordered_codebook=ordered_codebook,
            n_bands=args.n_bands,
            harmonics=args.harmonics,
            n_components=args.n_components,
            n_delay=args.n_delay,
            onset_shift=args.onset_shift,
        )
        for subject in subjects
    ]
    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    unit_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(subject_tasks))) as executor:
        futures = {executor.submit(run_subject, task): task.subject for task in subject_tasks}
        for future in as_completed(futures):
            subject = futures[future]
            result = future.result()
            summary_rows.extend(result["summary_rows"])
            prediction_rows.extend(result["prediction_rows"])
            unit_rows.extend(result["unit_rows"])
            print(f"[Liang2020 diagnostic TDCA] subject={subject:02d} complete", flush=True)

    summary_rows.sort(key=lambda row: (row["experiment"], row["subject"], row["window"]))
    prediction_rows.sort(key=lambda row: (row["experiment"], row["subject"], row["window"], row["run"], row["target_id"]))
    unit_rows.sort(key=lambda row: (row["experiment"], row["subject"], row["window"]))
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "predictions.csv", prediction_rows)
    write_csv(output / "unit_manifest.csv", unit_rows)
    summary = pd.DataFrame(summary_rows)
    aggregate = aggregate_summary(summary, origin_subject=1)
    aggregate.to_csv(output / "aggregate_summary.csv", index=False)
    plot_curves(aggregate, figures)
    plot_confusions(pd.DataFrame(prediction_rows), figures)
    manifest = {
        "task": "ssvep_dual_frequency_phase_liang2020",
        "run_type": "diagnostic_full_tdca_method1",
        "run_state": "complete",
        "subjects": subjects,
        "experiments": list(experiments),
        "conditions": CONDITIONS,
        "windows": {key: list(value) for key, value in windows.items()},
        "sampling_rate": LIANG2020_SAMPLING_RATE,
        "onset_shift_seconds": args.onset_shift,
        "n_bands": args.n_bands,
        "bands": DEFAULT_BANDS[: args.n_bands],
        "harmonics": args.harmonics,
        "n_components": args.n_components,
        "n_delay_samples": args.n_delay,
        "workers": min(args.workers, len(subject_tasks)),
        "label_map": str(label_map_path),
        "label_map_origin_subject": 1,
        "label_map_status": "inferred from Exp3-C3 Subject 1 spectral amplitudes; not author-confirmed",
        "scope_limit": "Only Exp3-C3 and Exp4-C1 use the digitized Method 1 Figure 1 codebook. Exp3-C1/C2 need Figure 16 codebooks.",
        "summary_rows": len(summary_rows),
        "prediction_rows": len(prediction_rows),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
