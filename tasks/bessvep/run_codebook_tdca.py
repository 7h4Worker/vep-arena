from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vep_arena.data.embc_jbhi import (  # noqa: E402
    analysis_epochs,
    dataset_spec,
    load_subject,
    load_target_frequency_pairs,
    resolve_dataset_root,
)
from vep_arena.methods.tdca import TDCA  # noqa: E402
from vep_arena.methods.traditional import TRCA, filterbank_weights  # noqa: E402


DEFAULTS = {
    "embc9": {"subjects": ("S04", "S06"), "windows": (0.5, 1.0, 2.0), "bands": 3},
    "jbhi16": {"subjects": ("S04", "S06"), "windows": (0.4, 1.0, 2.0), "bands": 5},
    "jbhi35": {"subjects": ("S04", "S06"), "windows": (0.4, 1.0, 2.0), "bands": 5},
}

BASELINE_TASKS = {
    "embc9": "BS01_embc_9t",
    "jbhi16": "BS02_16t",
    "jbhi35": "BS03_35t",
}


def padded_epochs(dataset: str, subject_id: str, window: float, bands: int, delay: int) -> np.ndarray:
    subject = load_subject(dataset, subject_id, resolve_dataset_root(dataset))
    spec = dataset_spec(dataset)
    extra = delay / spec.analysis_sampling_rate
    load_window = min(spec.stimulation_seconds, window + extra)
    epochs = analysis_epochs(subject, load_window, n_bands=bands)
    needed = int(round(window * spec.analysis_sampling_rate)) + delay
    if epochs.shape[-1] < needed:
        epochs = np.pad(epochs, [(0, 0), (0, 0), (0, 0), (0, 0), (0, needed - epochs.shape[-1])])
    return epochs[..., :needed]


def eeg_references(train_x: np.ndarray, train_y: np.ndarray, classes: int, samples: int) -> list[np.ndarray]:
    references = []
    for target in range(classes):
        class_mean = train_x[train_y == target, 0, :, :samples].mean(axis=0)
        class_mean = class_mean - class_mean.mean(axis=1, keepdims=True)
        references.append(class_mean)
    return references


def spectral_references(
    dataset: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    classes: int,
    samples: int,
    sampling_rate: int,
) -> tuple[list[np.ndarray], int]:
    pairs = load_target_frequency_pairs(dataset, resolve_dataset_root(dataset))
    time = np.arange(samples, dtype=np.float64) / sampling_rate
    references = []
    void_fallbacks = 0
    for target, pair in enumerate(pairs):
        active = tuple(dict.fromkeys(frequency for frequency in pair if frequency > 0))
        if not active:
            class_mean = train_x[train_y == target, 0, :, :samples].mean(axis=0)
            references.append(class_mean - class_mean.mean(axis=1, keepdims=True))
            void_fallbacks += 1
            continue
        rows = []
        for harmonic in (1, 2, 3):
            for frequency in active:
                rows.append(np.sin(2 * np.pi * harmonic * frequency * time))
                rows.append(np.cos(2 * np.pi * harmonic * frequency * time))
        references.append(np.stack(rows))
    if len(references) != classes:
        raise RuntimeError("Codebook reference count does not match target count.")
    return references, void_fallbacks


def run_unit(
    dataset: str,
    subject_id: str,
    window: float,
    bands: int,
    delay: int,
    components: int,
    include_etrca: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    started = time.perf_counter()
    spec = dataset_spec(dataset)
    samples = int(round(window * spec.analysis_sampling_rate))
    epochs = padded_epochs(dataset, subject_id, window, bands, delay)
    labels = np.arange(spec.targets, dtype=np.int64)
    rows = []
    for test_block in range(spec.blocks):
        train_blocks = [block for block in range(spec.blocks) if block != test_block]
        train_x = epochs[:, train_blocks].reshape(-1, bands, 9, epochs.shape[-1])
        train_y = np.repeat(labels, len(train_blocks))
        test_x = epochs[:, test_block]
        eeg_refs = eeg_references(train_x, train_y, spec.targets, samples)
        spectral_refs, void_fallbacks = spectral_references(
            dataset, train_x, train_y, spec.targets, samples, spec.analysis_sampling_rate
        )
        tdca_predictions = {}
        for method, references in (("EEG_TDCA", eeg_refs), ("SPECTRAL_TDCA", spectral_refs)):
            tdca = TDCA(n_components=components, n_delay=delay, fb_weights=filterbank_weights(bands))
            tdca.fit(train_x, train_y, references)
            tdca_predictions[method], _ = tdca.predict(test_x)
        predictions_by_method = list(tdca_predictions.items())
        if include_etrca:
            etrca = TRCA(n_fbs=bands, ensemble=True)
            etrca.fit(train_x[..., :samples], train_y)
            etrca_pred, _ = etrca.predict(test_x[..., :samples])
            predictions_by_method.append(("ETRCA", etrca_pred))
        for method, predictions in predictions_by_method:
            for target, prediction in enumerate(predictions):
                rows.append(
                    {
                        "dataset": dataset,
                        "subject": subject_id,
                        "window_seconds": window,
                        "method": method,
                        "fold": test_block + 1,
                        "true": target + 1,
                        "pred": int(prediction) + 1,
                        "correct": int(prediction == target),
                        "void_reference_fallback": int(method == "SPECTRAL_TDCA" and void_fallbacks > 0),
                    }
                )
    method_count = 3 if include_etrca else 2
    unit = {
        "dataset": dataset,
        "subject": subject_id,
        "window_seconds": window,
        "filter_bands": bands,
        "n_delay": delay,
        "n_components": components,
        "folds": spec.blocks,
        "methods": ["EEG_TDCA", "SPECTRAL_TDCA"] + (["ETRCA"] if include_etrca else []),
        "expected_predictions": spec.blocks * spec.targets * method_count,
        "prediction_count": len(rows),
        "runtime_seconds": time.perf_counter() - started,
        "status": "complete" if len(rows) == spec.blocks * spec.targets * method_count else "partial",
    }
    return rows, unit


def plot(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True, sharey=True)
    for axis, dataset in zip(axes, DEFAULTS):
        group = summary[summary["dataset"] == dataset]
        for method, method_group in group.groupby("method"):
            axis.plot(method_group["window_seconds"], method_group["accuracy"], marker="o", label=method)
        axis.set_title(dataset)
        axis.set_xlabel("Window (s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[-1].legend()
    fig.savefig(output, dpi=210)
    plt.close(fig)


def full_configs() -> dict[str, dict[str, object]]:
    configs: dict[str, dict[str, object]] = {}
    for dataset, task_name in BASELINE_TASKS.items():
        manifest_path = TASK / "results" / task_name / "full" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or int(manifest.get("error_count", -1)) != 0:
            raise RuntimeError(f"Baseline manifest is not complete: {manifest_path}")
        configs[dataset] = {
            "subjects": tuple(str(value) for value in manifest["subjects"]),
            "windows": tuple(float(value) for value in manifest["windows_seconds"]),
            "bands": int(manifest["filter_bands"]),
            "baseline_manifest": str(manifest_path),
        }
    return configs


def unit_slug(dataset: str, subject_id: str, window: float) -> str:
    return f"{dataset}_{subject_id}_w{int(round(window * 1000)):04d}"


def part_paths(parts_dir: Path, slug: str) -> tuple[Path, Path]:
    return parts_dir / f"{slug}_unit.json", parts_dir / f"{slug}_predictions.csv"


def part_complete(parts_dir: Path, slug: str) -> bool:
    unit_path, prediction_path = part_paths(parts_dir, slug)
    if not unit_path.is_file() or not prediction_path.is_file():
        return False
    try:
        unit = json.loads(unit_path.read_text(encoding="utf-8"))
        row_count = sum(1 for _ in prediction_path.open(encoding="utf-8")) - 1
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return unit.get("status") == "complete" and row_count == int(unit.get("expected_predictions", -1))


def write_part(parts_dir: Path, rows: list[dict[str, object]], unit: dict[str, object]) -> None:
    slug = unit_slug(str(unit["dataset"]), str(unit["subject"]), float(unit["window_seconds"]))
    unit_path, prediction_path = part_paths(parts_dir, slug)
    prediction_tmp = prediction_path.with_suffix(".csv.tmp")
    unit_tmp = unit_path.with_suffix(".json.tmp")
    pd.DataFrame(rows).to_csv(prediction_tmp, index=False)
    unit_tmp.write_text(json.dumps(unit, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction_tmp.replace(prediction_path)
    unit_tmp.replace(unit_path)


def run_full_worker(task: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    return run_unit(
        str(task["dataset"]),
        str(task["subject"]),
        float(task["window"]),
        int(task["bands"]),
        int(task["delay"]),
        int(task["components"]),
        include_etrca=False,
    )


def aggregate_subject_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.groupby(["dataset", "subject", "method", "window_seconds"], as_index=False).agg(
        accuracy=("correct", "mean"), predictions=("correct", "size")
    )


def aggregate_group(subject_summary: pd.DataFrame) -> pd.DataFrame:
    return subject_summary.groupby(["dataset", "method", "window_seconds"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        accuracy_sem=("accuracy", lambda values: float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0),
        subjects=("subject", "nunique"),
        predictions=("predictions", "sum"),
    )


def etrca_subject_summary(configs: dict[str, dict[str, object]]) -> pd.DataFrame:
    frames = []
    for dataset, task_name in BASELINE_TASKS.items():
        path = TASK / "results" / task_name / "full" / "predictions.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["method"] == "ETRCA"].copy()
        frame["dataset"] = dataset
        frames.append(frame)
    return aggregate_subject_accuracy(pd.concat(frames, ignore_index=True))


def run_full(args: argparse.Namespace) -> None:
    configs = full_configs()
    output = args.output or TASK / "results" / "tdca_full"
    output.mkdir(parents=True, exist_ok=True)
    parts_dir = output / "parts"
    figures_dir = output / "figures"
    parts_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    tasks = [
        {
            "dataset": dataset,
            "subject": subject,
            "window": window,
            "bands": config["bands"],
            "delay": args.n_delay,
            "components": args.n_components,
        }
        for dataset, config in configs.items()
        for subject in config["subjects"]
        for window in config["windows"]
    ]
    expected_slugs = {unit_slug(str(task["dataset"]), str(task["subject"]), float(task["window"])) for task in tasks}
    pending = [
        task
        for task in tasks
        if not (args.resume and part_complete(parts_dir, unit_slug(str(task["dataset"]), str(task["subject"]), float(task["window"]))))
    ]
    manifest = {
        "status": "running",
        "evidence_role": "diagnostic full receiver evaluation; not a paper reproduction",
        "reference_variants": {
            "EEG_TDCA": "class-specific EEG temporal subspace estimated inside each training fold",
            "SPECTRAL_TDCA": "zero-phase sine/cosine harmonics for unique active frequencies; training-fold EEG fallback only for void-void",
        },
        "stimulus_phase_dimension": False,
        "test_fold_used_for_reference": False,
        "cv": "subject-specific leave-one-block-out matching each baseline task",
        "n_delay": args.n_delay,
        "n_components": args.n_components,
        "parameter_status": "fixed from S04/S06 discovery smoke before full run",
        "workers": args.workers,
        "resume": args.resume,
        "expected_units": len(tasks),
        "pending_units_at_start": len(pending),
        "expected_predictions": sum(dataset_spec(str(task["dataset"])).targets * dataset_spec(str(task["dataset"])).blocks * 2 for task in tasks),
        "baseline_manifests": {dataset: config["baseline_manifest"] for dataset, config in configs.items()},
        "blas_threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    errors: list[str] = []
    if args.workers == 1:
        for task in pending:
            try:
                rows, unit = run_full_worker(task)
                write_part(parts_dir, rows, unit)
            except Exception as exc:
                errors.append(f"{task['dataset']} {task['subject']} {task['window']}s: {type(exc).__name__}: {exc}")
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            futures = {executor.submit(run_full_worker, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    rows, unit = future.result()
                    write_part(parts_dir, rows, unit)
                except Exception as exc:
                    errors.append(f"{task['dataset']} {task['subject']} {task['window']}s: {type(exc).__name__}: {exc}")
    unit_rows = []
    prediction_frames = []
    for slug in sorted(expected_slugs):
        if not part_complete(parts_dir, slug):
            continue
        unit_path, prediction_path = part_paths(parts_dir, slug)
        unit_rows.append(json.loads(unit_path.read_text(encoding="utf-8")))
        prediction_frames.append(pd.read_csv(prediction_path))
    units = pd.DataFrame(unit_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    subject_summary = aggregate_subject_accuracy(predictions)
    tdca_summary = aggregate_group(subject_summary)
    etrca_subjects = etrca_subject_summary(configs)
    comparison_subjects = pd.concat([subject_summary, etrca_subjects], ignore_index=True)
    comparison = aggregate_group(comparison_subjects)
    units.to_csv(output / "unit_manifest.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    subject_summary.to_csv(output / "subject.csv", index=False)
    tdca_summary.to_csv(output / "summary.csv", index=False)
    comparison_subjects.to_csv(output / "comparison_subject.csv", index=False)
    comparison.to_csv(output / "comparison_summary.csv", index=False)
    (output / "errors.log").write_text("\n".join(errors), encoding="utf-8")
    plot(comparison, figures_dir / "tdca_full_vs_etrca.png")
    complete = len(units) == len(tasks) and len(predictions) == manifest["expected_predictions"] and not errors
    manifest.update(
        {
            "status": "complete" if complete else "partial",
            "completed_units": len(units),
            "completed_predictions": len(predictions),
            "error_count": len(errors),
            "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "completed_units", "expected_units", "completed_predictions", "expected_predictions", "error_count")}, indent=2))
    if not complete:
        raise RuntimeError("TDCA full run is partial; inspect manifest.json and errors.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe EEG-reference TDCA smoke for EMBC/JBHI.")
    parser.add_argument("--n-delay", type=int, default=2)
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.full:
        run_full(args)
        return
    args.output = args.output or TASK / "results" / "tdca_smoke"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figures").mkdir(exist_ok=True)
    prediction_rows = []
    units = []
    errors = []
    for dataset, config in DEFAULTS.items():
        for subject_id in config["subjects"]:
            for window in config["windows"]:
                try:
                    rows, unit = run_unit(dataset, subject_id, window, config["bands"], args.n_delay, args.n_components)
                    prediction_rows.extend(rows)
                    units.append(unit)
                except Exception as exc:
                    errors.append(f"{dataset} {subject_id} {window:g}s: {type(exc).__name__}: {exc}")
    predictions = pd.DataFrame(prediction_rows)
    unit_frame = pd.DataFrame(units)
    summary = predictions.groupby(["dataset", "method", "window_seconds"], as_index=False).agg(
        accuracy=("correct", "mean"), subjects=("subject", "nunique"), predictions=("correct", "size")
    )
    predictions.to_csv(args.output / "predictions.csv", index=False)
    unit_frame.to_csv(args.output / "unit_manifest.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    (args.output / "errors.log").write_text("\n".join(errors), encoding="utf-8")
    plot(summary, args.output / "figures" / "tdca_vs_etrca.png")
    expected_units = sum(len(config["subjects"]) * len(config["windows"]) for config in DEFAULTS.values())
    manifest = {
        "status": "complete" if len(units) == expected_units and not errors else "partial",
        "evidence_role": "diagnostic receiver smoke; not a paper reproduction",
        "reference_variants": {
            "EEG_TDCA": "class-specific EEG temporal subspace estimated inside each training fold",
            "SPECTRAL_TDCA": "zero-phase sine/cosine harmonics for the unique active frequencies in each ordered label pair; training-fold EEG fallback only for void-void",
        },
        "stimulus_phase_dimension": False,
        "spectral_reference_phase": "fixed zero; phase is not a target/codebook dimension",
        "analytic_zero_phase_codebook_reference": True,
        "external_measured_reference": False,
        "test_fold_used_for_reference": False,
        "cv": "subject-specific leave-one-block-out",
        "n_delay": args.n_delay,
        "n_components": args.n_components,
        "parameter_status": "smoke-selected after JBHI35 2 s audit; delay=5/components=8 was unstable",
        "expected_units": expected_units,
        "completed_units": len(units),
        "prediction_count": len(predictions),
        "error_count": len(errors),
        "blas_threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest["status"] != "complete":
        raise RuntimeError("TDCA smoke is partial; inspect errors.log")


if __name__ == "__main__":
    main()
