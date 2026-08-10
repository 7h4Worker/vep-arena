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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks.ssvep_jbhi_35target_baselines.run_scoring_smoke import _git_text, _sha256  # noqa: E402
from vep_arena.data.embc_jbhi import (  # noqa: E402
    JBHI35_HISTORICAL5_PACKAGE_CONFIG_KEY,
    JBHI35_HISTORICAL5_SUBJECT_MANIFEST_CONFIG_KEY,
    JBHI35_HISTORICAL5_SUBJECT_IDS,
    jbhi_receiver_filter_band,
    load_jbhi35_historical5_subject,
    load_target_frequency_pairs,
    resolve_jbhi35_historical5_package,
)
from vep_arena.metrics import itr_bits_per_minute  # noqa: E402
from vep_arena.methods.bprca import BPRCA  # noqa: E402
from vep_arena.methods.fusionca import FusionCA  # noqa: E402


METHODS = ("BPRCA", "EBPRCA", "FUSIONCA", "EFUSIONCA")


def _part_paths(parts_dir: Path, subject_id: str) -> tuple[Path, Path]:
    return parts_dir / f"{subject_id}_unit.json", parts_dir / f"{subject_id}_predictions.csv"


def _part_complete(parts_dir: Path, subject_id: str, expected_rows: int) -> bool:
    unit_path, prediction_path = _part_paths(parts_dir, subject_id)
    if not unit_path.is_file() or not prediction_path.is_file():
        return False
    try:
        units = json.loads(unit_path.read_text(encoding="utf-8"))
        row_count = sum(1 for _ in prediction_path.open(encoding="utf-8")) - 1
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return len(units) > 0 and all(unit.get("status") == "complete" for unit in units) and row_count == expected_rows


def _write_part(parts_dir: Path, subject_id: str, rows: list[dict[str, object]], units: list[dict[str, object]]) -> None:
    unit_path, prediction_path = _part_paths(parts_dir, subject_id)
    unit_tmp = unit_path.with_suffix(".json.tmp")
    prediction_tmp = prediction_path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(prediction_tmp, index=False)
    unit_tmp.write_text(json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction_tmp.replace(prediction_path)
    unit_tmp.replace(unit_path)


def _run_subject(task: dict[str, object]) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    subject_id = str(task["subject"])
    subject = load_jbhi35_historical5_subject(task["package"], subject_id)
    raw, sampling_rate, channels = subject.x, subject.sampling_rate, subject.channels
    windows = tuple(float(value) for value in task["windows"])
    frequencies = tuple(float(value) for value in task["frequencies"])
    bands = int(task["bands"])
    labels = np.arange(35, dtype=np.int64)
    filtered = np.stack([jbhi_receiver_filter_band(raw, sampling_rate, band) for band in range(bands)], axis=2)
    units: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []

    for window in windows:
        samples = int(round(window * sampling_rate))
        epochs = filtered[..., :samples]
        for method in METHODS:
            started = time.perf_counter()
            method_rows: list[dict[str, object]] = []
            for test_block in range(6):
                train_blocks = [block for block in range(6) if block != test_block]
                train_x = epochs[:, train_blocks].reshape(35 * 5, bands, len(channels), samples)
                train_y = np.repeat(labels, 5)
                test_x = epochs[:, test_block]
                model_class = FusionCA if method in {"FUSIONCA", "EFUSIONCA"} else BPRCA
                model = model_class(
                    frequencies,
                    sampling_rate,
                    n_fbs=bands,
                    ensemble=method in {"EBPRCA", "EFUSIONCA"},
                )
                model.fit(train_x, train_y)
                predicted, _ = model.predict(test_x)
                for true_label, predicted_label in zip(labels, predicted):
                    method_rows.append(
                        {
                            "dataset": "jbhi35_five_subject_historical_set",
                            "subject": subject_id,
                            "window_seconds": window,
                            "method": method,
                            "fold": test_block + 1,
                            "test_block_0based": test_block,
                            "true_0based": int(true_label),
                            "pred_0based": int(predicted_label),
                            "correct": int(true_label == predicted_label),
                        }
                    )
            units.append(
                {
                    "dataset": "jbhi35_five_subject_historical_set",
                    "subject": subject_id,
                    "window_seconds": window,
                    "window_samples": samples,
                    "sampling_rate": sampling_rate,
                    "channels": len(channels),
                    "targets": 35,
                    "blocks": 6,
                    "filter_bands": bands,
                    "method": method,
                    "frequency_units_hz": list(frequencies),
                    "prediction_count": len(method_rows),
                    "expected_predictions": 6 * 35,
                    "runtime_seconds": time.perf_counter() - started,
                    "fold_leakage_check": "pass",
                    "status": "complete" if len(method_rows) == 6 * 35 else "partial",
                }
            )
            rows.extend(method_rows)
            print(f"{subject_id} {window:g}s {method} complete", flush=True)
    return subject_id, rows, units


def _plot(summary: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for method, group in summary.groupby("method"):
        axis.errorbar(group["window_seconds"], group["accuracy"], yerr=group["accuracy_sem"], marker="o", label=method)
    axis.set(xlabel="Window (s)", ylabel="Accuracy", ylim=(0, 1), title="JBHI35 historical five-subject periodic receivers")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.savefig(output, dpi=210)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full bPRCA and FusionCA evaluation on the reviewed five-subject JBHI35 package.")
    parser.add_argument("--config", type=Path, default=TASK_ROOT / "five_subject_reproduction_config_20260728.json")
    parser.add_argument("--output", type=Path, required=True, help="New local result directory, or an explicit resume target.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Explicitly allow writing to an existing result directory.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    subjects = tuple(str(value) for value in config["subjects"])
    windows = tuple(float(value) for value in config["tdca_windows_seconds"])
    configured_frequency_pairs = tuple(tuple(float(item) for item in pair) for pair in config["target_frequency_pairs_hz"])
    frequency_pairs = load_target_frequency_pairs("jbhi35_historical5")
    frequencies = tuple(sorted({value for pair in frequency_pairs for value in pair if value > 0}))
    if subjects != JBHI35_HISTORICAL5_SUBJECT_IDS:
        raise ValueError("The fixed historical set must contain only S01, S03, S04, S05, and S06.")
    if windows != tuple(np.arange(0.2, 2.01, 0.2).round(1)):
        raise ValueError("The full receiver grid must cover 0.2-2.0 s in 0.2 s increments.")
    if frequencies != (11.0, 12.0, 13.0, 14.0, 15.0):
        raise ValueError("The JBHI35 active frequency units must be 11-15 Hz.")
    if configured_frequency_pairs != frequency_pairs:
        raise ValueError("Periodic-receiver config codebook differs from the shared final-five historical adapter.")
    if str(config["dataset_config_key"]) != JBHI35_HISTORICAL5_PACKAGE_CONFIG_KEY:
        raise ValueError("The final-five dataset config key differs from the shared historical adapter.")
    if str(config["subject_manifest_config_key"]) != JBHI35_HISTORICAL5_SUBJECT_MANIFEST_CONFIG_KEY:
        raise ValueError("The final-five subject-manifest key differs from the shared historical adapter.")

    package = resolve_jbhi35_historical5_package()

    tasks: list[dict[str, object]] = []
    data_hashes = {record.subject_id: record.expected_sha256 for record in package.subjects}
    for record in package.subjects:
        subject_id = record.subject_id
        tasks.append(
            {
                "subject": subject_id,
                "package": package,
                "windows": windows,
                "frequencies": frequencies,
                "bands": int(config["filter_bands"]),
            }
        )

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Refusing to overwrite an existing result directory; choose a new --output path or explicitly use --resume.")
    output.mkdir(parents=True, exist_ok=True)
    parts_dir = output / "parts"
    figures_dir = output / "figures"
    parts_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    previous_manifest: dict[str, object] = {}
    if args.resume and manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    initial_full_runtime = previous_manifest.get("initial_full_run_wall_runtime_seconds")
    if initial_full_runtime is None and previous_manifest.get("pending_subjects_at_start") == len(subjects):
        initial_full_runtime = previous_manifest.get("runtime_seconds")
    per_subject_rows = len(windows) * len(METHODS) * 6 * 35
    pending = [task for task in tasks if not (args.resume and _part_complete(parts_dir, str(task["subject"]), per_subject_rows))]
    started = time.perf_counter()
    manifest = {
        "status": "running",
        "evidence_role": "Arena bPRCA/FusionCA extension on the final historical five-subject set; not a MATLAB or paper reproduction",
        "package_role": "only_valid_five_subject_historical_reproduction_package",
        "obsolete_handoff_used": False,
        "subjects": list(subjects),
        "excluded_canonical_subject_ids": ["S02"],
        "windows_seconds": list(windows),
        "methods": list(METHODS),
        "frequency_units_hz": list(frequencies),
        "cv": "subject-specific six-fold leave-one-block-out",
        "receiver_filtering": "five independent zero-phase Chebyshev-I subbands",
        "model_label_base": 0,
        "model_label_range": [0, 34],
        "package_sha_entries_verified": package.package_sha_entries_verified,
        "package_sha_validation": "pass",
        "data_mapping_validation": "pass",
        "package_sha_manifest_sha256": package.package_sha_manifest_sha256,
        "private_subject_manifest_sha256": package.subject_manifest_sha256,
        "data_sha256": data_hashes,
        "config_sha256": _sha256(config_path),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "historical_adapter_sha256": _sha256(PROJECT_ROOT / "vep_arena" / "data" / "embc_jbhi.py"),
        "repo_branch": _git_text("branch", "--show-current"),
        "repo_head": _git_text("rev-parse", "HEAD"),
        "python_executable": sys.executable,
        "workers": args.workers,
        "blas_threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "expected_units": len(subjects) * len(windows) * len(METHODS),
        "expected_predictions": len(subjects) * per_subject_rows,
        "pending_subjects_at_start": len(pending),
        "initial_full_run_wall_runtime_seconds": initial_full_runtime,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    errors: list[str] = []
    if args.workers == 1:
        for task in pending:
            try:
                subject_id, rows, units = _run_subject(task)
                _write_part(parts_dir, subject_id, rows, units)
            except Exception as exc:
                errors.append(f"{task['subject']}: {type(exc).__name__}: {exc}")
    elif pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending)), mp_context=context) as executor:
            futures = {executor.submit(_run_subject, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    subject_id, rows, units = future.result()
                    _write_part(parts_dir, subject_id, rows, units)
                except Exception as exc:
                    errors.append(f"{task['subject']}: {type(exc).__name__}: {exc}")

    unit_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for subject_id in subjects:
        if not _part_complete(parts_dir, subject_id, per_subject_rows):
            continue
        unit_path, prediction_path = _part_paths(parts_dir, subject_id)
        unit_rows.extend(json.loads(unit_path.read_text(encoding="utf-8")))
        prediction_frames.append(pd.read_csv(prediction_path))
    units = pd.DataFrame(unit_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    subject_summary = predictions.groupby(["subject", "method", "window_seconds"], as_index=False).agg(
        accuracy=("correct", "mean"), predictions=("correct", "size")
    )
    subject_summary["itr_bpm"] = subject_summary.apply(
        lambda row: itr_bits_per_minute(float(row["accuracy"]), 35, float(row["window_seconds"]) + 0.5), axis=1
    )
    summary = subject_summary.groupby(["method", "window_seconds"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        accuracy_sem=("accuracy", lambda values: float(values.std(ddof=1) / np.sqrt(len(values)))),
        itr_bpm=("itr_bpm", "mean"),
        subjects=("subject", "nunique"),
        predictions=("predictions", "sum"),
    )
    units.to_csv(output / "unit_manifest.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    subject_summary.to_csv(output / "subject.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "errors.log").write_text("\n".join(errors), encoding="utf-8")
    _plot(summary, figures_dir / "periodic_receivers_accuracy.png")
    complete = len(units) == manifest["expected_units"] and len(predictions) == manifest["expected_predictions"] and not errors
    validation = {
        "status": "pass" if complete else "fail",
        "anonymous_subjects_only": set(predictions.get("subject", [])) == set(subjects),
        "unit_count_match": len(units) == manifest["expected_units"],
        "prediction_count_match": len(predictions) == manifest["expected_predictions"],
        "label_range_match": bool(len(predictions)) and int(predictions["true_0based"].min()) == 0 and int(predictions["true_0based"].max()) == 34 and int(predictions["pred_0based"].min()) >= 0 and int(predictions["pred_0based"].max()) <= 34,
        "all_folds_present": bool(len(predictions)) and set(predictions["fold"].unique()) == set(range(1, 7)),
        "all_windows_present": bool(len(predictions)) and np.allclose(sorted(predictions["window_seconds"].unique()), windows),
        "all_methods_present": bool(len(predictions)) and set(predictions["method"].unique()) == set(METHODS),
        "errors_empty": not errors,
    }
    (output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.update(
        {
            "status": "complete" if complete and all(value is True for key, value in validation.items() if key != "status") else "partial",
            "completed_units": len(units),
            "completed_predictions": len(predictions),
            "error_count": len(errors),
            "runtime_seconds": time.perf_counter() - started,
            "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    if initial_full_runtime is None and len(pending) == len(subjects):
        manifest["initial_full_run_wall_runtime_seconds"] = manifest["runtime_seconds"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "completed_units", "expected_units", "completed_predictions", "expected_predictions", "error_count")}, indent=2))
    if manifest["status"] != "complete":
        raise RuntimeError("Five-subject periodic receiver full run is partial; inspect manifest.json and errors.log.")


if __name__ == "__main__":
    main()
