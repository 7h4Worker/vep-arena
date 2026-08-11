from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy import linalg, signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks.bessvep.run_35t_scoring_smoke import (  # noqa: E402
    _band_correlations,
    _git_text,
    _sha256,
    _write_csv,
)
from vep_arena.data.embc_jbhi import (  # noqa: E402
    JBHI35_HISTORICAL5_PACKAGE_CONFIG_KEY,
    JBHI35_HISTORICAL5_SUBJECT_MANIFEST_CONFIG_KEY,
    JBHI35_HISTORICAL5_SUBJECT_IDS,
    dataset_spec,
    jbhi_receiver_filter_band,
    load_jbhi35_historical5_subject,
    resolve_jbhi35_historical5_package,
)
from vep_arena.metrics import itr_bits_per_minute  # noqa: E402
from vep_arena.methods.traditional import TRCA  # noqa: E402
from vep_arena.methods.trca_core import corr_rows  # noqa: E402

def _legacy_matlab_filter_band(x: np.ndarray, sampling_rate: int, band: int) -> np.ndarray:
    low = 6.0 + 8.0 * band
    nyquist = sampling_rate / 2.0
    numerator, denominator = signal.cheby1(
        6,
        0.5,
        [low / nyquist, 90.0 / nyquist],
        btype="bandpass",
        output="ba",
    )
    padlen = 3 * (max(len(numerator), len(denominator)) - 1)
    return signal.filtfilt(
        numerator,
        denominator,
        np.asarray(x, dtype=np.float64),
        axis=-1,
        padlen=padlen,
    )


def _legacy_matlab_trca_filter(trials: np.ndarray) -> np.ndarray:
    values = np.asarray(trials, dtype=np.float64)
    centered_trials = values - np.mean(values, axis=-1, keepdims=True)
    summed = np.sum(centered_trials, axis=0)
    between = summed @ summed.T
    unfolded = np.transpose(centered_trials, (1, 0, 2)).reshape(values.shape[1], -1)
    total = unfolded @ unfolded.T
    eigenvalues, eigenvectors = linalg.eigh(between, total, check_finite=False)
    spatial_filter = eigenvectors[:, int(np.argmax(eigenvalues))]
    # Preserve the generalized-eigenvector normalization. Rescaling each class
    # filter independently changes its relative weight in ensemble TRCA.
    return spatial_filter


def _fit_legacy_matlab_band(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    templates = np.mean(train_x, axis=1)
    filters = np.stack([_legacy_matlab_trca_filter(class_trials) for class_trials in train_x])
    return templates, filters


def _legacy_matlab_band_correlations(
    templates: np.ndarray,
    filters: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    trials = test_x.shape[0]
    classes = templates.shape[0]
    ensemble_filters = filters.T
    projected_trials = np.matmul(np.transpose(test_x, (0, 2, 1)), ensemble_filters).reshape(trials, -1)
    output = np.zeros((trials, classes), dtype=np.float64)
    for target in range(classes):
        projected_template = (templates[target].T @ ensemble_filters).reshape(-1)
        output[:, target] = corr_rows(projected_trials, projected_template)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 2 s eTRCA aggregate counts against the fixed five-subject JBHI35 package.")
    parser.add_argument("--config", type=Path, default=TASK_ROOT / "five_subject_reproduction_config_20260728.json")
    parser.add_argument("--output", type=Path, required=True, help="New, empty local result directory.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(config["subjects"]) != JBHI35_HISTORICAL5_SUBJECT_IDS:
        raise ValueError("The fixed five-subject historical set must use the reviewed anonymous sequence.")
    if float(config["window_seconds"]) != 2.0 or int(config["filter_bands"]) != 5:
        raise ValueError("The bounded reproduction protocol is fixed at 2.0 s and five filter bands.")
    if str(config["dataset_config_key"]) != JBHI35_HISTORICAL5_PACKAGE_CONFIG_KEY:
        raise ValueError("The final-five dataset config key differs from the shared historical adapter.")
    if str(config["subject_manifest_config_key"]) != JBHI35_HISTORICAL5_SUBJECT_MANIFEST_CONFIG_KEY:
        raise ValueError("The final-five subject-manifest key differs from the shared historical adapter.")

    package = resolve_jbhi35_historical5_package()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing result directory; choose a new --output path.")
    output.mkdir(parents=True, exist_ok=True)
    spec = dataset_spec("jbhi35_historical5")
    labels = np.arange(35, dtype=np.int64)
    variants = tuple(str(value) for value in config["scoring_variants"])
    expected_variants = (
        "arena_independent_signed",
        "arena_independent_absolute",
        "legacy_matlab_signed",
        "legacy_matlab_absolute",
    )
    if variants != expected_variants:
        raise ValueError("Scoring variants differ from the reviewed protocol.")

    started = time.perf_counter()
    units: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    errors: list[str] = []
    data_hashes: dict[str, str] = {}
    for subject_record in package.subjects:
        subject_id = subject_record.subject_id
        try:
            subject = load_jbhi35_historical5_subject(package, subject_id)
            raw, sampling_rate, channels = subject.x, subject.sampling_rate, subject.channels
            data_hashes[subject_id] = subject_record.expected_sha256
            prediction_counts = {variant: 0 for variant in variants}
            correct_counts = {variant: 0 for variant in variants}
            families = ("arena_independent", "legacy_matlab")
            for family in families:
                for test_block in range(6):
                    print(f"{subject_id} {family} fold {test_block + 1}/6", flush=True)
                    train_blocks = [block for block in range(6) if block != test_block]
                    train_y = np.repeat(labels, 5)
                    train_raw = raw[:, train_blocks]
                    test_raw = raw[:, test_block]
                    train_current = train_raw
                    signed_scores = np.zeros((35, 35), dtype=np.float64)
                    absolute_scores = np.zeros((35, 35), dtype=np.float64)
                    weights = np.asarray([(band + 1) ** (-1.25) + 0.25 for band in range(5)])
                    for band in range(5):
                        if family == "legacy_matlab":
                            train_filtered = _legacy_matlab_filter_band(train_current, sampling_rate, band)
                            train_current = train_filtered
                            test_filtered = _legacy_matlab_filter_band(test_raw, sampling_rate, band)
                            templates, filters = _fit_legacy_matlab_band(train_filtered)
                            correlations = _legacy_matlab_band_correlations(
                                templates,
                                filters,
                                test_filtered,
                            )
                        else:
                            train_filtered = jbhi_receiver_filter_band(train_raw, sampling_rate, band)
                            test_filtered = jbhi_receiver_filter_band(test_raw, sampling_rate, band)
                            train_x = train_filtered.reshape(35 * 5, 1, 9, 2000)
                            model = TRCA(n_fbs=1, ensemble=True).fit(train_x, train_y)
                            correlations = _band_correlations(model, test_filtered[:, None])[:, 0]
                        signed_scores += weights[band] * correlations
                        absolute_scores += weights[band] * np.abs(correlations)
                        del train_filtered, test_filtered, correlations
                        if family == "legacy_matlab":
                            del templates, filters
                        else:
                            del train_x, model
                        gc.collect()
                    score_matrices = {
                        f"{family}_signed": signed_scores,
                        f"{family}_absolute": absolute_scores,
                    }
                    for variant, scores in score_matrices.items():
                        predicted = np.argmax(scores, axis=1).astype(np.int64)
                        correct = int(np.sum(predicted == labels))
                        prediction_counts[variant] += predicted.size
                        correct_counts[variant] += correct
                        folds.append(
                            {
                                "subject": subject_id,
                                "variant": variant,
                                "fold": test_block + 1,
                                "test_block_0based": test_block,
                                "train_blocks_0based": "|".join(str(value) for value in train_blocks),
                                "test_block_in_training": False,
                                "correct": correct,
                                "prediction_count": predicted.size,
                                "accuracy": correct / 35,
                            }
                        )
                        for true_label, predicted_label in zip(labels, predicted):
                            predictions.append(
                                {
                                    "subject": subject_id,
                                    "variant": variant,
                                    "fold": test_block + 1,
                                    "true_0based": int(true_label),
                                    "pred_0based": int(predicted_label),
                                    "correct": int(true_label == predicted_label),
                                }
                            )
            del raw
            gc.collect()
            if subject_record.matlab_signed_accuracy is None:
                raise ValueError("Final-five private manifest is missing the aggregate MATLAB reference accuracy.")
            matlab_accuracy = subject_record.matlab_signed_accuracy
            matlab_correct = int(round(matlab_accuracy * 210))
            for variant in variants:
                accuracy = correct_counts[variant] / 210
                reference_match = (
                    correct_counts[variant] == matlab_correct if variant == "legacy_matlab_signed" else None
                )
                units.append(
                    {
                        "dataset": "jbhi35_five_subject_reproduction",
                        "subject": subject_id,
                        "variant": variant,
                        "window_seconds": 2.0,
                        "window_samples": 2000,
                        "sampling_rate": sampling_rate,
                        "channels": len(channels),
                        "targets": 35,
                        "blocks": 6,
                        "filter_bands": 5,
                        "prediction_count": prediction_counts[variant],
                        "expected_predictions": 210,
                        "accuracy": accuracy,
                        "itr_bpm": itr_bits_per_minute(accuracy, 35, 2.0 + spec.itr_shift_seconds),
                        "matlab_signed_reference_accuracy": matlab_accuracy,
                        "difference_from_matlab_signed_pp": 100.0 * (accuracy - matlab_accuracy),
                        "matlab_signed_reference_correct": matlab_correct,
                        "matlab_aggregate_correct_count_match": reference_match,
                        "model_label_min": 0,
                        "model_label_max": 34,
                        "fold_leakage_check": "pass",
                        "status": "complete" if prediction_counts[variant] == 210 else "partial",
                    }
                )
        except Exception as exc:
            errors.append(f"{subject_id}: {type(exc).__name__}: {exc}")

    _write_csv(output / "unit_manifest.csv", units)
    _write_csv(output / "subject.csv", units)
    _write_csv(output / "folds.csv", folds)
    _write_csv(output / "predictions.csv", predictions)
    (output / "errors.log").write_text("\n".join(errors), encoding="utf-8")
    code_paths = {
        "task": Path(__file__).resolve(),
        "config": config_path,
        "historical_adapter": PROJECT_ROOT / "vep_arena" / "data" / "embc_jbhi.py",
        "scoring_helper": TASK_ROOT / "run_scoring_smoke.py",
        "traditional": PROJECT_ROOT / "vep_arena" / "methods" / "traditional.py",
        "trca_core": PROJECT_ROOT / "vep_arena" / "methods" / "trca_core.py",
    }
    expected_units = len(package.subjects) * len(variants)
    expected_predictions = expected_units * 210
    reproduction_rows = [row for row in units if row["variant"] == "legacy_matlab_signed"]
    aggregate_reference_match = len(reproduction_rows) == len(package.subjects) and all(
        bool(row["matlab_aggregate_correct_count_match"]) for row in reproduction_rows
    )
    complete = (
        len(units) == expected_units
        and len(predictions) == expected_predictions
        and not errors
    )
    manifest = {
        "status": "complete" if complete else "partial",
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "repo_branch": _git_text("branch", "--show-current"),
        "repo_head": _git_text("rev-parse", "HEAD"),
        "package_name": package.root.name,
        "package_sha_manifest_sha256": package.package_sha_manifest_sha256,
        "package_sha_entries_verified": package.package_sha_entries_verified,
        "package_sha_validation": "pass",
        "data_mapping_validation": "pass",
        "package_role": "only_valid_five_subject_historical_reproduction_package",
        "obsolete_handoff_used": False,
        "subjects": config["subjects"],
        "excluded_canonical_subject_ids": ["S02"],
        "source_schema": "package_4d_9x2000x35x6",
        "split": config["split"],
        "window_seconds": 2.0,
        "window_samples": 2000,
        "stored_epoch_start_seconds": 0.13,
        "runtime_latency_offset_seconds": 0.0,
        "receiver_filterbank": {**spec.receiver_filterbank, "selected_bands": 5},
        "variants": {
            "arena_independent_signed": "independent train/test subbands; sum_b weight_b*r_b",
            "arena_independent_absolute": "independent train/test subbands; sum_b weight_b*abs(r_b)",
            "legacy_matlab_signed": "legacy-reference-oriented B/A filtfilt, cascaded training subbands, and generalized eigenfilter; sum_b weight_b*r_b; not a numerical-equivalence claim",
            "legacy_matlab_absolute": "legacy-reference-oriented B/A filtfilt, cascaded training subbands, and generalized eigenfilter; sum_b weight_b*abs(r_b); not a numerical-equivalence claim",
        },
        "matlab_aggregate_reference_validation": "match" if aggregate_reference_match else "mismatch",
        "matlab_numerical_equivalence": "not_established: the package supplies aggregate correct counts, not per-trial predictions or score matrices",
        "matlab_reference_acceptance": "all five anonymous subjects must match the package 2 s signed eTRCA aggregate correct-prediction counts; this is not a claim of per-trial or numerical equivalence",
        "historical_absScore_role": "legacy_reference_only_not_reproduced_by_filename",
        "model_label_base": 0,
        "model_label_range": [0, 34],
        "data_sha256": data_hashes,
        "private_subject_manifest_sha256": package.subject_manifest_sha256,
        "code_sha256": {name: _sha256(path) for name, path in code_paths.items()},
        "python_executable": sys.executable,
        "blas_threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "expected_units": expected_units,
        "completed_units": len(units),
        "expected_predictions": expected_predictions,
        "completed_predictions": len(predictions),
        "error_count": len(errors),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "units": units, "output": str(output)}, ensure_ascii=False, indent=2))
    if not complete:
        raise RuntimeError(f"Five-subject reproduction is partial; inspect {output / 'errors.log'}")


if __name__ == "__main__":
    main()
