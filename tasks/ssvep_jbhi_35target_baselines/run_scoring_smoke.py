from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.embc_jbhi import analysis_epochs, dataset_spec, load_subject, resolve_dataset_root
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.traditional import TRCA
from vep_arena.methods.trca_core import corr_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _band_correlations(model: TRCA, test_x: np.ndarray) -> np.ndarray:
    if model.templates is None or model.filters is None:
        raise RuntimeError("ETRCA model is not fitted.")
    trials, bands = test_x.shape[:2]
    classes = model.templates.shape[0]
    output = np.zeros((trials, bands, classes), dtype=np.float64)
    for band in range(bands):
        filters = model.filters[band].T
        projected_trials = np.matmul(np.transpose(test_x[:, band], (0, 2, 1)), filters).reshape(trials, -1)
        for target in range(classes):
            projected_template = (model.templates[target, band].T @ filters).reshape(-1)
            output[:, band, target] = corr_rows(projected_trials, projected_template)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded JBHI35 0.2 s signed/absolute ETRCA smoke.")
    parser.add_argument("--config", type=Path, default=TASK_ROOT / "smoke_config_20260728.json")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=TASK_ROOT / "results" / "handoff_20260728_scoring_smoke")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config != {
        "dataset": "jbhi35",
        "subjects": ["S03", "S06"],
        "window_seconds": 0.2,
        "filter_bands": 5,
        "method": "ETRCA",
        "scoring_rules": ["signed_correlation", "absolute_correlation"],
        "split": "subject-specific six-fold leave-one-block-out",
        "runtime_latency_offset_seconds": 0.0,
    }:
        raise ValueError("Smoke configuration must remain the reviewed S03/S06 0.2 s protocol.")

    dataset = str(config["dataset"])
    subjects = tuple(str(value) for value in config["subjects"])
    window = float(config["window_seconds"])
    bands = int(config["filter_bands"])
    root = resolve_dataset_root(dataset, args.root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = dataset_spec(dataset)
    labels = np.arange(spec.targets, dtype=np.int64)
    started = time.perf_counter()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()

    units: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    errors: list[str] = []
    for subject_id in subjects:
        try:
            subject = load_subject(dataset, subject_id, root)
            if subject.targets != tuple(range(35)):
                raise RuntimeError(f"{subject_id} did not expose Python labels 0-34.")
            epochs = analysis_epochs(subject, window, n_bands=bands)
            by_scoring: dict[str, list[int]] = {name: [] for name in config["scoring_rules"]}
            correct_by_scoring = {name: 0 for name in config["scoring_rules"]}
            for test_block in range(spec.blocks):
                train_blocks = [block for block in range(spec.blocks) if block != test_block]
                if test_block in train_blocks or len(train_blocks) != 5:
                    raise RuntimeError("Fold leakage check failed.")
                train_x = epochs[:, train_blocks].reshape(spec.targets * 5, bands, 9, epochs.shape[-1])
                train_y = np.repeat(labels, 5)
                model = TRCA(n_fbs=bands, ensemble=True).fit(train_x, train_y)
                correlations = _band_correlations(model, epochs[:, test_block])
                score_matrices = {
                    "signed_correlation": np.einsum("f,tfc->tc", model.weights, correlations),
                    "absolute_correlation": np.einsum("f,tfc->tc", model.weights, np.abs(correlations)),
                }
                for scoring, scores in score_matrices.items():
                    predicted = np.argmax(scores, axis=1).astype(np.int64)
                    by_scoring[scoring].extend(predicted.tolist())
                    fold_correct = int(np.sum(predicted == labels))
                    correct_by_scoring[scoring] += fold_correct
                    folds.append(
                        {
                            "subject": subject_id,
                            "scoring": scoring,
                            "fold": test_block + 1,
                            "test_block_0based": test_block,
                            "train_blocks_0based": "|".join(str(value) for value in train_blocks),
                            "test_block_in_training": False,
                            "prediction_count": spec.targets,
                            "correct": fold_correct,
                            "accuracy": fold_correct / spec.targets,
                        }
                    )
                    for true_label, predicted_label in zip(labels, predicted):
                        predictions.append(
                            {
                                "subject": subject_id,
                                "scoring": scoring,
                                "fold": test_block + 1,
                                "block_0based": test_block,
                                "true_0based": int(true_label),
                                "pred_0based": int(predicted_label),
                                "true_1based_for_matlab": int(true_label) + 1,
                                "pred_1based_for_matlab": int(predicted_label) + 1,
                                "correct": int(true_label == predicted_label),
                            }
                        )
            for scoring in config["scoring_rules"]:
                expected = spec.targets * spec.blocks
                actual = len(by_scoring[scoring])
                accuracy = correct_by_scoring[scoring] / expected
                units.append(
                    {
                        "dataset": dataset,
                        "subject": subject_id,
                        "method": "ETRCA",
                        "scoring": scoring,
                        "window_seconds": window,
                        "window_samples": epochs.shape[-1],
                        "sampling_rate": spec.analysis_sampling_rate,
                        "targets": spec.targets,
                        "blocks": spec.blocks,
                        "filter_bands": bands,
                        "folds": spec.blocks,
                        "prediction_count": actual,
                        "expected_predictions": expected,
                        "accuracy": accuracy,
                        "itr_bpm": itr_bits_per_minute(
                            accuracy,
                            spec.targets,
                            window + spec.itr_shift_seconds,
                        ),
                        "fold_leakage_check": "pass",
                        "model_label_min": 0,
                        "model_label_max": 34,
                        "status": "complete" if actual == expected else "partial",
                    }
                )
        except Exception as exc:
            errors.append(f"{subject_id}: {type(exc).__name__}: {exc}")

    _write_csv(output / "unit_manifest.csv", units)
    _write_csv(output / "subject.csv", units)
    _write_csv(output / "folds.csv", folds)
    _write_csv(output / "predictions.csv", predictions)
    (output / "errors.log").write_text("\n".join(errors), encoding="utf-8")

    data_hashes = {subject: _sha256(root / f"{subject}.mat") for subject in subjects}
    data_hashes["target_mapping_35.csv"] = _sha256(root / "target_mapping_35.csv")
    code_paths = {
        "adapter": PROJECT_ROOT / "vep_arena" / "data" / "embc_jbhi.py",
        "task": Path(__file__).resolve(),
        "config": config_path,
        "traditional": PROJECT_ROOT / "vep_arena" / "methods" / "traditional.py",
        "trca_core": PROJECT_ROOT / "vep_arena" / "methods" / "trca_core.py",
    }
    expected_units = len(subjects) * len(config["scoring_rules"])
    expected_predictions = expected_units * spec.targets * spec.blocks
    complete = len(units) == expected_units and len(predictions) == expected_predictions and not errors
    manifest = {
        "status": "complete" if complete else "partial",
        "started_at_utc": started_at,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "repo_branch": _git_text("branch", "--show-current"),
        "repo_head": _git_text("rev-parse", "HEAD"),
        "dataset": dataset,
        "dataset_root": str(root),
        "source_schema": "canonical_4d_v1",
        "subjects": list(subjects),
        "window_seconds": window,
        "window_samples": int(round(window * spec.analysis_sampling_rate)),
        "stored_epoch_start_seconds": spec.stored_epoch_start_seconds,
        "runtime_latency_offset_seconds": config["runtime_latency_offset_seconds"],
        "split": config["split"],
        "train_blocks_per_fold": 5,
        "test_blocks_per_fold": 1,
        "method": config["method"],
        "scoring_rules": {
            "signed_correlation": "sum_b weight_b * r_b",
            "absolute_correlation": "sum_b weight_b * abs(r_b)",
        },
        "historical_absScore_role": "legacy_reference_only_not_a_reproduced_baseline",
        "receiver_filterbank": {**spec.receiver_filterbank, "selected_bands": bands},
        "upstream_preprocessing": spec.preprocessing_profile,
        "model_label_base": 0,
        "model_label_range": [0, 34],
        "matlab_comparison_label_base": 1,
        "data_sha256": data_hashes,
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
        raise RuntimeError(f"Smoke is partial; inspect {output / 'errors.log'}")


if __name__ == "__main__":
    main()
