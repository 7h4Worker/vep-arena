from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vep_arena.channel.capacity import capacity_ba, capacity_c1, mutual_info_uniform
from vep_arena.channel.confusion import confusion_counts, normalize_confusion
from vep_arena.metrics import sem


METHODS = ("CCA", "FBCCA", "TRCA", "ETRCA")
WINDOWS = (0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
FULL_V2_WINDOWS = (0.2, 0.3, 0.4, 0.5, 1.0)
CLASSES = 40
SUBJECTS = tuple(range(1, 36))
BLOCKS = tuple(range(1, 7))
REQUIRED_COLUMNS = {"method", "window", "subject", "block", "true", "pred"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_windows(frame: pd.DataFrame, windows: tuple[float, ...]) -> pd.DataFrame:
    wanted = np.asarray(windows, dtype=float)
    mask = np.isclose(frame["window"].to_numpy(dtype=float)[:, None], wanted[None, :]).any(axis=1)
    selected = frame.loc[mask].copy()
    for window in windows:
        selected.loc[np.isclose(selected["window"], window), "window"] = window
    return selected


def validate_prediction_grid(frame: pd.DataFrame, windows: tuple[float, ...]) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Prediction grid contains null values in required columns.")

    expected_rows = len(METHODS) * len(windows) * len(SUBJECTS) * len(BLOCKS) * CLASSES
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(frame)}.")
    if set(frame["method"].astype(str)) != set(METHODS):
        raise ValueError(f"Unexpected methods: {sorted(frame['method'].astype(str).unique())}")
    actual_windows = sorted(frame["window"].astype(float).unique())
    if len(actual_windows) != len(windows) or not np.allclose(actual_windows, windows):
        raise ValueError(f"Unexpected windows: {actual_windows}")
    if set(frame["subject"].astype(int)) != set(SUBJECTS):
        raise ValueError("Subject grid is not exactly 1-35.")
    if set(frame["block"].astype(int)) != set(BLOCKS):
        raise ValueError("Block grid is not exactly 1-6.")
    for column in ("true", "pred"):
        values = frame[column].astype(int)
        if values.min() != 0 or values.max() != CLASSES - 1:
            raise ValueError(f"{column} labels are not exactly within 0-{CLASSES - 1}.")

    keys = ["method", "window", "subject", "block", "true"]
    if frame.duplicated(keys).any():
        raise ValueError(f"Prediction grid has duplicate keys: {keys}")
    group_sizes = frame.groupby(["method", "window", "subject", "block"], observed=True).size()
    if not (group_sizes == CLASSES).all():
        raise ValueError("Each method/window/subject/block cell must contain 40 trials.")
    true_counts = frame.groupby(["method", "window", "subject", "block"], observed=True)["true"].nunique()
    if not (true_counts == CLASSES).all():
        raise ValueError("Each method/window/subject/block cell must contain every true label once.")


def load_unified_predictions(full_v2_path: Path, w075_path: Path) -> pd.DataFrame:
    full_v2 = pd.read_csv(full_v2_path)
    w075 = pd.read_csv(w075_path)
    full_selected = _select_windows(full_v2, FULL_V2_WINDOWS)
    full_selected = full_selected[full_selected["method"].isin(METHODS)].copy()
    w075_selected = _select_windows(w075, (0.75,))
    w075_selected = w075_selected[w075_selected["method"].isin(METHODS)].copy()
    validate_prediction_grid(full_selected, FULL_V2_WINDOWS)
    validate_prediction_grid(w075_selected, (0.75,))
    unified = pd.concat([full_selected, w075_selected], ignore_index=True)
    validate_prediction_grid(unified, WINDOWS)
    return unified


def analyze_predictions(frame: pd.DataFrame, source_by_window: dict[float, dict[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    subject_rows: list[dict[str, object]] = []
    for (method, window, subject), group in frame.groupby(["method", "window", "subject"], sort=True):
        accuracy = float((group["true"].to_numpy() == group["pred"].to_numpy()).mean())
        bits_per_trial = capacity_c1(CLASSES, accuracy)
        subject_rows.append(
            {
                "method": str(method),
                "window_seconds": float(window),
                "subject": int(subject),
                "trials": int(len(group)),
                "accuracy": accuracy,
                "itr_bits_per_min": bits_per_trial * 60.0 / (float(window) + 0.5),
            }
        )
    subjects = pd.DataFrame(subject_rows)

    result_rows: list[dict[str, object]] = []
    confusion_payload: dict[str, np.ndarray] = {}
    for (method, window), group in frame.groupby(["method", "window"], sort=True):
        subject_group = subjects[(subjects["method"] == method) & np.isclose(subjects["window_seconds"], window)]
        counts = confusion_counts(group["true"], group["pred"], CLASSES)
        transition = normalize_confusion(counts, alpha=0.0)
        ba = capacity_ba(transition)
        key = f"{str(method).lower()}_w{float(window):.2f}".replace(".", "p")
        confusion_payload[f"{key}_counts"] = counts
        confusion_payload[f"{key}_transition"] = transition
        source = source_by_window[float(window)]
        result_rows.append(
            {
                "method": str(method),
                "window_seconds": float(window),
                "subjects": int(subject_group["subject"].nunique()),
                "trials": int(len(group)),
                "accuracy_mean": float(subject_group["accuracy"].mean()),
                "accuracy_sem": sem(subject_group["accuracy"].to_numpy()),
                "itr_bits_per_min_mean": float(subject_group["itr_bits_per_min"].mean()),
                "itr_bits_per_min_sem": sem(subject_group["itr_bits_per_min"].to_numpy()),
                "c_ba_bits_per_trial": float(ba.capacity),
                "mi_uniform_bits_per_trial": float(mutual_info_uniform(transition)),
                "ba_converged": bool(ba.converged),
                "ba_iterations": int(ba.iterations),
                "source_run": source["run"],
                "source_predictions_sha256": source["sha256"],
            }
        )
    results = pd.DataFrame(result_rows)
    method_order = {method: index for index, method in enumerate(METHODS)}
    results["_method_order"] = results["method"].map(method_order)
    results = results.sort_values(["_method_order", "window_seconds"]).drop(columns="_method_order").reset_index(drop=True)
    subjects = subjects.sort_values(["method", "window_seconds", "subject"]).reset_index(drop=True)
    return results, subjects, confusion_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unified P0 decoder-number table.")
    parser.add_argument("--full-v2-predictions", type=Path, required=True)
    parser.add_argument("--w075-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    full_sha = sha256_file(args.full_v2_predictions)
    w075_sha = sha256_file(args.w075_predictions)
    unified = load_unified_predictions(args.full_v2_predictions, args.w075_predictions)
    source_by_window = {
        **{
            window: {"run": args.full_v2_predictions.parent.name, "sha256": full_sha}
            for window in FULL_V2_WINDOWS
        },
        0.75: {"run": args.w075_predictions.parent.name, "sha256": w075_sha},
    }
    results, subjects, confusion_payload = analyze_predictions(unified, source_by_window)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "unified_decoder_numbers.csv", index=False, float_format="%.12g")
    subjects.to_csv(args.output_dir / "subject_decoder_numbers.csv", index=False, float_format="%.12g")
    np.savez_compressed(args.output_dir / "confusion_matrices.npz", **confusion_payload)
    provenance = {
        "status": "complete",
        "methods": list(METHODS),
        "windows_seconds": list(WINDOWS),
        "classes": CLASSES,
        "subjects": list(SUBJECTS),
        "blocks": list(BLOCKS),
        "itr_trial_seconds": "window_seconds + 0.5",
        "capacity_scope": "pooled 40x40 confusion matrix for each method/window",
        "sources": {
            "full_v2": {"path": str(args.full_v2_predictions), "sha256": full_sha},
            "w075": {"path": str(args.w075_predictions), "sha256": w075_sha},
        },
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(args.output_dir / "unified_decoder_numbers.csv")


if __name__ == "__main__":
    main()
