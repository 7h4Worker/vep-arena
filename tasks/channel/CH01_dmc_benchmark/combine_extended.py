"""Combine coarse/fine extended predictions and build capacity tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TASK_ROOT = Path(__file__).resolve().parent
EXTENDED_ROOT = TASK_ROOT / "results" / "extended"
PREDICTION_COLUMNS = ["method", "window", "subject", "block", "true", "pred"]
PREDICTION_KEY = ["method", "window", "subject", "block", "true"]
ROWS_PER_CELL = 6 * 40
DEFAULT_METHODS = "CCA,FBCCA,ECCA,TRCA,ETRCA"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage(path: Path, methods: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest_path = path / "manifest.json"
    predictions_path = path / "predictions.csv"
    if not manifest_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(f"Incomplete stage directory: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Stage is not complete: {path}")
    frame = pd.read_csv(predictions_path, usecols=PREDICTION_COLUMNS)
    frame["method"] = frame["method"].astype(str).str.upper()
    frame["window"] = frame["window"].astype(float).round(4)
    frame = frame[frame["method"].isin(methods)].copy()
    missing_methods = sorted(set(methods) - set(frame["method"].unique()))
    if missing_methods:
        raise ValueError(f"{path} is missing methods: {', '.join(missing_methods)}")
    if frame.duplicated(PREDICTION_KEY).any():
        raise ValueError(f"{predictions_path} contains duplicate prediction keys.")
    return frame, manifest


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def validate_cells(frame: pd.DataFrame, subjects: list[int], windows: list[float], methods: list[str]) -> None:
    expected_cells = len(subjects) * len(windows) * len(methods)
    groups = frame.groupby(["subject", "window", "method"], sort=False)
    if groups.ngroups != expected_cells:
        raise ValueError(f"Combined result has {groups.ngroups} cells; expected {expected_cells}.")
    sizes = groups.size()
    if not sizes.eq(ROWS_PER_CELL).all():
        bad = sizes[~sizes.eq(ROWS_PER_CELL)].head().to_dict()
        raise ValueError(f"Combined result contains incomplete cells: {bad}")
    if len(frame) != expected_cells * ROWS_PER_CELL:
        raise ValueError("Combined prediction row count does not match the complete grid.")


def build_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = (
        frame.assign(correct=frame["true"].eq(frame["pred"]))
        .groupby(["method", "window", "subject"], as_index=False)
        .agg(accuracy=("correct", "mean"), samples=("correct", "size"))
    )
    summary = (
        subject.groupby(["method", "window"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_sem=("accuracy", lambda values: float(values.std(ddof=1) / np.sqrt(len(values)))),
            subjects=("subject", "nunique"),
            samples=("samples", "sum"),
        )
        .sort_values(["method", "window"])
    )
    return summary, subject.sort_values(["method", "window", "subject"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine extended Benchmark stages")
    parser.add_argument("--coarse", type=Path, default=EXTENDED_ROOT / "coarse_0.1s")
    parser.add_argument("--fine", type=Path, default=EXTENDED_ROOT / "fine_8ms")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--output-dir", type=Path, default=EXTENDED_ROOT / "combined")
    parser.add_argument("--skip-capacity", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    methods = list(dict.fromkeys(item.strip().upper() for item in args.methods.split(",") if item.strip()))
    if not methods:
        raise ValueError("At least one method is required.")
    coarse, coarse_manifest = load_stage(args.coarse, methods)
    fine, fine_manifest = load_stage(args.fine, methods)

    coarse_context = float(coarse_manifest["preprocessing"]["context_seconds"])
    fine_context = float(fine_manifest["preprocessing"]["context_seconds"])
    if coarse_context != fine_context:
        raise ValueError(f"Preprocessing contexts differ: coarse={coarse_context}, fine={fine_context}.")
    coarse_subjects = sorted(int(value) for value in coarse["subject"].unique())
    fine_subjects = sorted(int(value) for value in fine["subject"].unique())
    if coarse_subjects != fine_subjects:
        raise ValueError("Coarse and fine stages contain different subject sets.")

    stacked = pd.concat(
        [coarse.assign(source_stage="coarse"), fine.assign(source_stage="fine")],
        ignore_index=True,
    )
    overlapping = stacked[stacked.duplicated(PREDICTION_KEY, keep=False)]
    conflicts = overlapping.groupby(PREDICTION_KEY)["pred"].nunique()
    if not conflicts.empty and not conflicts.eq(1).all():
        raise ValueError(f"Coarse/fine predictions conflict on {int((conflicts > 1).sum())} overlapping keys.")
    combined = (
        stacked.drop_duplicates(PREDICTION_KEY, keep="first")[PREDICTION_COLUMNS]
        .sort_values(["method", "window", "subject", "block", "true"])
        .reset_index(drop=True)
    )
    windows = sorted(float(value) for value in combined["window"].unique())
    validate_cells(combined, coarse_subjects, windows, methods)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    atomic_csv(combined, predictions_path)
    summary, subject = build_summary(combined)
    atomic_csv(summary, args.output_dir / "summary.csv")
    atomic_csv(subject, args.output_dir / "subject.csv")

    overlap_cells = overlapping[PREDICTION_KEY[:-2]].drop_duplicates()
    manifest = {
        "task": "benchmark_decision_channel_capacity_extended_combined",
        "status": "complete",
        "sources": [
            {
                "stage": "coarse",
                "path": str(args.coarse / "predictions.csv"),
                "sha256": file_sha256(args.coarse / "predictions.csv"),
            },
            {
                "stage": "fine",
                "path": str(args.fine / "predictions.csv"),
                "sha256": file_sha256(args.fine / "predictions.csv"),
            },
        ],
        "methods": methods,
        "subjects": coarse_subjects,
        "windows": windows,
        "preprocessing_context_seconds": coarse_context,
        "overlap_cells_verified": int(len(overlap_cells)),
        "prediction_rows": int(len(combined)),
        "expected_prediction_rows": len(methods) * len(coarse_subjects) * len(windows) * ROWS_PER_CELL,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.skip_capacity:
        subprocess.run(
            [
                sys.executable,
                str(TASK_ROOT / "analyze.py"),
                "--predictions",
                str(predictions_path),
                "--output-dir",
                str(args.output_dir / "analysis"),
                "--skip-pruning",
            ],
            cwd=TASK_ROOT.parents[1],
            check=True,
        )
        if not args.skip_plots:
            subprocess.run(
                [
                    sys.executable,
                    str(TASK_ROOT / "plot.py"),
                    "--analysis-dir",
                    str(args.output_dir / "analysis"),
                    "--figures-dir",
                    str(args.output_dir / "figures"),
                    "--skip-report",
                ],
                cwd=TASK_ROOT.parents[1],
                check=True,
            )
    print(
        f"Combined {len(coarse_subjects)} subjects, {len(methods)} methods, "
        f"{len(windows)} windows, {len(combined)} predictions -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
