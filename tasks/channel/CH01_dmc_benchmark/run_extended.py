"""Run fixed-context Benchmark sweeps over coarse or fine time grids.

The original 0.2-1.0 s benchmark run cached the longest requested epoch and
sliced shorter windows from it.  This runner keeps that protocol, but fixes the
preprocessing context at 5.0 s for both stages so coarse and fine predictions
can be combined without window-dependent filtering edge effects.

Default methods are the five canonical Benchmark receivers.  Periodic
receivers remain available through ``--methods`` for explicit diagnostic runs.
Each completed subject is checkpointed under ``parts/`` and ``--resume`` only
skips cells containing all six blocks and all forty true classes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.config import BENCHMARK_FREQS, CACHE_ROOT, DATA_ROOT, BenchmarkSpec
from vep_arena.data.epochs import CanonicalEpochStore, EpochRequest, epoch_fingerprint
from vep_arena.data.presets import benchmark_9ch_default
from vep_arena.methods.bprca import BPRCA
from vep_arena.methods.fusionca import FusionCA
from vep_arena.methods.traditional import CCA, ECCA, FBCCA, TRCA
from vep_arena.metrics import itr_bits_per_minute


TASK_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = TASK_ROOT / "results" / "extended"
DEFAULT_EPOCH_CACHE = CACHE_ROOT / "canonical_epochs"

SPEC = BenchmarkSpec()
FREQUENCIES = BENCHMARK_FREQS
N_FBS = 5
HARMONICS = 5
DEFAULT_CACHE_WINDOW = 5.0
ROWS_PER_CELL = SPEC.blocks * SPEC.classes
PREDICTION_COLUMNS = ["method", "window", "subject", "block", "true", "pred"]
PREDICTION_KEY = ["method", "window", "subject", "block", "true"]
RUNTIME_COLUMNS = ["subject", "method", "window", "seconds"]
RUNTIME_KEY = ["subject", "method", "window"]

STANDARD_METHODS = ("CCA", "FBCCA", "ECCA", "TRCA", "ETRCA")
PERIODIC_METHODS = ("BPRCA", "EBPRCA", "FUSIONCA", "EFUSIONCA")
SUPPORTED_METHODS = STANDARD_METHODS + PERIODIC_METHODS


def coarse_windows() -> list[float]:
    """Return 0.1-5.0 s in 0.1 s increments."""

    return [round(samples / SPEC.sampling_rate, 4) for samples in range(25, 1251, 25)]


def fine_windows() -> list[float]:
    """Return the exact 25-101 sample grid in two-sample increments."""

    return [round(samples / SPEC.sampling_rate, 4) for samples in range(25, 102, 2)]


def parse_subjects(text: str) -> list[int]:
    subjects: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            subjects.extend(range(int(start), int(end) + 1))
        else:
            subjects.append(int(part))
    subjects = sorted(set(subjects))
    if not subjects or subjects[0] < 1 or subjects[-1] > SPEC.subjects:
        raise ValueError(f"Subjects must be within 1-{SPEC.subjects}.")
    return subjects


def parse_windows(text: str | None, stage: str) -> list[float]:
    if text is None:
        return coarse_windows() if stage == "coarse" else fine_windows()
    windows = sorted(set(round(float(item.strip()), 4) for item in text.split(",") if item.strip()))
    if not windows or windows[0] <= 0:
        raise ValueError("Windows must contain positive durations.")
    return windows


def parse_methods(text: str | None) -> list[str]:
    methods = list(STANDARD_METHODS) if text is None else [item.strip().upper() for item in text.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one method is required.")
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return list(dict.fromkeys(methods))


def make_model(name: str, window: float):
    if name == "CCA":
        return CCA(window=window, harmonics=HARMONICS, spec=SPEC)
    if name == "FBCCA":
        return FBCCA(window=window, harmonics=HARMONICS, n_fbs=N_FBS, spec=SPEC)
    if name == "ECCA":
        return ECCA(window=window, harmonics=HARMONICS, n_fbs=N_FBS, spec=SPEC)
    if name == "TRCA":
        return TRCA(n_fbs=N_FBS, ensemble=False)
    if name == "ETRCA":
        return TRCA(n_fbs=N_FBS, ensemble=True)
    if name == "BPRCA":
        return BPRCA(FREQUENCIES, SPEC.sampling_rate, N_FBS, ensemble=False)
    if name == "EBPRCA":
        return BPRCA(FREQUENCIES, SPEC.sampling_rate, N_FBS, ensemble=True)
    if name == "FUSIONCA":
        return FusionCA(FREQUENCIES, SPEC.sampling_rate, N_FBS, ensemble=False)
    if name == "EFUSIONCA":
        return FusionCA(FREQUENCIES, SPEC.sampling_rate, N_FBS, ensemble=True)
    raise ValueError(f"Unknown method: {name}")


def run_cell(subject: int, window: float, method: str, epochs: np.ndarray) -> tuple[list[dict[str, object]], float]:
    labels = np.arange(SPEC.classes, dtype=np.int64)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for test_block in range(SPEC.blocks):
        train_blocks = [block for block in range(SPEC.blocks) if block != test_block]
        train_x = epochs[:, train_blocks].reshape(-1, epochs.shape[2], epochs.shape[3], epochs.shape[4])
        train_y = np.repeat(labels, len(train_blocks))
        test_x = epochs[:, test_block]

        model = make_model(method, window)
        model.fit(train_x, train_y)
        predicted, _scores = model.predict(test_x)
        for true_label, predicted_label in zip(labels, predicted):
            rows.append(
                {
                    "method": method,
                    "window": window,
                    "subject": subject,
                    "block": test_block + 1,
                    "true": int(true_label),
                    "pred": int(predicted_label),
                }
            )
    return rows, time.perf_counter() - started


def run_subject(task: dict[str, object]) -> dict[str, object]:
    subject = int(task["subject"])
    pending = {str(method): [float(window) for window in windows] for method, windows in dict(task["pending"]).items()}
    data_root = Path(str(task["data_root"]))
    epoch_cache = Path(str(task["epoch_cache"]))
    cache_window = float(task["cache_window"])
    preset = benchmark_9ch_default(data_root)
    store = CanonicalEpochStore(epoch_cache)

    raw_epochs: np.ndarray | None = None
    filterbank_epochs: np.ndarray | None = None
    if pending.get("CCA"):
        raw_epochs = store.load_or_create(
            EpochRequest(
                preset=preset,
                subject=subject,
                window=cache_window,
                kind="raw",
                cache_window=cache_window,
            )
        )
    if any(method != "CCA" and windows for method, windows in pending.items()):
        filterbank_epochs = store.load_or_create(
            EpochRequest(
                preset=preset,
                subject=subject,
                window=cache_window,
                kind="filterbank",
                n_fbs=N_FBS,
                cache_window=cache_window,
            )
        )

    rows: list[dict[str, object]] = []
    runtimes: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for method, windows in pending.items():
        source = raw_epochs if method == "CCA" else filterbank_epochs
        if source is None:
            continue
        for window in windows:
            samples = SPEC.sample_length(window)
            if method == "CCA":
                epochs = source[..., :samples][:, :, None, :, :]
            else:
                epochs = source[..., :samples]
            try:
                cell_rows, seconds = run_cell(subject, window, method, epochs)
                rows.extend(cell_rows)
                runtimes.append({"subject": subject, "method": method, "window": window, "seconds": seconds})
            except Exception as error:
                errors.append(
                    {
                        "subject": subject,
                        "method": method,
                        "window": window,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    return {"subject": subject, "rows": rows, "runtimes": runtimes, "errors": errors}


def empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def empty_runtime() -> pd.DataFrame:
    return pd.DataFrame(columns=RUNTIME_COLUMNS)


def load_runtime(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "runtime.csv"
    if not path.exists():
        return empty_runtime()
    frame = pd.read_csv(path)
    missing = sorted(set(RUNTIME_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    frame = frame[RUNTIME_COLUMNS].copy()
    frame["subject"] = frame["subject"].astype(int)
    frame["method"] = frame["method"].astype(str).str.upper()
    frame["window"] = frame["window"].astype(float).round(4)
    frame["seconds"] = frame["seconds"].astype(float)
    if frame.duplicated(RUNTIME_KEY).any():
        raise ValueError(f"{path} contains duplicate runtime keys.")
    return frame.sort_values(RUNTIME_KEY).reset_index(drop=True)


def merge_runtime(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if new_rows.empty:
        return existing
    normalized = new_rows[RUNTIME_COLUMNS].copy()
    normalized["subject"] = normalized["subject"].astype(int)
    normalized["method"] = normalized["method"].astype(str).str.upper()
    normalized["window"] = normalized["window"].astype(float).round(4)
    normalized["seconds"] = normalized["seconds"].astype(float)
    new_keys = normalized[RUNTIME_KEY].drop_duplicates()
    retained = existing.merge(new_keys.assign(_replace=True), on=RUNTIME_KEY, how="left")
    retained = retained[retained["_replace"].isna()].drop(columns="_replace")
    merged = pd.concat([retained, normalized], ignore_index=True)
    if merged.duplicated(RUNTIME_KEY).any():
        raise ValueError("Runtime merge produced duplicate keys.")
    return merged.sort_values(RUNTIME_KEY).reset_index(drop=True)


def normalize_predictions(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = sorted(set(PREDICTION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")
    normalized = frame[PREDICTION_COLUMNS].copy()
    normalized["method"] = normalized["method"].astype(str).str.upper()
    normalized["window"] = normalized["window"].astype(float).round(4)
    for column in ("subject", "block", "true", "pred"):
        normalized[column] = normalized[column].astype(int)
    if normalized.duplicated(PREDICTION_KEY).any():
        raise ValueError(f"{source} contains duplicate prediction keys.")
    return normalized


def load_existing_by_subject(out_dir: Path) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    parts_dir = out_dir / "parts"
    for path in sorted(parts_dir.glob("subject_*.csv")) if parts_dir.exists() else []:
        frame = normalize_predictions(pd.read_csv(path), path)
        subjects = frame["subject"].unique()
        if len(subjects) != 1:
            raise ValueError(f"{path} must contain exactly one subject.")
        frames[int(subjects[0])] = frame

    predictions_path = out_dir / "predictions.csv"
    if predictions_path.exists():
        combined = normalize_predictions(pd.read_csv(predictions_path), predictions_path)
        for subject, frame in combined.groupby("subject", sort=True):
            frames.setdefault(int(subject), frame.copy())
    return frames


def complete_cells(frames: dict[int, pd.DataFrame]) -> set[tuple[int, float, str]]:
    done: set[tuple[int, float, str]] = set()
    for subject, frame in frames.items():
        for (method, window), group in frame.groupby(["method", "window"], sort=False):
            if len(group) != ROWS_PER_CELL:
                continue
            if group["block"].nunique() != SPEC.blocks or group["true"].nunique() != SPEC.classes:
                continue
            if group.groupby(["block", "true"]).size().eq(1).all():
                done.add((subject, float(window), str(method)))
    return done


def replace_cells(existing: pd.DataFrame, new_rows: pd.DataFrame, pending: dict[str, list[float]]) -> pd.DataFrame:
    retained = existing.copy()
    for method, windows in pending.items():
        retained = retained[~((retained["method"] == method) & retained["window"].isin(windows))]
    merged = pd.concat([retained, new_rows], ignore_index=True)
    if merged.duplicated(PREDICTION_KEY).any():
        raise ValueError("Checkpoint merge produced duplicate prediction keys.")
    return merged.sort_values(["method", "window", "subject", "block", "true"]).reset_index(drop=True)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_subject_checkpoint(out_dir: Path, subject: int, frame: pd.DataFrame) -> None:
    atomic_csv(frame, out_dir / "parts" / f"subject_{subject:02d}.csv")


def combine_predictions(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return empty_predictions()
    combined = pd.concat([frames[subject] for subject in sorted(frames)], ignore_index=True)
    if combined.duplicated(PREDICTION_KEY).any():
        raise ValueError("Combined predictions contain duplicate keys.")
    return combined.sort_values(["method", "window", "subject", "block", "true"]).reset_index(drop=True)


def build_summary(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_rows: list[dict[str, object]] = []
    for (method, window, subject), group in predictions.groupby(["method", "window", "subject"], sort=True):
        accuracy = float((group["true"] == group["pred"]).mean())
        subject_rows.append(
            {
                "method": method,
                "window": float(window),
                "subject": int(subject),
                "accuracy": accuracy,
                "itr_bpm": itr_bits_per_minute(accuracy, SPEC.classes, float(window) + SPEC.cue_seconds),
                "samples": int(len(group)),
            }
        )
    subject = pd.DataFrame(subject_rows)
    summary = (
        subject.groupby(["method", "window"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_sem=("accuracy", lambda values: float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0),
            itr_bpm=("itr_bpm", "mean"),
            subjects=("subject", "nunique"),
            samples=("samples", "sum"),
        )
        .sort_values(["method", "window"])
    )
    return summary, subject


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    subjects: list[int],
    windows: list[float],
    methods: list[str],
    predictions: pd.DataFrame,
    done: set[tuple[int, float, str]],
    errors: list[dict[str, object]],
    started_at: str,
    seconds: float,
) -> dict[str, object]:
    requested_cells = {(subject, window, method) for subject in subjects for window in windows for method in methods}
    completed_requested = requested_cells & done
    status = "complete" if completed_requested == requested_cells and not errors else "partial"
    preset = benchmark_9ch_default(args.data_root)
    fingerprint = epoch_fingerprint(
        EpochRequest(
            preset=preset,
            subject=subjects[-1],
            window=args.cache_window,
            kind="filterbank",
            n_fbs=N_FBS,
            cache_window=args.cache_window,
        )
    )
    manifest = {
        "task": "benchmark_decision_channel_capacity_extended",
        "stage": args.stage,
        "status": status,
        "resume": bool(args.resume),
        "subjects_requested": subjects,
        "windows_requested": windows,
        "methods_requested": methods,
        "methods_present": sorted(predictions["method"].unique().tolist()) if not predictions.empty else [],
        "classes": SPEC.classes,
        "blocks": SPEC.blocks,
        "expected_cells": len(requested_cells),
        "completed_cells": len(completed_requested),
        "expected_prediction_rows": len(requested_cells) * ROWS_PER_CELL,
        "prediction_rows_in_requested_scope": int(
            len(
                predictions[
                    predictions["subject"].isin(subjects)
                    & predictions["window"].isin(windows)
                    & predictions["method"].isin(methods)
                ]
            )
        ),
        "prediction_rows_total": int(len(predictions)),
        "error_count": len(errors),
        "preprocessing": {
            "policy": "fixed_max_window_slice",
            "context_seconds": args.cache_window,
            "notch": "50 Hz iircomb Q=35",
            "filterbank": "SSVEP-Analysis-Toolbox Benchmark filterbank",
            "n_filterbanks": N_FBS,
            "harmonics": HARMONICS,
        },
        "epoch_cache": str(args.epoch_cache),
        "epoch_fingerprint": fingerprint,
        "started_at_utc": started_at,
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "invocation_seconds": seconds,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended fixed-context Benchmark window sweep")
    parser.add_argument("--stage", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--methods", default=None, help="Comma-separated methods; defaults to canonical five")
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--windows", default=None, help="Optional comma-separated override for protocol checks")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-window", type=float, default=DEFAULT_CACHE_WINDOW)
    parser.add_argument("--epoch-cache", type=Path, default=DEFAULT_EPOCH_CACHE)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least one.")
    args.data_root = DATA_ROOT
    subjects = parse_subjects(args.subjects)
    windows = parse_windows(args.windows, args.stage)
    methods = parse_methods(args.methods)
    if args.smoke:
        subjects = subjects[:2]
        windows = windows[:5]
    if args.cache_window < max(windows):
        raise ValueError("--cache-window must be at least the longest requested window.")

    default_name = "coarse_0.1s" if args.stage == "coarse" else "fine_8ms"
    if args.smoke:
        default_name += "_smoke"
    out_dir = RESULTS_DIR / (args.output_name or default_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if not args.resume and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; use --resume or choose --output-name.")
    if args.resume and any(out_dir.iterdir()) and not manifest_path.exists():
        raise ValueError(f"Cannot resume {out_dir}: manifest.json is missing.")
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_context = float(previous.get("preprocessing", {}).get("context_seconds", -1))
        if previous_context != args.cache_window:
            raise ValueError(f"Resume context mismatch: existing={previous_context}, requested={args.cache_window}.")

    existing = load_existing_by_subject(out_dir)
    done = complete_cells(existing)
    tasks: list[dict[str, object]] = []
    for subject in subjects:
        pending = {
            method: [window for window in windows if (subject, window, method) not in done]
            for method in methods
        }
        pending = {method: method_windows for method, method_windows in pending.items() if method_windows}
        if pending:
            tasks.append(
                {
                    "subject": subject,
                    "pending": pending,
                    "data_root": str(args.data_root),
                    "epoch_cache": str(args.epoch_cache),
                    "cache_window": args.cache_window,
                }
            )

    total_cells = len(subjects) * len(windows) * len(methods)
    completed_before = len({cell for cell in done if cell[0] in subjects and cell[1] in windows and cell[2] in methods})
    print(f"Extended Benchmark: stage={args.stage}")
    print(f"  Windows: {len(windows)} ({windows[0]:.3f}s - {windows[-1]:.3f}s)")
    print(f"  Fixed preprocessing context: {args.cache_window:.1f}s")
    print(f"  Methods: {methods}")
    print(f"  Subjects: {len(subjects)} (S{subjects[0]:02d}-S{subjects[-1]:02d})")
    print(f"  Cells: {total_cells} ({total_cells - completed_before} remaining)")
    print(f"  Subject tasks: {len(tasks)}; workers: {args.workers}")
    print(f"  Output: {out_dir}", flush=True)

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    started = time.perf_counter()
    errors: list[dict[str, object]] = []
    runtime = load_runtime(out_dir)
    if not (out_dir / "runtime.csv").exists():
        atomic_csv(runtime, out_dir / "runtime.csv")
    initial_predictions = combine_predictions(existing)
    write_manifest(
        manifest_path,
        args=args,
        subjects=subjects,
        windows=windows,
        methods=methods,
        predictions=initial_predictions,
        done=done,
        errors=errors,
        started_at=started_at,
        seconds=0.0,
    )

    def consume(result: dict[str, object], task: dict[str, object], completed: int) -> None:
        nonlocal runtime
        subject = int(result["subject"])
        new_rows = normalize_predictions(pd.DataFrame(result["rows"], columns=PREDICTION_COLUMNS), out_dir)
        existing_subject = existing.get(subject, empty_predictions())
        merged = replace_cells(existing_subject, new_rows, dict(task["pending"]))
        existing[subject] = merged
        save_subject_checkpoint(out_dir, subject, merged)
        subject_errors = list(result["errors"])
        errors.extend(subject_errors)
        if result["runtimes"]:
            runtime = merge_runtime(runtime, pd.DataFrame(result["runtimes"]))
            atomic_csv(runtime, out_dir / "runtime.csv")
        elapsed = time.perf_counter() - started
        eta = elapsed / completed * (len(tasks) - completed) if completed else 0.0
        print(
            f"  [{completed}/{len(tasks)}] S{subject:02d}: {len(new_rows) // ROWS_PER_CELL} cells "
            f"checkpointed; errors={len(subject_errors)}; elapsed={elapsed:.0f}s; eta~{eta:.0f}s",
            flush=True,
        )

    if args.workers == 1:
        for completed, task in enumerate(tasks, start=1):
            consume(run_subject(task), task, completed)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_subject, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    subject = int(task["subject"])
                    result = {
                        "subject": subject,
                        "rows": [],
                        "runtimes": [],
                        "errors": [
                            {
                                "subject": subject,
                                "method": "__task__",
                                "window": np.nan,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        ],
                    }
                consume(result, task, completed)

    predictions = combine_predictions(existing)
    atomic_csv(predictions, out_dir / "predictions.csv")
    summary, subject_summary = build_summary(predictions)
    atomic_csv(summary, out_dir / "summary.csv")
    atomic_csv(subject_summary, out_dir / "subject.csv")
    pd.DataFrame(errors, columns=["subject", "method", "window", "error_type", "error"]).to_csv(
        out_dir / "errors.csv", index=False
    )

    done = complete_cells(existing)
    elapsed = time.perf_counter() - started
    manifest = write_manifest(
        manifest_path,
        args=args,
        subjects=subjects,
        windows=windows,
        methods=methods,
        predictions=predictions,
        done=done,
        errors=errors,
        started_at=started_at,
        seconds=elapsed,
    )
    print(f"Done in {elapsed:.0f}s: status={manifest['status']}, rows={len(predictions)}, errors={len(errors)}")
    if manifest["status"] != "complete":
        raise RuntimeError(f"Extended benchmark is partial; inspect {out_dir / 'errors.csv'}.")


if __name__ == "__main__":
    main()
