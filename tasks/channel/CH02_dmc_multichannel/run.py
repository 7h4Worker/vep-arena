from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parent
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from common import (
    CHANNEL_CONFIGS,
    CONFIG_BY_SLUG,
    DEFAULT_CACHE_ROOT,
    DEFAULT_RESULT_ROOT,
    DEFAULT_SOURCE_9CH,
    METHODS,
    ROWS_PER_CELL,
    SPEC,
    WINDOWS,
    atomic_json,
    atomic_npy,
    cache_paths,
    classify_confusion,
    confusion_path,
    experiment_manifest,
    parse_configs,
    parse_methods,
    parse_subjects,
    parse_windows,
    utc_now,
    valid_confusion,
    validate_subject_cache,
)

import numpy as np
import pandas as pd


RUNTIME_COLUMNS = (
    "subject",
    "method",
    "channel_config",
    "channels",
    "window",
    "seconds",
    "completed_at_utc",
)
RUNTIME_KEY = ("subject", "method", "channel_config", "window")
ERROR_COLUMNS = (
    "subject",
    "method",
    "channel_config",
    "channels",
    "window",
    "error_type",
    "error",
    "updated_at_utc",
)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_existing_9ch(
    source: Path,
    result_root: Path,
    subjects: tuple[int, ...],
    methods: tuple[str, ...],
    windows: tuple[float, ...],
) -> dict[str, object]:
    required_columns = {"subject", "method", "window", "block", "true", "pred"}
    frame = pd.read_csv(source)
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{source} is missing columns: {', '.join(missing_columns)}")
    frame["subject"] = frame["subject"].astype(int)
    frame["method"] = frame["method"].astype(str).str.upper()
    frame["window"] = frame["window"].astype(float).round(4)
    frame["true"] = frame["true"].astype(int)
    frame["pred"] = frame["pred"].astype(int)
    frame = frame[
        frame["subject"].isin(subjects)
        & frame["method"].isin(methods)
        & frame["window"].isin(windows)
    ].copy()

    config = CONFIG_BY_SLUG["occipital9"]
    expected_cells = len(subjects) * len(methods) * len(windows)
    grouped = frame.groupby(["subject", "method", "window"], sort=True)
    if grouped.ngroups != expected_cells:
        raise ValueError(f"9ch source exposes {grouped.ngroups} cells; expected {expected_cells}.")

    written = 0
    skipped = 0
    for completed, ((subject, method, window), group) in enumerate(grouped, start=1):
        path = confusion_path(result_root, int(subject), str(method), config, float(window))
        if valid_confusion(path):
            skipped += 1
            continue
        if len(group) != ROWS_PER_CELL:
            raise ValueError(f"S{subject:02d} {method} w{window:g} has {len(group)} rows; expected {ROWS_PER_CELL}.")
        counts = np.zeros((SPEC.classes, SPEC.classes), dtype=np.int64)
        np.add.at(counts, (group["true"].to_numpy(), group["pred"].to_numpy()), 1)
        if int(counts.sum()) != ROWS_PER_CELL:
            raise ValueError(f"S{subject:02d} {method} w{window:g} confusion sum is invalid.")
        atomic_npy(path, counts)
        written += 1
        if completed % 500 == 0:
            print(f"9ch extraction [{completed}/{expected_cells}] written={written} skipped={skipped}", flush=True)

    manifest = {
        "status": "complete",
        "source": str(source),
        "source_sha256": file_sha256(source),
        "source_rows": len(frame),
        "expected_cells": expected_cells,
        "written": written,
        "skipped": skipped,
        "channel_config": config.manifest(),
        "subjects": list(subjects),
        "methods": list(methods),
        "windows": list(windows),
        "updated_at_utc": utc_now(),
    }
    atomic_json(result_root / "occipital9_extraction_manifest.json", manifest)
    return manifest


def chunked(values: tuple[float, ...], size: int) -> list[tuple[float, ...]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def sweep_batch(task: dict[str, object]) -> dict[str, object]:
    subject = int(task["subject"])
    method = str(task["method"])
    config = CONFIG_BY_SLUG[str(task["channel_config"])]
    windows = tuple(float(value) for value in task["windows"])
    cache_root = Path(str(task["cache_root"]))
    result_root = Path(str(task["result_root"]))
    if not validate_subject_cache(cache_root, subject):
        raise FileNotFoundError(f"S{subject:02d} fixed5 full64 cache is incomplete under {cache_root}.")

    paths = cache_paths(cache_root, subject)
    source = np.load(paths["raw"] if method == "CCA" else paths["filterbank"], mmap_mode="r")
    channel_indices = np.asarray(config.indices_1based, dtype=np.int64) - 1
    completed: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    skipped = 0

    for window in windows:
        path = confusion_path(result_root, subject, method, config, window)
        if valid_confusion(path):
            skipped += 1
            continue
        try:
            samples = SPEC.sample_length(window)
            if method == "CCA":
                selected = np.take(source[..., :samples], channel_indices, axis=2)
                epochs = selected[:, :, None, :, :]
            else:
                epochs = np.take(source[..., :samples], channel_indices, axis=3)
            counts, seconds = classify_confusion(method, window, epochs)
            atomic_npy(path, counts)
            completed.append(
                {
                    "subject": subject,
                    "method": method,
                    "channel_config": config.slug,
                    "channels": config.channels,
                    "window": window,
                    "seconds": seconds,
                    "completed_at_utc": utc_now(),
                }
            )
            del epochs
        except Exception as error:
            errors.append(
                {
                    "subject": subject,
                    "method": method,
                    "channel_config": config.slug,
                    "channels": config.channels,
                    "window": window,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "updated_at_utc": utc_now(),
                }
            )
    return {"completed": completed, "errors": errors, "skipped": skipped}


def load_table(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return frame[list(columns)].copy()


def merge_runtime(existing: pd.DataFrame, records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return existing
    new = pd.DataFrame(records, columns=RUNTIME_COLUMNS)
    combined = pd.concat([existing, new], ignore_index=True)
    combined["subject"] = combined["subject"].astype(int)
    combined["window"] = combined["window"].astype(float).round(4)
    combined = combined.drop_duplicates(list(RUNTIME_KEY), keep="last")
    return combined.sort_values(list(RUNTIME_KEY)).reset_index(drop=True)


def append_errors(existing: pd.DataFrame, records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return existing
    combined = pd.concat([existing, pd.DataFrame(records, columns=ERROR_COLUMNS)], ignore_index=True)
    combined["subject"] = combined["subject"].astype(int)
    combined["window"] = combined["window"].astype(float).round(4)
    return combined.drop_duplicates(
        ["subject", "method", "channel_config", "window", "error_type", "error"], keep="last"
    ).sort_values(["subject", "method", "channel_config", "window"]).reset_index(drop=True)


def prune_resolved_errors(errors: pd.DataFrame, result_root: Path) -> pd.DataFrame:
    if errors.empty:
        return errors
    unresolved: list[bool] = []
    for row in errors.itertuples(index=False):
        config = CONFIG_BY_SLUG.get(str(row.channel_config))
        if config is None:
            unresolved.append(True)
            continue
        path = confusion_path(
            result_root,
            int(row.subject),
            str(row.method),
            config,
            float(row.window),
        )
        unresolved.append(not valid_confusion(path))
    return errors.loc[unresolved].reset_index(drop=True)


def build_sweep_tasks(
    result_root: Path,
    cache_root: Path,
    subjects: tuple[int, ...],
    methods: tuple[str, ...],
    configs,
    windows: tuple[float, ...],
    chunk_size: int,
) -> tuple[list[dict[str, object]], int]:
    tasks: list[dict[str, object]] = []
    already_complete = 0
    for config in configs:
        if config.slug == "occipital9":
            continue
        for method in methods:
            for window_batch in chunked(windows, chunk_size):
                for subject in subjects:
                    pending = tuple(
                        window
                        for window in window_batch
                        if not valid_confusion(confusion_path(result_root, subject, method, config, window))
                    )
                    already_complete += len(window_batch) - len(pending)
                    if pending:
                        tasks.append(
                            {
                                "subject": subject,
                                "method": method,
                                "channel_config": config.slug,
                                "windows": pending,
                                "cache_root": str(cache_root),
                                "result_root": str(result_root),
                            }
                        )
    return tasks, already_complete


def run_sweep(
    result_root: Path,
    cache_root: Path,
    subjects: tuple[int, ...],
    methods: tuple[str, ...],
    configs,
    windows: tuple[float, ...],
    workers: int,
    chunk_size: int,
) -> None:
    missing_caches = [subject for subject in subjects if not validate_subject_cache(cache_root, subject)]
    if missing_caches:
        raise FileNotFoundError(f"Missing fixed5 caches for subjects: {missing_caches}")
    tasks, already_complete = build_sweep_tasks(
        result_root, cache_root, subjects, methods, configs, windows, chunk_size
    )
    requested_cells = len(subjects) * len(methods) * len([config for config in configs if config.slug != "occipital9"]) * len(windows)
    print(
        f"Multichannel sweep: requested={requested_cells}, complete={already_complete}, "
        f"pending_batches={len(tasks)}, workers={workers}",
        flush=True,
    )

    runtime_path = result_root / "runtime.csv"
    error_path = result_root / "errors.csv"
    state_path = result_root / "sweep_state.json"
    runtime = load_table(runtime_path, RUNTIME_COLUMNS)
    errors = prune_resolved_errors(load_table(error_path, ERROR_COLUMNS), result_root)
    atomic_csv(error_path, errors)
    if not tasks:
        atomic_json(
            state_path,
            {
                "status": "complete" if errors.empty else "partial",
                "requested_cells": requested_cells,
                "already_complete_cells": already_complete,
                "errors": len(errors),
                "finished_at_utc": utc_now(),
            },
        )
        if not errors.empty:
            raise SystemExit(f"Sweep has {len(errors)} unresolved errors; see {error_path}")
        return
    saved_cells = 0
    skipped_cells = 0

    atomic_json(
        state_path,
        {
            "status": "running",
            "pending_batches": len(tasks),
            "requested_cells": requested_cells,
            "already_complete_cells": already_complete,
            "started_at_utc": utc_now(),
        },
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(sweep_batch, task): task for task in tasks}
        for batch_index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                result = future.result()
                completed_records = list(result["completed"])
                error_records = list(result["errors"])
                skipped_cells += int(result["skipped"])
            except Exception as error:
                completed_records = []
                error_records = [
                    {
                        "subject": int(task["subject"]),
                        "method": str(task["method"]),
                        "channel_config": str(task["channel_config"]),
                        "channels": CONFIG_BY_SLUG[str(task["channel_config"])].channels,
                        "window": float(window),
                        "error_type": type(error).__name__,
                        "error": f"batch_failure: {error}",
                        "updated_at_utc": utc_now(),
                    }
                    for window in task["windows"]
                ]
            saved_cells += len(completed_records)
            runtime = merge_runtime(runtime, completed_records)
            errors = append_errors(errors, error_records)
            errors = prune_resolved_errors(errors, result_root)
            atomic_csv(runtime_path, runtime)
            atomic_csv(error_path, errors)
            atomic_json(
                state_path,
                {
                    "status": "running",
                    "completed_batches": batch_index,
                    "total_batches": len(tasks),
                    "saved_cells_this_run": saved_cells,
                    "skipped_cells_this_run": skipped_cells,
                    "errors": len(errors),
                    "last_task": task,
                    "updated_at_utc": utc_now(),
                },
            )
            print(
                f"[{batch_index}/{len(tasks)}] {task['channel_config']} {task['method']} "
                f"S{int(task['subject']):02d}: saved={len(completed_records)} errors={len(error_records)}",
                flush=True,
            )

    final_status = "complete" if errors.empty else "partial"
    atomic_json(
        state_path,
        {
            "status": final_status,
            "total_batches": len(tasks),
            "saved_cells_this_run": saved_cells,
            "skipped_cells_this_run": skipped_cells,
            "errors": len(errors),
            "finished_at_utc": utc_now(),
        },
    )
    if not errors.empty:
        raise SystemExit(f"Sweep finished with {len(errors)} recorded errors; see {error_path}")


def bool_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - {"true", "false", "1", "0"})
    if unknown:
        raise ValueError(f"Invalid boolean values: {unknown}")
    return normalized.isin({"true", "1"}).to_numpy(dtype=bool)


def validate_results(
    result_root: Path,
    require_analysis: bool,
    subjects: tuple[int, ...],
    methods: tuple[str, ...],
    configs,
    windows: tuple[float, ...],
) -> dict[str, object]:
    missing: list[str] = []
    invalid: list[str] = []
    checked = 0
    for config in configs:
        for method in methods:
            for window in windows:
                for subject in subjects:
                    path = confusion_path(result_root, subject, method, config, window)
                    checked += 1
                    if not path.exists():
                        missing.append(str(path))
                    elif not valid_confusion(path):
                        invalid.append(str(path))

    subject_rows = None
    aggregate_rows = None
    ba_not_converged = None
    if require_analysis:
        decision_root = result_root / "decision_channel"
        subject_path = decision_root / "capacity_by_subject_method_channels_window.csv"
        aggregate_path = decision_root / "capacity_aggregate.csv"
        if not subject_path.exists() or not aggregate_path.exists():
            missing.extend(str(path) for path in (subject_path, aggregate_path) if not path.exists())
        else:
            subject_frame = pd.read_csv(subject_path)
            aggregate_frame = pd.read_csv(aggregate_path)
            subject_rows = len(subject_frame)
            aggregate_rows = len(aggregate_frame)
            ba_not_converged = int((~bool_values(subject_frame["ba_converged"])).sum()) + int(
                (~bool_values(aggregate_frame["ba_converged"])).sum()
            )
            if subject_rows != checked:
                invalid.append(f"subject capacity rows={subject_rows}, expected={checked}")
            expected_aggregate = len(configs) * len(methods) * len(windows)
            if aggregate_rows != expected_aggregate:
                invalid.append(f"aggregate capacity rows={aggregate_rows}, expected={expected_aggregate}")
            if ba_not_converged:
                invalid.append(f"BA non-converged rows={ba_not_converged}")

    status = "complete" if not missing and not invalid and (not require_analysis or subject_rows is not None) else "partial"
    manifest = experiment_manifest()
    manifest.update(
        {
            "status": status,
            "confusions_checked": checked,
            "missing_count": len(missing),
            "invalid_count": len(invalid),
            "missing_examples": missing[:20],
            "invalid_examples": invalid[:20],
            "capacity_subject_rows": subject_rows,
            "capacity_aggregate_rows": aggregate_rows,
            "ba_not_converged": ba_not_converged,
            "validation_scope": {
                "subjects": list(subjects),
                "methods": list(methods),
                "channel_configs": [config.slug for config in configs],
                "windows": list(windows),
            },
            "updated_at_utc": utc_now(),
        }
    )
    atomic_json(result_root / "manifest.json", manifest)
    return manifest


def status(result_root: Path) -> None:
    paths = [result_root / "manifest.json", result_root / "sweep_state.json"]
    payload = {}
    for path in paths:
        if path.exists():
            payload[path.name] = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract-9")
    extract_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_9CH)
    extract_parser.add_argument("--subjects", default="1-35")
    extract_parser.add_argument("--methods", default=",".join(METHODS))
    extract_parser.add_argument("--windows", default=",".join(str(window) for window in WINDOWS))

    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--subjects", default="1-35")
    sweep_parser.add_argument("--methods", default=",".join(METHODS))
    sweep_parser.add_argument("--configs", default="posterior21,posterior32,wholehead32,full64")
    sweep_parser.add_argument("--windows", default=",".join(str(window) for window in WINDOWS))
    sweep_parser.add_argument("--workers", type=int, default=4)
    sweep_parser.add_argument("--chunk-size", type=int, default=5)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--require-analysis", action="store_true")
    validate_parser.add_argument("--subjects", default="1-35")
    validate_parser.add_argument("--methods", default=",".join(METHODS))
    validate_parser.add_argument("--configs", default=",".join(config.slug for config in CHANNEL_CONFIGS))
    validate_parser.add_argument("--windows", default=",".join(str(window) for window in WINDOWS))
    subparsers.add_parser("status")
    args = parser.parse_args()
    args.result_root.mkdir(parents=True, exist_ok=True)

    if args.command == "extract-9":
        result = extract_existing_9ch(
            args.source,
            args.result_root,
            parse_subjects(args.subjects),
            parse_methods(args.methods),
            parse_windows(args.windows),
        )
        print(json.dumps(result, indent=2))
    elif args.command == "sweep":
        if args.workers < 1 or args.chunk_size < 1:
            raise ValueError("workers and chunk-size must be positive.")
        run_sweep(
            args.result_root,
            args.cache_root,
            parse_subjects(args.subjects),
            parse_methods(args.methods),
            parse_configs(args.configs),
            parse_windows(args.windows),
            args.workers,
            args.chunk_size,
        )
    elif args.command == "validate":
        result = validate_results(
            args.result_root,
            args.require_analysis,
            parse_subjects(args.subjects),
            parse_methods(args.methods),
            parse_configs(args.configs),
            parse_windows(args.windows),
        )
        print(json.dumps(result, indent=2))
        if result["status"] != "complete":
            raise SystemExit(1)
    else:
        status(args.result_root)


if __name__ == "__main__":
    main()
