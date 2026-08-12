from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parent
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from common import (
    CHANNEL_CONFIGS,
    DEFAULT_RESULT_ROOT,
    METHODS,
    SPEC,
    WINDOWS,
    atomic_json,
    confusion_path,
    load_valid_confusion,
    parse_configs,
    parse_methods,
    parse_subjects,
    parse_windows,
    utc_now,
    window_slug,
)

import numpy as np
import pandas as pd


PROJECT_ROOT = TASK_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.channel.capacity import (
    capacity_ba,
    capacity_c0,
    capacity_c1,
    conditional_entropy_rows,
    mutual_info_uniform,
)
from vep_arena.channel.confusion import normalize_confusion


SUBJECT_COLUMNS = (
    "subject",
    "method",
    "channel_config",
    "channels",
    "window",
    "samples",
    "accuracy",
    "c0",
    "c1",
    "c_ba",
    "ba_iterations",
    "ba_gap",
    "ba_converged",
    "i_uniform",
    "h_y_given_x_uniform",
    "delta_asm",
)
AGGREGATE_COLUMNS = (
    "method",
    "channel_config",
    "channels",
    "window",
    "subjects",
    "samples",
    "accuracy",
    "c0",
    "c1",
    "c_ba",
    "ba_iterations",
    "ba_gap",
    "ba_converged",
    "i_uniform",
    "h_y_given_x_uniform",
    "delta_asm",
    "accuracy_subject_mean",
    "accuracy_subject_std",
    "accuracy_subject_sem",
    "c_ba_subject_mean",
    "c_ba_subject_std",
    "c_ba_subject_sem",
    "i_uniform_subject_mean",
    "i_uniform_subject_std",
    "i_uniform_subject_sem",
    "delta_asm_subject_mean",
    "delta_asm_subject_std",
    "delta_asm_subject_sem",
)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def bool_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - {"true", "false", "1", "0"})
    if unknown:
        raise ValueError(f"Invalid boolean values: {unknown}")
    return normalized.isin({"true", "1"}).to_numpy(dtype=bool)


def analyze_counts(counts: np.ndarray) -> dict[str, object]:
    samples = int(counts.sum())
    accuracy = float(np.trace(counts) / samples) if samples else 0.0
    transition = normalize_confusion(counts, alpha=0.0)
    ba = capacity_ba(transition)
    i_uniform = float(mutual_info_uniform(transition))
    h_y_given_x_uniform = float(np.mean(conditional_entropy_rows(transition)))
    return {
        "samples": samples,
        "accuracy": accuracy,
        "c0": float(capacity_c0(counts.shape[0])),
        "c1": float(capacity_c1(counts.shape[0], accuracy)),
        "c_ba": float(ba.capacity),
        "ba_iterations": int(ba.iterations),
        "ba_gap": float(ba.gap),
        "ba_converged": bool(ba.converged),
        "i_uniform": i_uniform,
        "h_y_given_x_uniform": h_y_given_x_uniform,
        "delta_asm": max(float(ba.capacity) - i_uniform, 0.0),
    }


def part_paths(result_root: Path, config_slug: str, method: str, window: float) -> tuple[Path, Path]:
    part_root = result_root / "decision_channel" / "parts"
    stem = f"{config_slug}_{method}_w{window_slug(window)}"
    return part_root / f"{stem}_subject.csv", part_root / f"{stem}_aggregate.json"


def valid_part(result_root: Path, config_slug: str, method: str, window: float, subjects: tuple[int, ...]) -> bool:
    subject_path, aggregate_path = part_paths(result_root, config_slug, method, window)
    if not subject_path.exists() or not aggregate_path.exists():
        return False
    try:
        frame = pd.read_csv(subject_path)
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        return (
            len(frame) == len(subjects)
            and set(SUBJECT_COLUMNS).issubset(frame.columns)
            and set(frame["subject"].astype(int)) == set(subjects)
            and set(AGGREGATE_COLUMNS).issubset(aggregate)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def summary_stats(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sem = std / math.sqrt(len(values)) if values.size else 0.0
    return mean, std, sem


def analyze_group(task: dict[str, object]) -> dict[str, object]:
    result_root = Path(str(task["result_root"]))
    config_slug = str(task["channel_config"])
    config = next(config for config in CHANNEL_CONFIGS if config.slug == config_slug)
    method = str(task["method"])
    window = float(task["window"])
    subjects = tuple(int(subject) for subject in task["subjects"])
    subject_records: list[dict[str, object]] = []
    pooled = np.zeros((SPEC.classes, SPEC.classes), dtype=np.int64)

    for subject in subjects:
        counts = load_valid_confusion(confusion_path(result_root, subject, method, config, window))
        pooled += counts
        record = {
            "subject": subject,
            "method": method,
            "channel_config": config.slug,
            "channels": config.channels,
            "window": window,
        }
        record.update(analyze_counts(counts))
        subject_records.append(record)

    subject_frame = pd.DataFrame(subject_records, columns=SUBJECT_COLUMNS)
    aggregate = {
        "method": method,
        "channel_config": config.slug,
        "channels": config.channels,
        "window": window,
        "subjects": len(subjects),
    }
    aggregate.update(analyze_counts(pooled))
    for metric in ("accuracy", "c_ba", "i_uniform", "delta_asm"):
        mean, std, sem = summary_stats(subject_frame[metric].to_numpy(dtype=float))
        aggregate[f"{metric}_subject_mean"] = mean
        aggregate[f"{metric}_subject_std"] = std
        aggregate[f"{metric}_subject_sem"] = sem

    subject_path, aggregate_path = part_paths(result_root, config.slug, method, window)
    atomic_csv(subject_path, subject_frame)
    atomic_json(aggregate_path, aggregate)
    return {
        "channel_config": config.slug,
        "method": method,
        "window": window,
        "subject_rows": len(subject_frame),
        "ba_converged": bool(subject_frame["ba_converged"].all() and aggregate["ba_converged"]),
    }


def combine_parts(
    result_root: Path,
    configs,
    methods: tuple[str, ...],
    windows: tuple[float, ...],
    subjects: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_frames: list[pd.DataFrame] = []
    aggregate_records: list[dict[str, object]] = []
    for config in configs:
        for method in methods:
            for window in windows:
                if not valid_part(result_root, config.slug, method, window, subjects):
                    raise ValueError(f"Analysis part is incomplete: {config.slug} {method} w{window:g}")
                subject_path, aggregate_path = part_paths(result_root, config.slug, method, window)
                subject_frames.append(pd.read_csv(subject_path)[list(SUBJECT_COLUMNS)])
                aggregate_records.append(json.loads(aggregate_path.read_text(encoding="utf-8")))

    subject_frame = pd.concat(subject_frames, ignore_index=True)
    subject_frame = subject_frame.sort_values(
        ["channel_config", "method", "window", "subject"]
    ).reset_index(drop=True)
    aggregate_frame = pd.DataFrame(aggregate_records, columns=AGGREGATE_COLUMNS)
    aggregate_frame = aggregate_frame.sort_values(["channel_config", "method", "window"]).reset_index(drop=True)
    decision_root = result_root / "decision_channel"
    atomic_csv(decision_root / "capacity_by_subject_method_channels_window.csv", subject_frame)
    atomic_csv(decision_root / "capacity_aggregate.csv", aggregate_frame)
    return subject_frame, aggregate_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--configs", default=",".join(config.slug for config in CHANNEL_CONFIGS))
    parser.add_argument("--windows", default=",".join(str(window) for window in WINDOWS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    methods = parse_methods(args.methods)
    configs = parse_configs(args.configs)
    windows = parse_windows(args.windows)
    tasks = [
        {
            "result_root": str(args.result_root),
            "channel_config": config.slug,
            "method": method,
            "window": window,
            "subjects": subjects,
        }
        for config in configs
        for method in methods
        for window in windows
        if not valid_part(args.result_root, config.slug, method, window, subjects)
    ]
    total_groups = len(configs) * len(methods) * len(windows)
    print(f"BA analysis: groups={total_groups}, pending={len(tasks)}, workers={args.workers}", flush=True)

    errors: list[dict[str, object]] = []
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(analyze_group, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    result = future.result()
                    print(
                        f"[{completed}/{len(tasks)}] {result['channel_config']} {result['method']} "
                        f"w{result['window']:g}: rows={result['subject_rows']} converged={result['ba_converged']}",
                        flush=True,
                    )
                except Exception as error:
                    errors.append(
                        {
                            "channel_config": task["channel_config"],
                            "method": task["method"],
                            "window": task["window"],
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    print(
                        f"[{completed}/{len(tasks)}] {task['channel_config']} {task['method']} "
                        f"w{task['window']:g}: ERROR {type(error).__name__}: {error}",
                        flush=True,
                    )

    if errors:
        error_path = args.result_root / "decision_channel" / "errors.json"
        atomic_json(error_path, {"status": "failed", "errors": errors, "updated_at_utc": utc_now()})
        raise SystemExit(f"BA analysis failed for {len(errors)} groups; see {error_path}")

    subject_frame, aggregate_frame = combine_parts(args.result_root, configs, methods, windows, subjects)
    manifest = {
        "status": "complete",
        "alpha": 0.0,
        "subjects": list(subjects),
        "methods": list(methods),
        "channel_configs": [config.manifest() for config in configs],
        "windows": list(windows),
        "subject_rows": len(subject_frame),
        "aggregate_rows": len(aggregate_frame),
        "subject_ba_not_converged": int((~bool_values(subject_frame["ba_converged"])).sum()),
        "aggregate_ba_not_converged": int((~bool_values(aggregate_frame["ba_converged"])).sum()),
        "aggregate_semantics": {
            "primary_capacity_columns": "computed from confusion counts pooled across subjects",
            "subject_suffix_columns": "mean/std/sem of independently computed subject metrics",
        },
        "updated_at_utc": utc_now(),
    }
    atomic_json(args.result_root / "decision_channel" / "analysis_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
