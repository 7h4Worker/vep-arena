from __future__ import annotations

import argparse
import gc
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


TASK_ROOT = Path(__file__).resolve().parent
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from common import (
    CACHE_WINDOW,
    DEFAULT_CACHE_ROOT,
    EXPECTED_FILTERBANK_SHAPE,
    EXPECTED_RAW_SHAPE,
    N_FBS,
    SPEC,
    atomic_json,
    atomic_npy,
    cache_paths,
    parse_subjects,
    utc_now,
    valid_array_file,
    validate_subject_cache,
)


PROJECT_ROOT = TASK_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.config import DATA_ROOT
from vep_arena.data.benchmark import load_subject_filterbank, load_subject_toolbox_raw
from vep_arena.data.presets import benchmark_64ch_default


def prepare_subject(subject: int, data_root: str, cache_root: str) -> dict[str, object]:
    resolved_data_root = Path(data_root)
    resolved_cache_root = Path(cache_root)
    paths = cache_paths(resolved_cache_root, subject)
    paths["root"].mkdir(parents=True, exist_ok=True)
    preset = benchmark_64ch_default(resolved_data_root)
    built: list[str] = []

    if not valid_array_file(paths["raw"], EXPECTED_RAW_SHAPE):
        raw = load_subject_toolbox_raw(
            resolved_data_root,
            subject,
            CACHE_WINDOW,
            channels=preset.channels,
            spec=SPEC,
        ).astype(np.float32, copy=False)
        if raw.shape != EXPECTED_RAW_SHAPE:
            raise ValueError(f"S{subject:02d} raw cache shape {raw.shape} does not match {EXPECTED_RAW_SHAPE}.")
        atomic_npy(paths["raw"], raw)
        built.append("raw")
        del raw
        gc.collect()

    if not valid_array_file(paths["filterbank"], EXPECTED_FILTERBANK_SHAPE):
        filterbank = load_subject_filterbank(
            resolved_data_root,
            subject,
            CACHE_WINDOW,
            n_fbs=N_FBS,
            channels=preset.channels,
            spec=SPEC,
        ).astype(np.float32, copy=False)
        if filterbank.shape != EXPECTED_FILTERBANK_SHAPE:
            raise ValueError(
                f"S{subject:02d} filterbank cache shape {filterbank.shape} does not match {EXPECTED_FILTERBANK_SHAPE}."
            )
        atomic_npy(paths["filterbank"], filterbank)
        built.append("filterbank")
        del filterbank
        gc.collect()

    manifest = {
        "status": "complete",
        "subject": subject,
        "preset": preset.manifest(),
        "cache_window": CACHE_WINDOW,
        "raw": {"path": str(paths["raw"]), "shape": list(EXPECTED_RAW_SHAPE), "dtype": "float32"},
        "filterbank": {
            "path": str(paths["filterbank"]),
            "shape": list(EXPECTED_FILTERBANK_SHAPE),
            "dtype": "float32",
            "n_filterbanks": N_FBS,
        },
        "updated_at_utc": utc_now(),
    }
    atomic_json(paths["manifest"], manifest)
    if not validate_subject_cache(resolved_cache_root, subject):
        raise ValueError(f"S{subject:02d} cache validation failed after writing.")
    return {"subject": subject, "built": built, "status": "complete"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    pending = [subject for subject in subjects if not validate_subject_cache(args.cache_root, subject)]
    print(f"Fixed5 full64 cache: subjects={len(subjects)}, pending={len(pending)}, workers={args.workers}", flush=True)
    if not pending:
        return

    errors: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(prepare_subject, subject, str(args.data_root), str(args.cache_root)): subject
            for subject in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            subject = futures[future]
            try:
                result = future.result()
                print(f"[{completed}/{len(pending)}] S{subject:02d}: built={result['built']}", flush=True)
            except Exception as error:
                errors.append({"subject": subject, "error_type": type(error).__name__, "error": str(error)})
                print(f"[{completed}/{len(pending)}] S{subject:02d}: ERROR {type(error).__name__}: {error}", flush=True)

    if errors:
        error_path = args.cache_root / "errors.json"
        atomic_json(error_path, {"status": "failed", "errors": errors, "updated_at_utc": utc_now()})
        raise SystemExit(f"Cache preparation failed for {len(errors)} subjects; see {error_path}")


if __name__ == "__main__":
    main()
