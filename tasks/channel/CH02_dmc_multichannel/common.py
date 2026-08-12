from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from channel_configs import CHANNEL_CONFIGS, CONFIG_BY_SLUG, ChannelConfig
from vep_arena.config import BENCHMARK_FREQS, CACHE_ROOT, DATA_ROOT, BenchmarkSpec
from vep_arena.methods.traditional import CCA, ECCA, FBCCA, TRCA


TASK_ROOT = Path(__file__).resolve().parent
PARENT_TASK_ROOT = PROJECT_ROOT / "tasks" / "channel" / "CH01_dmc_benchmark"
DEFAULT_RESULT_ROOT = PARENT_TASK_ROOT / "results" / "extended" / "multichannel"
DEFAULT_CACHE_ROOT = CACHE_ROOT / "benchmark_multichannel_fixed5_cache"
DEFAULT_SOURCE_9CH = PARENT_TASK_ROOT / "results" / "extended" / "coarse_0.1s" / "predictions.csv"

SPEC = BenchmarkSpec()
METHODS = ("CCA", "FBCCA", "ECCA", "TRCA", "ETRCA")
N_FBS = 5
HARMONICS = 5
CACHE_WINDOW = 5.0
ROWS_PER_CELL = SPEC.blocks * SPEC.classes
EXPECTED_RAW_SHAPE = (SPEC.classes, SPEC.blocks, 64, SPEC.sample_length(CACHE_WINDOW))
EXPECTED_FILTERBANK_SHAPE = (SPEC.classes, SPEC.blocks, N_FBS, 64, SPEC.sample_length(CACHE_WINDOW))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def coarse_windows() -> tuple[float, ...]:
    return tuple(round(samples / SPEC.sampling_rate, 4) for samples in range(25, 1251, 25))


WINDOWS = coarse_windows()


def parse_subjects(text: str) -> tuple[int, ...]:
    subjects: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = token.split("-", 1)
            subjects.extend(range(int(start), int(stop) + 1))
        else:
            subjects.append(int(token))
    normalized = tuple(sorted(set(subjects)))
    if not normalized or normalized[0] < 1 or normalized[-1] > SPEC.subjects:
        raise ValueError(f"Subjects must be within 1-{SPEC.subjects}.")
    return normalized


def parse_methods(text: str) -> tuple[str, ...]:
    methods = tuple(dict.fromkeys(token.strip().upper() for token in text.split(",") if token.strip()))
    unknown = sorted(set(methods) - set(METHODS))
    if not methods or unknown:
        raise ValueError(f"Methods must be drawn from {METHODS}; unknown={unknown}.")
    return methods


def parse_configs(text: str) -> tuple[ChannelConfig, ...]:
    slugs = tuple(dict.fromkeys(token.strip().lower() for token in text.split(",") if token.strip()))
    unknown = sorted(set(slugs) - set(CONFIG_BY_SLUG))
    if not slugs or unknown:
        raise ValueError(f"Channel configs must be drawn from {tuple(CONFIG_BY_SLUG)}; unknown={unknown}.")
    return tuple(CONFIG_BY_SLUG[slug] for slug in slugs)


def parse_windows(text: str) -> tuple[float, ...]:
    windows = tuple(sorted(set(round(float(token.strip()), 4) for token in text.split(",") if token.strip())))
    unknown = sorted(set(windows) - set(WINDOWS))
    if not windows or unknown:
        raise ValueError(f"Windows must be on the 0.1-5.0 s coarse grid; unknown={unknown}.")
    return windows


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def cache_paths(cache_root: Path, subject: int) -> dict[str, Path]:
    subject_root = cache_root / f"S{subject:02d}"
    return {
        "root": subject_root,
        "raw": subject_root / "raw_64ch_fixed5_float32.npy",
        "filterbank": subject_root / "filterbank5_64ch_fixed5_float32.npy",
        "manifest": subject_root / "manifest.json",
    }


def valid_array_file(path: Path, expected_shape: tuple[int, ...]) -> bool:
    if not path.exists():
        return False
    try:
        array = np.load(path, mmap_mode="r")
        return array.shape == expected_shape and array.dtype == np.float32
    except (OSError, ValueError):
        return False


def validate_subject_cache(cache_root: Path, subject: int) -> bool:
    paths = cache_paths(cache_root, subject)
    return (
        paths["manifest"].exists()
        and valid_array_file(paths["raw"], EXPECTED_RAW_SHAPE)
        and valid_array_file(paths["filterbank"], EXPECTED_FILTERBANK_SHAPE)
    )


def window_slug(window: float) -> str:
    return f"{window:.1f}"


def confusion_path(
    result_root: Path,
    subject: int,
    method: str,
    config: ChannelConfig,
    window: float,
) -> Path:
    filename = (
        f"confusion_S{subject:02d}_{method.upper()}_{config.slug}_"
        f"{config.channels}ch_w{window_slug(window)}.npy"
    )
    return result_root / "confusions" / filename


def load_valid_confusion(path: Path) -> np.ndarray:
    counts = np.load(path)
    if counts.shape != (SPEC.classes, SPEC.classes):
        raise ValueError(f"{path} has shape {counts.shape}; expected {(SPEC.classes, SPEC.classes)}.")
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError(f"{path} must contain integer counts, got {counts.dtype}.")
    if np.any(counts < 0):
        raise ValueError(f"{path} contains negative counts.")
    if int(counts.sum()) != ROWS_PER_CELL:
        raise ValueError(f"{path} sums to {int(counts.sum())}; expected {ROWS_PER_CELL}.")
    return np.asarray(counts, dtype=np.int64)


def valid_confusion(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        load_valid_confusion(path)
        return True
    except (OSError, ValueError):
        return False


def make_model(method: str, window: float):
    if method == "CCA":
        return CCA(window=window, harmonics=HARMONICS, spec=SPEC)
    if method == "FBCCA":
        return FBCCA(window=window, harmonics=HARMONICS, n_fbs=N_FBS, spec=SPEC)
    if method == "ECCA":
        return ECCA(window=window, harmonics=HARMONICS, n_fbs=N_FBS, spec=SPEC)
    if method == "TRCA":
        return TRCA(n_fbs=N_FBS, ensemble=False)
    if method == "ETRCA":
        return TRCA(n_fbs=N_FBS, ensemble=True)
    raise ValueError(f"Unknown method: {method}")


def classify_confusion(method: str, window: float, epochs: np.ndarray) -> tuple[np.ndarray, float]:
    labels = np.arange(SPEC.classes, dtype=np.int64)
    counts = np.zeros((SPEC.classes, SPEC.classes), dtype=np.int64)
    started = time.perf_counter()
    for test_block in range(SPEC.blocks):
        train_blocks = [block for block in range(SPEC.blocks) if block != test_block]
        train_x = epochs[:, train_blocks].reshape(-1, epochs.shape[2], epochs.shape[3], epochs.shape[4])
        train_y = np.repeat(labels, len(train_blocks))
        test_x = epochs[:, test_block]
        model = make_model(method, window)
        model.fit(train_x, train_y)
        predicted, _scores = model.predict(test_x)
        predicted = np.asarray(predicted, dtype=np.int64)
        if predicted.shape != labels.shape or np.any(predicted < 0) or np.any(predicted >= SPEC.classes):
            raise ValueError(f"{method} returned invalid predictions with shape {predicted.shape}.")
        np.add.at(counts, (labels, predicted), 1)
    if int(counts.sum()) != ROWS_PER_CELL:
        raise ValueError(f"Confusion matrix sums to {int(counts.sum())}; expected {ROWS_PER_CELL}.")
    return counts, time.perf_counter() - started


def experiment_manifest() -> dict[str, object]:
    return {
        "task": "benchmark_multichannel_decision_channel_capacity",
        "dataset": "Tsinghua Benchmark SSVEP",
        "subjects": list(range(1, SPEC.subjects + 1)),
        "methods": list(METHODS),
        "windows": list(WINDOWS),
        "channel_configs": [config.manifest() for config in CHANNEL_CONFIGS],
        "classes": SPEC.classes,
        "blocks": SPEC.blocks,
        "expected_confusions": SPEC.subjects * len(METHODS) * len(CHANNEL_CONFIGS) * len(WINDOWS),
        "preprocessing": {
            "policy": "fixed_max_window_slice",
            "context_seconds": CACHE_WINDOW,
            "cue_seconds": SPEC.cue_seconds,
            "latency_seconds": SPEC.latency_seconds,
            "notch": "50 Hz iircomb Q=35",
            "filterbank": "SSVEP-Analysis-Toolbox Benchmark filterbank",
            "n_filterbanks": N_FBS,
            "harmonics": HARMONICS,
        },
        "confusion_samples_per_cell": ROWS_PER_CELL,
        "capacity_alpha": 0.0,
    }
