from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from run import (
    BASE_FREQS,
    DEFAULT_DATASET,
    FB_NUM,
    LATENCY,
    LAG,
    RAW_REL,
    bp40_fb2,
    fit_tdca_single,
    itr_bits_per_minute,
    matlab_data250hz,
    predict_tdca,
    projection_matrices,
)


TASK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TASK_DIR / "results" / "offline_tdca_grid"
AUTHOR_SUBJECTS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S11", "S12", "S13", "S14", "S15"]

TARGET_SETS = {
    "40": {
        "name": "up",
        "indices": np.arange(3, 200, 5),
        "freqs": BASE_FREQS,
    },
    "80": {
        "name": "down_up",
        "indices": np.asarray([i for i in range(200) if (i + 1) % 5 not in (1, 3, 0)], dtype=np.int64),
        "freqs": np.repeat(BASE_FREQS, 2),
    },
    "120": {
        "name": "right_left_up",
        "indices": np.asarray([i for i in range(200) if (i + 1) % 5 not in (2, 0)], dtype=np.int64),
        "freqs": np.repeat(BASE_FREQS, 3),
    },
    "160": {
        "name": "right_down_left_up",
        "indices": np.asarray([i for i in range(200) if (i + 1) % 5 != 0], dtype=np.int64),
        "freqs": np.repeat(BASE_FREQS, 4),
    },
    "200": {
        "name": "right_down_left_up_center",
        "indices": np.arange(200, dtype=np.int64),
        "freqs": np.repeat(BASE_FREQS, 5),
    },
}

CHANNEL_SETS = {
    "9": np.asarray([21, 27, 28, 29, 31, 32, 33, 59, 63], dtype=np.int64),
    "21": np.asarray(list(range(17, 36)) + [59, 63], dtype=np.int64),
    "32": np.asarray(list(range(17, 36)) + list(range(53, 66)), dtype=np.int64),
    "66": np.arange(66, dtype=np.int64),
}


@dataclass(frozen=True)
class GridSpec:
    subject: str
    target_set: str
    channel_set: str
    window_ms: int

    @property
    def key(self) -> tuple[str, str, str, int]:
        return self.subject, self.target_set, self.channel_set, self.window_ms

    @property
    def slug(self) -> str:
        return f"{self.subject}_{self.target_set}target_{self.channel_set}ch_w{self.window_ms}"


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_subjects(text: str) -> list[str]:
    if text.lower() == "author":
        return AUTHOR_SUBJECTS
    return [s if s.upper().startswith("S") else f"S{int(s)}" for s in parse_csv(text)]


def filterbank(raw: np.ndarray) -> np.ndarray:
    channels, samples, targets, blocks = raw.shape
    out = np.empty((channels, samples, targets, blocks, FB_NUM), dtype=np.float32)
    for fb in range(FB_NUM):
        print(f"filter fb={fb + 1}/{FB_NUM}", flush=True)
        out[..., fb] = bp40_fb2(raw, fb)
    return out


def load_or_create_filtered(dataset_root: Path, subject: str, force_filter: bool = False) -> np.ndarray:
    deriv = dataset_root / "derivatives" / "tdca_sample"
    cache_dir = deriv / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"filtered_{subject}_200target_66ch_18blocks_float32.npy"
    if cache.exists() and not force_filter:
        print(f"{subject}: loaded float32 cache {cache}", flush=True)
        return np.load(cache, mmap_mode="r")

    h5_cache = deriv / f"filtered_{subject}_200target_66ch_18blocks.h5"
    if h5_cache.exists() and not force_filter:
        print(f"{subject}: converting h5 cache to float32 {h5_cache}", flush=True)
        with h5py.File(h5_cache, "r") as f:
            arr = np.asarray(f["bpdatahAll"], dtype=np.float32)
        np.save(cache, arr)
        return np.load(cache, mmap_mode="r")

    raw_path = dataset_root / RAW_REL / "data" / "offline" / f"{subject}.mat"
    raw = matlab_data250hz(raw_path).astype(np.float32, copy=False)
    print(f"{subject}: raw {raw.shape}", flush=True)
    filtered = filterbank(raw)
    np.save(cache, filtered)
    return np.load(cache, mmap_mode="r")


def existing_keys(trials_path: Path) -> set[tuple[str, str, str, int]]:
    if not trials_path.exists():
        return set()
    out = set()
    with trials_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "complete":
                out.add((row["subject"], row["target_set"], row["channel_set"], int(row["window_ms"])))
    return out


def append_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_row(dataset_root: Path, filtered: np.ndarray, spec: GridSpec) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    started = time.perf_counter()
    target_cfg = TARGET_SETS[spec.target_set]
    target_indices = target_cfg["indices"]
    channels = CHANNEL_SETS[spec.channel_set]
    freqs = target_cfg["freqs"]
    targets = len(target_indices)
    time_samples = spec.window_ms // 4
    needed = time_samples + LATENCY + LAG
    if needed > filtered.shape[1]:
        row = {
            "method": "TDCA",
            "subject": spec.subject,
            "target_set": spec.target_set,
            "target_name": target_cfg["name"],
            "channel_set": spec.channel_set,
            "window_ms": spec.window_ms,
            "targets": targets,
            "channels": len(channels),
            "blocks": 18,
            "accuracy": "",
            "itr_bpm": "",
            "seconds": time.perf_counter() - started,
            "status": "unavailable",
            "reason": f"needs {needed} samples; file has {filtered.shape[1]}",
        }
        return row, [], np.zeros((targets, targets), dtype=np.int64)

    data = np.asarray(filtered[np.ix_(channels, np.arange(filtered.shape[1]), target_indices, np.arange(18), np.arange(FB_NUM))], dtype=np.float32)
    p_cond = projection_matrices(time_samples, freqs)
    correct = np.zeros(FB_NUM, dtype=np.int64)
    total = 0
    pred_rows = []
    confusion = np.zeros((targets, targets), dtype=np.int64)
    for cv in range(18):
        train_blocks = [idx for idx in range(18) if idx != cv]
        model = fit_tdca_single(np.asarray(data[:, :, :, train_blocks, :], dtype=np.float32), time_samples, p_cond)
        datatest = data[:, :, :, cv, :]
        for cond in range(targets):
            pred = predict_tdca(np.asarray(datatest[:, :, cond, :], dtype=np.float32), model, time_samples, p_cond)
            final_pred = int(pred[-1])
            correct += pred == cond
            confusion[cond, final_pred] += 1
            total += 1
            pred_rows.append(
                {
                    "method": "TDCA",
                    "subject": spec.subject,
                    "block": cv + 1,
                    "trial_index": total,
                    "target_set": spec.target_set,
                    "target_name": target_cfg["name"],
                    "channel_set": spec.channel_set,
                    "window_ms": spec.window_ms,
                    "time_samples": time_samples,
                    "true": cond + 1,
                    "pred": final_pred + 1,
                    "original_true": int(target_indices[cond]) + 1,
                    "original_pred": int(target_indices[final_pred]) + 1,
                    "correct": int(final_pred == cond),
                }
            )
        print(f"{spec.slug} cv={cv + 1}/18 acc={correct[-1] / total:.6f}", flush=True)

    acc = float(correct[-1] / total)
    row = {
        "method": "TDCA",
        "subject": spec.subject,
        "target_set": spec.target_set,
        "target_name": target_cfg["name"],
        "channel_set": spec.channel_set,
        "window_ms": spec.window_ms,
        "targets": targets,
        "channels": len(channels),
        "blocks": 18,
        "accuracy": acc,
        "itr_bpm": itr_bits_per_minute(targets, acc, spec.window_ms / 1000 + 0.5),
        "seconds": time.perf_counter() - started,
        "status": "complete",
        "reason": "",
    }
    return row, pred_rows, confusion


def run_spec_task(task: dict[str, object]) -> tuple[GridSpec, dict[str, object], list[dict[str, object]], np.ndarray]:
    spec = GridSpec(
        str(task["subject"]),
        str(task["target_set"]),
        str(task["channel_set"]),
        int(task["window_ms"]),
    )
    dataset_root = Path(str(task["dataset_root"]))
    filtered = load_or_create_filtered(dataset_root, spec.subject, bool(task["force_filter"]))
    row, pred_rows, confusion = run_row(dataset_root, filtered, spec)
    return spec, row, pred_rows, confusion


def rebuild_summary(out_dir: Path) -> None:
    trials_path = out_dir / "trials.csv"
    if not trials_path.exists():
        return
    rows = []
    with trials_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] == "complete":
                rows.append(row)
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["target_set"], row["channel_set"], row["window_ms"]), []).append(row)
    out_rows = []
    for (method, target_set, channel_set, window_ms), group in sorted(groups.items()):
        acc = np.asarray([float(r["accuracy"]) for r in group], dtype=np.float64)
        itr = np.asarray([float(r["itr_bpm"]) for r in group], dtype=np.float64)
        out_rows.append(
            {
                "method": method,
                "target_set": target_set,
                "channel_set": channel_set,
                "window_ms": window_ms,
                "subjects": len(group),
                "accuracy": float(acc.mean()),
                "accuracy_sem": float(acc.std(ddof=1) / math.sqrt(len(acc))) if len(acc) > 1 else 0.0,
                "itr_bpm": float(itr.mean()),
                "itr_sem": float(itr.std(ddof=1) / math.sqrt(len(itr))) if len(itr) > 1 else 0.0,
            }
        )
    fields = ["method", "target_set", "channel_set", "window_ms", "subjects", "accuracy", "accuracy_sem", "itr_bpm", "itr_sem"]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)


def consolidate_predictions(out_dir: Path) -> None:
    parts = sorted((out_dir / "prediction_parts").glob("*.csv"))
    if not parts:
        return
    out = out_dir / "predictions.csv"
    first = True
    with out.open("w", newline="", encoding="utf-8") as fout:
        writer = None
        for part in parts:
            with part.open(newline="", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                if first:
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    first = False
                writer.writerows(reader)


def write_config_outputs(
    out_dir: Path,
    spec: GridSpec,
    row: dict[str, object],
    pred_rows: list[dict[str, object]],
    confusion: np.ndarray,
    trial_fields: list[str],
    pred_fields: list[str],
    runtime_fields: list[str],
) -> None:
    append_csv(out_dir / "trials.csv", [row], trial_fields)
    append_csv(
        out_dir / "runtime.csv",
        [
            {
                "subject": spec.subject,
                "target_set": spec.target_set,
                "channel_set": spec.channel_set,
                "window_ms": spec.window_ms,
                "seconds": row["seconds"],
                "status": row["status"],
            }
        ],
        runtime_fields,
    )
    if pred_rows:
        part = out_dir / "prediction_parts" / f"predictions_{spec.slug}.csv"
        part.parent.mkdir(parents=True, exist_ok=True)
        with part.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pred_fields)
            writer.writeheader()
            writer.writerows(pred_rows)
        conf_dir = out_dir / "confusions"
        conf_dir.mkdir(parents=True, exist_ok=True)
        np.save(conf_dir / f"confusion_{spec.slug}.npy", confusion)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--subjects", default="author")
    parser.add_argument("--target-sets", default="40,80,120,160,200")
    parser.add_argument("--channel-sets", default="9,21,32,66")
    parser.add_argument("--windows-ms", default="100,200,300,400,500")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--force-filter", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--consolidate-only", action="store_true")
    args = parser.parse_args()

    if args.workers > 1:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.consolidate_only:
        consolidate_predictions(out_dir)
        rebuild_summary(out_dir)
        return

    subjects = parse_subjects(args.subjects)
    target_sets = parse_csv(args.target_sets)
    channel_sets = parse_csv(args.channel_sets)
    windows = [int(x) for x in parse_csv(args.windows_ms)]
    specs = [GridSpec(s, t, c, w) for s in subjects for t in target_sets for c in channel_sets for w in windows]
    done = existing_keys(out_dir / "trials.csv")
    specs = [spec for spec in specs if spec.key not in done]
    if args.max_rows > 0:
        specs = specs[: args.max_rows]

    manifest = {
        "runner": str(Path(__file__).resolve()),
        "dataset": "ssvep_hd_200target offline release",
        "paper_script": "TDCA_Classification.m",
        "subjects": subjects,
        "target_sets": target_sets,
        "channel_sets": channel_sets,
        "windows_ms": windows,
        "latency_samples": LATENCY,
        "lag": LAG,
        "filter_banks": FB_NUM,
        "workers": args.workers,
        "parallel_unit": "config" if args.workers > 1 else "serial",
        "status": "partial",
        "output_contract": "docs/result_artifact_contract.md",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    trial_fields = [
        "method",
        "subject",
        "target_set",
        "target_name",
        "channel_set",
        "window_ms",
        "targets",
        "channels",
        "blocks",
        "accuracy",
        "itr_bpm",
        "seconds",
        "status",
        "reason",
    ]
    pred_fields = [
        "method",
        "subject",
        "block",
        "trial_index",
        "target_set",
        "target_name",
        "channel_set",
        "window_ms",
        "time_samples",
        "true",
        "pred",
        "original_true",
        "original_pred",
        "correct",
    ]
    runtime_fields = ["subject", "target_set", "channel_set", "window_ms", "seconds", "status"]
    if args.workers <= 1:
        current_subject = None
        filtered = None
        for spec in specs:
            if spec.subject != current_subject:
                current_subject = spec.subject
                filtered = load_or_create_filtered(args.dataset_root, spec.subject, args.force_filter)
            row, pred_rows, confusion = run_row(args.dataset_root, filtered, spec)
            write_config_outputs(out_dir, spec, row, pred_rows, confusion, trial_fields, pred_fields, runtime_fields)
            rebuild_summary(out_dir)
            print(f"done {spec.slug} status={row['status']} acc={row['accuracy']}", flush=True)
    else:
        tasks = [
            {
                "dataset_root": str(args.dataset_root),
                "subject": spec.subject,
                "target_set": spec.target_set,
                "channel_set": spec.channel_set,
                "window_ms": spec.window_ms,
                "force_filter": args.force_filter,
            }
            for spec in specs
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_map = {pool.submit(run_spec_task, task): task for task in tasks}
            for future in as_completed(future_map):
                spec, row, pred_rows, confusion = future.result()
                write_config_outputs(out_dir, spec, row, pred_rows, confusion, trial_fields, pred_fields, runtime_fields)
                rebuild_summary(out_dir)
                print(f"done {spec.slug} status={row['status']} acc={row['accuracy']}", flush=True)

    consolidate_predictions(out_dir)
    rebuild_summary(out_dir)
    manifest["status"] = "complete" if not [spec for spec in [GridSpec(s, t, c, w) for s in subjects for t in target_sets for c in channel_sets for w in windows] if spec.key not in existing_keys(out_dir / "trials.csv")] else "partial"
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
