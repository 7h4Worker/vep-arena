from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from run import DEFAULT_DATASET, FB_NUM, LATENCY, itr_bits_per_minute
from run_offline_tdca_grid import (
    AUTHOR_SUBJECTS,
    CHANNEL_SETS,
    TARGET_SETS,
    load_or_create_filtered,
    parse_csv,
    parse_subjects,
)
from vep_arena.methods.traditional import TRCA


TASK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TASK_DIR / "results" / "offline_trca_grid"
SUPPORTED_METHODS = {"TRCA": False, "ETRCA": True}


@dataclass(frozen=True)
class GridSpec:
    method: str
    subject: str
    target_set: str
    channel_set: str
    window_ms: int

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return self.method, self.subject, self.target_set, self.channel_set, self.window_ms

    @property
    def slug(self) -> str:
        return f"{self.method.lower()}_{self.subject}_{self.target_set}target_{self.channel_set}ch_w{self.window_ms}"


def method_names(text: str) -> list[str]:
    methods = [x.strip().upper() for x in text.split(",") if x.strip()]
    unknown = [name for name in methods if name not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"Unsupported method(s): {', '.join(unknown)}")
    return methods


def existing_keys(trials_path: Path) -> set[tuple[str, str, str, str, int]]:
    if not trials_path.exists():
        return set()
    out = set()
    with trials_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "complete":
                out.add((row["method"], row["subject"], row["target_set"], row["channel_set"], int(row["window_ms"])))
    return out


def append_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def prepare_trca_epochs(filtered: np.ndarray, spec: GridSpec) -> tuple[np.ndarray | None, dict[str, object] | None]:
    target_cfg = TARGET_SETS[spec.target_set]
    target_indices = target_cfg["indices"]
    channels = CHANNEL_SETS[spec.channel_set]
    targets = len(target_indices)
    time_samples = spec.window_ms // 4
    needed = time_samples + LATENCY
    if needed > filtered.shape[1]:
        return None, {
            "method": spec.method,
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
            "seconds": 0.0,
            "status": "unavailable",
            "reason": f"needs {needed} samples; file has {filtered.shape[1]}",
        }

    data = np.asarray(
        filtered[
            np.ix_(
                channels,
                np.arange(LATENCY, LATENCY + time_samples),
                target_indices,
                np.arange(18),
                np.arange(FB_NUM),
            )
        ],
        dtype=np.float32,
    )
    return np.transpose(data, (2, 3, 4, 0, 1)), None


def run_row(filtered: np.ndarray, spec: GridSpec) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    started = time.perf_counter()
    target_cfg = TARGET_SETS[spec.target_set]
    target_indices = target_cfg["indices"]
    channels = CHANNEL_SETS[spec.channel_set]
    targets = len(target_indices)
    labels = np.arange(targets, dtype=np.int64)
    epochs, unavailable = prepare_trca_epochs(filtered, spec)
    if unavailable is not None:
        unavailable["seconds"] = time.perf_counter() - started
        return unavailable, [], np.zeros((targets, targets), dtype=np.int64)
    assert epochs is not None

    pred_rows: list[dict[str, object]] = []
    confusion = np.zeros((targets, targets), dtype=np.int64)
    correct = 0
    total = 0
    ensemble = SUPPORTED_METHODS[spec.method]

    for cv in range(18):
        train_blocks = [idx for idx in range(18) if idx != cv]
        train_x = epochs[:, train_blocks].reshape(-1, epochs.shape[2], epochs.shape[3], epochs.shape[4])
        train_y = np.repeat(labels, len(train_blocks))
        test_x = epochs[:, cv]
        model = TRCA(n_fbs=FB_NUM, ensemble=ensemble)
        model.fit(train_x, train_y)
        pred, _scores = model.predict(test_x)
        for cond, final_pred in enumerate(pred):
            final_pred = int(final_pred)
            correct += int(final_pred == cond)
            confusion[cond, final_pred] += 1
            total += 1
            pred_rows.append(
                {
                    "method": spec.method,
                    "subject": spec.subject,
                    "block": cv + 1,
                    "trial_index": total,
                    "target_set": spec.target_set,
                    "target_name": target_cfg["name"],
                    "channel_set": spec.channel_set,
                    "window_ms": spec.window_ms,
                    "time_samples": spec.window_ms // 4,
                    "true": cond + 1,
                    "pred": final_pred + 1,
                    "original_true": int(target_indices[cond]) + 1,
                    "original_pred": int(target_indices[final_pred]) + 1,
                    "correct": int(final_pred == cond),
                }
            )
        print(f"{spec.slug} cv={cv + 1}/18 acc={correct / total:.6f}", flush=True)

    acc = float(correct / total)
    row = {
        "method": spec.method,
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
        str(task["method"]),
        str(task["subject"]),
        str(task["target_set"]),
        str(task["channel_set"]),
        int(task["window_ms"]),
    )
    dataset_root = Path(str(task["dataset_root"]))
    filtered = load_or_create_filtered(dataset_root, spec.subject, bool(task["force_filter"]))
    row, pred_rows, confusion = run_row(filtered, spec)
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
                "method": spec.method,
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


def all_specs(subjects: list[str], target_sets: list[str], channel_sets: list[str], windows: list[int], methods: list[str]) -> list[GridSpec]:
    return [GridSpec(m, s, t, c, w) for m in methods for s in subjects for t in target_sets for c in channel_sets for w in windows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--subjects", default="author")
    parser.add_argument("--target-sets", default="40,80,120,160,200")
    parser.add_argument("--channel-sets", default="9,21,32,66")
    parser.add_argument("--windows-ms", default="100,200,300,400,500")
    parser.add_argument("--methods", default="TRCA")
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
    methods = method_names(args.methods)
    specs_all = all_specs(subjects, target_sets, channel_sets, windows, methods)
    done = existing_keys(out_dir / "trials.csv")
    specs = [spec for spec in specs_all if spec.key not in done]
    if args.max_rows > 0:
        specs = specs[: args.max_rows]

    manifest = {
        "runner": str(Path(__file__).resolve()),
        "dataset": "ssvep_hd_200target offline release",
        "methods": methods,
        "subjects": subjects,
        "target_sets": target_sets,
        "channel_sets": channel_sets,
        "windows_ms": windows,
        "latency_samples": LATENCY,
        "filter_banks": FB_NUM,
        "workers": args.workers,
        "parallel_unit": "config" if args.workers > 1 else "serial",
        "status": "partial",
        "notes": [
            "TRCA/ETRCA uses the same filtered cache as the validated TDCA task.",
            "Epochs are cropped from the 140 ms latency-corrected onset, without TDCA lag/projection augmentation.",
        ],
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
    runtime_fields = ["method", "subject", "target_set", "channel_set", "window_ms", "seconds", "status"]

    if args.workers <= 1:
        current_subject = None
        filtered = None
        for spec in specs:
            if spec.subject != current_subject:
                current_subject = spec.subject
                filtered = load_or_create_filtered(args.dataset_root, spec.subject, args.force_filter)
            assert filtered is not None
            row, pred_rows, confusion = run_row(filtered, spec)
            write_config_outputs(out_dir, spec, row, pred_rows, confusion, trial_fields, pred_fields, runtime_fields)
            rebuild_summary(out_dir)
            print(f"done {spec.slug} status={row['status']} acc={row['accuracy']}", flush=True)
    else:
        tasks = [
            {
                "dataset_root": str(args.dataset_root),
                "method": spec.method,
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
    remaining = [spec for spec in specs_all if spec.key not in existing_keys(out_dir / "trials.csv")]
    manifest["status"] = "complete" if not remaining else "partial"
    manifest["remaining_rows"] = len(remaining)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
