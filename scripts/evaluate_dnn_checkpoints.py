# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-18
# Last updated: 2026-06-18
# Description: Evaluate saved DNN checkpoints through the Arena data path.
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import cheby1, sosfiltfilt
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import BENCHMARK_CHANNELS_9, DATA_ROOT, RESULT_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import load_subject_raw
from vep_arena.metrics import itr_bits_per_minute


def parse_range(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        elif part:
            values.append(int(part))
    return values


def make_dnn_filter_bank(subbands: int, fs: int) -> list[np.ndarray]:
    return [
        cheby1(N=2, rp=1, Wn=[8 * idx, 90], btype="bandpass", fs=fs, output="sos")
        for idx in range(1, subbands + 1)
    ]


def preprocess_subject_for_dnn(
    data_root: Path,
    subject: int,
    window: float,
    channels: tuple[int, ...],
    spec: BenchmarkSpec,
    subbands: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = load_subject_raw(data_root, subject)
    channel_idx = np.asarray(channels, dtype=np.int64) - 1
    sample_window = spec.sample_slice(window)
    sub = raw[channel_idx, sample_window, :, :]
    filters = make_dnn_filter_bank(subbands, spec.sampling_rate)
    samples = spec.sample_length(window)
    x = np.zeros((spec.classes * spec.blocks, subbands, len(channels), samples), dtype=np.float32)
    y = np.zeros((spec.classes * spec.blocks,), dtype=np.int64)
    block_numbers = np.zeros((spec.classes * spec.blocks,), dtype=np.int64)
    row = 0
    for block in range(spec.blocks):
        for cls in range(spec.classes):
            trial = sub[:, :, cls, block]
            for fb, sos in enumerate(filters):
                x[row, fb] = sosfiltfilt(sos, trial, axis=-1).astype(np.float32)
            y[row] = cls
            block_numbers[row] = block
            row += 1
    return x, y, block_numbers


def load_split_from_arena(
    data_root: Path,
    subjects: list[int],
    test_block: int,
    window: float,
    channels: tuple[int, ...],
    spec: BenchmarkSpec,
    subbands: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_xs: list[np.ndarray] = []
    train_ys: list[np.ndarray] = []
    test_xs: list[np.ndarray] = []
    test_ys: list[np.ndarray] = []
    train_subjects: list[np.ndarray] = []
    test_subjects: list[np.ndarray] = []
    block_idx = test_block - 1
    for subject in subjects:
        x, y, block_numbers = preprocess_subject_for_dnn(data_root, subject, window, channels, spec, subbands)
        test_mask = block_numbers == block_idx
        train_xs.append(x[~test_mask])
        train_ys.append(y[~test_mask])
        test_xs.append(x[test_mask])
        test_ys.append(y[test_mask])
        train_subjects.append(np.full(int((~test_mask).sum()), subject, dtype=np.int64))
        test_subjects.append(np.full(int(test_mask.sum()), subject, dtype=np.int64))
    return (
        np.concatenate(train_xs),
        np.concatenate(train_ys),
        np.concatenate(test_xs),
        np.concatenate(test_ys),
        np.concatenate(train_subjects),
        np.concatenate(test_subjects),
    )


def registry_rows(registry_path: Path, window: float, blocks: list[int]) -> list[dict[str, str]]:
    with registry_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selected = [
        row
        for row in rows
        if round(float(row["signal_length"]), 6) == round(window, 6) and int(row["block"]) in blocks
    ]
    if not selected:
        raise ValueError(f"No checkpoint rows found for window={window:g}, blocks={blocks}")
    return sorted(selected, key=lambda row: int(row["block"]))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sem(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def evaluate_batches(model, x: np.ndarray, batch_size: int, device) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dummy_y = torch.zeros((len(x),), dtype=torch.long)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), dummy_y), batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--dnn-project-root", type=Path, default=Path("D:/ProjData/proj_python/dnn_ssvep_pytorch"))
    parser.add_argument("--registry", type=Path, default=Path("D:/ProjData/proj_python/dnn_ssvep_pytorch/results_clean/model_registry.csv"))
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "dnn_checkpoint_eval")
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--blocks", default="1-6")
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--subbands", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    sys.path.insert(0, str(args.dnn_project_root))
    import torch
    from dnn_ssvep.model import DNNSsvep

    spec = BenchmarkSpec()
    subjects = parse_range(args.subjects)
    blocks = parse_range(args.blocks)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trial_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for row in registry_rows(args.registry, args.window, blocks):
        block = int(row["block"])
        model_path = Path(str(row["model_path"]))
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        split_start = time.perf_counter()
        train_x, train_y, test_x, test_y, train_subjects, test_subjects = load_split_from_arena(
            args.data_root,
            subjects,
            block,
            args.window,
            BENCHMARK_CHANNELS_9,
            spec,
            args.subbands,
        )
        load_seconds = time.perf_counter() - split_start
        model = DNNSsvep(
            channels=len(BENCHMARK_CHANNELS_9),
            samples=spec.sample_length(args.window),
            subbands=args.subbands,
            classes=spec.classes,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        infer_start = time.perf_counter()
        pred = evaluate_batches(model, test_x, args.batch_size, device)
        infer_seconds = time.perf_counter() - infer_start
        acc = accuracy_score(test_y, pred)
        itr = itr_bits_per_minute(float(acc), spec.classes, args.window + spec.cue_seconds)
        trial_rows.append(
            {
                "method": "DNN-global-pt",
                "window": args.window,
                "subject": "__all__",
                "block": block,
                "accuracy": float(acc),
                "itr": float(itr),
                "samples": int(len(test_y)),
                "seconds": load_seconds + infer_seconds,
                "model_path": str(model_path),
            }
        )
        runtime_rows.extend(
            [
                {"method": "DNN-global-pt", "window": args.window, "block": block, "stage": "arena_dnn_preprocess", "seconds": load_seconds},
                {"method": "DNN-global-pt", "window": args.window, "block": block, "stage": "checkpoint_inference", "seconds": infer_seconds},
            ]
        )
        for subject in subjects:
            mask = test_subjects == subject
            subject_acc = accuracy_score(test_y[mask], pred[mask])
            subject_itr = itr_bits_per_minute(float(subject_acc), spec.classes, args.window + spec.cue_seconds)
            trial_rows.append(
                {
                    "method": "DNN-global-pt",
                    "window": args.window,
                    "subject": subject,
                    "block": block,
                    "accuracy": float(subject_acc),
                    "itr": float(subject_itr),
                    "samples": int(mask.sum()),
                    "seconds": 0.0,
                    "model_path": str(model_path),
                }
            )
        for true_label, pred_label, subject in zip(test_y, pred, test_subjects):
            pred_rows.append(
                {
                    "method": "DNN-global-pt",
                    "window": args.window,
                    "subject": int(subject),
                    "block": block,
                    "true": int(true_label),
                    "pred": int(pred_label),
                }
            )
        print(f"DNN-global-pt w={args.window:g} block={block} acc={acc:.4f}", flush=True)

    subject_trials = [row for row in trial_rows if row["subject"] != "__all__"]
    block_summary = [row for row in trial_rows if row["subject"] == "__all__"]
    accuracies = [float(row["accuracy"]) for row in subject_trials]
    itrs = [float(row["itr"]) for row in subject_trials]
    summary_rows = [
        {
            "method": "DNN-global-pt",
            "window": args.window,
            "accuracy": float(np.mean(accuracies)),
            "accuracy_sem": sem(accuracies),
            "itr": float(np.mean(itrs)),
            "itr_sem": sem(itrs),
            "subjects": len({int(row["subject"]) for row in subject_trials}),
            "samples": int(sum(int(row["samples"]) for row in subject_trials)),
        }
    ]
    write_csv(
        args.output_dir / "trials.csv",
        trial_rows,
        ["method", "window", "subject", "block", "accuracy", "itr", "samples", "seconds", "model_path"],
    )
    write_csv(args.output_dir / "summary.csv", summary_rows, ["method", "window", "accuracy", "accuracy_sem", "itr", "itr_sem", "subjects", "samples"])
    write_csv(
        args.output_dir / "block.csv",
        block_summary,
        ["method", "window", "subject", "block", "accuracy", "itr", "samples", "seconds", "model_path"],
    )
    write_csv(args.output_dir / "predictions.csv", pred_rows, ["method", "window", "subject", "block", "true", "pred"])
    write_csv(args.output_dir / "runtime.csv", runtime_rows, ["method", "window", "block", "stage", "seconds"])
    true_values = [int(row["true"]) for row in pred_rows]
    pred_values = [int(row["pred"]) for row in pred_rows]
    np.save(args.output_dir / "confusion_dnn_global_pt.npy", confusion_matrix(true_values, pred_values, labels=np.arange(spec.classes)))
    manifest = {
        "task_name": args.output_dir.name,
        "method": "DNN-global-pt",
        "status": "complete",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "Arena-side DNN preprocessing plus saved global checkpoint inference.",
        "window": args.window,
        "subjects": subjects,
        "blocks": blocks,
        "device": str(device),
        "torch": torch.__version__,
        "dnn_project_root": str(args.dnn_project_root),
        "registry": str(args.registry),
        "preprocessing": {
            "crop_start_seconds": spec.cue_seconds + spec.latency_seconds,
            "subbands": args.subbands,
            "filterbank": "Chebyshev-I order-2 bandpass filters, [8*i, 90] Hz, scipy sosfiltfilt.",
            "notch": "None.",
            "channels": list(BENCHMARK_CHANNELS_9),
        },
        "limitations": [
            "Saved checkpoints are global stage-1 models only.",
            "Historical imported DNN scores include subject-specific fine-tuning that was not saved in these .pt files.",
        ],
        "seconds": time.perf_counter() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
