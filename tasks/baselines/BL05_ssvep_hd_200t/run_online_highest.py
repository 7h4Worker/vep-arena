from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np

from run import (
    BASE_FREQS,
    CHANNEL66,
    DEFAULT_DATASET,
    FB_NUM,
    FS,
    LATENCY,
    LAG,
    RAW_REL,
    bp40_fb2,
    fit_tdca_single,
    projection_matrices,
    predict_tdca,
)


def load_online(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        arr = np.asarray(f["data250Hz"], dtype=np.float32)
    return np.transpose(arr, (3, 2, 1, 0))


def filterbank_online(raw: np.ndarray) -> np.ndarray:
    channels, samples, targets, blocks = raw.shape
    out = np.empty((channels, samples, targets, blocks, FB_NUM), dtype=np.float32)
    for fb in range(FB_NUM):
        print(f"filter fb={fb + 1}/{FB_NUM}", flush=True)
        out[..., fb] = bp40_fb2(raw, fb)
    return out


def itr_bits_per_minute(classes: int, accuracy: float, trial_seconds: float) -> float:
    if accuracy <= 0:
        bits = math.log2(classes - 1)
    elif accuracy >= 1:
        bits = math.log2(classes)
    else:
        bits = math.log2(classes) + accuracy * math.log2(accuracy) + (1 - accuracy) * math.log2((1 - accuracy) / (classes - 1))
    return bits * 60 / trial_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--window-ms", type=int, default=250)
    parser.add_argument("--force-filter", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.dataset_root / "derivatives" / "tdca_sample" / "online_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_cache = cache_dir / f"filtered_online_training_{args.subject}_160target_66ch_float32.npy"
    test_cache = cache_dir / f"filtered_online_testing_{args.subject}_160target_66ch_float32.npy"
    data_root = args.dataset_root / RAW_REL / "data" / "online"

    if train_cache.exists() and test_cache.exists() and not args.force_filter:
        train = np.load(train_cache, mmap_mode="r")
        test = np.load(test_cache, mmap_mode="r")
        print(f"loaded caches {train.shape} {test.shape}", flush=True)
    else:
        train_raw = load_online(data_root / "training" / f"{args.subject}.mat")[CHANNEL66]
        test_raw = load_online(data_root / "testing" / f"{args.subject}.mat")[CHANNEL66]
        print(f"raw train={train_raw.shape} test={test_raw.shape}", flush=True)
        train = filterbank_online(train_raw)
        test = filterbank_online(test_raw)
        np.save(train_cache, train)
        np.save(test_cache, test)
        print(f"saved caches {train_cache} {test_cache}", flush=True)
        train = np.load(train_cache, mmap_mode="r")
        test = np.load(test_cache, mmap_mode="r")

    # The 160-target online setting excludes every fifth spatial location:
    # right, down, left, and up; no center.
    target_num = train.shape[2]
    if target_num != 160:
        raise ValueError(f"Expected 160 targets, got {target_num}")
    freqs = np.repeat(BASE_FREQS, 4)

    time_samples = int(math.floor(args.window_ms / 1000 * FS + 0.5))
    needed = time_samples + LATENCY + LAG
    if needed > train.shape[1] or needed > test.shape[1]:
        raise ValueError(f"{args.window_ms} ms needs {needed} samples, train/test have {train.shape[1]}/{test.shape[1]}")

    p_cond = projection_matrices(time_samples, freqs)
    model = fit_tdca_single(np.asarray(train, dtype=np.float32), time_samples, p_cond)
    correct = np.zeros(FB_NUM, dtype=np.int64)
    total = 0
    predictions = []
    for block in range(test.shape[3]):
        for cond in range(target_num):
            pred = predict_tdca(np.asarray(test[:, :, cond, block, :], dtype=np.float32), model, time_samples, p_cond)
            correct += pred == cond
            total += 1
            predictions.append({"block": block + 1, "true": cond + 1, "pred": int(pred[-1]) + 1})
        print(f"test_block={block + 1}/{test.shape[3]} fb5_acc={correct[-1] / total:.6f}", flush=True)

    acc = float(correct[-1] / total)
    itr = itr_bits_per_minute(target_num, acc, args.window_ms / 1000 + 0.5)
    seconds = time.perf_counter() - started
    row = {
        "subject": args.subject,
        "split": "online_training_to_online_testing",
        "targets": target_num,
        "channels": 66,
        "filter_banks": FB_NUM,
        "train_blocks": int(train.shape[3]),
        "test_blocks": int(test.shape[3]),
        "window_ms": args.window_ms,
        "time_samples": time_samples,
        "accuracy": acc,
        "itr_bpm": itr,
        "seconds": seconds,
    }

    csv_path = out_dir / f"online_highest_{args.subject}_160target_66ch_w{args.window_ms}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    pred_path = out_dir / f"online_highest_{args.subject}_160target_66ch_w{args.window_ms}_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["block", "true", "pred"])
        writer.writeheader()
        writer.writerows(predictions)
    manifest = {
        "setting": "paper highest online ITR setting candidate",
        "subject": args.subject,
        "targets": 160,
        "target_set": "ConSeq4: right, down, left, up; excludes every fifth center target",
        "window_ms": args.window_ms,
        "train_file": str(data_root / "training" / f"{args.subject}.mat"),
        "test_file": str(data_root / "testing" / f"{args.subject}.mat"),
        "result_csv": str(csv_path),
        "prediction_csv": str(pred_path),
        "row": row,
    }
    (out_dir / f"online_highest_{args.subject}_160target_66ch_w{args.window_ms}_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(row, flush=True)
    print(csv_path, flush=True)


if __name__ == "__main__":
    main()
