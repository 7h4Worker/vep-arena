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
import scipy.io as sio
from scipy.linalg import eig, qr
from scipy.signal import cheb1ord, cheby1, filtfilt


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(r"D:\ProjData\datasets\ssvep_hd_200target")
RAW_REL = Path(r"raw\code_data\code&data")
FS = 250
FB_NUM = 5
LATENCY = round(140 / 4)
LAG = 5
CHANNEL66 = np.arange(66)
CONSEQ_200 = np.arange(200)
BASE_FREQS = np.asarray(
    list(range(8, 16))
    + [x + 0.2 for x in range(8, 16)]
    + [x + 0.4 for x in range(8, 16)]
    + [x + 0.6 for x in range(8, 16)]
    + [x + 0.8 for x in range(8, 16)],
    dtype=np.float64,
)
FREQS_200 = np.repeat(BASE_FREQS, 5)


@dataclass(frozen=True)
class RunSpec:
    dataset_root: Path
    subject: str
    window_ms: int
    out_dir: Path
    cache_dir: Path
    force_filter: bool


def matlab_data250hz(path: Path) -> np.ndarray:
    try:
        with h5py.File(path, "r") as f:
            arr = np.asarray(f["data250Hz"], dtype=np.float32)
        return np.transpose(arr, (3, 2, 1, 0))
    except OSError:
        return np.asarray(sio.loadmat(path)["data250Hz"], dtype=np.float32)


def bp40_fb2(x: np.ndarray, n_fb: int) -> np.ndarray:
    nyq = FS / 2
    fls1 = [6, 14, 22, 30, 38]
    fls2 = [4, 12, 20, 28, 36]
    fhs1 = [90, 90, 90, 90, 90]
    fhs2 = [100, 100, 100, 100, 100]
    wp = [fls1[n_fb] / nyq, fhs1[n_fb] / nyq]
    ws = [fls2[n_fb] / nyq, fhs2[n_fb] / nyq]
    order, wn = cheb1ord(wp, ws, 3, 40)
    b, a = cheby1(order, 0.5, wn, btype="bandpass")
    return filtfilt(b, a, x, axis=1).astype(np.float32, copy=False)


def filterbank(raw: np.ndarray) -> np.ndarray:
    channels, samples, targets, blocks = raw.shape
    out = np.empty((channels, samples, targets, blocks, FB_NUM), dtype=np.float32)
    for fb in range(FB_NUM):
        print(f"filter fb={fb + 1}/{FB_NUM}", flush=True)
        out[..., fb] = bp40_fb2(raw, fb)
    return out


def load_or_filter_subject(spec: RunSpec) -> np.ndarray:
    spec.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = spec.cache_dir / f"filtered_{spec.subject}_200target_66ch_18blocks_float32.npy"
    if cache.exists() and not spec.force_filter:
        print(f"{spec.subject}: loading filter cache {cache}", flush=True)
        return np.load(cache, mmap_mode="r")

    data_path = spec.dataset_root / RAW_REL / "data" / "offline" / f"{spec.subject}.mat"
    raw = matlab_data250hz(data_path)[CHANNEL66][:, :, CONSEQ_200, :18]
    print(f"{spec.subject}: loaded raw {raw.shape}", flush=True)
    filtered = filterbank(raw)
    np.save(cache, filtered)
    print(f"{spec.subject}: saved filter cache {cache}", flush=True)
    return np.load(cache, mmap_mode="r")


def projection_matrices(samples: int, freqs: np.ndarray) -> np.ndarray:
    n = np.arange(1, samples + 1, dtype=np.float64) / FS
    mats = np.empty((len(freqs), samples, samples), dtype=np.float32)
    for i, freq in enumerate(freqs):
        cols = []
        for harmonic in range(1, 6):
            cols.append(np.sin(2 * np.pi * harmonic * freq * n))
            cols.append(np.cos(2 * np.pi * harmonic * freq * n))
        y = np.stack(cols, axis=1)
        y -= y.mean(axis=0, keepdims=True)
        q, _ = qr(y, mode="economic")
        mats[i] = (q @ q.T).astype(np.float32)
    return mats


def lagged_block_view(bpdata_fb: np.ndarray, time_samples: int) -> np.ndarray:
    n_chan, _, n_cond, n_block = bpdata_fb.shape
    out = np.empty((n_chan * LAG, time_samples, n_cond, n_block), dtype=np.float32)
    for lag_idx in range(1, LAG + 1):
        rows = slice((lag_idx - 1) * n_chan, lag_idx * n_chan)
        start = LATENCY + lag_idx - 1
        stop = time_samples + LATENCY + lag_idx - 1
        out[rows] = bpdata_fb[:, start:stop, :, :]
    return out


def sorted_eigvec(sb: np.ndarray, sw: np.ndarray) -> np.ndarray:
    vals, vecs = eig(sb, sw + np.eye(sw.shape[0]) * 1e-7)
    order = np.argsort(np.real(vals))[::-1]
    return np.real(vecs[:, order]).astype(np.float32)


def fit_tdca_single(
    data: np.ndarray,
    time_samples: int,
    p_cond: np.ndarray,
) -> dict[str, np.ndarray]:
    n_chan, _, n_cond, n_block, n_band = data.shape
    feat_dim = n_chan * LAG
    bpdata = data[:, : time_samples + LATENCY + LAG, :, :, :]
    templates = np.empty((feat_dim, time_samples * 2, n_cond, n_band), dtype=np.float32)
    filters = np.empty((feat_dim, feat_dim, n_band), dtype=np.float32)

    for fb in range(n_band):
        bpdatah = lagged_block_view(bpdata[..., fb], time_samples)
        train_templates = np.empty((feat_dim, time_samples * 2, n_cond), dtype=np.float32)
        trca_xm = np.empty_like(train_templates)
        sw = np.zeros((feat_dim, feat_dim), dtype=np.float64)

        for cond in range(n_cond):
            trial = bpdatah[:, :, cond, :]
            projected = np.einsum("dtb,ts->dsb", trial, p_cond[cond], optimize=True)
            hp = np.concatenate([trial, projected], axis=1)
            tmpl = hp.mean(axis=2)
            train_templates[:, :, cond] = tmpl
            trca_xm[:, :, cond] = tmpl - tmpl.mean(axis=1, keepdims=True)

            x = hp - hp.mean(axis=1, keepdims=True)
            x = x - x.mean(axis=2, keepdims=True)
            x2 = x.reshape(feat_dim, -1, order="F").astype(np.float64, copy=False)
            sw += (x2 @ x2.T) / (n_block * n_cond)

        trca_xma = trca_xm.mean(axis=2, keepdims=True)
        hb = (trca_xm - trca_xma).reshape(feat_dim, -1, order="F").astype(np.float64, copy=False) / math.sqrt(n_cond)
        sb = hb @ hb.T
        templates[..., fb] = train_templates
        filters[..., fb] = sorted_eigvec(sb, sw)

    return {"templates": templates, "filters": filters}


def predict_tdca(epoch: np.ndarray, model: dict[str, np.ndarray], time_samples: int, p_cond: np.ndarray) -> np.ndarray:
    n_chan, _, n_band = epoch.shape
    target_num = p_cond.shape[0]
    weights = np.asarray([(i + 1) ** (-1.25) + 0.25 for i in range(10)], dtype=np.float32)
    rr = np.zeros((n_band, target_num), dtype=np.float32)
    templates = model["templates"]
    filters = model["filters"]

    for fb in range(n_band):
        test_parts = []
        bpdata = epoch[:, : time_samples + LATENCY, fb]
        for lag_idx in range(1, LAG + 1):
            start = LATENCY + lag_idx - 1
            stop = time_samples + LATENCY
            part = bpdata[:, start:stop]
            if part.shape[1] < time_samples:
                part = np.pad(part, ((0, 0), (0, time_samples - part.shape[1])))
            test_parts.append(part)
        testdatah = np.concatenate(test_parts, axis=0)
        xh = testdatah.T
        filt = filters[:, :n_chan, fb]
        base_u = xh @ filt
        projected_x = np.einsum("cts,td->csd", p_cond, xh, optimize=True)
        projected_u = np.einsum("csd,dr->csr", projected_x, filt, optimize=True)
        u1 = np.concatenate(
            [np.broadcast_to(base_u, (target_num, base_u.shape[0], base_u.shape[1])), projected_u],
            axis=1,
        )
        tmpl_t = templates[..., fb].transpose(2, 1, 0)
        v1 = np.einsum("ctd,dr->ctr", tmpl_t, filt, optimize=True)
        numerator = np.sum(u1 * v1, axis=(1, 2))
        denominator = np.sqrt(np.sum(u1 * u1, axis=(1, 2)) * np.sum(v1 * v1, axis=(1, 2)))
        rr[fb] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)

    preds = np.zeros(n_band, dtype=np.int64)
    for fb in range(n_band):
        scores = weights[: fb + 1] @ (np.sign(rr[: fb + 1]) * np.abs(rr[: fb + 1]) ** 2)
        preds[fb] = int(np.argmax(scores))
    return preds


def itr_bits_per_minute(classes: int, accuracy: float, trial_seconds: float) -> float:
    if accuracy <= 0:
        bits = math.log2(classes - 1)
    elif accuracy >= 1:
        bits = math.log2(classes)
    else:
        bits = math.log2(classes) + accuracy * math.log2(accuracy) + (1 - accuracy) * math.log2((1 - accuracy) / (classes - 1))
    return bits * 60 / trial_seconds


def unavailable_row(spec: RunSpec, reason: str) -> dict[str, object]:
    return {
        "subject": spec.subject,
        "targets": 200,
        "channels": 66,
        "filter_banks": FB_NUM,
        "blocks": 18,
        "window_ms": spec.window_ms,
        "accuracy": "",
        "itr_bpm": "",
        "seconds": 0.0,
        "status": "unavailable",
        "reason": reason,
    }


def run_subject_window(spec: RunSpec) -> dict[str, object]:
    started = time.perf_counter()
    data_path = spec.dataset_root / RAW_REL / "data" / "offline" / f"{spec.subject}.mat"
    raw_shape = matlab_data250hz(data_path).shape
    time_samples = spec.window_ms // 4
    required = time_samples + LATENCY + LAG
    if required > raw_shape[1]:
        return unavailable_row(
            spec,
            f"requires {required} samples (window {time_samples} + latency {LATENCY} + lag {LAG}), file has {raw_shape[1]}",
        )

    filtered = load_or_filter_subject(spec)
    p_cond = projection_matrices(time_samples, FREQS_200)
    correct = np.zeros(FB_NUM, dtype=np.int64)
    total = 0
    for cv in range(18):
        train_blocks = [idx for idx in range(18) if idx != cv]
        datatrain = np.asarray(filtered[:, :, :, train_blocks, :], dtype=np.float32)
        datatest = filtered[:, :, :, cv, :]
        model = fit_tdca_single(datatrain, time_samples, p_cond)
        for cond in range(200):
            pred = predict_tdca(np.asarray(datatest[:, :, cond, :], dtype=np.float32), model, time_samples, p_cond)
            correct += pred == cond
            total += 1
        print(f"{spec.subject} {spec.window_ms}ms cv={cv + 1}/18 fb5_acc={correct[-1] / total:.6f}", flush=True)

    acc = float(correct[-1] / total)
    return {
        "subject": spec.subject,
        "targets": 200,
        "channels": 66,
        "filter_banks": FB_NUM,
        "blocks": 18,
        "window_ms": spec.window_ms,
        "accuracy": acc,
        "itr_bpm": itr_bits_per_minute(200, acc, spec.window_ms / 1000 + 0.5),
        "seconds": time.perf_counter() - started,
        "status": "complete",
        "reason": "",
    }


def parse_subjects(text: str) -> list[str]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(item if item.upper().startswith("S") else f"S{int(item)}")
    return out


def parse_windows(text: str) -> list[int]:
    return [int(float(x.strip())) for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subject", "targets", "channels", "filter_banks", "blocks", "window_ms", "accuracy", "itr_bpm", "seconds", "status", "reason"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--subjects", default="S1,S2,S3")
    parser.add_argument("--windows-ms", default="500,1000")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force-filter", action="store_true")
    parser.add_argument("--task-name", default="ssvep_hd_200target_tdca_sample")
    args = parser.parse_args()

    task_dir = PROJECT / "tasks" / args.task_name
    out_dir = task_dir / "results"
    cache_dir = args.dataset_root / "derivatives" / "tdca_sample" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        RunSpec(args.dataset_root, subject, window, out_dir, cache_dir, args.force_filter)
        for subject in parse_subjects(args.subjects)
        for window in parse_windows(args.windows_ms)
    ]

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    if args.workers <= 1:
        for spec in specs:
            rows.append(run_subject_window(spec))
            write_csv(out_dir / "summary.csv", rows)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_map = {pool.submit(run_subject_window, spec): spec for spec in specs}
            for future in as_completed(future_map):
                spec = future_map[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = unavailable_row(spec, f"error: {exc}")
                rows.append(row)
                rows.sort(key=lambda r: (str(r["subject"]), int(r["window_ms"])))
                write_csv(out_dir / "summary.csv", rows)
                print(f"done {row['subject']} {row['window_ms']}ms status={row['status']} acc={row['accuracy']}", flush=True)

    manifest = {
        "task_name": args.task_name,
        "dataset": "ssvep_hd_200target",
        "protocol": "offline 200-target, 66-channel, 18-fold leave-one-block-out TDCA",
        "subjects": parse_subjects(args.subjects),
        "windows_ms": parse_windows(args.windows_ms),
        "workers": args.workers,
        "cache_dir": str(cache_dir),
        "result_csv": str(out_dir / "summary.csv"),
        "seconds": time.perf_counter() - started,
        "notes": [
            "1000 ms is marked unavailable for the released offline files because each trial has 185 samples.",
            "500 ms follows the package TDCA settings: latency=35 samples, lag=5, 5 filter banks, 200 targets, 66 channels.",
            "The implementation uses float32 filter caches and per-condition covariance accumulation to avoid MATLAB-style repmat memory spikes.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_csv(out_dir / "summary.csv", rows)
    print(out_dir / "summary.csv", flush=True)


if __name__ == "__main__":
    main()
