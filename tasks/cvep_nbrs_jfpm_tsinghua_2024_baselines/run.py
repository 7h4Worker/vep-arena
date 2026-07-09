from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import os
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from vep_arena.metrics import itr_bits_per_minute, sem
from vep_arena.methods.multistimulus import MSTRCA
from vep_arena.methods.trca_core import corr_rows, trca_filter


TASK = Path(__file__).resolve().parent
DEFAULT_DATASET = Path(r"D:\ProjData\datasets\cvep_nbrs_jfpm_tsinghua_2024")
PREPROCESSING_VERSION = "paper_iirnotch50_cheby1_filterbank_latency_crop_v2"
FS = 250
DISPLAY_FS = 120
CLASSES = 40
BLOCKS = 3
CHANNELS = 59
EPOCH_SAMPLES = 1500
STIM_START = FS
GAZE_SHIFT_SECONDS = 0.5
CHANNEL_PRESETS = {
    "all59": list(range(59)),
    "posterior17": list(range(42, 59)),
    "po_o10": list(range(49, 59)),
    "occipital9": list(range(50, 59)),
    "o3": [56, 57, 58],
}

PARADIGM_CONFIG = {
    "NBRS-15": {
        "n_bands": 3,
        "lowcuts": [10.0, 25.0, 40.0],
        "highcut": 80.0,
        "weight_a": 3.0,
        "weight_b": 0.5,
        "fbcca_latency": 0.12,
        "trca_latency": 0.14,
        "code_file": "NBRS-15.mat",
        "latency_points": [0, 2],
        "reference": "code",
    },
    "NBRS-8": {
        "n_bands": 3,
        "lowcuts": [6.0, 14.0, 22.0],
        "highcut": 50.0,
        "weight_a": 2.5,
        "weight_b": 0.25,
        "fbcca_latency": 0.12,
        "trca_latency": 0.14,
        "code_file": "NBRS-8.mat",
        "latency_points": [0, 1, 2, 3],
        "reference": "code",
    },
    "JFPM-8": {
        "n_bands": 6,
        "lowcuts": [6.0, 14.0, 22.0, 30.0, 38.0, 46.0],
        "highcut": 90.0,
        "weight_a": 1.25,
        "weight_b": 0.25,
        "fbcca_latency": 0.14,
        "trca_latency": 0.14,
        "harmonics": 5,
        "reference": "sinusoid",
    },
}


@dataclass(frozen=True)
class Unit:
    subject: int
    paradigm: str

    @property
    def slug(self) -> str:
        return f"S{self.subject:03d}_{self.paradigm}"


def parse_range(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            values.extend(range(int(lo), int(hi) + 1))
        else:
            values.append(int(part))
    return values


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_windows(text: str) -> list[float]:
    if ":" in text:
        start, step, stop = [float(x) for x in text.split(":")]
        n = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 10) for i in range(n)]
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def paradigm_weights(paradigm: str) -> np.ndarray:
    cfg = PARADIGM_CONFIG[paradigm]
    n = int(cfg["n_bands"])
    a = float(cfg["weight_a"])
    b = float(cfg["weight_b"])
    return np.asarray([(idx + 1) ** (-a) + b for idx in range(n)], dtype=np.float64)


def find_batch_zip(dataset: Path, subject: int) -> Path:
    article = dataset / "raw" / "article_24864243_v3"
    for path in sorted(article.glob("S*-S*.zip")):
        name = path.stem
        left, right = name.split("-")
        lo = int(left[1:])
        hi = int(right[1:])
        if lo <= subject <= hi:
            return path
    raise FileNotFoundError(f"Subject S{subject} batch zip not found under {article}")


def load_subject_paradigm(dataset: Path, subject: int, paradigm: str) -> np.ndarray:
    batch = find_batch_zip(dataset, subject)
    member = f"S{subject}/{paradigm}/data.mat"
    with zipfile.ZipFile(batch) as zf:
        with zf.open(member) as f:
            mat = loadmat(io.BytesIO(f.read()))
    data = np.asarray(mat["data"], dtype=np.float32)
    if data.shape != (CHANNELS, EPOCH_SAMPLES, BLOCKS, CLASSES):
        raise ValueError(f"Unexpected data shape for S{subject} {paradigm}: {data.shape}")
    return data


def load_code(dataset: Path, paradigm: str) -> np.ndarray:
    cfg = PARADIGM_CONFIG[paradigm]
    path = dataset / "metadata" / "supplementary_extracted" / str(cfg["code_file"])
    code = np.asarray(loadmat(path)["code"], dtype=np.float64)
    if code.shape[0] != CLASSES:
        raise ValueError(f"Unexpected code shape for {paradigm}: {code.shape}")
    return code


def notch_50hz(x: np.ndarray) -> np.ndarray:
    b, a = signal.iirnotch(50, 35, fs=FS)
    return signal.filtfilt(b, a, x, axis=-1, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1))


def cheby_bandpass(x: np.ndarray, low: float, high: float) -> np.ndarray:
    nyq = FS / 2
    wp = [low / nyq, high / nyq]
    ws = [max(0.5, low - 2.0) / nyq, min(nyq - 1.0, high + 10.0) / nyq]
    gstop = 40
    while gstop >= 20:
        try:
            order, wn = signal.cheb1ord(wp, ws, 3, gstop)
            sos = signal.cheby1(order, 0.5, wn, btype="bandpass", output="sos")
            return signal.sosfiltfilt(sos, x, axis=-1)
        except ValueError:
            gstop -= 1
    raise ValueError(f"Bandpass failed for {low}-{high} Hz")


def parse_channels(value: str) -> tuple[str, list[int]]:
    key = value.strip().lower()
    if key in CHANNEL_PRESETS:
        return key, CHANNEL_PRESETS[key]
    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 1 or idx > CHANNELS:
            raise ValueError(f"Channel index must be 1-{CHANNELS}: {idx}")
        indices.append(idx - 1)
    if not indices:
        raise ValueError("No channels selected.")
    return "custom_" + "_".join(str(i + 1) for i in indices), indices


def build_filterbank(data: np.ndarray, paradigm: str, channel_indices: list[int]) -> np.ndarray:
    """Return classes x blocks x bands x channels x samples."""

    cfg = PARADIGM_CONFIG[paradigm]
    n_channels = len(channel_indices)
    selected = data[np.asarray(channel_indices, dtype=np.int64)]
    epochs = np.transpose(selected, (3, 2, 0, 1)).reshape(CLASSES * BLOCKS, n_channels, EPOCH_SAMPLES)
    epochs = notch_50hz(epochs.astype(np.float64, copy=False))
    bands = []
    for low in cfg["lowcuts"]:
        bands.append(cheby_bandpass(epochs, float(low), float(cfg["highcut"])))
    fb = np.stack(bands, axis=1)
    return fb.reshape(CLASSES, BLOCKS, int(cfg["n_bands"]), n_channels, EPOCH_SAMPLES).astype(np.float32)


def crop_epochs(fb: np.ndarray, window: float, latency: float) -> np.ndarray:
    samples = int(round(window * FS))
    start = STIM_START + int(round(latency * FS))
    stop = start + samples
    if stop > fb.shape[-1]:
        raise ValueError(f"Window {window:g}s with latency {latency:g}s needs stop {stop}, only {fb.shape[-1]} samples.")
    return fb[..., start:stop].copy()


def center_rows(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=-1, keepdims=True)


def orth_rows(x: np.ndarray) -> np.ndarray:
    z = center_rows(np.asarray(x, dtype=np.float64)).T
    if np.linalg.norm(z) <= 1e-12:
        return np.zeros((z.shape[0], 1), dtype=np.float64)
    q, r = np.linalg.qr(z, mode="reduced")
    keep = np.abs(np.diag(r)) > 1e-10
    return q[:, keep] if np.any(keep) else q[:, :1] * 0.0


def cca_corr_from_basis(x: np.ndarray, y_q: np.ndarray) -> float:
    x_q = orth_rows(x)
    vals = np.linalg.svd(x_q.T @ y_q, compute_uv=False)
    return float(np.clip(vals[0], 0.0, 1.0)) if vals.size else 0.0


def resample_code_row(row: np.ndarray, needed: int) -> np.ndarray:
    t_src = np.arange(row.size, dtype=np.float64) / DISPLAY_FS
    t_dst = np.arange(needed, dtype=np.float64) / FS
    return np.interp(t_dst, t_src, row, period=None)


def shifted_vector(x: np.ndarray, shift: int, samples: int) -> np.ndarray:
    if shift <= 0:
        return x[:samples]
    if x.size < samples + shift:
        x = np.pad(x, (0, samples + shift - x.size), mode="edge")
    return x[shift : shift + samples]


def code_references(dataset: Path, paradigm: str, samples: int) -> list[np.ndarray]:
    cfg = PARADIGM_CONFIG[paradigm]
    code = load_code(dataset, paradigm)
    max_shift = max(int(s) for s in cfg["latency_points"])
    refs = []
    for cls in range(CLASSES):
        src = resample_code_row(code[cls], samples + max_shift + 4)
        rows = []
        for shift in cfg["latency_points"]:
            vec = shifted_vector(src, int(shift), samples)
            rows.append(vec)
            rows.append(vec * vec)
        refs.append(orth_rows(np.asarray(rows, dtype=np.float64)))
    return refs


def sinusoid_references(paradigm: str, samples: int) -> list[np.ndarray]:
    cfg = PARADIGM_CONFIG[paradigm]
    harmonics = int(cfg["harmonics"])
    t = np.arange(samples, dtype=np.float64) / FS
    refs = []
    for cls in range(CLASSES):
        freq = 8.0 + 0.2 * cls
        phase = 0.5 * math.pi * cls
        rows = []
        for h in range(1, harmonics + 1):
            angle = 2 * math.pi * h * freq * t + h * phase
            rows.append(np.sin(angle))
            rows.append(np.cos(angle))
        refs.append(orth_rows(np.asarray(rows, dtype=np.float64)))
    return refs


def make_references(dataset: Path, paradigm: str, window: float) -> list[np.ndarray]:
    samples = int(round(window * FS))
    if PARADIGM_CONFIG[paradigm]["reference"] == "code":
        return code_references(dataset, paradigm, samples)
    return sinusoid_references(paradigm, samples)


def predict_fbcca_code(dataset: Path, paradigm: str, fb: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    cfg = PARADIGM_CONFIG[paradigm]
    epochs = crop_epochs(fb, window, float(cfg["fbcca_latency"]))
    refs = make_references(dataset, paradigm, window)
    weights = paradigm_weights(paradigm)
    test = epochs.reshape(CLASSES * BLOCKS, epochs.shape[2], epochs.shape[3], epochs.shape[-1])
    scores = np.zeros((test.shape[0], CLASSES), dtype=np.float64)
    for trial_idx, trial in enumerate(test):
        for band_idx in range(trial.shape[0]):
            for cls, ref_q in enumerate(refs):
                rho = cca_corr_from_basis(trial[band_idx], ref_q)
                scores[trial_idx, cls] += weights[band_idx] * rho * rho
    return np.argmax(scores, axis=1).reshape(CLASSES, BLOCKS), scores.reshape(CLASSES, BLOCKS, CLASSES)


class CVEPTRCA:
    def __init__(self, paradigm: str) -> None:
        self.weights = paradigm_weights(paradigm)
        self.templates: np.ndarray | None = None
        self.filters: np.ndarray | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> "CVEPTRCA":
        classes = int(np.max(train_y)) + 1
        _, n_bands, channels, samples = train_x.shape
        self.templates = np.zeros((classes, n_bands, channels, samples), dtype=np.float64)
        self.filters = np.zeros((n_bands, classes, channels), dtype=np.float64)
        for cls in range(classes):
            cls_trials = train_x[train_y == cls]
            self.templates[cls] = np.mean(cls_trials, axis=0)
            for band_idx in range(n_bands):
                self.filters[band_idx, cls] = trca_filter(cls_trials[:, band_idx])
        return self

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.templates is None or self.filters is None:
            raise RuntimeError("TRCA model is not fitted.")
        trials = test_x.shape[0]
        classes = self.templates.shape[0]
        band_scores = np.zeros((trials, test_x.shape[1], classes), dtype=np.float64)
        for band_idx in range(test_x.shape[1]):
            for cls in range(classes):
                w = self.filters[band_idx, cls]
                projected_trials = test_x[:, band_idx].transpose(0, 2, 1) @ w
                projected_template = self.templates[cls, band_idx].T @ w
                band_scores[:, band_idx, cls] = corr_rows(projected_trials, projected_template)
        scores = np.einsum("b,tbc->tc", self.weights[: test_x.shape[1]], band_scores)
        return np.argmax(scores, axis=1), scores


def predict_trca(paradigm: str, fb: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    cfg = PARADIGM_CONFIG[paradigm]
    epochs = crop_epochs(fb, window, float(cfg["trca_latency"]))
    labels = np.arange(CLASSES, dtype=np.int64)
    pred = np.zeros((CLASSES, BLOCKS), dtype=np.int64)
    score_cube = np.zeros((CLASSES, BLOCKS, CLASSES), dtype=np.float64)
    for block in range(BLOCKS):
        train_blocks = [idx for idx in range(BLOCKS) if idx != block]
        train_x = np.take(epochs, train_blocks, axis=1).reshape(
            CLASSES * len(train_blocks), epochs.shape[2], epochs.shape[3], epochs.shape[-1]
        )
        train_y = np.repeat(labels, len(train_blocks))
        test_x = epochs[:, block]
        model = CVEPTRCA(paradigm).fit(train_x, train_y)
        block_pred, scores = model.predict(test_x)
        pred[:, block] = block_pred
        score_cube[:, block] = scores
    return pred, score_cube


def predict_mstrca(paradigm: str, fb: np.ndarray, window: float, neighbor_width: int = 2) -> tuple[np.ndarray, np.ndarray]:
    cfg = PARADIGM_CONFIG[paradigm]
    epochs = crop_epochs(fb, window, float(cfg["trca_latency"]))
    labels = np.arange(CLASSES, dtype=np.int64)
    pred = np.zeros((CLASSES, BLOCKS), dtype=np.int64)
    score_cube = np.zeros((CLASSES, BLOCKS, CLASSES), dtype=np.float64)
    for block in range(BLOCKS):
        train_blocks = [idx for idx in range(BLOCKS) if idx != block]
        train_x = np.take(epochs, train_blocks, axis=1).reshape(
            CLASSES * len(train_blocks), epochs.shape[2], epochs.shape[3], epochs.shape[-1]
        )
        train_y = np.repeat(labels, len(train_blocks))
        test_x = epochs[:, block]
        model = MSTRCA(
            neighbor_width=neighbor_width,
            weights=paradigm_weights(paradigm),
            class_order=labels.tolist(),
            ensemble=True,
        ).fit(train_x, train_y)
        block_pred, scores = model.predict(test_x)
        pred[:, block] = block_pred
        score_cube[:, block] = scores
    return pred, score_cube


def rows_from_prediction(
    subject: int,
    paradigm: str,
    method: str,
    window: float,
    pred: np.ndarray,
    scores: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trial_rows = []
    pred_rows = []
    labels = np.arange(CLASSES, dtype=np.int64)
    for block in range(BLOCKS):
        block_pred = pred[:, block]
        acc = float(np.mean(block_pred == labels))
        itr = itr_bits_per_minute(acc, CLASSES, window + GAZE_SHIFT_SECONDS)
        trial_rows.append(
            {
                "subject": subject,
                "paradigm": paradigm,
                "method": method,
                "window": window,
                "block": block + 1,
                "accuracy": acc,
                "itr": itr,
                "samples": CLASSES,
            }
        )
        for target, pred_label in enumerate(block_pred):
            pred_rows.append(
                {
                    "subject": subject,
                    "paradigm": paradigm,
                    "method": method,
                    "window": window,
                    "block": block + 1,
                    "target": int(target),
                    "pred": int(pred_label),
                    "correct": int(pred_label == target),
                    "score_true": float(scores[target, block, target]),
                    "score_pred": float(scores[target, block, pred_label]),
                }
            )
    return trial_rows, pred_rows


def run_unit(task: dict[str, object]) -> dict[str, object]:
    dataset = Path(str(task["dataset"]))
    subject = int(task["subject"])
    paradigm = str(task["paradigm"])
    windows = [float(x) for x in task["windows"]]
    methods = [str(x).upper() for x in task["methods"]]
    channel_indices = [int(x) for x in task["channel_indices"]]
    started = time.perf_counter()
    raw = load_subject_paradigm(dataset, subject, paradigm)
    fb = build_filterbank(raw, paradigm, channel_indices)
    trial_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = [
        {
                "subject": subject,
                "paradigm": paradigm,
                "method": "__all__",
                "window": "",
                "stage": f"load_filterbank_{len(channel_indices)}ch",
                "seconds": time.perf_counter() - started,
                "pid": os.getpid(),
        }
    ]
    logs = []
    for window in windows:
        for method in methods:
            t0 = time.perf_counter()
            if method == "FBCCA-CODE":
                pred, scores = predict_fbcca_code(dataset, paradigm, fb, window)
            elif method == "TRCA":
                pred, scores = predict_trca(paradigm, fb, window)
            elif method == "MSTRCA":
                pred, scores = predict_mstrca(paradigm, fb, window)
            else:
                raise ValueError(f"Unsupported method: {method}")
            tr, pr = rows_from_prediction(subject, paradigm, method, window, pred, scores)
            trial_rows.extend(tr)
            pred_rows.extend(pr)
            acc = float(np.mean(pred == np.arange(CLASSES)[:, None]))
            seconds = time.perf_counter() - t0
            runtime_rows.append(
                {
                    "subject": subject,
                    "paradigm": paradigm,
                    "method": method,
                    "window": window,
                    "stage": "fit_predict",
                    "seconds": seconds,
                    "pid": os.getpid(),
                }
            )
            logs.append(f"S{subject:03d} {paradigm} {method} w={window:g} acc={acc:.4f} sec={seconds:.1f}")
    return {
        "subject": subject,
        "paradigm": paradigm,
        "trial_rows": trial_rows,
        "pred_rows": pred_rows,
        "runtime_rows": runtime_rows,
        "logs": logs,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def part_paths(result_dir: Path, unit: Unit) -> tuple[Path, Path, Path]:
    base = result_dir / "parts" / unit.slug
    return base.with_suffix(".trials.csv"), base.with_suffix(".predictions.csv"), base.with_suffix(".runtime.csv")


def is_complete_part(result_dir: Path, unit: Unit) -> bool:
    trials, preds, runtime = part_paths(result_dir, unit)
    return trials.exists() and preds.exists() and runtime.exists()


def write_part(result_dir: Path, result: dict[str, object]) -> None:
    unit = Unit(int(result["subject"]), str(result["paradigm"]))
    trials, preds, runtime = part_paths(result_dir, unit)
    write_csv(
        trials,
        list(result["trial_rows"]),
        ["subject", "paradigm", "method", "window", "block", "accuracy", "itr", "samples"],
    )
    write_csv(
        preds,
        list(result["pred_rows"]),
        ["subject", "paradigm", "method", "window", "block", "target", "pred", "correct", "score_true", "score_pred"],
    )
    write_csv(
        runtime,
        list(result["runtime_rows"]),
        ["subject", "paradigm", "method", "window", "stage", "seconds", "pid"],
    )


def write_failure(result_dir: Path, payload: dict[str, object], exc: BaseException) -> None:
    path = result_dir / "failures.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["time", "subject", "paradigm", "methods", "windows", "error", "traceback"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "time": dt.datetime.now(dt.timezone.utc).isoformat(),
                "subject": payload["subject"],
                "paradigm": payload["paradigm"],
                "methods": ",".join(payload["methods"]),
                "windows": ",".join(str(w) for w in payload["windows"]),
                "error": repr(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            }
        )


def combine_parts(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    part_dir = result_dir / "parts"
    trial_files = sorted(part_dir.glob("*.trials.csv"))
    pred_files = sorted(part_dir.glob("*.predictions.csv"))
    runtime_files = sorted(part_dir.glob("*.runtime.csv"))
    trials = pd.concat([pd.read_csv(p) for p in trial_files], ignore_index=True) if trial_files else pd.DataFrame()
    preds = pd.concat([pd.read_csv(p) for p in pred_files], ignore_index=True) if pred_files else pd.DataFrame()
    runtimes = pd.concat([pd.read_csv(p) for p in runtime_files], ignore_index=True) if runtime_files else pd.DataFrame()
    if not trials.empty:
        trials.to_csv(result_dir / "trials.csv", index=False)
    if not preds.empty:
        preds.to_csv(result_dir / "predictions.csv", index=False)
    if not runtimes.empty:
        runtimes.to_csv(result_dir / "runtime.csv", index=False)
    return trials, preds, runtimes


def summarize(result_dir: Path, total_units: int) -> None:
    trials, preds, _ = combine_parts(result_dir)
    if trials.empty:
        return
    subject = (
        trials.groupby(["paradigm", "method", "window", "subject"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), itr=("itr", "mean"))
        .sort_values(["paradigm", "method", "window", "subject"])
    )
    summary_rows = []
    for (paradigm, method, window), group in subject.groupby(["paradigm", "method", "window"]):
        summary_rows.append(
            {
                "paradigm": paradigm,
                "method": method,
                "window": float(window),
                "accuracy": float(group["accuracy"].mean()),
                "accuracy_sem": sem(group["accuracy"].to_numpy()),
                "itr": float(group["itr"].mean()),
                "itr_sem": sem(group["itr"].to_numpy()),
                "subjects": int(group["subject"].nunique()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["paradigm", "method", "window"])
    subject.to_csv(result_dir / "subject.csv", index=False)
    summary.to_csv(result_dir / "summary.csv", index=False)
    write_confusions(result_dir, preds)
    plot_summary(result_dir, summary)
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    complete_parts = len(list((result_dir / "parts").glob("*.trials.csv")))
    manifest["complete_units"] = complete_parts
    manifest["expected_units"] = total_units
    manifest["status"] = "complete" if complete_parts == total_units else "partial"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_confusions(result_dir: Path, preds: pd.DataFrame) -> None:
    if preds.empty:
        return
    out = result_dir / "confusions"
    out.mkdir(parents=True, exist_ok=True)
    for (paradigm, method, window), group in preds.groupby(["paradigm", "method", "window"]):
        mat = np.zeros((CLASSES, CLASSES), dtype=np.int64)
        for target, pred in zip(group["target"].to_numpy(dtype=int), group["pred"].to_numpy(dtype=int)):
            mat[target, pred] += 1
        np.save(out / f"{paradigm}_{method}_w{float(window):g}.npy", mat)


def plot_summary(result_dir: Path, summary: pd.DataFrame) -> None:
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for metric, sem_col, ylabel, filename in [
        ("accuracy", "accuracy_sem", "Accuracy", "accuracy_curve.png"),
        ("itr", "itr_sem", "ITR (bits/min)", "itr_curve.png"),
    ]:
        paradigms = list(summary["paradigm"].drop_duplicates())
        fig, axes = plt.subplots(1, len(paradigms), figsize=(5.2 * len(paradigms), 4.2), sharey=False)
        if len(paradigms) == 1:
            axes = [axes]
        for ax, paradigm in zip(axes, paradigms):
            rows_p = summary[summary["paradigm"] == paradigm]
            for method in rows_p["method"].drop_duplicates():
                rows = rows_p[rows_p["method"] == method]
                ax.errorbar(rows["window"], rows[metric], yerr=rows[sem_col], marker="o", capsize=3, label=method)
            ax.set_title(paradigm)
            ax.set_xlabel("Window (s)")
            ax.set_ylabel(ylabel)
            if metric == "accuracy":
                ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)

    pivot = summary.pivot_table(index=["paradigm", "method"], columns="window", values="accuracy")
    fig, ax = plt.subplots(figsize=(9.5, max(3.2, 0.42 * len(pivot) + 1.5)))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Accuracy")
    ax.set_xticks(range(len(pivot.columns)), [f"{x:.1f}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{p}/{m}" for p, m in pivot.index])
    ax.set_xlabel("Window (s)")
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--task-name", default="full")
    parser.add_argument("--subjects", default="1-100")
    parser.add_argument("--paradigms", default="NBRS-15,NBRS-8,JFPM-8")
    parser.add_argument("--methods", default="FBCCA-CODE,TRCA")
    parser.add_argument("--windows", default="0.4,0.8,1.2,1.6,2.0,2.4,3.0,4.0")
    parser.add_argument(
        "--channels",
        default="all59",
        help="Channel preset or 1-based comma list. Presets: all59, posterior17, po_o10, occipital9, o3.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summary-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    subjects = parse_range(args.subjects)
    paradigms = parse_csv(args.paradigms)
    methods = [m.upper() for m in parse_csv(args.methods)]
    windows = parse_windows(args.windows)
    channel_name, channel_indices = parse_channels(args.channels)
    for paradigm in paradigms:
        if paradigm not in PARADIGM_CONFIG:
            raise ValueError(f"Unknown paradigm: {paradigm}")
    for method in methods:
        if method not in {"FBCCA-CODE", "TRCA", "MSTRCA"}:
            raise ValueError(f"Unknown method: {method}")

    result_dir = TASK / "results" / args.task_name
    result_dir.mkdir(parents=True, exist_ok=True)
    units = [Unit(subject, paradigm) for subject in subjects for paradigm in paradigms]
    pending = [unit for unit in units if not (args.resume and is_complete_part(result_dir, unit))]
    manifest = {
        "dataset": str(args.dataset),
        "task_name": args.task_name,
        "subjects": subjects,
        "paradigms": paradigms,
        "methods": methods,
        "windows": windows,
        "channels": channel_name,
        "channel_indices_1based": [idx + 1 for idx in channel_indices],
        "workers": args.workers,
        "parallel_unit": "subject_paradigm",
        "protocol": "subject-specific leave-one-block-out, 3 folds",
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing": {
            "released_data": "250 Hz raw EEG without further processing, shaped channels x samples x blocks x targets.",
            "channel_selection": channel_name,
            "notch": "50 Hz second-order IIR notch via scipy.signal.iirnotch(Q=35) and filtfilt.",
            "filterbank": "Chebyshev type I band-pass filters with paradigm-specific low/high cutoffs from the paper.",
            "crop": "Filter full 6 s epoch, then retain stimulus onset + algorithm-specific latency through window.",
        },
        "notes": {
            "FBCCA-CODE": "NBRS uses latency-shifted code references plus second-order Hadamard templates; JFPM uses sinusoidal harmonic references.",
            "TRCA": "Arena TRCA approximation with paper-specific filter banks and weights; not strict msTRCA yet.",
            "MSTRCA": "Arena multi-stimulus ensemble TRCA using target-order neighborhoods to estimate spatial filters.",
            "itr": "window + 0.5 s gaze shift",
        },
        "expected_units": len(units),
        "pending_units_at_start": len(pending),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"result_dir={result_dir}", flush=True)
    print(
        f"units total={len(units)} pending={len(pending)} workers={args.workers} "
        f"channels={channel_name} ({len(channel_indices)}ch)",
        flush=True,
    )

    task_payloads = [
        {
            "dataset": str(args.dataset),
            "subject": unit.subject,
            "paradigm": unit.paradigm,
            "windows": windows,
            "methods": methods,
            "channel_indices": channel_indices,
        }
        for unit in pending
    ]
    completed_since_summary = 0
    if args.workers > 1 and task_payloads:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_map = {pool.submit(run_unit, payload): payload for payload in task_payloads}
            for fut in as_completed(future_map):
                payload = future_map[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    write_failure(result_dir, payload, exc)
                    print(f"FAILED S{int(payload['subject']):03d} {payload['paradigm']}: {exc!r}", flush=True)
                    continue
                write_part(result_dir, result)
                completed_since_summary += 1
                for line in result["logs"]:
                    print(line, flush=True)
                if completed_since_summary >= max(1, args.summary_every):
                    summarize(result_dir, len(units))
                    completed_since_summary = 0
    else:
        for payload in task_payloads:
            try:
                result = run_unit(payload)
            except Exception as exc:
                write_failure(result_dir, payload, exc)
                print(f"FAILED S{int(payload['subject']):03d} {payload['paradigm']}: {exc!r}", flush=True)
                continue
            write_part(result_dir, result)
            completed_since_summary += 1
            for line in result["logs"]:
                print(line, flush=True)
            if completed_since_summary >= max(1, args.summary_every):
                summarize(result_dir, len(units))
                completed_since_summary = 0

    summarize(result_dir, len(units))
    print(result_dir, flush=True)


if __name__ == "__main__":
    main()
