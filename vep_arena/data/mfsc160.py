from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
from scipy import signal


MFSC160_KEY = "ssvep_160target_mfsc_chen2021"
MFSC160_ROOT = Path("D:/ProjData/datasets/ssvep_160target_mfsc_chen2021")
MFSC160_SAMPLING_RATE = 250
MFSC160_CLASSES = 160
MFSC160_CHANNELS = ("Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2")
MFSC160_BASE_FREQS = tuple(float(freq) for freq in range(8, 16))
MFSC160_PHASES_PI = (0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 1.5)
MFSC160_LATENCY_SECONDS = 0.14
MFSC160_LATENCY_SAMPLES = int(round(MFSC160_LATENCY_SECONDS * MFSC160_SAMPLING_RATE))
MFSC160_SLOT_SECONDS = 1.0
MFSC160_SLOT_SAMPLES = int(round(MFSC160_SLOT_SECONDS * MFSC160_SAMPLING_RATE))
MFSC160_DEFAULT_FILTER_BANKS = 6


@dataclass(frozen=True)
class Mfsc160File:
    path: Path
    split: str
    subject: int
    block: int


def resolve_dataset_root(key: str, default: Path) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "configs" / "datasets" / "local_paths.json"
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        datasets = config.get("datasets", {})
        if key in datasets:
            return Path(str(datasets[key]))
    return default


def mfsc160_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else resolve_dataset_root(MFSC160_KEY, MFSC160_ROOT)


def _natural_key(path: Path | str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(path))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def parse_subjects(text: str, split: str = "offline") -> list[int]:
    text = text.strip()
    if text.lower() in {"all", "author"}:
        return list(range(1, 9)) if split == "offline" else list(range(1, 13))
    subjects: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(item) for item in part.split("-", 1))
            subjects.extend(range(lo, hi + 1))
        else:
            subjects.append(int(part))
    return subjects


def read_codebook(root: Path | str | None = None) -> np.ndarray:
    code = sio.loadmat(mfsc160_root(root) / "metadata" / "reqCodeword.mat", squeeze_me=True)["reqCodeword"]
    code = np.asarray(code, dtype=np.int64)
    if code.shape != (MFSC160_CLASSES, 4):
        raise ValueError(f"Expected reqCodeword shape (160, 4), got {code.shape}")
    return code


def iter_mfsc160_files(
    root: Path | str | None = None,
    split: str = "offline",
    subjects: Iterable[int] | None = None,
) -> list[Mfsc160File]:
    split = split.lower()
    if split not in {"offline", "online"}:
        raise ValueError(f"Unknown MFSC160 split: {split}")
    base = mfsc160_root(root) / "extracted" / ("Offline" if split == "offline" else "Online")
    wanted = {int(item) for item in subjects} if subjects is not None else None
    rows: list[Mfsc160File] = []
    for subject_dir in sorted(base.iterdir(), key=_natural_key):
        if not subject_dir.is_dir():
            continue
        match = re.search(r"(\d+)", subject_dir.name)
        if not match:
            continue
        subject = int(match.group(1))
        if wanted is not None and subject not in wanted:
            continue
        for path in sorted(subject_dir.glob("*.mat"), key=_natural_key):
            rows.append(Mfsc160File(path=path, split=split, subject=subject, block=int(path.stem)))
    return rows


def load_mfsc160_block(path: Path | str) -> np.ndarray:
    data = np.asarray(sio.loadmat(path, squeeze_me=True)["EEG_downsample"], dtype=np.float32)
    if data.shape != (MFSC160_CLASSES, len(MFSC160_CHANNELS), 1035):
        raise ValueError(f"Expected EEG_downsample shape (160, 9, 1035), got {data.shape} in {path}")
    return data


def load_mfsc160_subject(root: Path | str | None = None, split: str = "offline", subject: int = 1) -> np.ndarray:
    files = iter_mfsc160_files(root=root, split=split, subjects=[subject])
    if not files:
        raise FileNotFoundError(f"No MFSC160 {split} files found for subject {subject}")
    blocks = [load_mfsc160_block(item.path) for item in files]
    return np.stack(blocks, axis=1)


def mfsc160_filter_sos(n_bands: int = MFSC160_DEFAULT_FILTER_BANKS) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    nyq = MFSC160_SAMPLING_RATE / 2
    for band in range(1, n_bands + 1):
        wp = [8 * band / nyq, 90 / nyq]
        ws = [(8 * band - 2) / nyq, 100 / nyq]
        order, wn = signal.cheb1ord(wp, ws, 3, 40)
        out.append(signal.cheby1(order, 0.5, wn, btype="bandpass", output="sos"))
    return out


def filter_mfsc160_subject(raw: np.ndarray, n_bands: int = MFSC160_DEFAULT_FILTER_BANKS) -> np.ndarray:
    if raw.ndim != 4:
        raise ValueError(f"Expected raw subject data targets x blocks x channels x samples, got {raw.shape}")
    targets, blocks, channels, samples = raw.shape
    out = np.empty((targets, blocks, n_bands, channels, samples), dtype=np.float32)
    for idx, sos in enumerate(mfsc160_filter_sos(n_bands)):
        out[:, :, idx] = signal.sosfiltfilt(sos, raw, axis=-1).astype(np.float32, copy=False)
    return out


def load_or_filter_mfsc160_subject(
    root: Path | str | None = None,
    split: str = "offline",
    subject: int = 1,
    n_bands: int = MFSC160_DEFAULT_FILTER_BANKS,
    force_filter: bool = False,
) -> np.ndarray:
    base = mfsc160_root(root)
    cache_dir = base / "derivatives" / "mfsc160_tdca" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"filtered_{split}_S{subject}_{n_bands}fb_float32.npy"
    if cache.exists() and not force_filter:
        return np.load(cache, mmap_mode="r")
    raw = load_mfsc160_subject(root=base, split=split, subject=subject)
    filtered = filter_mfsc160_subject(raw, n_bands=n_bands)
    np.save(cache, filtered)
    return np.load(cache, mmap_mode="r")


def mfsc160_sequence_references(
    codebook: np.ndarray,
    segment_samples: int,
    harmonics: int = 5,
    fs: int = MFSC160_SAMPLING_RATE,
) -> list[np.ndarray]:
    t = np.arange(segment_samples, dtype=np.float64) / fs
    refs: list[np.ndarray] = []
    for code in codebook:
        rows: list[np.ndarray] = []
        for harmonic in range(1, harmonics + 1):
            sin_parts = []
            cos_parts = []
            for symbol in code:
                freq = MFSC160_BASE_FREQS[int(symbol)]
                phase = MFSC160_PHASES_PI[int(symbol)] * np.pi
                sin_parts.append(np.sin(2 * np.pi * harmonic * freq * t + harmonic * phase))
                cos_parts.append(np.cos(2 * np.pi * harmonic * freq * t + harmonic * phase))
            rows.append(np.concatenate(sin_parts))
            rows.append(np.concatenate(cos_parts))
        return_ref = np.asarray(rows, dtype=np.float64)
        refs.append(return_ref)
    return refs


def mfsc160_epoch_tensor(filtered: np.ndarray, segment_seconds: float) -> np.ndarray:
    segment_samples = int(round(segment_seconds * MFSC160_SAMPLING_RATE))
    if segment_samples <= 0 or segment_samples > MFSC160_SLOT_SAMPLES:
        raise ValueError(f"segment_seconds must be in (0, 1.0], got {segment_seconds}")
    starts = [MFSC160_LATENCY_SAMPLES + slot * MFSC160_SLOT_SAMPLES for slot in range(4)]
    snippets = []
    for start in starts:
        stop = start + segment_samples
        if stop > filtered.shape[-1]:
            raise ValueError(f"Segment stop {stop} exceeds available samples {filtered.shape[-1]}")
        snippets.append(filtered[..., start:stop])
    return np.concatenate(snippets, axis=-1)
