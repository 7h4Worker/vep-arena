from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio


BROADBAND_WN_KEY = "cvep_broadband_white_noise_bci_zenodo8300517"
BROADBAND_WN_ROOT = Path("D:/ProjData/datasets/cvep_broadband_white_noise_bci_zenodo8300517")
BROADBAND_WN_SAMPLING_RATE = 250
BROADBAND_WN_CLASSES = 160
BROADBAND_WN_SAMPLES = 250
BROADBAND_WN_DEFAULT_TAG = "WN"
BROADBAND_WN_PAPER_CHANNELS = (
    "PZ",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "POZ",
    "PO3",
    "PO4",
    "PO5",
    "PO6",
    "PO7",
    "PO8",
    "O1",
    "OZ",
    "O2",
    "CB1",
    "CB2",
)


@dataclass(frozen=True)
class BroadbandSession:
    subject: int
    session: int
    tag: str
    x: np.ndarray
    y: np.ndarray
    channels: tuple[str, ...]
    path: Path


def resolve_dataset_root(key: str, default: Path) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "configs" / "datasets" / "local_paths.json"
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        datasets = config.get("datasets", {})
        if key in datasets:
            return Path(str(datasets[key]))
    return default


def broadband_wn_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else resolve_dataset_root(BROADBAND_WN_KEY, BROADBAND_WN_ROOT)


def _natural_key(path: Path | str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path)))


def parse_subjects(text: str) -> list[int]:
    text = text.strip()
    if text.lower() in {"all", "author"}:
        return list(range(1, 11))
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


def subject_file(root: Path | str | None, subject: int) -> Path:
    return broadband_wn_root(root) / "extracted" / "data" / "seperate" / "sweep" / f"S_{subject}.pkl"


def available_subjects(root: Path | str | None = None) -> tuple[int, ...]:
    base = broadband_wn_root(root) / "extracted" / "data" / "seperate" / "sweep"
    subjects = []
    for path in sorted(base.glob("S_*.pkl"), key=_natural_key):
        match = re.search(r"S_(\d+)\.pkl$", path.name)
        if match:
            subjects.append(int(match.group(1)))
    return tuple(subjects)


def load_subject_sessions(
    root: Path | str | None = None,
    subject: int = 1,
    tag: str | None = BROADBAND_WN_DEFAULT_TAG,
) -> list[BroadbandSession]:
    path = subject_file(root, subject)
    with path.open("rb") as handle:
        sessions = pickle.load(handle)
    out: list[BroadbandSession] = []
    for idx, session in enumerate(sessions, start=1):
        session_tag, x, y, channels = session
        if tag is not None and str(session_tag).upper() != tag.upper():
            continue
        out.append(
            BroadbandSession(
                subject=subject,
                session=idx,
                tag=str(session_tag).upper(),
                x=np.asarray(x, dtype=np.float64),
                y=np.asarray(y, dtype=np.int64),
                channels=tuple(str(ch).upper() for ch in channels),
                path=path,
            )
        )
    return out


def read_stimulus(root: Path | str | None = None, tag: str = BROADBAND_WN_DEFAULT_TAG) -> np.ndarray:
    path = broadband_wn_root(root) / "extracted" / "data" / "stimulation" / "sweep" / "STI.mat"
    data = sio.loadmat(path, squeeze_me=True)
    key = tag.upper()
    if key not in data:
        raise KeyError(f"{key} not found in {path}")
    return np.asarray(data[key], dtype=np.float64)


def channel_indices(channels: Iterable[str], channel_set: str = "paper21") -> list[int]:
    names = [str(ch).upper() for ch in channels]
    if channel_set == "all64":
        return list(range(len(names)))
    if channel_set == "paper21":
        wanted = list(BROADBAND_WN_PAPER_CHANNELS)
    elif channel_set == "occipital9":
        wanted = ["PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"]
    else:
        wanted = [item.strip().upper() for item in channel_set.split(",") if item.strip()]
    missing = [ch for ch in wanted if ch not in names]
    if missing:
        raise ValueError(f"Missing broadband WN channels: {missing}")
    return [names.index(ch) for ch in wanted]
