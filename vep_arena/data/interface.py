from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class DatasetInfo:
    """Small dataset contract inspired by SSVEP-Analysis-Toolbox.

    Arena keeps the object light: it describes the data and exposes a stable
    tensor shape, but it does not download files or own the evaluator.
    """

    id: str
    name: str
    root: Path
    subjects: tuple[int, ...]
    blocks: tuple[int, ...]
    targets: tuple[int, ...]
    channels: tuple[str, ...]
    sampling_rate: int
    frequencies: tuple[float, ...]
    phases: tuple[float, ...]
    prestim_seconds: float
    break_seconds: float
    latency_seconds: float | None


@dataclass(frozen=True)
class TrialBatch:
    """Trials in Arena's common SSVEP shape.

    Shape convention:
    - x: trials x bands x channels x samples
    - y: trials, zero-based target labels
    """

    x: np.ndarray
    y: np.ndarray
    subject: int
    blocks: tuple[int, ...]
    targets: tuple[int, ...]
    window: float
    preprocess: str
    info: DatasetInfo


class SSVEPDataset(Protocol):
    info: DatasetInfo

    def load_subject(self, subject: int) -> np.ndarray:
        """Return subject data as blocks x targets x channels x samples."""

    def get_trials(
        self,
        subject: int,
        blocks: list[int] | tuple[int, ...],
        targets: list[int] | tuple[int, ...],
        channels: str | list[int] | tuple[int, ...],
        window: float,
        preprocess: str = "raw",
        n_bands: int = 1,
    ) -> TrialBatch:
        """Return trials x bands x channels x samples plus labels."""


def classes_blocks_to_trials(epochs: np.ndarray, targets: tuple[int, ...], blocks: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Convert classes x blocks x ... into trials x ... with zero-based labels."""

    selected = epochs[np.asarray(targets, dtype=np.int64)][:, np.asarray(blocks, dtype=np.int64)]
    x = selected.reshape(len(targets) * len(blocks), *selected.shape[2:])
    y = np.repeat(np.asarray(targets, dtype=np.int64), len(blocks))
    return x, y


def ensure_band_axis(x: np.ndarray) -> np.ndarray:
    """Convert trials x channels x samples to trials x 1 x channels x samples."""

    if x.ndim == 3:
        return x[:, None, :, :]
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected 3D or 4D trial tensor, got shape {x.shape}.")
