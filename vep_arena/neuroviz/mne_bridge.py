# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-20
# Last updated: 2026-06-20
# Description: Convert Arena epoch tensors into MNE objects for QA figures.
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EpochMetadata:
    """Metadata needed to convert Arena tensors into MNE EpochsArray."""

    subject: int
    block: int | None
    window: float
    preprocess: str
    event_name: str = "target"


def require_mne():
    """Import MNE lazily so Arena's benchmark code can run without MNE installed."""

    try:
        import mne  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MNE is not available in this Python environment. "
            "Use D:/ProjData/envs/erp_ssvep_lab/python.exe or install mne in the active environment."
        ) from exc
    return mne


def make_info(
    channel_names: Sequence[str],
    sampling_rate: float,
    *,
    channel_types: str | Sequence[str] = "eeg",
    montage: str | None = "standard_1005",
):
    """Create an MNE Info object with an optional standard montage."""

    mne = require_mne()
    info = mne.create_info(list(channel_names), sfreq=float(sampling_rate), ch_types=channel_types)
    if montage:
        info.set_montage(montage, on_missing="ignore")
    return info


def epochs_array_from_class_block_trials(
    trials: np.ndarray,
    *,
    channel_names: Sequence[str],
    sampling_rate: float,
    metadata: EpochMetadata,
    class_offset: int = 1,
    montage: str | None = "standard_1005",
):
    """Convert `classes x blocks x channels x samples` trials into MNE EpochsArray.

    The returned events use one event id per class: `target/1`, `target/2`, ...
    If `metadata.block` is provided, only that block is exported.
    """

    mne = require_mne()
    arr = np.asarray(trials, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"Expected classes x blocks x channels x samples, got shape {arr.shape}.")

    if metadata.block is not None:
        block_idx = metadata.block - 1
        if block_idx < 0 or block_idx >= arr.shape[1]:
            raise ValueError(f"Block {metadata.block} is outside available blocks 1..{arr.shape[1]}.")
        data = arr[:, block_idx]
        labels = np.arange(arr.shape[0], dtype=np.int64) + class_offset
    else:
        data = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3])
        labels = np.repeat(np.arange(arr.shape[0], dtype=np.int64) + class_offset, arr.shape[1])

    info = make_info(channel_names, sampling_rate, montage=montage)
    events = np.column_stack(
        [
            np.arange(len(labels), dtype=np.int64),
            np.zeros(len(labels), dtype=np.int64),
            labels.astype(np.int64),
        ]
    )
    event_id = {f"{metadata.event_name}/{int(label)}": int(label) for label in sorted(set(labels.tolist()))}
    epochs = mne.EpochsArray(data, info, events=events, event_id=event_id, tmin=0.0, baseline=None, verbose=False)
    epochs.info["description"] = (
        f"subject={metadata.subject}; block={metadata.block}; window={metadata.window:g}; "
        f"preprocess={metadata.preprocess}"
    )
    return epochs


def evoked_for_event(epochs, event_name: str):
    """Return an Evoked average for an MNE event key."""

    return epochs[event_name].average()

