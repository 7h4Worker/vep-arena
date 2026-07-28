# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vep_arena.config import PROJECT_ROOT
from vep_arena.data.benchmark import load_subject_filterbank, load_subject_toolbox_raw
from vep_arena.data.presets import DatasetPreset


FILTERBANK_VERSION = "ssvep_analysis_toolbox_benchmark_8k"
RAW_VERSION = "ssvep_analysis_toolbox_notch_latency_crop"


def channel_slug(channels: tuple[int, ...]) -> str:
    """Return a stable cache-file slug without expanding long channel lists."""

    expanded = "-".join(str(ch) for ch in channels)
    if len(expanded) <= 48:
        return expanded
    digest = hashlib.sha1(",".join(str(ch) for ch in channels).encode("utf-8")).hexdigest()[:10]
    return f"{len(channels)}ch_{digest}"


@dataclass(frozen=True)
class EpochRequest:
    preset: DatasetPreset
    subject: int
    window: float
    kind: str = "raw"
    n_fbs: int = 1
    extra_samples: int = 0
    cache_window: float | None = None

    @property
    def samples(self) -> int:
        return self.preset.spec.sample_length(self.window) + self.extra_samples

    @property
    def cache_samples(self) -> int:
        window = self.cache_window if self.cache_window is not None else self.window
        return self.preset.spec.sample_length(window) + self.extra_samples

    def manifest(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "window": self.window,
            "kind": self.kind,
            "n_fbs": self.n_fbs,
            "filterbank_version": FILTERBANK_VERSION if self.kind == "filterbank" else None,
            "raw_version": RAW_VERSION if self.kind == "raw" else None,
            "extra_samples": self.extra_samples,
            "samples": self.samples,
            "cache_window": self.cache_window,
            "cache_samples": self.cache_samples,
            "preset": self.preset.manifest(),
        }


class CanonicalEpochStore:
    """Cache and retrieve dataset-preset-aligned epoch tensors."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PROJECT_ROOT / "runs" / "canonical_epochs"

    def path_for(self, request: EpochRequest) -> Path:
        channels = channel_slug(request.preset.channels)
        cache_window = request.cache_window if request.cache_window is not None else request.window
        name = (
            f"{request.preset.name}"
            f"_s{request.subject:02d}"
            f"_maxw{cache_window:g}"
            f"_{request.kind}"
            f"_fb{request.n_fbs}"
            f"_{FILTERBANK_VERSION if request.kind == 'filterbank' else RAW_VERSION}"
            f"_extra{request.extra_samples}"
            f"_ch{channels}.npz"
        )
        return self.root / request.preset.name / name

    def manifest_path_for(self, request: EpochRequest) -> Path:
        return self.path_for(request).with_suffix(".manifest.json")

    def load_or_create(self, request: EpochRequest, force: bool = False) -> np.ndarray:
        path = self.path_for(request)
        manifest_path = self.manifest_path_for(request)
        if path.exists() and manifest_path.exists() and not force:
            return slice_epochs(np.load(path)["epochs"], request)

        path.parent.mkdir(parents=True, exist_ok=True)
        epochs = build_epochs_for_cache(request).astype(np.float32, copy=False)
        np.savez_compressed(path, epochs=epochs)
        manifest = request.manifest()
        manifest["shape"] = list(epochs.shape)
        manifest["dtype"] = str(epochs.dtype)
        manifest["cache_policy"] = "subject_max_window_slice"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return slice_epochs(epochs, request)

    def manifest_for(self, request: EpochRequest) -> dict[str, object]:
        manifest_path = self.manifest_path_for(request)
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = request.manifest()
        manifest["shape"] = None
        return manifest


def build_epochs(request: EpochRequest) -> np.ndarray:
    return slice_epochs(build_epochs_for_cache(request), request)


def build_epochs_for_cache(request: EpochRequest) -> np.ndarray:
    preset = request.preset
    cache_window = request.cache_window if request.cache_window is not None else request.window
    if request.kind == "raw":
        if request.extra_samples:
            raise ValueError("raw canonical epochs do not support extra_samples yet.")
        return load_subject_toolbox_raw(
            preset.data_root,
            request.subject,
            cache_window,
            channels=preset.channels,
            spec=preset.spec,
        )
    if request.kind == "filterbank":
        return load_subject_filterbank(
            preset.data_root,
            request.subject,
            cache_window,
            n_fbs=request.n_fbs,
            channels=preset.channels,
            spec=preset.spec,
            extra_samples=request.extra_samples,
        )
    raise ValueError(f"Unknown epoch kind: {request.kind}")


def slice_epochs(epochs: np.ndarray, request: EpochRequest) -> np.ndarray:
    samples = request.samples
    if epochs.shape[-1] < samples:
        raise ValueError(f"Cached epoch has {epochs.shape[-1]} samples but request needs {samples}.")
    return epochs[..., :samples].copy()


def epoch_fingerprint(request: EpochRequest) -> dict[str, object]:
    preset = request.preset
    return {
        "preset": preset.name,
        "dataset": preset.dataset,
        "kind": request.kind,
        "filterbank_version": FILTERBANK_VERSION if request.kind == "filterbank" else None,
        "raw_version": RAW_VERSION if request.kind == "raw" else None,
        "subject": request.subject,
        "window": request.window,
        "channels": list(preset.channels),
        "sampling_rate": preset.spec.sampling_rate,
        "cue_seconds": preset.spec.cue_seconds,
        "latency_seconds": preset.spec.latency_seconds,
        "crop_start_seconds": preset.crop_start_seconds,
        "n_fbs": request.n_fbs,
        "extra_samples": request.extra_samples,
        "samples": request.samples,
        "cache_window": request.cache_window,
        "cache_samples": request.cache_samples,
    }
