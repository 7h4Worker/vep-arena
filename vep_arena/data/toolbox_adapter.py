from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

import numpy as np

from vep_arena.data.interface import DatasetInfo, TrialBatch


TOOLBOX_ENV = "SSVEP_TOOLBOX_PATH"
TOOLBOX_CANDIDATES = (
    Path(os.environ.get(TOOLBOX_ENV, "")),
    Path("D:/ProjData/proj_python/_reference/SSVEP-Analysis-Toolbox"),
    Path(os.environ.get("TEMP", "")) / "ssvep_toolbox_ref",
)
OCCIPITAL_9CH = ("PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2")


@dataclass(frozen=True)
class ToolboxMeta:
    kind: str
    id: str
    name: str
    root: Path
    subjects: tuple[int, ...]
    blocks: tuple[int, ...]
    targets: tuple[int, ...]
    channels: tuple[str, ...]
    sampling_rate: int
    frequencies: tuple[float, ...]
    phases_pi: tuple[float, ...]
    prestim_seconds: float
    break_seconds: float
    latency_seconds: float

    def as_info(self, channels: tuple[str, ...] | None = None) -> DatasetInfo:
        return DatasetInfo(
            id=self.id,
            name=self.name,
            root=self.root,
            subjects=self.subjects,
            blocks=self.blocks,
            targets=self.targets,
            channels=channels or self.channels,
            sampling_rate=self.sampling_rate,
            frequencies=self.frequencies,
            phases=self.phases_pi,
            prestim_seconds=self.prestim_seconds,
            break_seconds=self.break_seconds,
            latency_seconds=self.latency_seconds,
        )


@dataclass
class ToolboxDatasetAdapter:
    toolbox_dataset: object
    meta: ToolboxMeta
    preprocess_fun: Callable[[object, np.ndarray], np.ndarray]
    filterbank_fun: Callable[[object, np.ndarray, int], np.ndarray]
    default_channels: str | tuple[int, ...] = "occipital_9ch"

    @property
    def info(self) -> DatasetInfo:
        names = tuple(self.toolbox_dataset.channels[idx] for idx in self.channel_indices(self.default_channels))
        return self.meta.as_info(channels=names)

    def load_subject(self, subject: int) -> np.ndarray:
        return np.asarray(self.toolbox_dataset.get_sub_data(subject - 1), dtype=np.float64)

    def channel_indices(self, channels: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
        if channels == "all":
            return tuple(range(len(self.toolbox_dataset.channels)))
        if channels == "occipital_9ch":
            idx = [self.toolbox_dataset.get_ch_idx(ch) for ch in OCCIPITAL_9CH]
            missing = [ch for ch, value in zip(OCCIPITAL_9CH, idx) if value is None]
            if missing:
                raise ValueError(f"Dataset {self.meta.id} does not have channels: {missing}")
            return tuple(int(value) for value in idx)
        if isinstance(channels, str):
            raise ValueError(f"Unknown channel preset: {channels}")
        values = tuple(int(ch) for ch in channels)
        if not values:
            raise ValueError("At least one channel is required.")
        if min(values) == 0:
            return values
        return tuple(value - 1 for value in values)

    def set_preprocess(self, preprocess: str, n_bands: int) -> None:
        if preprocess == "raw":
            self.toolbox_dataset.reset_preprocess()
            self.toolbox_dataset.reset_filterbank()
            return
        if preprocess in {"toolbox_preprocess", "preprocess"}:
            self.toolbox_dataset.regist_preprocess(self.preprocess_fun)
            self.toolbox_dataset.reset_filterbank()
            return
        if preprocess in {"toolbox_fb", "filterbank"}:
            self.toolbox_dataset.regist_preprocess(self.preprocess_fun)
            self.toolbox_dataset.regist_filterbank(lambda dataself, x: self.filterbank_fun(dataself, x, n_bands))
            return
        raise ValueError(f"Unknown preprocess: {preprocess}")

    def get_trials(
        self,
        subject: int,
        blocks: list[int] | tuple[int, ...],
        targets: list[int] | tuple[int, ...],
        channels: str | list[int] | tuple[int, ...] = "occipital_9ch",
        window: float = 1.0,
        preprocess: str = "raw",
        n_bands: int = 1,
        filter_window: float | None = None,
        extra_samples: int = 0,
    ) -> TrialBatch:
        zero_blocks = [int(block) - 1 for block in blocks]
        zero_targets = [int(target) for target in targets]
        ch_idx = list(self.channel_indices(channels))
        self.set_preprocess(preprocess, n_bands)
        requested_window = window + extra_samples / self.meta.sampling_rate
        sig_len = max(filter_window or window, requested_window)
        x_list, y_list = self.toolbox_dataset.get_data(
            sub_idx=subject - 1,
            blocks=zero_blocks,
            trials=zero_targets,
            channels=ch_idx,
            sig_len=sig_len,
            t_latency=self.meta.latency_seconds,
            shuffle=False,
        )
        x = np.stack([np.asarray(item, dtype=np.float64) for item in x_list], axis=0)
        samples = int(round(window * self.meta.sampling_rate)) + extra_samples
        x = x[..., :samples]
        y = np.asarray(y_list, dtype=np.int64)
        return TrialBatch(
            x=x,
            y=y,
            subject=subject,
            blocks=tuple(int(block) for block in blocks),
            targets=tuple(zero_targets),
            window=window,
            preprocess=preprocess,
            info=self.meta.as_info(channels=tuple(self.toolbox_dataset.channels[idx] for idx in ch_idx)),
        )


def ensure_toolbox_path(toolbox_path: Path | None = None) -> Path:
    _patch_numpy_compat()
    candidates = (toolbox_path,) if toolbox_path is not None else TOOLBOX_CANDIDATES
    for path in candidates:
        if path and (path / "SSVEPAnalysisToolbox").is_dir():
            resolved = path.resolve()
            if str(resolved) not in sys.path:
                sys.path.insert(0, str(resolved))
            return resolved
    searched = ", ".join(str(path) for path in candidates if path)
    raise ImportError(f"SSVEP-Analysis-Toolbox source not found. Set {TOOLBOX_ENV}. Searched: {searched}")


def toolbox_dataset_info(kind: str, root: Path | None = None, toolbox_path: Path | None = None) -> DatasetInfo:
    return _meta(kind, root=root, toolbox_path=toolbox_path).as_info()


def toolbox_beta(
    root: Path = Path("D:/ProjData/datasets/ssvep_beta"),
    toolbox_path: Path | None = None,
    download: bool = True,
) -> ToolboxDatasetAdapter:
    dataset_cls, benchmarkpreprocess = _toolbox_beta_symbols(toolbox_path)
    dataset = _construct_toolbox_dataset(dataset_cls, root, download)
    return ToolboxDatasetAdapter(
        toolbox_dataset=dataset,
        meta=_meta("beta", root, toolbox_path),
        preprocess_fun=benchmarkpreprocess.preprocess,
        filterbank_fun=benchmarkpreprocess.filterbank,
        default_channels="occipital_9ch",
    )


def toolbox_wearable_wet(
    root: Path = Path("D:/ProjData/datasets/ssvep_wearable"),
    toolbox_path: Path | None = None,
    download: bool = True,
) -> ToolboxDatasetAdapter:
    dataset_cls, wearablepreprocess = _toolbox_wearable_symbols("wet", toolbox_path)
    dataset = _construct_toolbox_dataset(dataset_cls, root, download)
    return ToolboxDatasetAdapter(
        toolbox_dataset=dataset,
        meta=_meta("wearable_wet", root, toolbox_path),
        preprocess_fun=wearablepreprocess.preprocess,
        filterbank_fun=wearablepreprocess.filterbank,
        default_channels="all",
    )


def toolbox_wearable_dry(
    root: Path = Path("D:/ProjData/datasets/ssvep_wearable"),
    toolbox_path: Path | None = None,
    download: bool = True,
) -> ToolboxDatasetAdapter:
    dataset_cls, wearablepreprocess = _toolbox_wearable_symbols("dry", toolbox_path)
    dataset = _construct_toolbox_dataset(dataset_cls, root, download)
    return ToolboxDatasetAdapter(
        toolbox_dataset=dataset,
        meta=_meta("wearable_dry", root, toolbox_path),
        preprocess_fun=wearablepreprocess.preprocess,
        filterbank_fun=wearablepreprocess.filterbank,
        default_channels="all",
    )


def _patch_numpy_compat() -> None:
    if "object" not in np.__dict__:
        setattr(np, "object", object)


def _toolbox_beta_symbols(toolbox_path: Path | None):
    ensure_toolbox_path(toolbox_path)
    dataset_module = import_module("SSVEPAnalysisToolbox.datasets.betadataset")
    preprocess_module = import_module("SSVEPAnalysisToolbox.utils.benchmarkpreprocess")
    return dataset_module.BETADataset, preprocess_module


def _toolbox_wearable_symbols(kind: str, toolbox_path: Path | None):
    ensure_toolbox_path(toolbox_path)
    dataset_module = import_module("SSVEPAnalysisToolbox.datasets.wearabledataset")
    preprocess_module = import_module("SSVEPAnalysisToolbox.utils.wearablepreprocess")
    if kind == "wet":
        return dataset_module.WearableDataset_wet, preprocess_module
    if kind == "dry":
        return dataset_module.WearableDataset_dry, preprocess_module
    raise ValueError(f"Unknown wearable kind: {kind}")


def _construct_toolbox_dataset(dataset_cls, root: Path, download: bool):
    root.mkdir(parents=True, exist_ok=True)
    if download:
        return dataset_cls(path=str(root), path_support_file=str(root))

    basedataset = import_module("SSVEPAnalysisToolbox.datasets.basedataset")
    old_download_all = basedataset.BaseDataset.download_all
    old_download_support_files = basedataset.BaseDataset.download_support_files
    try:
        basedataset.BaseDataset.download_all = lambda self, total_retry_time=10: None
        basedataset.BaseDataset.download_support_files = lambda self, total_retry_time=10: None
        return dataset_cls(path=str(root), path_support_file=str(root))
    finally:
        basedataset.BaseDataset.download_all = old_download_all
        basedataset.BaseDataset.download_support_files = old_download_support_files


def _meta(kind: str, root: Path | None = None, toolbox_path: Path | None = None) -> ToolboxMeta:
    root = root or _default_root(kind)
    if kind == "beta":
        dataset_cls, _ = _toolbox_beta_symbols(toolbox_path)
        return ToolboxMeta(
            kind=kind,
            id="toolbox_beta",
            name="BETA SSVEP via SSVEP-Analysis-Toolbox",
            root=root,
            subjects=tuple(range(1, len(dataset_cls._SUBJECTS) + 1)),
            blocks=tuple(range(1, 5)),
            targets=tuple(range(len(dataset_cls._FREQS))),
            channels=tuple(ch.upper() for ch in dataset_cls._CHANNELS),
            sampling_rate=250,
            frequencies=tuple(float(x) for x in dataset_cls._FREQS),
            phases_pi=tuple(float(x) for x in dataset_cls._PHASES),
            prestim_seconds=0.5,
            break_seconds=0.5,
            latency_seconds=0.13,
        )
    if kind in {"wearable_wet", "wearable_dry"}:
        dataset_cls, _ = _toolbox_wearable_symbols("wet" if kind == "wearable_wet" else "dry", toolbox_path)
        return ToolboxMeta(
            kind=kind,
            id=f"toolbox_{kind}",
            name=f"{kind.replace('_', ' ').title()} SSVEP via SSVEP-Analysis-Toolbox",
            root=root,
            subjects=tuple(range(1, len(dataset_cls._SUBJECTS) + 1)),
            blocks=tuple(range(1, 11)),
            targets=tuple(range(len(dataset_cls._FREQS))),
            channels=tuple(ch.upper() for ch in dataset_cls._CHANNELS),
            sampling_rate=250,
            frequencies=tuple(float(x) for x in dataset_cls._FREQS),
            phases_pi=tuple(float(x) for x in dataset_cls._PHASES),
            prestim_seconds=0.5,
            break_seconds=0.2,
            latency_seconds=0.14,
        )
    raise ValueError(f"Unknown toolbox dataset kind: {kind}")


def _default_root(kind: str) -> Path:
    if kind == "beta":
        return Path("D:/ProjData/datasets/ssvep_beta")
    if kind in {"wearable_wet", "wearable_dry"}:
        return Path("D:/ProjData/datasets/ssvep_wearable")
    raise ValueError(f"Unknown toolbox dataset kind: {kind}")
