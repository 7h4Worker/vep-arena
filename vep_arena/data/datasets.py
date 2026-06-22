from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vep_arena.config import (
    BENCHMARK_CHANNELS_9,
    BENCHMARK_FREQS,
    BENCHMARK_PHASES_PI,
    DATA_ROOT,
    BenchmarkSpec,
)
from vep_arena.data.benchmark import load_subject_filterbank, load_subject_raw, load_subject_trials
from vep_arena.data.interface import DatasetInfo, TrialBatch, classes_blocks_to_trials, ensure_band_axis


BENCHMARK_CHANNEL_NAMES = (
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2",
    "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "M1", "TP7",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "M2", "P7", "P5",
    "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4",
    "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
)

CHANNEL_PRESETS = {
    "occipital_9ch": BENCHMARK_CHANNELS_9,
}


@dataclass(frozen=True)
class BenchmarkDataset:
    root: Path = DATA_ROOT
    spec: BenchmarkSpec = BenchmarkSpec()
    channels: tuple[int, ...] = BENCHMARK_CHANNELS_9
    id: str = "benchmark_9ch"

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            id=self.id,
            name="Tsinghua Benchmark SSVEP",
            root=self.root,
            subjects=tuple(range(1, self.spec.subjects + 1)),
            blocks=tuple(range(1, self.spec.blocks + 1)),
            targets=tuple(range(self.spec.classes)),
            channels=tuple(BENCHMARK_CHANNEL_NAMES[idx - 1] for idx in self.channels),
            sampling_rate=self.spec.sampling_rate,
            frequencies=tuple(float(x) for x in BENCHMARK_FREQS),
            phases=tuple(float(x) for x in BENCHMARK_PHASES_PI),
            prestim_seconds=self.spec.cue_seconds,
            break_seconds=0.5,
            latency_seconds=self.spec.latency_seconds,
        )

    def channel_indices(self, channels: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(channels, str):
            try:
                return CHANNEL_PRESETS[channels]
            except KeyError as exc:
                raise ValueError(f"Unknown channel preset: {channels}") from exc
        return tuple(int(ch) for ch in channels)

    def load_subject(self, subject: int) -> np.ndarray:
        raw = load_subject_raw(self.root, subject)
        data = raw[np.asarray(self.channels, dtype=np.int64) - 1]
        return np.transpose(data, (3, 2, 0, 1)).copy()

    def get_trials(
        self,
        subject: int,
        blocks: list[int] | tuple[int, ...],
        targets: list[int] | tuple[int, ...],
        channels: str | list[int] | tuple[int, ...] = "occipital_9ch",
        window: float = 1.0,
        preprocess: str = "raw",
        n_bands: int = 1,
    ) -> TrialBatch:
        channel_indices = self.channel_indices(channels)
        zero_blocks = tuple(int(block) - 1 for block in blocks)
        zero_targets = tuple(int(target) for target in targets)
        if preprocess == "raw":
            epochs = load_subject_trials(self.root, subject, window, channels=channel_indices, spec=self.spec)
            x, y = classes_blocks_to_trials(epochs, zero_targets, zero_blocks)
            x = ensure_band_axis(x)
        elif preprocess in {"toolbox_fb", "filterbank"}:
            epochs = load_subject_filterbank(self.root, subject, window, n_fbs=n_bands, channels=channel_indices, spec=self.spec)
            x, y = classes_blocks_to_trials(epochs, zero_targets, zero_blocks)
        else:
            raise ValueError(f"Unknown preprocess: {preprocess}")
        return TrialBatch(
            x=x,
            y=y,
            subject=subject,
            blocks=tuple(int(block) for block in blocks),
            targets=zero_targets,
            window=window,
            preprocess=preprocess,
            info=self.info,
        )


def benchmark_9ch(root: Path = DATA_ROOT) -> BenchmarkDataset:
    return BenchmarkDataset(root=root)


def beta_9ch(root: Path = Path("D:/ProjData/datasets/ssvep_beta")):
    from vep_arena.data.beta import BetaDataset

    return BetaDataset(root=root)
