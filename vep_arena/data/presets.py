from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from vep_arena.config import BENCHMARK_CHANNELS_9, DATA_ROOT, BenchmarkSpec


@dataclass(frozen=True)
class DatasetPreset:
    """Dataset-level timing and channel protocol.

    Algorithms should consume epochs produced from a preset instead of deciding
    cue, latency, and crop offsets locally.
    """

    name: str
    dataset: str
    data_root: Path
    channels: tuple[int, ...]
    spec: BenchmarkSpec
    description: str

    @property
    def crop_start_seconds(self) -> float:
        return self.spec.cue_seconds + self.spec.latency_seconds

    def manifest(self) -> dict[str, object]:
        return {
            "preset": self.name,
            "dataset": self.dataset,
            "data_root": str(self.data_root),
            "channels": list(self.channels),
            "spec": asdict(self.spec),
            "crop_start_seconds": self.crop_start_seconds,
            "description": self.description,
        }


def benchmark_9ch_default(data_root: Path = DATA_ROOT) -> DatasetPreset:
    """Tsinghua Benchmark 9-channel canonical preset.

    Raw trials include 0.5 s pre-stim/cue. The canonical epoch starts at
    cue offset plus the 0.14 s visual latency, i.e. 0.64 s from raw trial start.
    """

    return DatasetPreset(
        name="benchmark_9ch_cue0.5_latency0.14",
        dataset="Tsinghua Benchmark SSVEP",
        data_root=data_root,
        channels=BENCHMARK_CHANNELS_9,
        spec=BenchmarkSpec(),
        description="Canonical Benchmark 9ch epochs: skip 0.5 s cue/pre-stim and 0.14 s visual latency.",
    )


def benchmark_9ch_with_latency(latency_seconds: float, data_root: Path = DATA_ROOT) -> DatasetPreset:
    """Explicit variant for offset ablations; use only when the task calls for it."""

    spec = BenchmarkSpec(latency_seconds=latency_seconds)
    return DatasetPreset(
        name=f"benchmark_9ch_cue0.5_latency{latency_seconds:g}",
        dataset="Tsinghua Benchmark SSVEP",
        data_root=data_root,
        channels=BENCHMARK_CHANNELS_9,
        spec=spec,
        description=f"Benchmark 9ch offset-ablation preset with visual latency {latency_seconds:g} s.",
    )
