from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("D:/ProjData/datasets/ssvep_benchmark")
RESULT_ROOT = PROJECT_ROOT / "results" / "benchmark_9ch"
RUN_ROOT = PROJECT_ROOT / "runs"


BENCHMARK_CHANNELS_9 = (48, 54, 55, 56, 57, 58, 61, 62, 63)
BENCHMARK_FREQS = (
    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
    8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
    8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
    8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8,
)
BENCHMARK_PHASES_PI = (
    0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 1.5,
    0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 1.5, 0.0,
    1.0, 1.5, 0.0, 0.5, 1.0, 1.5, 0.0, 0.5,
    1.5, 0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0,
    0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 1.5,
)
WINDOWS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class BenchmarkSpec:
    subjects: int = 35
    blocks: int = 6
    classes: int = 40
    sampling_rate: int = 250
    cue_seconds: float = 0.5
    latency_seconds: float = 0.14

    def sample_length(self, seconds: float) -> int:
        return round(self.sampling_rate * seconds)

    def sample_slice(self, seconds: float) -> slice:
        start = round((self.cue_seconds + self.latency_seconds) * self.sampling_rate)
        return slice(start, start + self.sample_length(seconds))
