from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.datasets import BENCHMARK_CHANNEL_NAMES


INDEX_BY_NAME = {name: index for index, name in enumerate(BENCHMARK_CHANNEL_NAMES, start=1)}


@dataclass(frozen=True)
class ChannelConfig:
    slug: str
    label: str
    indices_1based: tuple[int, ...]
    role: str
    nested_parent: str | None

    @property
    def channels(self) -> int:
        return len(self.indices_1based)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(BENCHMARK_CHANNEL_NAMES[index - 1] for index in self.indices_1based)

    def manifest(self) -> dict[str, object]:
        payload = asdict(self)
        payload["indices_1based"] = list(self.indices_1based)
        payload["names"] = list(self.names)
        payload["channels"] = self.channels
        return payload


OCCIPITAL_9 = ChannelConfig(
    slug="occipital9",
    label="Existing occipital 9ch",
    indices_1based=(48, 54, 55, 56, 57, 58, 61, 62, 63),
    role="existing fixed-context baseline",
    nested_parent=None,
)

POSTERIOR_21 = ChannelConfig(
    slug="posterior21",
    label="Nested posterior 21ch",
    indices_1based=tuple(range(44, 65)),
    role="parieto-occipital expansion",
    nested_parent=OCCIPITAL_9.slug,
)

POSTERIOR_32 = ChannelConfig(
    slug="posterior32",
    label="Nested posterior-dense 32ch",
    indices_1based=tuple(
        sorted(
            set(POSTERIOR_21.indices_1based)
            | {
                INDEX_BY_NAME["C3"],
                INDEX_BY_NAME["C4"],
                INDEX_BY_NAME["TP7"],
                INDEX_BY_NAME["CP5"],
                INDEX_BY_NAME["CP3"],
                INDEX_BY_NAME["CP1"],
                INDEX_BY_NAME["CPZ"],
                INDEX_BY_NAME["CP2"],
                INDEX_BY_NAME["CP4"],
                INDEX_BY_NAME["CP6"],
                INDEX_BY_NAME["TP8"],
            }
        )
    ),
    role="posterior-dense nested expansion",
    nested_parent=POSTERIOR_21.slug,
)

WHOLE_HEAD_32_NAMES = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7",
    "FC3", "FCZ", "FC4", "FT8", "T7", "C3", "CZ", "C4",
    "T8", "TP7", "CP3", "CPZ", "CP4", "TP8", "P7", "P3",
    "PZ", "P4", "P8", "PO3", "PO4", "O1", "OZ", "O2",
)
WHOLE_HEAD_32 = ChannelConfig(
    slug="wholehead32",
    label="Whole-head sparse 32ch",
    indices_1based=tuple(INDEX_BY_NAME[name] for name in WHOLE_HEAD_32_NAMES),
    role="same-count spatial-coverage control",
    nested_parent=None,
)

FULL_64 = ChannelConfig(
    slug="full64",
    label="Full recording 64ch",
    indices_1based=tuple(range(1, 65)),
    role="all recorded channels",
    nested_parent=POSTERIOR_32.slug,
)

CHANNEL_CONFIGS = (OCCIPITAL_9, POSTERIOR_21, POSTERIOR_32, WHOLE_HEAD_32, FULL_64)
CONFIG_BY_SLUG = {config.slug: config for config in CHANNEL_CONFIGS}


def validate_channel_configs() -> None:
    expected_counts = {
        "occipital9": 9,
        "posterior21": 21,
        "posterior32": 32,
        "wholehead32": 32,
        "full64": 64,
    }
    if set(CONFIG_BY_SLUG) != set(expected_counts):
        raise ValueError("Channel configuration slugs do not match the experiment contract.")
    for config in CHANNEL_CONFIGS:
        if config.channels != expected_counts[config.slug]:
            raise ValueError(f"{config.slug} contains {config.channels} channels.")
        if len(set(config.indices_1based)) != config.channels:
            raise ValueError(f"{config.slug} contains duplicate channels.")
        if min(config.indices_1based) < 1 or max(config.indices_1based) > len(BENCHMARK_CHANNEL_NAMES):
            raise ValueError(f"{config.slug} contains an out-of-range channel index.")
    if not set(OCCIPITAL_9.indices_1based) < set(POSTERIOR_21.indices_1based):
        raise ValueError("occipital9 must be a strict subset of posterior21.")
    if not set(POSTERIOR_21.indices_1based) < set(POSTERIOR_32.indices_1based):
        raise ValueError("posterior21 must be a strict subset of posterior32.")
    if not set(POSTERIOR_32.indices_1based) < set(FULL_64.indices_1based):
        raise ValueError("posterior32 must be a strict subset of full64.")
    if set(POSTERIOR_32.indices_1based) == set(WHOLE_HEAD_32.indices_1based):
        raise ValueError("The two 32-channel spatial priors must remain distinct.")


validate_channel_configs()
