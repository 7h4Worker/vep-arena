from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.config import DATA_ROOT
from vep_arena.data.datasets import BENCHMARK_CHANNEL_NAMES


TASK_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = TASK_ROOT / "figures_channel_configs_v20260804"
MONTAGE_PATH = DATA_ROOT / "64-channels.loc"
MONTAGE_HEAD_SIZE_M = 0.095

INDEX_BY_NAME = {name: index for index, name in enumerate(BENCHMARK_CHANNEL_NAMES, start=1)}

EXISTING_9 = (48, 54, 55, 56, 57, 58, 61, 62, 63)
POSTERIOR_21 = tuple(range(44, 65))
POSTERIOR_32_ADDITIONS = (
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
)
POSTERIOR_32 = tuple(sorted(set(POSTERIOR_21) | set(POSTERIOR_32_ADDITIONS)))
WHOLE_HEAD_SPARSE_32_NAMES = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7",
    "FC3", "FCZ", "FC4", "FT8", "T7", "C3", "CZ", "C4",
    "T8", "TP7", "CP3", "CPZ", "CP4", "TP8", "P7", "P3",
    "PZ", "P4", "P8", "PO3", "PO4", "O1", "OZ", "O2",
)
WHOLE_HEAD_SPARSE_32 = tuple(INDEX_BY_NAME[name] for name in WHOLE_HEAD_SPARSE_32_NAMES)
FULL_64 = tuple(range(1, 65))

CONFIGS = {
    "existing_occipital_9": {
        "indices_1based": EXISTING_9,
        "role": "existing fixed 9ch baseline",
        "nested_parent": None,
    },
    "posterior_21_nested": {
        "indices_1based": POSTERIOR_21,
        "role": "parieto-occipital expansion",
        "nested_parent": "existing_occipital_9",
    },
    "posterior_32_nested": {
        "indices_1based": POSTERIOR_32,
        "role": "posterior-dense nested expansion",
        "nested_parent": "posterior_21_nested",
    },
    "whole_head_sparse_32": {
        "indices_1based": WHOLE_HEAD_SPARSE_32,
        "role": "whole-head sparse spacing control",
        "nested_parent": None,
    },
    "full_64": {
        "indices_1based": FULL_64,
        "role": "all recorded channels",
        "nested_parent": "posterior_32_nested",
    },
}


def channel_names(indices: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(BENCHMARK_CHANNEL_NAMES[index - 1] for index in indices)


def benchmark_montage() -> mne.channels.DigMontage:
    if not MONTAGE_PATH.exists():
        raise FileNotFoundError(f"Benchmark montage not found: {MONTAGE_PATH}")
    original = mne.channels.read_custom_montage(
        MONTAGE_PATH,
        head_size=MONTAGE_HEAD_SIZE_M,
    )
    original_names = tuple(name.upper() for name in original.ch_names)
    if original_names != BENCHMARK_CHANNEL_NAMES:
        raise ValueError(
            "Benchmark montage channel order does not match BENCHMARK_CHANNEL_NAMES. "
            f"montage={original_names}, expected={BENCHMARK_CHANNEL_NAMES}"
        )
    positions = original.get_positions()
    original_by_upper = {name.upper(): position for name, position in positions["ch_pos"].items()}
    ch_pos = {name: original_by_upper[name] for name in BENCHMARK_CHANNEL_NAMES}
    return mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        coord_frame="head",
    )


def plot_config(ax: plt.Axes, montage: mne.channels.DigMontage, title: str, indices: tuple[int, ...], show_names: bool) -> None:
    names = channel_names(indices)
    info = mne.create_info(names, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage)
    mne.viz.plot_sensors(
        info,
        kind="topomap",
        ch_type="eeg",
        title=None,
        show_names=show_names,
        pointsize=32,
        linewidth=1.5,
        axes=ax,
        show=False,
    )
    for collection in ax.collections:
        collection.set_clip_on(True)
    for line in ax.lines:
        line.set_clip_on(True)
    ax.set_title(f"{title} ({len(names)}ch)", fontsize=14, pad=12)


def write_manifest() -> Path:
    payload = {
        "status": "final",
        "index_base": 1,
        "benchmark_channel_order": list(BENCHMARK_CHANNEL_NAMES),
        "coordinate_source": {
            "dataset_relative_path": MONTAGE_PATH.name,
            "format": "EEGLAB .loc",
            "mne_loader": "mne.channels.read_custom_montage",
            "head_size_m": MONTAGE_HEAD_SIZE_M,
            "channel_order_verified": True,
        },
        "plot_only_aliases": {},
        "configs": {},
    }
    for key, value in CONFIGS.items():
        indices = tuple(value["indices_1based"])
        payload["configs"][key] = {
            "channels": len(indices),
            "indices_1based": list(indices),
            "names": list(channel_names(indices)),
            "role": value["role"],
            "nested_parent": value["nested_parent"],
        }
    path = OUTPUT_DIR / "channel_config_candidates.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    montage = benchmark_montage()
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), constrained_layout=True)
    panels = (
        ("Existing occipital", EXISTING_9, True),
        ("Nested posterior", POSTERIOR_21, True),
        ("Nested posterior-dense", POSTERIOR_32, False),
        ("Whole-head sparse", WHOLE_HEAD_SPARSE_32, False),
        ("Full recording", FULL_64, False),
    )
    plot_axes = (axes.flat[0], axes.flat[1], axes.flat[2], axes.flat[3], axes.flat[5])
    for ax, (title, indices, show_names) in zip(plot_axes, panels):
        plot_config(ax, montage, title, indices, show_names)

    text_ax = axes.flat[4]
    text_ax.axis("off")
    text_ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=text_ax.transAxes, facecolor="white", edgecolor="none", zorder=10000)
    )
    text_ax.text(
        0.02,
        0.98,
        "Design semantics\n\n"
        "Nested sweep:\n"
        "9ch ⊂ posterior 21ch ⊂ posterior 32ch ⊂ full 64ch\n\n"
        "Control configuration:\n"
        "whole-head sparse 32ch\n"
        "(same channel count, different spatial prior)\n\n"
        "Coordinates: original Benchmark 64-channels.loc\n"
        "CB1/CB2 use their recorded montage positions; no aliases.",
        va="top",
        ha="left",
        fontsize=13,
        linespacing=1.45,
        zorder=10001,
    )
    figure_path = OUTPUT_DIR / "benchmark_channel_config_candidates.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    manifest_path = write_manifest()
    print(figure_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
