from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks.ssvep_efficient_dual_frequency_sun2024.run_comb_candidates import (  # noqa: E402
    BASELINE_BANDS,
    FS,
    build_pair_specs,
    periodic_response,
    phase_aligned_response,
    soft_response,
)
from vep_arena.data.dual_frequency import sun_stimulus_codebook  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def hard_pair_response(frequencies: np.ndarray, centers: tuple[float, ...], half_width: float = 1.5) -> np.ndarray:
    response = np.zeros_like(frequencies)
    for center in centers:
        response[np.abs(frequencies - center) <= half_width] = 1.0
    response[(frequencies < 6.0) | (frequencies > 90.0)] = 0.0
    return response


def component_lines(axis, components: tuple[tuple[float, float], ...]) -> None:
    for frequency, _ in components:
        axis.axvline(frequency, color="0.7", linewidth=0.55, alpha=0.45)


def main() -> None:
    task_root = Path(__file__).resolve().parent
    output = task_root / "results" / "comb_candidates_smoke_20260720" / "figures" / "comb_filter_frequency_responses.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    pair_specs = build_pair_specs(sun_stimulus_codebook(kind="forty_targets"))
    wide = pair_specs[1]
    close = pair_specs[2]
    frequencies = np.linspace(0.0, 100.0, 4001)
    global_centers = tuple(
        sorted({float(freq) for spec in pair_specs.values() for freq, _ in spec["components"]})
    )

    fig, axes = plt.subplots(4, 2, figsize=(14, 13), dpi=170)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (low, high) in enumerate(BASELINE_BANDS):
        sos = signal.butter(4, (low, high), btype="bandpass", fs=FS, output="sos")
        freq, response = signal.sosfreqz(sos, worN=frequencies, fs=FS)
        axes[0, 0].plot(freq, np.abs(response) ** 2, label=f"{low:g}-{high:g} Hz", color=colors[index])
    axes[0, 0].set(title="Standard two-band TRCA baseline (filtfilt magnitude)", xlim=(0, 100), ylim=(-0.03, 1.08))
    axes[0, 0].legend()

    for window, color in ((0.6, colors[1]), (1.0, colors[2])):
        response = soft_response(frequencies, global_centers, sigma_hz=0.5 / window)
        axes[0, 1].plot(frequencies, response, label=f"T={window:.1f} s, sigma={0.5/window:.2f} Hz", color=color)
    axes[0, 1].set(title="Global soft comb magnitude", xlim=(0, 50), ylim=(-0.03, 1.08))
    axes[0, 1].legend()

    for column, (name, spec) in enumerate((("Wide pair 1/6: 14.74 + 10.10 Hz", wide), ("Close pair 2/36: 12.42 + 12.63 Hz", close))):
        centers = tuple(float(freq) for freq, _ in spec["components"])
        soft = soft_response(frequencies, centers, sigma_hz=0.5 / 0.6)
        phase = phase_aligned_response(frequencies, spec["components"], sigma_hz=0.5 / 0.6)
        hard = hard_pair_response(frequencies, centers)
        periodic = periodic_response(frequencies, spec["fundamentals"])

        axes[1, column].plot(frequencies, hard, label="hard +/-1.5 Hz", color="0.45", linewidth=1.0)
        axes[1, column].plot(frequencies, soft, label="soft magnitude", color=colors[1])
        axes[1, column].plot(frequencies, np.abs(phase), label="phase-aligned |H|", color=colors[3])
        component_lines(axes[1, column], spec["components"])
        axes[1, column].set(title=f"{name} - pair responses at 0.6 s", xlim=(6, 50), ylim=(-0.03, 1.08))
        axes[1, column].legend(fontsize=8)

        phase_angle = np.angle(phase)
        phase_angle[np.abs(phase) < 0.03] = np.nan
        axes[2, column].plot(frequencies, phase_angle / np.pi, color=colors[3])
        component_lines(axes[2, column], spec["components"])
        axes[2, column].set(title=f"{name} - phase-aligned angle", xlim=(6, 50), ylim=(-1.05, 1.05), ylabel="Phase / pi")

        axes[3, column].plot(frequencies, periodic, color=colors[4])
        component_lines(axes[3, column], spec["components"])
        axes[3, column].set(title=f"{name} - periodic comb (power=8)", xlim=(6, 90), ylim=(-0.03, 1.08))

    for axis in axes.flat:
        axis.set_xlabel("Frequency (Hz)")
        if axis not in axes[2, :]:
            axis.set_ylabel("Magnitude")
        axis.grid(alpha=0.22)
    fig.suptitle("Sun2024 diagnostic comb-filter frequency responses", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
