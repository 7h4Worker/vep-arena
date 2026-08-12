from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vep_arena.data.embc_jbhi import (  # noqa: E402
    available_subject_ids,
    load_subject,
    load_target_frequency_pairs,
    resolve_dataset_root,
)


FREQUENCIES = np.arange(11.0, 16.0)
HARMONICS = (1, 2, 3)


def exact_log_power(x: np.ndarray, sampling_rate: int) -> np.ndarray:
    centered = signal.detrend(np.asarray(x, dtype=np.float64), axis=-1, type="constant")
    samples = centered.shape[-1]
    time = np.arange(samples, dtype=np.float64) / sampling_rate
    output = np.empty((*centered.shape[:-1], len(FREQUENCIES), len(HARMONICS)), dtype=np.float64)
    for frequency_index, frequency in enumerate(FREQUENCIES):
        for harmonic_index, harmonic in enumerate(HARMONICS):
            kernel = np.exp(-2j * np.pi * harmonic * frequency * time)
            amplitude = 2.0 * np.abs(np.einsum("...t,t->...", centered, kernel)) / samples
            output[..., frequency_index, harmonic_index] = 10.0 * np.log10(amplitude**2 + 1e-18)
    return output


def row_correlation(test: np.ndarray, templates: np.ndarray) -> np.ndarray:
    test_centered = test - test.mean(axis=1, keepdims=True)
    template_centered = templates - templates.mean(axis=1, keepdims=True)
    numerator = test_centered @ template_centered.T
    denominator = np.linalg.norm(test_centered, axis=1, keepdims=True) * np.linalg.norm(
        template_centered, axis=1, keepdims=True
    ).T
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def leave_one_block_accuracy(features: np.ndarray) -> float:
    targets, blocks = features.shape[:2]
    true = np.arange(targets)
    correct = 0
    total = 0
    for block in range(blocks):
        templates = np.delete(features, block, axis=1).mean(axis=1).reshape(targets, -1)
        test = features[:, block].reshape(targets, -1)
        predicted = np.argmax(row_correlation(test, templates), axis=1)
        correct += int(np.sum(predicted == true))
        total += targets
    return correct / total


def expected_frequency_effect(power: np.ndarray, pairs: tuple[tuple[float, float], ...], channel: int | None) -> float:
    fundamental = power[..., 0]
    if channel is None:
        values = fundamental.mean(axis=2)
    else:
        values = fundamental[:, :, channel]
    present_values = []
    absent_values = []
    for target, pair in enumerate(pairs):
        active = {frequency for frequency in pair if frequency > 0}
        for frequency_index, frequency in enumerate(FREQUENCIES):
            destination = present_values if frequency in active else absent_values
            destination.extend(values[target, :, frequency_index].tolist())
    return float(np.mean(present_values) - np.mean(absent_values))


def unordered_code_accuracy(power: np.ndarray, pairs: tuple[tuple[float, float], ...]) -> float:
    fundamental = power[..., 0].mean(axis=2)
    correct = 0
    total = 0
    for target, pair in enumerate(pairs):
        expected = tuple(sorted({frequency for frequency in pair if frequency > 0}))
        cardinality = len(expected)
        for block in range(fundamental.shape[1]):
            indices = np.argsort(fundamental[target, block])[-cardinality:]
            predicted = tuple(sorted(FREQUENCIES[indices].tolist()))
            correct += int(predicted == expected)
            total += 1
    return correct / total


def expected_frequency_itpc_oz(
    x: np.ndarray,
    sampling_rate: int,
    pairs: tuple[tuple[float, float], ...],
    oz_index: int,
) -> float:
    samples = x.shape[-1]
    time = np.arange(samples, dtype=np.float64) / sampling_rate
    centered = signal.detrend(x[:, :, oz_index], axis=-1, type="constant")
    values = []
    for target, pair in enumerate(pairs):
        for frequency in sorted({frequency for frequency in pair if frequency > 0}):
            kernel = np.exp(-2j * np.pi * frequency * time)
            coefficient = centered[target] @ kernel
            phase = np.divide(
                coefficient,
                np.abs(coefficient),
                out=np.zeros_like(coefficient),
                where=np.abs(coefficient) > 1e-12,
            )
            values.append(float(np.abs(np.mean(phase))))
    return float(np.mean(values))


def main() -> None:
    root = resolve_dataset_root("jbhi35")
    pairs = load_target_frequency_pairs("jbhi35", root)
    etrca_path = TASK / "results" / "BS03_35t" / "full" / "subject.csv"
    etrca = pd.read_csv(etrca_path)
    etrca = etrca[(etrca["method"] == "ETRCA") & np.isclose(etrca["window_seconds"], 2.0)]
    etrca_by_subject = dict(zip(etrca["subject"], etrca["accuracy"]))

    rows: list[dict[str, object]] = []
    for subject_id in available_subject_ids("jbhi35", root):
        subject = load_subject("jbhi35", subject_id, root)
        power = exact_log_power(subject.x, subject.sampling_rate)
        temporal = signal.detrend(subject.x, axis=-1, type="constant")
        oz_index = subject.channels.index("Oz")
        rows.append(
            {
                "subject": subject_id,
                "legacy_arena_etrca_accuracy": float(etrca_by_subject[subject_id]),
                "frequency_effect_all_channels_db": expected_frequency_effect(power, pairs, None),
                "frequency_effect_oz_db": expected_frequency_effect(power, pairs, oz_index),
                "expected_frequency_itpc_oz": expected_frequency_itpc_oz(
                    subject.x, subject.sampling_rate, pairs, oz_index
                ),
                "unordered_frequency_set_accuracy": unordered_code_accuracy(power, pairs),
                "spectral_cross_block_label_accuracy": leave_one_block_accuracy(power),
                "temporal_cross_block_label_accuracy": leave_one_block_accuracy(temporal),
            }
        )

    output = TASK / "results" / "jbhi35_subject_quality_audit"
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "subject_quality.csv", index=False)

    metrics = (
        ("legacy_arena_etrca_accuracy", "Legacy Arena eTRCA", "%"),
        ("frequency_effect_oz_db", "Oz expected-frequency effect", "dB"),
        ("expected_frequency_itpc_oz", "Oz expected-frequency ITPC", "unit"),
        ("unordered_frequency_set_accuracy", "Frequency-set agreement", "%"),
        ("spectral_cross_block_label_accuracy", "Spectral label repeatability", "%"),
        ("temporal_cross_block_label_accuracy", "Temporal label repeatability", "%"),
    )
    fig, axes = plt.subplots(1, len(metrics), figsize=(16.5, 4.2), constrained_layout=True)
    for axis, (column, title, unit) in zip(axes, metrics):
        values = table[column].to_numpy()
        if unit == "%":
            values = 100.0 * values
        axis.bar(table["subject"], values, color="#4472C4")
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("JBHI 35-target subject audit: receiver outcome versus signal and label-consistency evidence")
    figure_path = figures / "subject_quality_diagnostics.png"
    fig.savefig(figure_path, dpi=210)
    plt.close(fig)

    manifest = {
        "status": "complete",
        "dataset": "jbhi35",
        "subjects": len(table),
        "identity_policy": "anonymous SNN identifiers only",
        "legacy_result_warning": "The eTRCA column is diagnostic input from the existing Arena result bundle, not a validated reference.",
        "unordered_frequency_set_chance": 1.0 / 7.0,
        "outputs": ["subject_quality.csv", "figures/subject_quality_diagnostics.png"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
