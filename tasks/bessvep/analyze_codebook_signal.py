from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.feature_selection import mutual_info_classif


TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vep_arena.data.embc_jbhi import (  # noqa: E402
    PrivateSSVEPSubject,
    available_subject_ids,
    dataset_spec,
    load_subject,
    load_target_frequency_pairs,
    resolve_dataset_root,
)
from vep_arena.neuroviz.mne_bridge import make_info  # noqa: E402


DATASETS = ("embc9", "jbhi16", "jbhi35")
LABELS = {"embc9": "EMBC 9", "jbhi16": "JBHI 16", "jbhi35": "JBHI 35"}
WINDOW_SECONDS = 2.0


def spectral_power(x: np.ndarray, fs: int, frequencies: tuple[float, ...]) -> np.ndarray:
    centered = signal.detrend(np.asarray(x, dtype=np.float64), axis=-1, type="constant")
    samples = centered.shape[-1]
    time = np.arange(samples, dtype=np.float64) / fs
    output = np.empty((*centered.shape[:-1], len(frequencies)), dtype=np.float64)
    for index, frequency in enumerate(frequencies):
        power = np.zeros(centered.shape[:-1], dtype=np.float64)
        for harmonic in (1, 2, 3):
            harmonic_frequency = harmonic * frequency
            if harmonic_frequency >= fs / 2:
                continue
            kernel = np.exp(-2j * np.pi * harmonic_frequency * time)
            amplitude = 2.0 * np.abs(np.einsum("...t,t->...", centered, kernel)) / samples
            power += amplitude**2
        output[..., index] = 10.0 * np.log10(power + 1e-18)
    return output


def within_subject_zscore(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1, values.shape[-2] * values.shape[-1])
    scale = flat.std(axis=0, ddof=1)
    scale[scale < 1e-12] = 1.0
    return ((flat - flat.mean(axis=0)) / scale).reshape(values.shape)


def frequency_state(pair: tuple[float, float], frequency: float) -> int:
    left = pair[0] == frequency
    right = pair[1] == frequency
    return int(left) + 2 * int(right)


def unfiltered_analysis_epochs(subject: PrivateSSVEPSubject, window_seconds: float) -> np.ndarray:
    spec = dataset_spec(subject.dataset)
    samples = int(round(window_seconds * spec.analysis_sampling_rate))
    if subject.dataset == "embc9":
        decimation = subject.sampling_rate // spec.analysis_sampling_rate
        return subject.x[..., ::decimation][..., :samples]
    return subject.x[..., :samples]


def extract_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, tuple[float, ...]]:
    root = resolve_dataset_root(dataset)
    pairs = load_target_frequency_pairs(dataset, root)
    frequencies = tuple(sorted({value for pair in pairs for value in pair if value > 0}))
    effect_rows: list[dict[str, object]] = []
    mi_rows: list[dict[str, object]] = []
    target_signatures: list[np.ndarray] = []
    for subject_id in available_subject_ids(dataset, root):
        subject = load_subject(dataset, subject_id, root)
        epochs = unfiltered_analysis_epochs(subject, WINDOW_SECONDS)
        power = spectral_power(epochs, subject.sampling_rate if dataset != "embc9" else 250, frequencies)
        standardized = within_subject_zscore(power)
        target_signatures.append(standardized.mean(axis=1).reshape(len(pairs), -1))
        for frequency_index, frequency in enumerate(frequencies):
            states_by_target = np.asarray([frequency_state(pair, frequency) for pair in pairs], dtype=np.int64)
            states = np.repeat(states_by_target, power.shape[1])
            feature = power[..., frequency_index].reshape(-1, power.shape[2])
            present = states > 0
            left_only = states == 1
            right_only = states == 2
            for channel_index, channel in enumerate(subject.channels):
                values = feature[:, channel_index]
                present_effect = float(values[present].mean() - values[~present].mean())
                eye_effect = float(values[left_only].mean() - values[right_only].mean())
                mi = float(
                    mutual_info_classif(
                        values[:, None], states, discrete_features=False, n_neighbors=3, random_state=0
                    )[0]
                )
                effect_rows.append(
                    {
                        "dataset": dataset,
                        "subject": subject_id,
                        "frequency_hz": frequency,
                        "channel": channel,
                        "present_minus_absent_db": present_effect,
                        "left_minus_right_db": eye_effect,
                    }
                )
                mi_rows.append(
                    {
                        "dataset": dataset,
                        "subject": subject_id,
                        "frequency_hz": frequency,
                        "channel": channel,
                        "state_mi_bits": mi,
                        "state_definition": "0=absent,1=left,2=right,3=both",
                    }
                )
    return pd.DataFrame(effect_rows), pd.DataFrame(mi_rows), np.mean(target_signatures, axis=0), frequencies


def plot_topomaps(dataset: str, effects: pd.DataFrame, mi: pd.DataFrame, frequencies: tuple[float, ...], output: Path) -> None:
    channels = tuple(effects["channel"].drop_duplicates())
    info = make_info(channels, 1000, montage="standard_1005")
    rows = 3
    columns = len(frequencies)
    fig, axes = plt.subplots(rows, columns, figsize=(3.05 * columns, 8.1), squeeze=False, constrained_layout=True)
    present_maps = []
    eye_maps = []
    mi_maps = []
    for frequency in frequencies:
        effect_group = effects[effects["frequency_hz"] == frequency].groupby("channel", sort=False).mean(numeric_only=True)
        mi_group = mi[mi["frequency_hz"] == frequency].groupby("channel", sort=False).mean(numeric_only=True)
        present_maps.append(effect_group.loc[list(channels), "present_minus_absent_db"].to_numpy())
        eye_maps.append(effect_group.loc[list(channels), "left_minus_right_db"].to_numpy())
        mi_maps.append(mi_group.loc[list(channels), "state_mi_bits"].to_numpy())
    present_limit = max(abs(np.concatenate(present_maps)).max(), 1e-6)
    eye_limit = max(abs(np.concatenate(eye_maps)).max(), 1e-6)
    mi_limit = max(np.concatenate(mi_maps).max(), 1e-6)
    row_specs = (
        (present_maps, "RdBu_r", (-present_limit, present_limit), "Present − absent (dB)"),
        (eye_maps, "RdBu_r", (-eye_limit, eye_limit), "Left − right only (dB)"),
        (mi_maps, "viridis", (0.0, mi_limit), "I(state; feature) (bits)"),
    )
    for row, (maps, cmap, limits, row_label) in enumerate(row_specs):
        image = None
        for column, (frequency, values) in enumerate(zip(frequencies, maps)):
            image, _ = mne.viz.plot_topomap(
                values,
                info,
                axes=axes[row, column],
                show=False,
                contours=0,
                cmap=cmap,
                vlim=limits,
                sensors=True,
            )
            if row == 0:
                axes[row, column].set_title(f"{frequency:g} Hz")
            if column == 0:
                axes[row, column].set_ylabel(row_label)
        fig.colorbar(image, ax=axes[row, :], shrink=0.72, pad=0.02)
    fig.suptitle(f"{LABELS[dataset]}: spectral unit topographies, {WINDOW_SECONDS:g} s")
    fig.savefig(output, dpi=210)
    plt.close(fig)


def pooled_confusion(dataset: str) -> np.ndarray:
    path = PROJECT / "tasks" / f"ssvep_{dataset.replace('embc9', 'embc_9target').replace('jbhi16', 'jbhi_16target').replace('jbhi35', 'jbhi_35target')}_baselines" / "results" / "full" / "predictions.csv"
    predictions = pd.read_csv(path)
    group = predictions[(predictions["method"] == "EFUSIONCA") & (predictions["window_seconds"] == 2.0)]
    targets = int(max(group["true"].max(), group["pred"].max()))
    matrix = np.zeros((targets, targets), dtype=np.float64)
    np.add.at(matrix, (group["true"].to_numpy(dtype=int) - 1, group["pred"].to_numpy(dtype=int) - 1), 1)
    return np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) > 0)


def plot_similarity_confusion(dataset: str, signatures: np.ndarray, output: Path) -> dict[str, object]:
    similarity = np.corrcoef(signatures)
    confusion = pooled_confusion(dataset)
    symmetric_confusion = (confusion + confusion.T) / 2.0
    mask = ~np.eye(similarity.shape[0], dtype=bool)
    rho, pvalue = stats.spearmanr(similarity[mask], symmetric_confusion[mask])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)
    first = axes[0].imshow(similarity, vmin=-1, vmax=1, cmap="coolwarm", aspect="equal")
    second = axes[1].imshow(confusion, vmin=0, vmax=1, cmap="viridis", aspect="equal")
    axes[0].set_title("Target spectral-spatial similarity")
    axes[1].set_title("eFusion confusion at 2 s")
    for axis in axes:
        axis.set_xlabel("Target")
        axis.set_ylabel("Target")
    fig.colorbar(first, ax=axes[0], fraction=0.046)
    fig.colorbar(second, ax=axes[1], fraction=0.046)
    fig.suptitle(f"{LABELS[dataset]}: signal geometry vs receiver errors (Spearman ρ={rho:.3f})")
    fig.savefig(output, dpi=210)
    plt.close(fig)
    return {"dataset": dataset, "spearman_rho": float(rho), "pvalue": float(pvalue), "off_diagonal_pairs": int(mask.sum())}


def main() -> None:
    output = TASK / "results" / "signal"
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    all_effects = []
    all_mi = []
    similarity_rows = []
    figure_names = []
    for dataset in DATASETS:
        effects, mi, signatures, frequencies = extract_dataset(dataset)
        all_effects.append(effects)
        all_mi.append(mi)
        topo_name = f"{dataset}_frequency_unit_topomaps.png"
        comparison_name = f"{dataset}_signal_similarity_vs_confusion.png"
        plot_topomaps(dataset, effects, mi, frequencies, figures / topo_name)
        similarity_rows.append(plot_similarity_confusion(dataset, signatures, figures / comparison_name))
        figure_names.extend([topo_name, comparison_name])
    effects = pd.concat(all_effects, ignore_index=True)
    mi = pd.concat(all_mi, ignore_index=True)
    effects.to_csv(tables / "frequency_unit_topography_subject.csv", index=False)
    mi.to_csv(tables / "frequency_unit_mi_subject.csv", index=False)
    pd.DataFrame(similarity_rows).to_csv(tables / "signal_similarity_confusion.csv", index=False)
    manifest = {
        "status": "complete",
        "window_seconds": WINDOW_SECONDS,
        "feature": "three-harmonic exact-frequency log power",
        "receiver_filterbank": "not applied; upstream-preprocessed analysis epochs only",
        "mi_definition": "per-subject kNN MI between scalar channel-frequency power and four-state left/right presence",
        "figures": figure_names,
        "tables": ["frequency_unit_topography_subject.csv", "frequency_unit_mi_subject.csv", "signal_similarity_confusion.csv"],
        "identity_policy": "anonymous SNN ids only",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
