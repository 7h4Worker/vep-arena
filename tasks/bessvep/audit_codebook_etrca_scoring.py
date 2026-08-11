from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vep_arena.data.embc_jbhi import analysis_epochs, available_subject_ids, load_subject, resolve_dataset_root  # noqa: E402
from vep_arena.methods.traditional import TRCA  # noqa: E402
from vep_arena.methods.traditional import filterbank_weights  # noqa: E402
from vep_arena.methods.trca_core import corr_rows  # noqa: E402
from vep_arena.methods.trca_core import trca_filter  # noqa: E402


def band_correlations(model: TRCA, test_x: np.ndarray) -> np.ndarray:
    if model.templates is None or model.filters is None:
        raise RuntimeError("TRCA is not fitted")
    trials, bands = test_x.shape[:2]
    classes = model.templates.shape[0]
    output = np.zeros((trials, bands, classes), dtype=np.float64)
    for band in range(bands):
        filters = model.filters[band].T
        projected_trials = np.matmul(np.transpose(test_x[:, band], (0, 2, 1)), filters).reshape(trials, -1)
        for target in range(classes):
            projected_template = (model.templates[target, band].T @ filters).reshape(-1)
            output[:, band, target] = corr_rows(projected_trials, projected_template)
    return output


def cascade_filterbank(x: np.ndarray, sampling_rate: int, bands: int) -> np.ndarray:
    filtered = np.asarray(x, dtype=np.float64)
    output = []
    for band in range(bands):
        low = 6.0 + 8.0 * band
        sos = signal.cheby1(6, 0.5, [low, 90.0], btype="bandpass", fs=sampling_rate, output="sos")
        filtered = signal.sosfiltfilt(sos, filtered, axis=-1)
        output.append(filtered)
    return np.stack(output, axis=1)


def fit_historical_cascade(train_raw: np.ndarray, train_y: np.ndarray) -> TRCA:
    classes = int(np.max(train_y)) + 1
    model = TRCA(n_fbs=5, ensemble=True)
    model.weights = filterbank_weights(5)
    model.templates = np.zeros((classes, 5, train_raw.shape[-2], train_raw.shape[-1]), dtype=np.float64)
    model.filters = np.zeros((5, classes, train_raw.shape[-2]), dtype=np.float64)
    for target in range(classes):
        bands = cascade_filterbank(train_raw[train_y == target], 1000, 5)
        model.templates[target] = bands.mean(axis=0)
        for band in range(5):
            model.filters[band, target] = trca_filter(bands[:, band])
    return model


def main() -> None:
    dataset = "jbhi35"
    root = resolve_dataset_root(dataset)
    labels = np.arange(35)
    rows = []
    for subject_id in available_subject_ids(dataset, root):
        subject = load_subject(dataset, subject_id, root)
        epochs = analysis_epochs(subject, 2.0, n_bands=5)
        predictions = {
            name: []
            for name in ("raw", "square", "signed_square", "historical_cascade", "historical_cascade_abs")
        }
        for test_block in range(6):
            train_blocks = [block for block in range(6) if block != test_block]
            train_x = epochs[:, train_blocks].reshape(-1, 5, 9, epochs.shape[-1])
            train_y = np.repeat(labels, 5)
            model = TRCA(n_fbs=5, ensemble=True).fit(train_x, train_y)
            correlations = band_correlations(model, epochs[:, test_block])
            train_raw = subject.x[:, train_blocks].reshape(-1, 9, subject.x.shape[-1])
            historical_model = fit_historical_cascade(train_raw, train_y)
            historical_correlations = band_correlations(historical_model, epochs[:, test_block])
            variants = {
                "raw": np.einsum("f,tfc->tc", model.weights, correlations),
                "square": np.einsum("f,tfc->tc", model.weights, correlations**2),
                "signed_square": np.einsum("f,tfc->tc", model.weights, np.sign(correlations) * correlations**2),
                "historical_cascade": np.einsum(
                    "f,tfc->tc", historical_model.weights, historical_correlations
                ),
                "historical_cascade_abs": np.einsum(
                    "f,tfc->tc", historical_model.weights, np.abs(historical_correlations)
                ),
            }
            for name, scores in variants.items():
                predictions[name].extend(np.argmax(scores, axis=1).tolist())
        true = np.tile(labels, 6)
        for name, predicted in predictions.items():
            rows.append(
                {
                    "dataset": dataset,
                    "subject": subject_id,
                    "window_seconds": 2.0,
                    "scoring": name,
                    "accuracy": float(np.mean(np.asarray(predicted) == true)),
                }
            )
    frame = pd.DataFrame(rows)
    output = TASK / "results" / "etrca_scoring_audit"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "subject.csv", index=False)
    frame.groupby("scoring", as_index=False).agg(accuracy=("accuracy", "mean"), accuracy_sem=("accuracy", "sem")).to_csv(
        output / "summary.csv", index=False
    )
    print(frame.pivot(index="subject", columns="scoring", values="accuracy"))
    print(frame.groupby("scoring")["accuracy"].mean())


if __name__ == "__main__":
    main()
