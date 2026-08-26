import numpy as np
import pytest

from vep_arena.methods.strca import sTRCA, temporal_laplacian


def _epochs(trials_per_class: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(19)
    sampling_rate = 250
    samples = 125
    time = np.arange(samples) / sampling_rate
    epochs = []
    labels = []
    for cls, frequency in enumerate((10.0, 12.0)):
        for _ in range(trials_per_class):
            signal = np.vstack(
                [np.sin(2 * np.pi * frequency * time), np.cos(2 * np.pi * frequency * time), np.zeros(samples)]
            )
            epochs.append((signal + 0.1 * rng.standard_normal(signal.shape))[None])
            labels.append(cls)
    return np.asarray(epochs), np.asarray(labels)


def test_temporal_laplacian_factorization() -> None:
    laplacian, temporal_filter = temporal_laplacian(40, 5)
    assert np.allclose(laplacian, temporal_filter @ temporal_filter.T, atol=1e-10)
    assert np.allclose(laplacian @ np.ones(40), 0.0, atol=1e-10)


def test_strca_fit_predict_contract() -> None:
    x, y = _epochs()
    model = sTRCA(n_fbs=1, frequencies=(10.0, 12.0), harmonics=2).fit(x, y)
    predictions, scores = model.predict(x)
    assert predictions.shape == (4,)
    assert scores.shape == (4, 2)
    assert np.mean(predictions == y) >= 0.75


def test_strca_rejects_single_trial_calibration() -> None:
    x, y = _epochs(trials_per_class=1)
    with pytest.raises(ValueError, match="at least two"):
        sTRCA(n_fbs=1, frequencies=(10.0, 12.0), harmonics=2).fit(x, y)
