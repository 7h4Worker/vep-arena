import numpy as np

from vep_arena.methods.sinref_trca import SinusoidalReferencedTRCA, sinusoidal_referenced_filter


def _epochs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(31)
    samples = 125
    time = np.arange(samples) / 250
    epochs = []
    labels = []
    for cls, frequency in enumerate((10.0, 12.0)):
        for _ in range(3):
            signal = np.vstack(
                [np.sin(2 * np.pi * frequency * time), np.cos(2 * np.pi * frequency * time), np.zeros(samples)]
            )
            epochs.append((signal + 0.1 * rng.standard_normal(signal.shape))[None])
            labels.append(cls)
    return np.asarray(epochs), np.asarray(labels)


def test_sinusoidal_referenced_filter_is_finite() -> None:
    x, y = _epochs()
    time = np.arange(1, 126) / 250
    reference = np.vstack([np.sin(2 * np.pi * 10 * time), np.cos(2 * np.pi * 10 * time)])
    spatial_filter = sinusoidal_referenced_filter(x[y == 0, 0], reference)
    assert spatial_filter.shape == (3,)
    assert np.all(np.isfinite(spatial_filter))
    assert np.isclose(np.linalg.norm(spatial_filter), 1.0)


def test_sinref_trca_fit_predict_contract() -> None:
    x, y = _epochs()
    model = SinusoidalReferencedTRCA(
        n_fbs=1, frequencies=(10.0, 12.0), harmonics=2
    ).fit(x, y)
    predictions, scores = model.predict(x)
    assert predictions.shape == (6,)
    assert scores.shape == (6, 2)
    assert np.mean(predictions == y) >= 2 / 3
