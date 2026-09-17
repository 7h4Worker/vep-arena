import numpy as np

from vep_arena.methods.xtrca import xTRCA, xtrca_filter


def _epochs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(37)
    samples = 125
    time = np.arange(samples) / 250
    epochs = []
    labels = []
    for cls, frequency in enumerate((10.0, 12.0)):
        for shift in (-3, 0, 3):
            signal = np.vstack(
                [np.sin(2 * np.pi * frequency * time), np.cos(2 * np.pi * frequency * time), np.zeros(samples)]
            )
            signal = np.roll(signal, shift, axis=-1)
            epochs.append((signal + 0.1 * rng.standard_normal(signal.shape))[None])
            labels.append(cls)
    return np.asarray(epochs), np.asarray(labels)


def test_xtrca_filter_optimizes_centered_shifts() -> None:
    x, y = _epochs()
    spatial_filter, shifts, aligned = xtrca_filter(x[y == 0, 0], search_samples=4)
    assert spatial_filter.shape == (3,)
    assert shifts.shape == (3,)
    assert aligned.shape == (3, 3, 125)
    assert abs(np.mean(shifts)) < 0.51


def test_xtrca_fit_predict_contract() -> None:
    x, y = _epochs()
    model = xTRCA(n_fbs=1, search_samples=4).fit(x, y)
    predictions, scores = model.predict(x)
    assert predictions.shape == (6,)
    assert scores.shape == (6, 2)
    assert np.mean(predictions == y) >= 2 / 3
