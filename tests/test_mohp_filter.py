import numpy as np

from vep_arena.methods.mohp_filter import MultiObjectiveHighPassFilter, mohp_filter


def _epochs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(29)
    samples = 125
    time = np.arange(samples) / 250
    epochs = []
    labels = []
    for cls, frequency in enumerate((10.0, 12.0, 14.0)):
        for _ in range(3):
            common = 0.4 * np.sin(2 * np.pi * 8 * time)
            signal = np.vstack(
                [np.sin(2 * np.pi * frequency * time) + common, np.cos(2 * np.pi * frequency * time) + common, common]
            )
            epochs.append((signal + 0.1 * rng.standard_normal(signal.shape))[None])
            labels.append(cls)
    return np.asarray(epochs), np.asarray(labels)


def test_mohp_filter_has_zero_sum() -> None:
    x, y = _epochs()
    spatial_filter = mohp_filter(
        x[y == 0, 0], [x[y == 1, 0], x[y == 2, 0]], solver="goal_attainment", max_iter=30
    )
    assert spatial_filter.shape == (3,)
    assert abs(np.sum(spatial_filter)) < 1e-8
    assert np.isclose(np.linalg.norm(spatial_filter), 1.0)


def test_mohp_fit_predict_contract() -> None:
    x, y = _epochs()
    model = MultiObjectiveHighPassFilter(n_fbs=1, solver="weighted_sum").fit(x, y)
    predictions, scores = model.predict(x)
    assert predictions.shape == (9,)
    assert scores.shape == (9, 3)
    assert np.mean(predictions == y) >= 2 / 3
    assert np.allclose(np.sum(model.filters, axis=-1), 0.0, atol=1e-8)
