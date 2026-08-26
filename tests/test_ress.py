import numpy as np

from vep_arena.methods.ress import RESS, ress_filter


def _epochs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    sampling_rate = 250
    samples = 250
    time = np.arange(samples) / sampling_rate
    mixing = np.asarray([[1.0, 0.2, -0.1], [0.1, 1.0, 0.3], [0.3, -0.2, 1.0]])
    trials = []
    labels = []
    for cls, frequency in enumerate((10.0, 12.0)):
        for _ in range(2):
            source = np.vstack(
                [np.sin(2 * np.pi * frequency * time), 0.5 * np.sin(4 * np.pi * frequency * time), np.zeros(samples)]
            )
            trial = mixing @ source + 0.15 * rng.standard_normal((3, samples))
            trials.append(trial[None, ...])
            labels.append(cls)
    return np.asarray(trials), np.asarray(labels)


def test_ress_filter_is_finite_for_one_trial() -> None:
    x, _ = _epochs()
    spatial_filter = ress_filter(x[:1, 0], 10.0, 1)
    assert spatial_filter.shape == (3,)
    assert np.all(np.isfinite(spatial_filter))
    assert np.isclose(np.linalg.norm(spatial_filter), 1.0)


def test_ress_fit_predict_contract() -> None:
    x, y = _epochs()
    model = RESS(n_fbs=1, frequencies=(10.0, 12.0)).fit(x, y)
    predictions, scores = model.predict(x)
    assert predictions.shape == (4,)
    assert scores.shape == (4, 2)
    assert np.mean(predictions == y) >= 0.75
