import numpy as np

from vep_arena.methods.prca import PRCA, periodic_components, synthetic_template


def _epochs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    sampling_rate = 250
    samples = 250
    time = np.arange(samples) / sampling_rate
    trials = []
    labels = []
    for cls, frequency in enumerate((10.0, 12.0)):
        signal = np.vstack(
            [np.sin(2 * np.pi * frequency * time), np.cos(2 * np.pi * frequency * time), np.zeros(samples)]
        )
        trials.append((signal + 0.08 * rng.standard_normal(signal.shape))[None])
        labels.append(cls)
    return np.asarray(trials), np.asarray(labels)


def test_periodic_components_follow_paper_rounding() -> None:
    x, _ = _epochs()
    periods, template = periodic_components(x[:1, 0], 10.0, 250)
    assert periods.shape == (10, 3, 25)
    assert template.shape == (3, 25)
    assert synthetic_template(template, 257).shape == (3, 257)


def test_prca_supports_single_trial_calibration() -> None:
    x, y = _epochs()
    model = PRCA(n_fbs=1, frequencies=(10.0, 12.0), ensemble=True).fit(x, y)
    predictions, scores = model.predict(x)
    assert predictions.shape == (2,)
    assert scores.shape == (2, 2)
    assert np.array_equal(predictions, y)
    short_predictions, short_scores = model.predict(x[..., :125])
    assert short_predictions.shape == (2,)
    assert short_scores.shape == (2, 2)
