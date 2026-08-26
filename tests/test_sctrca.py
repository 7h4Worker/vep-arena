from __future__ import annotations

import numpy as np

from vep_arena.methods.sctrca import scTRCA, similarity_constrained_filter


def _synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    random = np.random.default_rng(7)
    sampling_rate = 250
    samples = 250
    frequencies = (9.0, 13.0)
    time = np.arange(samples) / sampling_rate
    train = []
    labels = []
    test = []
    mixing = np.asarray([[1.0, 0.2], [0.5, 1.0], [-0.3, 0.7]])
    for cls, frequency in enumerate(frequencies):
        source = np.vstack((
            np.sin(2 * np.pi * frequency * time),
            0.5 * np.cos(4 * np.pi * frequency * time),
        ))
        for _ in range(3):
            train.append((mixing @ source + 0.3 * random.normal(size=(3, samples)))[None])
            labels.append(cls)
        test.append((mixing @ source + 0.3 * random.normal(size=(3, samples)))[None])
    return np.asarray(train), np.asarray(labels), np.asarray(test)


def test_similarity_constrained_filter_shapes() -> None:
    train, labels, _ = _synthetic()
    reference = np.vstack((np.sin(np.arange(250)), np.cos(np.arange(250))))
    spatial, reference_filter = similarity_constrained_filter(train[labels == 0, 0], reference)
    assert spatial.shape == (3,)
    assert reference_filter.shape == (2,)
    assert np.isfinite(spatial).all()


def test_sctrca_fit_predict_basic_and_ensemble() -> None:
    train, labels, test = _synthetic()
    for ensemble in (False, True):
        model = scTRCA(
            n_fbs=1, frequencies=(9.0, 13.0), sampling_rate=250,
            harmonics=3, ensemble=ensemble,
        ).fit(train, labels)
        predictions, scores = model.predict(test)
        assert predictions.tolist() == [0, 1]
        assert scores.shape == (2, 2)
        assert model.spatial_filters.shape == (1, 2, 3)
        assert model.reference_filters.shape == (1, 2, 6)
