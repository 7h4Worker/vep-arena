from __future__ import annotations

import numpy as np

from vep_arena.methods.bprca import BPRCA, _period_segments, _prca_filter
from vep_arena.methods.fusionca import FusionCA
from vep_arena.methods.trca_core import _solve_trca


def test_period_segments_preserve_trial_then_period_order() -> None:
    trials = np.arange(2 * 3 * 10, dtype=np.float64).reshape(2, 3, 10)
    segments = _period_segments(trials, 4)
    assert segments.shape == (4, 3, 4)
    np.testing.assert_array_equal(segments[0], trials[0, :, :4])
    np.testing.assert_array_equal(segments[1], trials[0, :, 4:8])
    np.testing.assert_array_equal(segments[2], trials[1, :, :4])


def test_prca_filter_returns_normalized_filter_and_period_prototype() -> None:
    rng = np.random.default_rng(7)
    trials = rng.normal(size=(4, 3, 100))
    spatial_filter, prototype = _prca_filter(trials, 20)
    assert spatial_filter.shape == (3,)
    assert prototype.shape == (3, 20)
    np.testing.assert_allclose(np.linalg.norm(spatial_filter), 1.0)


def test_prca_filter_matches_matlab_pairwise_covariance_definition() -> None:
    rng = np.random.default_rng(9)
    trials = rng.normal(size=(3, 4, 80))
    period_samples = 16
    segments = _period_segments(trials, period_samples)
    centered_segments = segments - np.mean(segments, axis=-1, keepdims=True)
    between = np.zeros((4, 4), dtype=np.float64)
    for first in range(centered_segments.shape[0] - 1):
        for second in range(first + 1, centered_segments.shape[0]):
            between += (
                centered_segments[first] @ centered_segments[second].T
                + centered_segments[second] @ centered_segments[first].T
            )
    segment_matrix = centered_segments.transpose(1, 0, 2).reshape(4, -1)
    expected_filter = _solve_trca(between, segment_matrix @ segment_matrix.T)

    actual_filter, actual_prototype = _prca_filter(trials, period_samples)
    np.testing.assert_allclose(abs(actual_filter @ expected_filter), 1.0, atol=1e-8)
    np.testing.assert_allclose(actual_prototype, np.mean(segments, axis=0))


def test_bprca_and_fusionca_classify_frequency_reuse_patterns() -> None:
    sampling_rate = 100
    samples = 200
    time = np.arange(samples) / sampling_rate
    rng = np.random.default_rng(11)
    mixes = np.asarray(
        [
            [[1.0, 0.2], [0.1, 0.8]],
            [[0.2, 1.0], [0.8, 0.1]],
            [[0.8, 0.7], [0.6, 0.9]],
        ]
    )
    class_epochs = []
    for mix in mixes:
        trials = []
        for _ in range(6):
            sources = np.stack(
                [np.sin(2 * np.pi * 10.0 * time), np.sin(2 * np.pi * 12.5 * time + 0.4)]
            )
            trials.append(mix @ sources + rng.normal(scale=0.08, size=(2, samples)))
        class_epochs.append(trials)
    epochs = np.asarray(class_epochs)[:, :, None]
    train_x = epochs[:, :5].reshape(15, 1, 2, samples)
    train_y = np.repeat(np.arange(3), 5)
    test_x = epochs[:, 5]

    for ensemble in (False, True):
        for fusion in (False, True):
            model = BPRCA((10.0, 12.5), sampling_rate, n_fbs=1, ensemble=ensemble, fusion=fusion)
            model.fit(train_x, train_y)
            predicted, scores = model.predict(test_x)
            np.testing.assert_array_equal(predicted, np.arange(3))
            assert scores.shape == (3, 3)


def test_fusionca_matches_legacy_bprca_fusion_mode() -> None:
    rng = np.random.default_rng(21)
    train_x = rng.normal(size=(12, 2, 3, 120))
    train_y = np.repeat(np.arange(3), 4)
    test_x = rng.normal(size=(3, 2, 3, 120))

    legacy = BPRCA((10.0, 12.0), 120, n_fbs=2, ensemble=True, fusion=True)
    explicit = FusionCA((10.0, 12.0), 120, n_fbs=2, ensemble=True)
    legacy.fit(train_x, train_y)
    explicit.fit(train_x, train_y)

    legacy_predicted, legacy_scores = legacy.predict(test_x)
    explicit_predicted, explicit_scores = explicit.predict(test_x)
    np.testing.assert_array_equal(explicit_predicted, legacy_predicted)
    np.testing.assert_allclose(explicit_scores, legacy_scores)
    assert explicit.name == "FUSIONCA"
    assert explicit.period_samples == (12, 10, 120)
