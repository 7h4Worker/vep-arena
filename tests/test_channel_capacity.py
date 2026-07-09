from __future__ import annotations

import math

import numpy as np

from vep_arena.channel.capacity import (
    capacity_ba,
    capacity_binary_closed,
    capacity_c0,
    capacity_c1,
    capacity_c2_closed,
    mutual_info_uniform,
)
from vep_arena.channel.class_selection import codebook_metrics, greedy_codebook_pruning
from vep_arena.channel.confusion import normalize_confusion


def symmetric_channel(classes: int, accuracy: float) -> np.ndarray:
    off = (1.0 - accuracy) / (classes - 1)
    P = np.full((classes, classes), off, dtype=np.float64)
    np.fill_diagonal(P, accuracy)
    return P


def test_identity_channel_capacity_matches_log_classes() -> None:
    P = np.eye(5)
    expected = math.log2(5)
    assert capacity_c0(5) == expected
    assert capacity_c1(5, 1.0) == expected
    assert abs(mutual_info_uniform(P) - expected) < 1e-10
    assert abs(capacity_ba(P).capacity - expected) < 1e-8
    closed = capacity_c2_closed(P)
    assert closed.valid
    assert abs(closed.capacity - expected) < 1e-8


def test_symmetric_error_channel_c1_matches_ba() -> None:
    P = symmetric_channel(6, 0.8)
    assert abs(capacity_ba(P).capacity - capacity_c1(6, 0.8)) < 1e-6


def test_binary_closed_matches_ba() -> None:
    P = np.array([[0.9, 0.1], [0.25, 0.75]], dtype=np.float64)
    assert abs(capacity_ba(P).capacity - capacity_binary_closed(0.1, 0.25)) < 1e-6


def test_normalize_confusion_row_stochastic() -> None:
    counts = np.array([[2, 1], [0, 3]], dtype=np.int64)
    P = normalize_confusion(counts)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_codebook_pruning_reports_path_metrics() -> None:
    counts = np.array(
        [
            [9, 1, 0],
            [1, 8, 1],
            [0, 2, 8],
        ],
        dtype=np.int64,
    )
    metrics = codebook_metrics(counts, [0, 1], alpha=0.01)
    assert metrics.size == 2
    assert metrics.capacity_ba > 0.0
    selected = greedy_codebook_pruning(counts, alpha=0.01, seed_classes="all")
    assert selected.best_capacity >= 0.0
    assert selected.path[-1].size == 3
    assert selected.path[-1].i_uniform >= 0.0
