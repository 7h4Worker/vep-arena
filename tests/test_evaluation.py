from __future__ import annotations

import numpy as np
import pandas as pd

from vep_arena.config import BenchmarkSpec
from vep_arena.evaluation import (
    completed_units,
    completed_windows,
    parse_range,
    parse_windows,
    summarize,
)


def test_parse_range_single():
    assert parse_range("5") == [5]


def test_parse_range_span():
    assert parse_range("1-5") == [1, 2, 3, 4, 5]


def test_parse_range_mixed():
    assert parse_range("1-3,7,10-12") == [1, 2, 3, 7, 10, 11, 12]


def test_parse_windows_default():
    result = parse_windows("default")
    assert result == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_parse_windows_colon():
    result = parse_windows("0.2:0.2:0.6")
    assert len(result) == 3
    np.testing.assert_allclose(result, [0.2, 0.4, 0.6], atol=1e-9)


def test_parse_windows_csv():
    result = parse_windows("0.5,1.0,1.5")
    assert result == [0.5, 1.0, 1.5]


def _make_trial_rows(methods, windows, subjects, blocks):
    rows = []
    for m in methods:
        for w in windows:
            for s in subjects:
                for b in blocks:
                    rows.append(
                        {
                            "method": m,
                            "window": w,
                            "subject": s,
                            "block": b,
                            "accuracy": 0.8,
                            "itr": 50.0,
                            "samples": 40,
                            "seconds": 1.0,
                        }
                    )
    return rows


def test_summarize_shape():
    rows = _make_trial_rows(["CCA", "TRCA"], [0.5, 1.0], [1, 2], [1, 2])
    trials = pd.DataFrame(rows)
    spec = BenchmarkSpec(subjects=2, blocks=2, classes=40)
    summary, subject_df, block_df = summarize(trials, spec)
    assert set(summary.columns) >= {"method", "window", "accuracy", "itr", "accuracy_sem"}
    assert len(summary) == 4  # 2 methods x 2 windows
    assert len(subject_df) == 8  # 2 methods x 2 windows x 2 subjects
    assert len(block_df) == 8  # 2 methods x 2 windows x 2 blocks


def test_completed_windows_full():
    rows = _make_trial_rows(["CCA"], [0.5], [1, 2], [1, 2])
    done = completed_windows(rows, ["CCA"], [1, 2], [1, 2])
    assert 0.5 in done
    assert "CCA" in done[0.5]


def test_completed_windows_partial():
    rows = _make_trial_rows(["CCA"], [0.5], [1], [1, 2])
    done = completed_windows(rows, ["CCA"], [1, 2], [1, 2])
    assert done == {}


def test_completed_windows_empty():
    assert completed_windows([], ["CCA"], [1], [1]) == {}


def test_completed_units_full():
    rows = _make_trial_rows(["TDCA"], [0.5], [1, 2], [1, 2, 3])
    done = completed_units(rows, [1, 2], [1, 2, 3])
    assert (0.5, 1) in done
    assert (0.5, 2) in done


def test_completed_units_empty():
    assert completed_units([], [1], [1]) == set()
