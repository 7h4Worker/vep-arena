from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tasks"
    / "benchmark_decision_channel_capacity"
    / "run_extended.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_run_extended", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run_extended = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_extended)


def test_merge_runtime_preserves_existing_and_replaces_retried_cell() -> None:
    existing = pd.DataFrame(
        [
            {"subject": 1, "method": "CCA", "window": 0.1, "seconds": 1.0},
            {"subject": 1, "method": "ETRCA", "window": 0.1, "seconds": 2.0},
        ]
    )
    new_rows = pd.DataFrame(
        [
            {"subject": 1, "method": "etrCA", "window": 0.1, "seconds": 3.0},
            {"subject": 1, "method": "TRCA", "window": 0.1, "seconds": 4.0},
        ]
    )

    merged = run_extended.merge_runtime(existing, new_rows)

    assert len(merged) == 3
    values = merged.set_index(["subject", "method", "window"])["seconds"].to_dict()
    assert values[(1, "CCA", 0.1)] == 1.0
    assert values[(1, "ETRCA", 0.1)] == 3.0
    assert values[(1, "TRCA", 0.1)] == 4.0


def test_window_grids_have_expected_overlap() -> None:
    coarse = set(run_extended.coarse_windows())
    fine = set(run_extended.fine_windows())

    assert len(coarse) == 50
    assert len(fine) == 39
    assert coarse & fine == {0.1, 0.3}
