from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tasks" / "channel" / "CH08_survey_p0p1" / "run_p0_analysis.py"
SPEC = importlib.util.spec_from_file_location("survey_v2_p0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def complete_grid(windows: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for method in MODULE.METHODS:
        for window in windows:
            for subject in MODULE.SUBJECTS:
                for block in MODULE.BLOCKS:
                    for label in range(MODULE.CLASSES):
                        rows.append((method, window, subject, block, label, label))
    return pd.DataFrame(rows, columns=["method", "window", "subject", "block", "true", "pred"])


def test_validate_prediction_grid_accepts_complete_grid() -> None:
    MODULE.validate_prediction_grid(complete_grid((0.75,)), (0.75,))


def test_validate_prediction_grid_rejects_duplicate() -> None:
    frame = complete_grid((0.75,))
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Expected"):
        MODULE.validate_prediction_grid(frame, (0.75,))


def test_analysis_of_perfect_predictions() -> None:
    frame = complete_grid((0.75,))
    source = {0.75: {"run": "test", "sha256": "0" * 64}}
    results, subjects, matrices = MODULE.analyze_predictions(frame, source)
    assert len(results) == len(MODULE.METHODS)
    assert len(subjects) == len(MODULE.METHODS) * len(MODULE.SUBJECTS)
    assert np.allclose(results["accuracy_mean"], 1.0)
    assert np.allclose(results["c_ba_bits_per_trial"], np.log2(MODULE.CLASSES))
    assert np.allclose(results["mi_uniform_bits_per_trial"], np.log2(MODULE.CLASSES))
    assert len(matrices) == 2 * len(MODULE.METHODS)
