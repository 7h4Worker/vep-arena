from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "tasks" / "ssvep_neural_communication_survey_v2" / "run_p1_frontload_analysis.py"
SPEC = importlib.util.spec_from_file_location("survey_v2_p1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_channel_metrics_for_perfect_predictions() -> None:
    labels = pd.Series(np.repeat(np.arange(4), 3))
    metrics = MODULE.channel_metrics(labels, labels, 4)
    assert metrics["accuracy"] == 1.0
    assert np.isclose(metrics["c_ba"], 2.0)
    assert np.isclose(metrics["mi_uniform"], 2.0)
    assert np.isclose(metrics["fano_c1"], 2.0)


def test_select_windows_uses_only_frozen_methods() -> None:
    frame = pd.DataFrame(
        {
            "method": ["CCA", "ECCA", "CCA"],
            "window": [0.1, 0.1, 0.2],
            "subject": [1, 1, 1],
            "block": [1, 1, 1],
            "true": [0, 0, 0],
            "pred": [0, 0, 0],
        }
    )
    selected = MODULE.select_windows(frame, (0.1,))
    assert selected[["method", "window"]].to_dict("records") == [{"method": "CCA", "window": 0.1}]
