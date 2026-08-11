from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


TASK_ROOT = Path(__file__).resolve().parents[1] / "tasks" / "benchmark_multichannel_decision_channel_capacity"
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

import analyze
import channel_configs
import common
import run


def test_channel_configs_have_expected_counts_and_nesting() -> None:
    configs = channel_configs.CONFIG_BY_SLUG
    assert {slug: config.channels for slug, config in configs.items()} == {
        "occipital9": 9,
        "posterior21": 21,
        "posterior32": 32,
        "wholehead32": 32,
        "full64": 64,
    }
    assert set(configs["occipital9"].indices_1based) < set(configs["posterior21"].indices_1based)
    assert set(configs["posterior21"].indices_1based) < set(configs["posterior32"].indices_1based)
    assert set(configs["posterior32"].indices_1based) < set(configs["full64"].indices_1based)
    assert set(configs["posterior32"].indices_1based) != set(configs["wholehead32"].indices_1based)


def test_coarse_grid_and_confusion_path_contract(tmp_path: Path) -> None:
    assert len(common.WINDOWS) == 50
    assert common.WINDOWS[0] == 0.1
    assert common.WINDOWS[-1] == 5.0
    path = common.confusion_path(
        tmp_path,
        1,
        "CCA",
        channel_configs.CONFIG_BY_SLUG["posterior32"],
        0.1,
    )
    assert path.name == "confusion_S01_CCA_posterior32_32ch_w0.1.npy"


def test_identity_confusion_has_full_ba_capacity() -> None:
    metrics = analyze.analyze_counts(np.eye(common.SPEC.classes, dtype=np.int64) * common.SPEC.blocks)
    assert metrics["samples"] == common.ROWS_PER_CELL
    assert metrics["accuracy"] == 1.0
    assert metrics["c_ba"] == pytest.approx(np.log2(common.SPEC.classes))
    assert metrics["i_uniform"] == pytest.approx(np.log2(common.SPEC.classes))
    assert metrics["delta_asm"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["ba_converged"] is True


def test_resolved_sweep_errors_are_pruned(tmp_path: Path) -> None:
    config = channel_configs.CONFIG_BY_SLUG["posterior21"]
    errors = pd.DataFrame(
        [
            {
                "subject": 1,
                "method": "CCA",
                "channel_config": config.slug,
                "channels": config.channels,
                "window": 0.1,
                "error_type": "RuntimeError",
                "error": "transient",
                "updated_at_utc": "2026-08-04T00:00:00+00:00",
            }
        ],
        columns=run.ERROR_COLUMNS,
    )
    path = common.confusion_path(tmp_path, 1, "CCA", config, 0.1)
    common.atomic_npy(path, np.eye(common.SPEC.classes, dtype=np.int64) * common.SPEC.blocks)
    assert run.prune_resolved_errors(errors, tmp_path).empty
