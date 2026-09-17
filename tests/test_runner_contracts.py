"""Isolated runner bookkeeping tests: synthetic loaders/models, not EEG acceptance."""
from __future__ import annotations
import ast
import os
import time
from pathlib import Path
from types import SimpleNamespace
import argparse
import numpy as np
import pandas as pd
import pytest
from vep_arena.config import BenchmarkSpec
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.run_contract import validate_predictions

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ["run_traditional_benchmark.py", "run_tdca.py", "run_toolbox_ssvep.py"]


@pytest.mark.parametrize("filename", RUNNERS)
def test_old_entrypoints_use_one_shared_implementation(filename):
    tree = ast.parse((ROOT / "scripts" / filename).read_text(encoding="utf-8"))
    definitions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert not definitions & {"summarize", "completed_windows", "completed_units", "write_outputs", "write_report"}
    imports = {alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "vep_arena.evaluation" for alias in node.names}
    assert {"parse_range", "parse_windows", "write_outputs"} <= imports
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(isinstance(call.func, ast.Name) and call.func.id == "prepare_run" for call in calls)
    assert all(len(call.args) >= 2 for call in calls if isinstance(call.func, ast.Name)
               and call.func.id in {"load_existing", "load_existing_rows"})


@pytest.mark.parametrize("filename", RUNNERS)
def test_worker_preserves_split_and_prediction_mapping(filename):
    spec = BenchmarkSpec(subjects=1, blocks=6, classes=2)
    seen = []
    tdca = filename == "run_tdca.py"

    class Model:
        fb_weights = np.ones(1)
        def fit(self, x, y, *args):
            self.train_blocks = set(x[:, 0, 0, 0])
            assert y.tolist() == [0, 1]
        def predict(self, x):
            test_blocks = set(x[:, 0, 0, 0])
            assert not self.train_blocks & test_blocks
            seen.append((self.train_blocks, test_blocks))
            scores = np.eye(2)
            return np.arange(2), scores[:, None, :] if tdca else scores

    class Store:
        def __init__(self, *args):
            pass
        def load_or_create(self, request, **kwargs):
            return np.broadcast_to(np.arange(1, 7)[None, :, None, None, None], (2, 6, 1, 2, 20)).copy()

    class Dataset:
        info = SimpleNamespace(targets=(0, 1), subjects=(1,), blocks=(2, 4), frequencies=(9.0, 13.0),
                               phases=(0.0, 0.0), sampling_rate=250, break_seconds=0.5, latency_seconds=0.14)
        def get_trials(self, subject, blocks, targets, channels, window, **kwargs):
            x = np.broadcast_to(np.repeat(blocks, 2)[:, None, None, None], (4, 1, 2, 20)).copy()
            return SimpleNamespace(x=x)

    namespace = dict(np=np, os=os, time=time, Path=Path, argparse=argparse,
        BenchmarkSpec=lambda: spec, CanonicalEpochStore=Store, EpochRequest=lambda **kw: kw,
        resolve_benchmark_preset=lambda *args: None, benchmark_9ch_default=lambda *args: None,
        make_model=lambda *args: Model(), TDCA=lambda **kwargs: Model(),
        epoch_fingerprint=lambda request: "synthetic-fixture", reference_signals=lambda *args, **kw: None,
        open_dataset=lambda *args, **kwargs: Dataset(), spec_from_info=lambda info: spec,
        itr_bits_per_minute=itr_bits_per_minute)
    source = ast.parse((ROOT / "scripts" / filename).read_text(encoding="utf-8"))
    worker = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "run_subject_window_task")
    exec(compile(ast.Module(body=[worker], type_ignores=[]), filename, "exec"), namespace)
    task = dict(data_root="synthetic", root="synthetic", dataset="beta", epoch_cache="unused", result_dir="unused",
        subject=1, window=0.5, cache_window=1.0, filter_window=1.0, methods=["TRCA"], blocks=[2, 4], targets=[0, 1],
        channels="fixture", harmonics=2, n_fbs=1, n_bands=1, n_components=1, n_delay=1, force_epochs=False,
        save_score_matrices=False, save_model_artifacts=False, save_trca_filters=False)
    result = namespace["run_subject_window_task"](task)
    assert seen == [({4}, {2}), ({2}, {4})]
    assert len(result["trial_rows"]) == 2 and len(result["pred_rows"]) == 4
    validate_predictions(pd.DataFrame(result["trial_rows"]), pd.DataFrame(result["pred_rows"]), 2)
