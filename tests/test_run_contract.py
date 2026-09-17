"""Synthetic structural regressions, not real EEG or paper validation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from vep_arena import evaluation as ev
from vep_arena.config import BenchmarkSpec, CACHE_ROOT, PROJECT_ROOT, RESULT_ROOT, RUN_ROOT
from vep_arena.run_contract import (atomic_json, audit_run, file_sha256, fingerprint,
    prepare_run, require_same_run, validate_predictions, source_inventory, verify_sources)


def rows(windows=(0.5,), subjects=(1, 2), blocks=(1, 2)):
    trials, preds = [], []
    for window in windows:
        for subject in subjects:
            for block in blocks:
                key = dict(method="TRCA", window=window, subject=subject, block=block)
                trials.append(dict(**key, accuracy=0.5, itr=1.0, samples=2, seconds=0.1))
                preds.extend([dict(**key, true=0, pred=0), dict(**key, true=1, pred=0)])
    return trials, preds


def manifest(windows=(0.5,)):
    return dict(dataset="BETA", protocol="subject-specific leave-one-block-out",
        methods=["TRCA"], windows=list(windows), subjects=[1, 2], blocks=[1, 2], classes=2, label_base=0,
        code={"git_commit": "synthetic-fixture", "dirty": False},
        source_verification="file_hashes_checked", source_fingerprint="synthetic-fixture",
        resolved_config={"harmonics": 5, "resume": False}, preprocessing={"latency_seconds": 0.13})


def write_fixture(root, monkeypatch, windows=(0.5,)):
    monkeypatch.setattr(ev, "plot_outputs", lambda *args: None)
    trial, pred = rows(windows)
    settings = manifest(windows)
    ev.write_outputs(root, root / "models", trial, pred, [], settings, BenchmarkSpec(classes=2), True)
    return settings


def test_legacy_path_aliases():
    assert RESULT_ROOT == PROJECT_ROOT / "results" / "benchmark_9ch"
    assert RUN_ROOT == PROJECT_ROOT / "runs"
    assert CACHE_ROOT == PROJECT_ROOT / ".cache"


@pytest.mark.parametrize("text", ["0.2:0:1", "0.2:-1:1", "0.2:0.1:0.1", "nan", "inf", "0", "", "0.5,0.5", "1:1e-320:2"])
def test_invalid_windows_fail_fast(text):
    with pytest.raises(ValueError):
        ev.parse_windows(text)


def test_window_syntax():
    assert ev.parse_windows(" 0.2:0.2:0.6 ") == [0.2, 0.4, 0.6]
    assert ev.parse_windows("default")[0] == 0.2


@pytest.mark.parametrize("text", ["", "1,1", "3-1", "1-2,2", "-1", "1,"])
def test_invalid_selections(text):
    with pytest.raises(ValueError):
        ev.parse_range(text)


def test_targets_can_be_zero_based():
    assert ev.parse_range("0-2,5") == [0, 1, 2, 5]


def test_actual_sample_count_not_full_dataset():
    trial, _ = rows(subjects=(1,), blocks=(2,))
    summary, subject, block = ev.summarize(pd.DataFrame(trial), BenchmarkSpec())
    assert summary.iloc[0].samples == subject.iloc[0].samples == block.iloc[0].samples == 2


def test_duplicate_rows_rejected():
    trial, _ = rows()
    with pytest.raises(ValueError, match="duplicate"):
        ev.completed_windows([trial[0]] * 4, ["TRCA"], [1, 2], [1, 2])
    with pytest.raises(ValueError, match="duplicate"):
        ev.completed_units([trial[0]] * 2, [1], [1, 2])


def test_wrong_scope_never_counts_as_completed():
    trial, _ = rows()
    assert ev.completed_windows(trial, ["TRCA"], [3, 4], [1, 2]) == {}
    assert ev.completed_windows(trial, ["TRCA"], [1, 2], [3, 4]) == {}
    assert ev.completed_units(trial, [1, 2], [3, 4]) == set()


def test_exact_scope_passes():
    trial, _ = rows()
    assert ev.completed_windows(trial, ["TRCA"], [1, 2], [1, 2]) == {0.5: {"TRCA"}}
    assert ev.completed_units(trial, [1, 2], [1, 2]) == {(0.5, 1), (0.5, 2)}


def test_mixed_methods_not_silently_combined():
    trial, _ = rows()
    trial[1]["method"] = "CCA"
    with pytest.raises(ValueError, match="one method"):
        ev.completed_units(trial, [1, 2], [1, 2])


def test_prediction_counts_and_metrics():
    trial, pred = rows()
    assert validate_predictions(pd.DataFrame(trial), pd.DataFrame(pred), 2) == "true"
    pred.pop()
    with pytest.raises(ValueError, match="count"):
        validate_predictions(pd.DataFrame(trial), pd.DataFrame(pred), 2)


def test_prediction_label_range():
    trial, pred = rows()
    pred[0]["pred"] = 2
    with pytest.raises(ValueError, match="range"):
        validate_predictions(pd.DataFrame(trial), pd.DataFrame(pred), 2)


def test_prediction_accuracy_recomputed():
    trial, pred = rows()
    trial[0]["accuracy"] = 1.0
    with pytest.raises(ValueError, match="accuracy"):
        validate_predictions(pd.DataFrame(trial), pd.DataFrame(pred), 2)


def test_repeated_targets_need_trial_identity():
    trial, pred = rows(subjects=(1,), blocks=(1,))
    pred[1]["true"], pred[1]["pred"] = 0, 1
    with pytest.raises(ValueError, match="duplicate"):
        validate_predictions(pd.DataFrame(trial), pd.DataFrame(pred), 2)
    for i, row in enumerate(pred):
        row["trial_id"] = f"event-{i}"
    assert validate_predictions(pd.DataFrame(trial), pd.DataFrame(pred), 2) == "trial_id"


def test_report_uses_declared_dataset(tmp_path):
    trial, _ = rows()
    summary, _, _ = ev.summarize(pd.DataFrame(trial), BenchmarkSpec())
    ev.write_report(summary, tmp_path, manifest())
    text = (tmp_path / "report.md").read_text()
    assert "BETA" in text and "0.13" in text
    assert "Tsinghua" not in text and "0.14" not in text


def test_mult_window_confusions_not_pooled(tmp_path, monkeypatch):
    settings = write_fixture(tmp_path, monkeypatch, (0.5, 1.0))
    assert len(settings["confusions"]) == 2
    assert all(np.load(tmp_path / item["path"]).sum() == 8 for item in settings["confusions"])
    assert not (tmp_path / "models" / "confusion_trca.npy").exists()


def test_resume_matching_config(tmp_path, monkeypatch):
    saved = write_fixture(tmp_path, monkeypatch)
    request = manifest()
    request["resolved_config"]["resume"] = True
    trials, preds, runtime = ev.load_existing_rows(tmp_path, request)
    assert len(trials) == 4 and len(preds) == 8 and runtime == []
    assert request["run_id"] == saved["run_id"]
    assert len(request["execution_history"]) == 1


@pytest.mark.parametrize("field,value", [("subjects", [3, 4]), ("windows", [0.5, 1.0]), ("channels", "3ch")])
def test_resume_changed_scope_rejected(tmp_path, monkeypatch, field, value):
    write_fixture(tmp_path, monkeypatch)
    request = manifest()
    request[field] = value
    with pytest.raises(ValueError, match="configuration changed"):
        ev.load_existing_rows(tmp_path, request)


def test_resume_changed_hyperparameter(tmp_path, monkeypatch):
    write_fixture(tmp_path, monkeypatch)
    request = manifest()
    request["resolved_config"]["harmonics"] = 3
    with pytest.raises(ValueError, match="configuration changed"):
        ev.load_existing_rows(tmp_path, request)


def test_tampered_checkpoint_rejected(tmp_path, monkeypatch):
    write_fixture(tmp_path, monkeypatch)
    with (tmp_path / "predictions.csv").open("a") as stream:
        stream.write("broken\n")
    with pytest.raises(ValueError, match="modified artifact"):
        ev.load_existing_rows(tmp_path, manifest())
    assert audit_run(tmp_path)["status"] == "fail"


def test_resume_needs_expected_config(tmp_path, monkeypatch):
    write_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="expected_manifest"):
        ev.load_existing_rows(tmp_path)


def test_legacy_manifest_not_silently_accepted():
    with pytest.raises(ValueError, match="resolved_config"):
        require_same_run({}, manifest())


def test_incomplete_run_cannot_be_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "plot_outputs", lambda *args: None)
    trial, pred = rows(subjects=(1,))
    with pytest.raises(ValueError, match="coverage"):
        ev.write_outputs(tmp_path, None, trial, pred, [], manifest(), BenchmarkSpec(classes=2), True)


def test_empty_run_not_complete(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        ev.write_outputs(tmp_path, None, [], [], [], manifest(), BenchmarkSpec(), True)


def test_analysis_needs_no_predictions(tmp_path):
    table = tmp_path / "spectrum.csv"
    table.write_text("hz,power\n10,0.5\n")
    settings = {"artifact_profile": "analysis", "sources": [{"id": "synthetic-fixture"}],
        "artifacts": [{"path": "spectrum.csv", "sha256": file_sha256(table)}]}
    atomic_json(tmp_path / "manifest.json", settings)
    audit = audit_run(tmp_path)
    assert audit["status"] == "pass" and audit["real_data_acceptance"] == "not_run"
    assert audit["task_math_acceptance"] == "not_run"


@pytest.mark.parametrize("payload", [{}, [], 123, "bad"])
def test_bad_manifest_fails_without_crashing(tmp_path, payload):
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    assert audit_run(tmp_path)["status"] == "fail"


def test_path_traversal_rejected(tmp_path):
    settings = {"artifact_profile": "analysis", "sources": ["fixture"],
        "artifacts": [{"path": "../private.csv", "sha256": "unused"}]}
    atomic_json(tmp_path / "manifest.json", settings)
    assert audit_run(tmp_path)["status"] == "fail"


def test_fingerprint_deterministic_and_strict():
    assert fingerprint({"b": 2, "a": [1]}) == fingerprint({"a": [1], "b": 2})
    with pytest.raises(ValueError):
        fingerprint({"bad": float("nan")})


def test_prepare_refuses_overwrite(tmp_path):
    (tmp_path / "old.csv").write_text("old\n")
    with pytest.raises(FileExistsError):
        prepare_run(manifest(), argparse.Namespace(resume=False), tmp_path, tmp_path, classes=2)


def test_valid_classification_audit(tmp_path, monkeypatch):
    write_fixture(tmp_path, monkeypatch)
    result = audit_run(tmp_path)
    assert result["status"] == "pass" and result["real_data_acceptance"] == "not_run"


def test_summary_rebuilt_even_with_updated_checksum(tmp_path, monkeypatch):
    settings = write_fixture(tmp_path, monkeypatch)
    summary = pd.read_csv(tmp_path / "summary.csv")
    summary.loc[0, "samples"] = 999
    summary.to_csv(tmp_path / "summary.csv", index=False)
    for item in settings["artifacts"]:
        if item["path"] == "summary.csv":
            item["sha256"] = file_sha256(tmp_path / item["path"])
    atomic_json(tmp_path / "manifest.json", settings)
    assert "summary mismatch" in audit_run(tmp_path)["errors"][0]


def test_real_plot_output_smoke(tmp_path):
    trial, _ = rows(subjects=(1,), blocks=(1,))
    summary, subject, block = ev.summarize(pd.DataFrame(trial), BenchmarkSpec())
    ev.plot_outputs(summary, subject, block, tmp_path)
    assert len(list((tmp_path / "figures").glob("*.png"))) == 6


def test_cli_writes_evidence_and_rejects_overwrite(tmp_path):
    import subprocess
    import sys
    evidence = tmp_path / "evidence.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_artifacts.py"
    command = [sys.executable, str(script), str(tmp_path / "missing"), "--output", str(evidence)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    first = evidence.read_bytes()
    assert json.loads(first)["status"] == "fail"
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and evidence.read_bytes() == first


def test_source_hashes_detect_changes(tmp_path):
    (tmp_path / "fixture.mat").write_bytes(b"fixture-not-real-eeg")
    inventory = source_inventory(tmp_path, ["fixture.mat"])
    assert verify_sources(tmp_path, inventory)["source_verification"] == "file_hashes_checked"
    (tmp_path / "fixture.mat").write_bytes(b"changed-fixture")
    with pytest.raises(ValueError, match="changed"):
        verify_sources(tmp_path, inventory)


@pytest.mark.parametrize("filenames", [[], ["missing.mat"], ["../outside.mat"], ["x.mat", "x.mat"]])
def test_invalid_source_inventory(tmp_path, filenames):
    with pytest.raises(ValueError):
        source_inventory(tmp_path, filenames)


def test_unverified_data_not_resumable(tmp_path, monkeypatch):
    write_fixture(tmp_path, monkeypatch)
    request = manifest()
    request["source_verification"] = "not_run"
    with pytest.raises(ValueError, match="source-manifest"):
        ev.load_existing_rows(tmp_path, request)


def test_dirty_code_not_resumable(tmp_path, monkeypatch):
    write_fixture(tmp_path, monkeypatch)
    request = manifest()
    request["code"]["dirty"] = True
    with pytest.raises(ValueError, match="clean code"):
        ev.load_existing_rows(tmp_path, request)


def test_orphan_artifacts_not_a_new_resume(tmp_path):
    (tmp_path / "unknown.csv").write_text("old\n")
    with pytest.raises(ValueError, match="Non-empty"):
        ev.load_existing_rows(tmp_path, manifest())


def test_declared_scope_checked_in_audit(tmp_path, monkeypatch):
    settings = write_fixture(tmp_path, monkeypatch)
    settings["subjects"] = [1, 2, 3]
    atomic_json(tmp_path / "manifest.json", settings)
    assert "coverage" in audit_run(tmp_path)["errors"][0]


def test_explicit_cache_namespace_changes_with_source(tmp_path, monkeypatch):
    import vep_arena.run_contract as rc
    monkeypatch.setattr(rc, "provenance", lambda p: {"git_commit": "fixture", "dirty": False})
    source = tmp_path / "raw"
    source.mkdir()
    (source / "a.mat").write_bytes(b"one")
    inventory_path = tmp_path / "sources.json"
    names = []
    for content in [b"one", b"two"]:
        (source / "a.mat").write_bytes(content)
        atomic_json(inventory_path, source_inventory(source, ["a.mat"]))
        args = argparse.Namespace(data_root=source, epoch_cache=tmp_path / "cache", source_manifest=inventory_path, resume=False)
        cfg = manifest()
        prepare_run(cfg, args, tmp_path, tmp_path / "out", classes=2)
        names.append(args.epoch_cache)
        assert cfg["splits"][0]["train_blocks"] == [2]
        assert cfg["method_inputs"]["TRCA"]["class_ids"] == [0, 1]
    assert names[0] != names[1]
