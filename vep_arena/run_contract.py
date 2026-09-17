"""Run identity and structural checks; never a real-data or paper acceptance claim.

This first profile covers the existing one-row-per-method/window/subject/block
runners. Repeated training combinations need a distinct profile with split IDs.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

UNIT_KEY = ["method", "window", "subject", "block"]
VOLATILE = {
    "status", "seconds", "started_at_utc", "finished_at_utc", "rows_written",
    "windows_written", "last_epoch_fingerprint", "artifacts", "artifact_profile",
    "structural_validation", "limitations", "run_id", "config_fingerprint",
    "execution_history", "schema_version", "confusions",
}


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite configuration value")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported configuration value: {type(value).__name__}")


def fingerprint(value) -> str:
    raw = json.dumps(json_value(value), sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    """Atomic replacement of one file, NOT a transaction across multiple files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_value(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def unique_frame(frame: pd.DataFrame, key: list[str], name: str) -> None:
    missing = set(key) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if frame[key].isna().any().any():
        raise ValueError(f"{name}: null identity")
    if frame.duplicated(key).any():
        raise ValueError(f"{name}: duplicate identity {key}")


def validate_trials(frame: pd.DataFrame) -> None:
    unique_frame(frame, UNIT_KEY, "trials")
    if frame.empty:
        raise ValueError("trials: empty run")
    for column in ["accuracy", "itr", "samples", "seconds", "window"]:
        if column not in frame:
            raise ValueError(f"trials: missing {column}")
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"trials: non-finite or negative {column}")
        if column == "accuracy" and (values > 1).any():
            raise ValueError("trials: accuracy outside [0, 1]")
        if column == "samples" and ((values <= 0) | (values != np.floor(values))).any():
            raise ValueError("trials: samples must be positive integers")
        if column == "window" and (values <= 0).any():
            raise ValueError("trials: window must be positive")


def validate_predictions(trials: pd.DataFrame, predictions: pd.DataFrame,
                         classes: int | None = None) -> str:
    """Legacy true-label key is valid only for one target trial per block.

    Repeated targets require an explicit trial_id/trial_index. This function does
    not fabricate raw source identity or score-column mappings.
    """
    validate_trials(trials)
    identity = next((c for c in ["trial_id", "trial_index"] if c in predictions), "true")
    unique_frame(predictions, [*UNIT_KEY, identity], "predictions")
    for column in ["true", "pred"]:
        if column not in predictions:
            raise ValueError(f"predictions: missing {column}")
        values = pd.to_numeric(predictions[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values != np.floor(values)).any():
            raise ValueError(f"predictions: {column} must be integer labels")
        if classes is not None and ((values < 0) | (values >= classes)).any():
            raise ValueError("predictions: labels outside declared zero-based class range")
    counts = predictions.groupby(UNIT_KEY, dropna=False).size().rename("actual_samples")
    actual = trials.set_index(UNIT_KEY).join(counts, how="outer")
    if actual[["samples", "actual_samples"]].isna().any().any():
        raise ValueError("predictions: missing or unknown evaluation units")
    if not (actual["samples"] == actual["actual_samples"]).all():
        raise ValueError("predictions: sample count mismatch")
    correct = predictions["true"].to_numpy() == predictions["pred"].to_numpy()
    if "correct" in predictions and not np.array_equal(predictions["correct"].to_numpy(), correct):
        raise ValueError("predictions: correct column disagrees with labels")
    measured = predictions.assign(_correct=correct).groupby(UNIT_KEY)["_correct"].mean()
    expected = trials.set_index(UNIT_KEY)["accuracy"].reindex(measured.index)
    if not np.allclose(measured, expected, rtol=1e-9, atol=1e-12):
        raise ValueError("predictions: reconstructed accuracy mismatch")
    return identity


def validate_coverage(trials: pd.DataFrame, manifest: dict, *, complete: bool) -> None:
    for field in ("methods", "windows", "subjects", "blocks"):
        values = manifest.get(field)
        if not isinstance(values, (list, tuple)) or not values or len(values) != len(set(values)):
            raise ValueError(f"Invalid declared scope: {field}")
    expected = set(product(*(manifest[f] for f in ("methods", "windows", "subjects", "blocks"))))
    actual = set(trials[UNIT_KEY].itertuples(index=False, name=None))
    if not actual <= expected or (complete and actual != expected):
        raise ValueError("Run coverage disagrees with declared scope")


def resume_config(manifest: dict) -> dict:
    config = {k: v for k, v in manifest.items() if k not in VOLATILE}
    if "resolved_config" in config:
        config["resolved_config"] = {k: v for k, v in config["resolved_config"].items() if k != "resume"}
    if isinstance(config.get("code"), dict):
        config["code"] = {k: v for k, v in config["code"].items() if k != "command"}
    return json_value(config)


def require_same_run(saved: dict, requested: dict) -> None:
    if "resolved_config" not in saved or "resolved_config" not in requested:
        raise ValueError("Safe resume requires resolved_config; use a new run for legacy outputs")
    for value in (saved, requested):
        if value.get("source_verification") != "file_hashes_checked" or not value.get("source_fingerprint"):
            raise ValueError("Resume requires a verified --source-manifest; start a new run otherwise")
        code = value.get("code", {})
        if not code.get("git_commit") or code.get("dirty") is not False:
            raise ValueError("Resume requires an identified clean code commit")
    saved_config = resume_config(saved)
    if saved.get("config_fingerprint") != fingerprint(saved_config):
        raise ValueError("Stored configuration fingerprint is missing or invalid")
    if fingerprint(saved_config) != fingerprint(resume_config(requested)):
        raise ValueError("Resume configuration changed; use a new output directory")


def provenance(project: Path) -> dict:
    def git(*arguments):
        try:
            return subprocess.check_output(["git", "-C", str(project), *arguments],
                stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    versions = {}
    for package in ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    status = git("status", "--porcelain", "--untracked-files=normal")
    lock = project / "uv.lock"
    return {"git_commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status),
            "command": [sys.executable, *sys.argv], "python": platform.python_version(),
            "platform": platform.platform(), "packages": versions,
            "lock_sha256": file_sha256(lock) if lock.is_file() else None}


def verify_files(root: Path, manifest: dict) -> None:
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Missing artifact integrity ledger; legacy output needs a separate audit")
    root, seen = root.resolve(), set()
    for item in artifacts:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or path in seen:
            raise ValueError("Invalid or duplicate artifact path")
        seen.add(path)
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            raise ValueError(f"Missing or modified artifact: {item['path']}")


def audit_run(root: Path) -> dict:
    """Read-only audit of declared files, block rows and reconstructed aggregates.

    Analysis/comparison profiles check only the ledger and declared references;
    they do NOT validate task-specific mathematics or comparison compatibility.
    """
    root, errors, manifest = Path(root), [], {}
    try:
        parsed = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("Manifest must be a JSON object")
        manifest = parsed
        profile = manifest.get("artifact_profile")
        if profile not in {"classification", "analysis", "comparison", "aggregate_only"}:
            raise ValueError("Missing/unknown artifact_profile; do not infer it from filenames")
        verify_files(root, manifest)
        if profile == "classification":
            if manifest.get("status") not in {"partial", "complete"}:
                raise ValueError("Classification status must be partial or complete")
            if manifest.get("label_base") != 0:
                raise ValueError("This block profile requires declared zero-based labels")
            classes = manifest.get("classes")
            if not isinstance(classes, int) or isinstance(classes, bool) or classes < 2:
                raise ValueError("Invalid declared class count")
            trials, preds = pd.read_csv(root / "trials.csv"), pd.read_csv(root / "predictions.csv")
            validate_predictions(trials, preds, classes)
            validate_coverage(trials, manifest, complete=manifest["status"] == "complete")
            ledger = {item["path"] for item in manifest["artifacts"]}
            if not {"trials.csv", "predictions.csv", "summary.csv", "runtime.csv"} <= ledger:
                raise ValueError("Classification integrity ledger is incomplete")
            from vep_arena.evaluation import summarize
            computed, _, _ = summarize(trials, None)
            stored = pd.read_csv(root / "summary.csv")
            key = ["method", "window"]
            unique_frame(stored, key, "summary")
            computed, stored = computed.set_index(key).sort_index(), stored.set_index(key).sort_index()
            if not computed.index.equals(stored.index):
                raise ValueError("Summary identity mismatch")
            for column in computed.columns:
                if column not in stored or not np.allclose(computed[column], stored[column], rtol=1e-8, atol=1e-10):
                    raise ValueError(f"Reconstructed summary mismatch: {column}")
            expected_pairs = set(preds[["method", "window"]].itertuples(index=False, name=None))
            actual_pairs = set()
            from vep_arena.channel.confusion import confusion_counts
            for item in manifest.get("confusions", []):
                pair = (item["method"], item["window"])
                if pair in actual_pairs or pair not in expected_pairs or item["path"] not in ledger:
                    raise ValueError("Invalid confusion identity")
                actual_pairs.add(pair)
                subset = preds[(preds.method == pair[0]) & (preds.window == pair[1])]
                expected_counts = confusion_counts(subset.true, subset.pred, classes)
                if not np.array_equal(np.load(root / item["path"], allow_pickle=False), expected_counts):
                    raise ValueError("Reconstructed confusion mismatch")
            if actual_pairs != expected_pairs:
                raise ValueError("Missing per-window confusion matrices")
        elif profile == "comparison":
            if not manifest.get("sources") or not manifest.get("compatibility"):
                raise ValueError("Comparison requires source references and an explicit compatibility outcome")
        elif profile == "analysis" and not manifest.get("sources"):
            raise ValueError("Analysis requires source references; no classification table is required")
        elif profile == "aggregate_only" and not manifest.get("limitations"):
            raise ValueError("Aggregate-only output must explain missing granularity")
    except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
        errors.append(str(exc))
    return {"schema_version": "1.0", "scope": "artifact_structure_only",
            "artifact_profile": manifest.get("artifact_profile"),
            "status": "fail" if errors else "pass", "errors": errors,
            "real_data_acceptance": "not_run", "paper_reproduction": "not_run",
            "task_math_acceptance": "not_run", "comparison_acceptance": "not_run"}


def prepare_run(manifest: dict, args, project: Path, result_dir: Path, *, classes: int) -> None:
    """Record effective settings and refuse accidental overwrite before execution."""
    if not getattr(args, "resume", False) and result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {result_dir}; choose a new run")
    methods = manifest.get("methods", [manifest.get("method")])
    if not methods or any(not m for m in methods) or len(methods) != len(set(methods)):
        raise ValueError("Methods must be non-empty and unique")
    if getattr(args, "workers", 1) < 1:
        raise ValueError("workers must be positive")
    for name in ["harmonics", "n_fbs", "n_bands", "n_components"]:
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if getattr(args, "n_delay", 0) < 0:
        raise ValueError("n_delay must not be negative")
    for name in ["subjects", "blocks", "windows"]:
        values = manifest.get(name, [])
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Invalid scope: {name}")
    source_path = getattr(args, "source_manifest", None)
    source_fields = {"source_verification": "not_run", "source_fingerprint": None}
    if source_path is not None:
        root = getattr(args, "data_root", None) or manifest.get("root")
        if root is None:
            raise ValueError("A data root is required for source verification")
        inventory = json.loads(Path(source_path).read_text(encoding="utf-8"))
        source_fields = verify_sources(Path(root), inventory)
        source_fields["source_inventory"] = inventory
    code = provenance(project)
    if (source_path is not None or getattr(args, "resume", False)) and (
        not code.get("git_commit") or code.get("dirty") is not False
    ):
        raise ValueError(
            "Verified or resumable runs require Git provenance from an identified clean commit; "
            "Git provenance is unavailable or the worktree is dirty"
        )
    manifest.update({"run_id": uuid.uuid4().hex, "methods": methods, "classes": classes,
                     "resolved_config": json_value(vars(args)), "code": code,
                     "label_base": 0, **source_fields})
    manifest["splits"] = [
        {"split_id": f"subject-{subject}-test-{block}", "subject": subject,
         "train_blocks": [b for b in manifest["blocks"] if b != block],
         "test_blocks": [block], "validation_blocks": [],
         "validation_policy": "no checkpoint selection in these classic runners"}
        for subject in manifest["subjects"] for block in manifest["blocks"]
    ]
    manifest["method_inputs"] = {
        method: {"preprocess": "raw_epoch" if method == "CCA" else "toolbox_filterbank",
                 "n_bands": 1 if method == "CCA" else getattr(args, "n_fbs", getattr(args, "n_bands", 5)),
                 "class_ids": list(range(classes)),
                 "extra_samples": getattr(args, "n_delay", 0) if method == "TDCA" else 0}
        for method in methods
    }
    # Existing epoch-cache recipes do not hash the raw sources. Isolate verified
    # runs by source AND code identity instead of silently trusting an old cache.
    if source_fields["source_fingerprint"] and hasattr(args, "epoch_cache"):
        cache_id = fingerprint({"source": source_fields["source_fingerprint"],
                                "code": manifest["code"]["git_commit"]})
        args.epoch_cache = Path(args.epoch_cache) / ("verified-" + cache_id[:24])
        manifest["epoch_cache"] = str(args.epoch_cache)
        manifest["resolved_config"]["epoch_cache"] = str(args.epoch_cache)


def source_inventory(root: Path, filenames: list[str]) -> dict:
    """Hash an explicit caller-supplied list, not a claim of complete dependency discovery."""
    if not filenames or len(filenames) != len(set(filenames)):
        raise ValueError("Source filenames must be non-empty and unique")
    root, entries = Path(root).resolve(), []
    for name in filenames:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid source path: {name}")
        entries.append({"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)})
    if len({e["path"] for e in entries}) != len(entries):
        raise ValueError("Duplicate source aliases")
    return {"schema_version": "1.0", "coverage": "caller_declared",
            "files": sorted(entries, key=lambda e: e["path"])}


def verify_sources(root: Path, inventory: dict) -> dict:
    if not isinstance(inventory, dict) or inventory.get("schema_version") != "1.0":
        raise ValueError("Unsupported source inventory")
    entries = inventory.get("files", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("Empty source inventory")
    expected = source_inventory(root, [entry["path"] for entry in entries])
    if expected["files"] != sorted(entries, key=lambda e: e["path"]):
        raise ValueError("Source contents changed or hash missing")
    return {"source_verification": "file_hashes_checked", "source_fingerprint": fingerprint(expected),
            "source_coverage": "caller_declared"}
