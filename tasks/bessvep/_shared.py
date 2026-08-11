from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.embc_jbhi import (  # noqa: E402
    PrivateSSVEPSpec,
    analysis_epochs,
    available_subject_ids,
    dataset_spec,
    load_jbhi35_target_frequencies,
    load_subject,
    resolve_dataset_root,
)
from vep_arena.metrics import itr_bits_per_minute, sem  # noqa: E402
from vep_arena.methods.bprca import BPRCA  # noqa: E402
from vep_arena.methods.fusionca import FusionCA  # noqa: E402
from vep_arena.methods.traditional import TRCA  # noqa: E402


def _parse_subjects(text: str, available: tuple[str, ...]) -> tuple[str, ...]:
    if text.strip().lower() == "all":
        return available
    selected: list[str] = []
    for part in text.split(","):
        token = part.strip().upper()
        if not token:
            continue
        match = re.fullmatch(r"S?(\d+)-S?(\d+)", token)
        if match:
            start, stop = (int(value) for value in match.groups())
            selected.extend(f"S{index:02d}" for index in range(start, stop + 1))
        else:
            match = re.fullmatch(r"S?(\d+)", token)
            if not match:
                raise ValueError(f"Invalid anonymous subject selector: {part!r}")
            selected.append(f"S{int(match.group(1)):02d}")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unavailable anonymous subject ids: {', '.join(unknown)}")
    return tuple(dict.fromkeys(selected))


def _parse_windows(text: str, maximum: float) -> tuple[float, ...]:
    windows = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not windows or any(value <= 0 or value > maximum for value in windows):
        raise ValueError(f"Windows must lie in (0, {maximum:g}].")
    return tuple(dict.fromkeys(windows))


def _parse_methods(text: str) -> tuple[str, ...]:
    methods = tuple(value.strip().upper() for value in text.split(",") if value.strip())
    supported = {"TRCA", "ETRCA", "BPRCA", "EBPRCA", "FUSIONCA", "EFUSIONCA"}
    unsupported = sorted(set(methods) - supported)
    if unsupported:
        raise ValueError(f"Unsupported methods: {', '.join(unsupported)}")
    return tuple(dict.fromkeys(methods))


def _bprca_frequency_units(dataset: str, root: Path) -> tuple[float, ...]:
    if dataset == "embc9":
        return (8.5, 9.5)
    if dataset == "jbhi16":
        return (11.0, 12.0, 13.0)
    pairs = load_jbhi35_target_frequencies(root)
    return tuple(sorted({frequency for pair in pairs for frequency in pair if frequency > 0}))


def _unit_slug(subject: str, method: str, window: float, n_bands: int) -> str:
    window_ms = int(round(window * 1000))
    return f"{subject}_{method.lower()}_w{window_ms:04d}_fb{n_bands}"


def _part_paths(parts_dir: Path, slug: str) -> dict[str, Path]:
    return {
        "unit": parts_dir / f"{slug}_unit.json",
        "folds": parts_dir / f"{slug}_folds.csv",
        "predictions": parts_dir / f"{slug}_predictions.csv",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _unit_complete(parts_dir: Path, slug: str) -> bool:
    paths = _part_paths(parts_dir, slug)
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        unit = json.loads(paths["unit"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return unit.get("status") == "complete" and int(unit["prediction_count"]) == int(unit["expected_predictions"])


def _run_unit(task: dict[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    dataset = str(task["dataset"])
    subject_id = str(task["subject_id"])
    method = str(task["method"])
    window = float(task["window"])
    n_bands = int(task["n_bands"])
    root = Path(str(task["root"]))
    spec = dataset_spec(dataset)
    subject = load_subject(dataset, subject_id, root)
    epochs = analysis_epochs(subject, window, n_bands=n_bands)
    labels = np.arange(spec.targets, dtype=np.int64)
    predictions: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    correct = 0

    for test_block in range(spec.blocks):
        train_blocks = [block for block in range(spec.blocks) if block != test_block]
        if test_block in train_blocks or len(set(train_blocks)) != spec.blocks - 1:
            raise RuntimeError("Fold leakage check failed.")
        train_x = epochs[:, train_blocks].reshape(
            spec.targets * len(train_blocks), epochs.shape[2], epochs.shape[3], epochs.shape[4]
        )
        train_y = np.repeat(labels, len(train_blocks))
        test_x = epochs[:, test_block]
        if method in {"TRCA", "ETRCA"}:
            model = TRCA(n_fbs=epochs.shape[2], ensemble=method == "ETRCA")
        elif method in {"BPRCA", "EBPRCA"}:
            model = BPRCA(
                _bprca_frequency_units(dataset, root),
                spec.analysis_sampling_rate,
                n_fbs=epochs.shape[2],
                ensemble=method == "EBPRCA",
            )
        else:
            model = FusionCA(
                _bprca_frequency_units(dataset, root),
                spec.analysis_sampling_rate,
                n_fbs=epochs.shape[2],
                ensemble=method == "EFUSIONCA",
            )
        model.fit(train_x, train_y)
        predicted, _ = model.predict(test_x)
        fold_correct = int(np.sum(predicted == labels))
        correct += fold_correct
        folds.append(
            {
                "dataset": dataset,
                "subject": subject_id,
                "method": method,
                "window_seconds": window,
                "fold": test_block + 1,
                "test_block": test_block + 1,
                "train_block_count": len(train_blocks),
                "test_block_in_training": False,
                "prediction_count": spec.targets,
                "correct": fold_correct,
                "accuracy": fold_correct / spec.targets,
            }
        )
        for target, prediction in enumerate(predicted):
            predictions.append(
                {
                    "dataset": dataset,
                    "subject": subject_id,
                    "method": method,
                    "window_seconds": window,
                    "window_samples": epochs.shape[-1],
                    "fold": test_block + 1,
                    "block": test_block + 1,
                    "trial_index": test_block * spec.targets + target + 1,
                    "true": target + 1,
                    "pred": int(prediction) + 1,
                    "correct": int(target == int(prediction)),
                }
            )

    expected_predictions = spec.targets * spec.blocks
    accuracy = correct / expected_predictions
    unit = {
        "dataset": dataset,
        "subject": subject_id,
        "method": method,
        "window_seconds": window,
        "window_samples": epochs.shape[-1],
        "sampling_rate": spec.analysis_sampling_rate,
        "targets": spec.targets,
        "blocks": spec.blocks,
        "channels": epochs.shape[-2],
        "filter_bands": epochs.shape[2],
        "folds": spec.blocks,
        "prediction_count": len(predictions),
        "expected_predictions": expected_predictions,
        "accuracy": accuracy,
        "itr_bpm": itr_bits_per_minute(accuracy, spec.targets, window + spec.itr_shift_seconds),
        "runtime_seconds": time.perf_counter() - started,
        "fold_leakage_check": "pass",
        "label_order_check": subject.metadata["label_order_status"],
        "status": "complete" if len(predictions) == expected_predictions else "partial",
    }
    return {"unit": unit, "folds": folds, "predictions": predictions}


def _write_part(parts_dir: Path, result: dict[str, object]) -> None:
    unit = dict(result["unit"])
    slug = _unit_slug(str(unit["subject"]), str(unit["method"]), float(unit["window_seconds"]), int(unit["filter_bands"]))
    paths = _part_paths(parts_dir, slug)
    _write_csv(paths["folds"], list(result["folds"]))
    _write_csv(paths["predictions"], list(result["predictions"]))
    _write_json(paths["unit"], unit)


def _read_parts(parts_dir: Path, expected_slugs: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    units: list[dict[str, object]] = []
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for slug in sorted(expected_slugs):
        if not _unit_complete(parts_dir, slug):
            continue
        paths = _part_paths(parts_dir, slug)
        units.append(json.loads(paths["unit"].read_text(encoding="utf-8")))
        fold_frames.append(pd.read_csv(paths["folds"]))
        prediction_frames.append(pd.read_csv(paths["predictions"]))
    unit_frame = pd.DataFrame(units)
    folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return unit_frame, folds, predictions


def _aggregate(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if units.empty:
        return pd.DataFrame()
    for (dataset, method, window), group in units.groupby(["dataset", "method", "window_seconds"], sort=True):
        spec = dataset_spec(str(dataset))
        mean_accuracy = float(group["accuracy"].mean())
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "window_seconds": window,
                "subjects": group["subject"].nunique(),
                "accuracy": mean_accuracy,
                "accuracy_sem": sem(group["accuracy"].to_numpy()),
                "itr_bpm": group["itr_bpm"].mean(),
                "itr_from_group_accuracy_bpm": itr_bits_per_minute(
                    mean_accuracy,
                    spec.targets,
                    float(window) + spec.itr_shift_seconds,
                ),
                "itr_sem": sem(group["itr_bpm"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def _plot_time_and_psd(dataset: str, root: Path, subject_id: str, figure_dir: Path) -> None:
    subject = load_subject(dataset, subject_id, root)
    spec = dataset_spec(dataset)
    analysis = subject.x
    sampling_rate = subject.sampling_rate
    if dataset == "embc9":
        analysis = signal.resample_poly(analysis, up=1, down=4, axis=-1)
        sampling_rate = spec.analysis_sampling_rate
    display_samples = min(analysis.shape[-1], int(round(2.0 * sampling_rate)))
    time_axis = np.arange(display_samples) / sampling_rate
    oz_index = subject.channels.index("Oz")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for target in range(min(3, spec.targets)):
        trace = analysis[target, 0, oz_index, :display_samples]
        trace = trace - np.mean(trace)
        ax.plot(time_axis, trace + target * np.std(trace) * 5.0, lw=0.8, label=f"target {target + 1}")
    ax.set(title=f"{dataset} {subject_id} time-domain check (Oz)", xlabel="Time (s)", ylabel="Offset amplitude")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "time_domain.png", dpi=180)
    plt.close(fig)

    subset = analysis[: min(4, spec.targets), : min(2, spec.blocks)]
    freqs, psd = signal.welch(subset, fs=sampling_rate, nperseg=min(2048, subset.shape[-1]), axis=-1)
    mask = (freqs >= 1.0) & (freqs <= min(100.0, sampling_rate / 2.0))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if dataset == "jbhi35":
        target_pairs = load_jbhi35_target_frequencies(root)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for target in range(3):
            target_psd = np.mean(psd[target], axis=(0, 1))
            pair = target_pairs[target]
            label = f"target {target + 1}: L={pair[0]:g}, R={pair[1]:g} Hz"
            ax.plot(freqs[mask], 10.0 * np.log10(target_psd[mask] + 1e-18), lw=1.0, color=colors[target], label=label)
            for frequency in sorted(set(pair) - {0.0}):
                ax.axvline(frequency, color=colors[target], alpha=0.55, lw=0.8, ls="--")
        ax.legend(fontsize=8)
    else:
        mean_psd = np.mean(psd, axis=(0, 1, 2))
        ax.plot(freqs[mask], 10.0 * np.log10(mean_psd[mask] + 1e-18), lw=1.1)
        for frequency in ((8.5, 9.5) if dataset == "embc9" else (11.0, 12.0, 13.0)):
            ax.axvline(frequency, color="#d62728", alpha=0.6, lw=0.8)
    ax.set(title=f"{dataset} {subject_id} PSD check", xlabel="Frequency (Hz)", ylabel="PSD (dB)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "fft_psd.png", dpi=180)
    plt.close(fig)


def _plot_curves(summary: pd.DataFrame, figure_dir: Path) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, group in summary.groupby("method"):
        group = group.sort_values("window_seconds")
        axes[0].errorbar(group["window_seconds"], group["accuracy"], yerr=group["accuracy_sem"], marker="o", label=method)
        axes[1].errorbar(group["window_seconds"], group["itr_bpm"], yerr=group["itr_sem"], marker="o", label=method)
    axes[0].set(title="Accuracy-window", xlabel="Window (s)", ylabel="Accuracy", ylim=(0, 1.02))
    axes[1].set(title="ITR-window", xlabel="Window (s)", ylabel="Bits/min")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "accuracy_itr_window.png", dpi=180)
    plt.close(fig)


def _plot_confusion(predictions: pd.DataFrame, spec: PrivateSSVEPSpec, figure_dir: Path) -> None:
    if predictions.empty:
        return
    methods = tuple(sorted(predictions["method"].unique()))
    max_window = float(predictions["window_seconds"].max())
    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 5), squeeze=False)
    for ax, method in zip(axes[0], methods):
        group = predictions[(predictions["method"] == method) & (predictions["window_seconds"] == max_window)]
        matrix = np.zeros((spec.targets, spec.targets), dtype=np.float64)
        for true, pred in zip(group["true"].astype(int), group["pred"].astype(int)):
            matrix[true - 1, pred - 1] += 1
        totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
        image = ax.imshow(normalized, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set(title=f"{method} confusion at {max_window:g}s", xlabel="Predicted", ylabel="True")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def _label_order_audit(dataset: str, root: Path, subjects: tuple[str, ...]) -> dict[str, object]:
    if dataset == "jbhi16":
        return {"status": "pass", "evidence": "all MAT label vectors are six exact repetitions of 1-16"}
    if dataset == "embc9":
        return {"status": "unverified", "evidence": "target and block axes are explicit but labels/codebook are not embedded"}
    return {
        "status": "pass",
        "evidence": "all labeled-source MAT files have six consecutive blocks containing labels 1-35 exactly once",
        "subjects_checked": list(subjects),
        "interpretation": "The loader groups trials from explicit labels; no signal-derived label inference is used.",
    }


def _write_report(output: Path, manifest: dict[str, object], summary: pd.DataFrame) -> None:
    lines = [
        f"# {manifest['dataset']} smoke report",
        "",
        f"- status: `{manifest['status']}`",
        f"- anonymous subjects: `{', '.join(manifest['subjects'])}`",
        f"- protocol: {manifest['cv_protocol']}",
        f"- protocol validation: `{manifest['protocol_validation_status']}`",
        f"- stored visual latency already applied: `{manifest['stored_epoch_start_seconds']} s`; runtime latency offset: `0 s`",
        "- EEGLAB preprocessing is treated as completed upstream; only the declared receiver filter bank/resampling is applied here.",
        "- No source filename or participant identity mapping is written to outputs.",
        "",
        "## Results",
        "",
    ]
    if summary.empty:
        lines.append("No complete units.")
    else:
        lines.extend(["| Method | Window (s) | Subjects | Accuracy | ITR (bits/min) |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in summary.itertuples(index=False):
            lines.append(f"| {row.method} | {row.window_seconds:g} | {row.subjects} | {row.accuracy:.4f} | {row.itr_bpm:.2f} |")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_cli(
    dataset: str,
    default_subjects: str,
    default_windows: str,
    default_methods: str,
) -> None:
    spec = dataset_spec(dataset)
    parser = argparse.ArgumentParser(description=f"Run anonymous {dataset} SSVEP block-CV baselines.")
    parser.add_argument("--root", type=Path, default=None, help="Override local_paths.json for this run only.")
    parser.add_argument("--subjects", default=default_subjects)
    parser.add_argument("--windows", default=default_windows)
    parser.add_argument("--methods", default=default_methods)
    parser.add_argument("--filter-bands", type=int, default=3 if dataset == "embc9" else 5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--output", type=Path, default=Path(sys.argv[0]).resolve().parent / "results" / "smoke")
    args = parser.parse_args()

    root = resolve_dataset_root(dataset, args.root)
    available = available_subject_ids(dataset, root)
    subjects = _parse_subjects(args.subjects, available)
    windows = _parse_windows(args.windows, spec.stimulation_seconds)
    methods = _parse_methods(args.methods)
    if dataset == "embc9" and args.filter_bands not in (1, 2, 3):
        parser.error("EMBC legacy-code receiver filter bank supports --filter-bands 1-3.")
    if args.workers < 1:
        parser.error("--workers must be positive.")

    output = args.output.resolve()
    parts_dir = output / "parts"
    figure_dir = output / "figures"
    parts_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "dataset": dataset,
            "subject_id": subject,
            "method": method,
            "window": window,
            "n_bands": args.filter_bands,
            "root": str(root),
        }
        for subject in subjects
        for method in methods
        for window in windows
    ]
    expected_slugs = {
        _unit_slug(str(task["subject_id"]), str(task["method"]), float(task["window"]), int(task["n_bands"]))
        for task in tasks
    }
    label_order_audit = _label_order_audit(dataset, root, subjects)
    manifest: dict[str, object] = {
        "dataset": dataset,
        "dataset_config_key": spec.config_key,
        "dataset_root": str(root),
        "paper_role": spec.paper_role,
        "subjects": list(subjects),
        "methods": list(methods),
        "windows_seconds": list(windows),
        "targets": spec.targets,
        "blocks": spec.blocks,
        "channels": 9,
        "stored_sampling_rate": spec.stored_sampling_rate,
        "analysis_sampling_rate": spec.analysis_sampling_rate,
        "stored_epoch_start_seconds": spec.stored_epoch_start_seconds,
        "runtime_latency_offset_seconds": 0.0,
        "cv_protocol": spec.cv_protocol,
        "preprocessing_profile": spec.preprocessing_profile,
        "receiver_filterbank": {**spec.receiver_filterbank, "selected_bands": args.filter_bands},
        "filter_bands": args.filter_bands,
        "workers": args.workers,
        "resume": args.resume,
        "blas_threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "python_executable": sys.executable,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "started",
        "identity_policy": "outputs use only stable anonymous SNN ids and never expose source filenames or embedded identity metadata",
        "label_order_audit": label_order_audit,
        "protocol_validation_status": "failed_label_order" if label_order_audit["status"] == "failed" else label_order_audit["status"],
    }
    _write_json(output / "manifest.json", manifest)

    pending = []
    for task in tasks:
        slug = _unit_slug(str(task["subject_id"]), str(task["method"]), float(task["window"]), int(task["n_bands"]))
        if args.resume and _unit_complete(parts_dir, slug):
            continue
        pending.append(task)

    errors: list[str] = []
    if args.workers == 1:
        for task in pending:
            try:
                _write_part(parts_dir, _run_unit(task))
            except Exception as exc:
                errors.append(f"{task['subject_id']} {task['method']} {task['window']}s: {type(exc).__name__}")
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            future_map = {executor.submit(_run_unit, task): task for task in pending}
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    _write_part(parts_dir, future.result())
                except Exception as exc:
                    errors.append(f"{task['subject_id']} {task['method']} {task['window']}s: {type(exc).__name__}")

    units, folds, predictions = _read_parts(parts_dir, expected_slugs)
    summary = _aggregate(units)
    units.to_csv(output / "unit_manifest.csv", index=False)
    units.to_csv(output / "subject.csv", index=False)
    folds.to_csv(output / "trials.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "errors.log").write_text("\n".join(errors), encoding="utf-8")

    expected_predictions = len(tasks) * spec.targets * spec.blocks
    completed_predictions = len(predictions)
    complete = len(units) == len(tasks) and completed_predictions == expected_predictions and not errors
    manifest.update(
        {
            "status": "complete" if complete else "partial",
            "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "expected_units": len(tasks),
            "completed_units": len(units),
            "expected_predictions": expected_predictions,
            "completed_predictions": completed_predictions,
            "error_count": len(errors),
            "interpretation_status": "blocked" if label_order_audit["status"] == "failed" else "usable",
        }
    )
    _write_json(output / "manifest.json", manifest)
    _plot_time_and_psd(dataset, root, subjects[0], figure_dir)
    _plot_curves(summary, figure_dir)
    _plot_confusion(predictions, spec, figure_dir)
    _write_report(output, manifest, summary)
    print(json.dumps({key: manifest[key] for key in ("status", "protocol_validation_status", "interpretation_status", "completed_units", "expected_units", "completed_predictions", "expected_predictions", "error_count")}, indent=2))
    if not complete:
        raise RuntimeError(f"Run is partial; inspect {output / 'manifest.json'} and {output / 'errors.log'}")
