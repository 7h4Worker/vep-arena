# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import BENCHMARK_FREQS, DATA_ROOT, RUN_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import load_subject_trials
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.sa_mvmd_trca import SAMVMDTRCAModel, sa_mvmd_components


PAPER_CHANNELS_9 = (48, 55, 54, 58, 56, 57, 61, 62, 63)
SUBJECT_BLOCK_FIELDS = ["method", "window", "subject", "block", "accuracy", "itr", "samples", "seconds"]
PREDICTION_FIELDS = ["method", "window", "subject", "block", "true", "pred", "score_true", "score_pred"]


def parse_range(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        elif part:
            values.append(int(part))
    return values


def parse_windows(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def decompose_true_trials(
    data: np.ndarray,
    *,
    spec: BenchmarkSpec,
    n_harmonics: int,
    selected_harmonics: int,
    alpha: float,
    rp: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    classes, blocks, channels, samples = data.shape
    out = np.zeros((classes, blocks, selected_harmonics + 1, channels, samples), dtype=np.float32)
    for target, freq in enumerate(BENCHMARK_FREQS):
        for block in range(blocks):
            out[target, block] = sa_mvmd_components(
                data[target, block],
                freq,
                fs=spec.sampling_rate,
                n_harmonics=n_harmonics,
                selected_harmonics=selected_harmonics,
                alpha=alpha,
                rp=rp,
                max_iter=max_iter,
                tol=tol,
            ).astype(np.float32)
    return out


def decompose_candidates(
    trial: np.ndarray,
    *,
    spec: BenchmarkSpec,
    n_harmonics: int,
    selected_harmonics: int,
    alpha: float,
    rp: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    classes = len(BENCHMARK_FREQS)
    channels, samples = trial.shape
    out = np.zeros((classes, selected_harmonics + 1, channels, samples), dtype=np.float32)
    for target, freq in enumerate(BENCHMARK_FREQS):
        out[target] = sa_mvmd_components(
            trial,
            freq,
            fs=spec.sampling_rate,
            n_harmonics=n_harmonics,
            selected_harmonics=selected_harmonics,
            alpha=alpha,
            rp=rp,
            max_iter=max_iter,
            tol=tol,
        ).astype(np.float32)
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    if not rows:
        return
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def completed_units(path: Path, methods: list[str]) -> set[tuple[float, int, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done: dict[tuple[float, int, int], set[str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (round(float(row["window"]), 6), int(row["subject"]), int(row["block"]))
            done.setdefault(key, set()).add(row["method"])
    required = set(methods)
    return {key for key, names in done.items() if required.issubset(names)}


def write_progress(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_confusions(output_dir: Path, prediction_rows: list[dict[str, object]], methods: list[str], spec: BenchmarkSpec) -> None:
    if not prediction_rows:
        return
    for method in methods:
        rows = [row for row in prediction_rows if row["method"] == method]
        if not rows:
            continue
        y_true = np.asarray([row["true"] for row in rows], dtype=np.int64)
        y_pred = np.asarray([row["pred"] for row in rows], dtype=np.int64)
        name = f"confusion_{method.lower().replace('-', '_')}.npy"
        np.save(output_dir / name, confusion_matrix(y_true, y_pred, labels=np.arange(spec.classes)))


def read_prediction_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "method": row["method"],
                    "window": float(row["window"]),
                    "subject": int(row["subject"]),
                    "block": int(row["block"]),
                    "true": int(row["true"]),
                    "pred": int(row["pred"]),
                    "score_true": float(row["score_true"]),
                    "score_pred": float(row["score_pred"]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT / "sa_mvmd_paper")
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--blocks", default="1-6")
    parser.add_argument("--windows", default="0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0")
    parser.add_argument("--methods", default="SA-MVMD-TRCA,SA-MVMD-eTRCA")
    parser.add_argument("--n-harmonics", type=int, default=7)
    parser.add_argument("--selected-harmonics", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=2000.0)
    parser.add_argument("--rp", type=float, default=100.0)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    spec = BenchmarkSpec()
    subjects = parse_range(args.subjects)
    blocks = parse_range(args.blocks)
    windows = parse_windows(args.windows)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subject_block_path = args.output_dir / "subject_block.csv"
    prediction_path = args.output_dir / "predictions.csv"
    if args.overwrite:
        for path in args.output_dir.glob("*.csv"):
            path.unlink()
        for path in args.output_dir.glob("confusion_*.npy"):
            path.unlink()

    done = set() if args.no_resume else completed_units(subject_block_path, methods)
    manifest: dict[str, object] = {
        "methods": methods,
        "dataset": "Benchmark",
        "protocol": "paper-style six-fold leave-one-block-out",
        "subjects": subjects,
        "blocks": blocks,
        "windows": windows,
        "channels": list(PAPER_CHANNELS_9),
        "n_harmonics": args.n_harmonics,
        "selected_harmonics": args.selected_harmonics,
        "selected_imfs": "IMF2-IMF6, plus reconstructed wideband",
        "alpha": args.alpha,
        "rp": args.rp,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "paper_pdf": "C:/Users/Admin/Desktop/papers-unread/Novel_Sinusoidal_Signal_Assisted_Multivariate_Variational_Mode_Decomposition_Combined_With_Task-Related_Component_Analysis_for_Enhancing_SSVEP-Based_BCI_Performance.pdf",
        "status": "running",
    }
    started = time.perf_counter()
    write_progress(args.output_dir / "manifest.json", manifest)

    for window in windows:
        for subject in subjects:
            pending_blocks = [
                block for block in blocks if (round(window, 6), subject, block) not in done
            ]
            if not pending_blocks:
                print(f"skip window={window:.1f} subject={subject:02d} complete", flush=True)
                continue
            subject_start = time.perf_counter()
            data = load_subject_trials(args.data_root, subject, window, channels=PAPER_CHANNELS_9, spec=spec)
            true_components = decompose_true_trials(
                data,
                spec=spec,
                n_harmonics=args.n_harmonics,
                selected_harmonics=args.selected_harmonics,
                alpha=args.alpha,
                rp=args.rp,
                max_iter=args.max_iter,
                tol=args.tol,
            )
            print(
                f"train-decompose window={window:.1f} subject={subject:02d} "
                f"seconds={time.perf_counter() - subject_start:.2f}",
                flush=True,
            )

            for block in pending_blocks:
                block_idx = block - 1
                train_blocks = [idx for idx in range(spec.blocks) if idx != block_idx]
                model = SAMVMDTRCAModel.fit(true_components[:, train_blocks].astype(np.float64))
                y_true = np.arange(spec.classes, dtype=np.int64)
                method_preds = {method: [] for method in methods}
                method_scores = {method: [] for method in methods}
                block_start = time.perf_counter()
                for true_target in range(spec.classes):
                    candidate_components = decompose_candidates(
                        data[true_target, block_idx],
                        spec=spec,
                        n_harmonics=args.n_harmonics,
                        selected_harmonics=args.selected_harmonics,
                        alpha=args.alpha,
                        rp=args.rp,
                        max_iter=args.max_iter,
                        tol=args.tol,
                    ).astype(np.float64)
                    if "SA-MVMD-TRCA" in methods:
                        pred, scores = model.predict_from_candidate_components(candidate_components, ensemble=False)
                        method_preds["SA-MVMD-TRCA"].append(pred)
                        method_scores["SA-MVMD-TRCA"].append(scores)
                    if "SA-MVMD-eTRCA" in methods:
                        pred, scores = model.predict_from_candidate_components(candidate_components, ensemble=True)
                        method_preds["SA-MVMD-eTRCA"].append(pred)
                        method_scores["SA-MVMD-eTRCA"].append(scores)

                elapsed = time.perf_counter() - block_start
                block_subject_rows: list[dict[str, object]] = []
                block_prediction_rows: list[dict[str, object]] = []
                for method in methods:
                    pred_arr = np.asarray(method_preds[method], dtype=np.int64)
                    score_arr = np.asarray(method_scores[method], dtype=np.float64)
                    acc = accuracy_score(y_true, pred_arr)
                    itr = itr_bits_per_minute(float(acc), spec.classes, window + spec.cue_seconds)
                    block_subject_rows.append(
                        {
                            "method": method,
                            "window": window,
                            "subject": subject,
                            "block": block,
                            "accuracy": float(acc),
                            "itr": float(itr),
                            "samples": spec.classes,
                            "seconds": elapsed,
                        }
                    )
                    for true_label, pred_label in zip(y_true, pred_arr):
                        block_prediction_rows.append(
                            {
                                "method": method,
                                "window": window,
                                "subject": subject,
                                "block": block,
                                "true": int(true_label),
                                "pred": int(pred_label),
                                "score_true": float(score_arr[int(true_label), int(true_label)]),
                                "score_pred": float(score_arr[int(true_label), int(pred_label)]),
                            }
                        )
                    print(
                        f"{method.lower()} window={window:.1f} subject={subject:02d} "
                        f"block={block} acc={acc:.3f} seconds={elapsed:.2f}",
                        flush=True,
                    )
                append_csv(subject_block_path, block_subject_rows, SUBJECT_BLOCK_FIELDS)
                append_csv(prediction_path, block_prediction_rows, PREDICTION_FIELDS)
                done.add((round(window, 6), subject, block))
                write_progress(
                    args.output_dir / "progress.json",
                    {
                        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "completed_units": len(done),
                        "total_units": len(windows) * len(subjects) * len(blocks),
                        "last": {"window": window, "subject": subject, "block": block},
                    },
                )

    write_confusions(args.output_dir, read_prediction_rows(prediction_path), methods, spec)
    manifest["seconds"] = time.perf_counter() - started
    manifest["status"] = "complete"
    manifest["completed_units"] = len(completed_units(subject_block_path, methods))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
