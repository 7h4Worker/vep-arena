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

from vep_arena.config import DATA_ROOT, RUN_ROOT, WINDOWS, BenchmarkSpec
from vep_arena.data.benchmark import load_subject_trials
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.mvmd import MVMDCCAClassifier, SimpleTRCA, mvmd_reconstruct_trial, reference_bank


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
    if text == "default":
        return list(WINDOWS)
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def transform_subject(
    data: np.ndarray,
    *,
    refs: np.ndarray | None,
    alpha: float,
    n_modes: int,
    max_iter: int,
    tol: float,
    drop_first: bool,
) -> np.ndarray:
    out = np.zeros_like(data)
    classes, blocks = data.shape[:2]
    for target in range(classes):
        for block in range(blocks):
            out[target, block] = mvmd_reconstruct_trial(
                data[target, block],
                alpha=alpha,
                n_modes=n_modes,
                max_iter=max_iter,
                tol=tol,
                refs=refs,
                drop_first=drop_first,
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT / "mvmd_methods")
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--blocks", default="1-6")
    parser.add_argument("--windows", default="0.5,1.0")
    parser.add_argument("--methods", default="MVMD-CCA,MVMD-TRCA,SA-MVMD-TRCA")
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--assist-harmonics", type=int, default=2)
    parser.add_argument("--n-modes", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=2000.0)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--drop-first-mode", action="store_true")
    args = parser.parse_args()

    spec = BenchmarkSpec()
    subjects = parse_range(args.subjects)
    blocks = parse_range(args.blocks)
    windows = parse_windows(args.windows)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subject_block_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "methods": methods,
        "dataset": "Benchmark",
        "protocol": "subject-specific leave-one-block-out for MVMD-TRCA/SA-MVMD-TRCA; calibration-free for MVMD-CCA",
        "subjects": subjects,
        "blocks": blocks,
        "windows": windows,
        "n_modes": args.n_modes,
        "alpha": args.alpha,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "harmonics": args.harmonics,
        "assist_harmonics": args.assist_harmonics,
        "drop_first_mode": args.drop_first_mode,
        "reference_code": "D:/ProjData/_reference/MDApproach_PS/MVMD.m",
        "notes": [
            "MVMD-CCA is an Arena implementation of the MVMD + CCA decoding line.",
            "MVMD-TRCA decomposes each trial with MVMD, reconstructs EEG modes, then trains ordinary TRCA.",
            "SA-MVMD-TRCA is an efficient local reproduction style: all Benchmark sine/cosine references are appended once as sinusoidal assistance, then ordinary TRCA is trained on reconstructed EEG.",
            "This is not official author code; use as a controlled local reproduction until paper code/settings are obtained.",
        ],
    }
    started = time.perf_counter()

    for window in windows:
        assist_refs = reference_bank(window, args.assist_harmonics, spec)
        mvmd_cca = (
            MVMDCCAClassifier(
                window,
                harmonics=args.harmonics,
                n_modes=args.n_modes,
                alpha=args.alpha,
                max_iter=args.max_iter,
                tol=args.tol,
                spec=spec,
            )
            if "MVMD-CCA" in methods
            else None
        )
        for subject in subjects:
            data = load_subject_trials(args.data_root, subject, window, spec=spec)
            mvmd_trca_data = None
            if "MVMD-TRCA" in methods:
                t0 = time.perf_counter()
                mvmd_trca_data = transform_subject(
                    data,
                    refs=None,
                    alpha=args.alpha,
                    n_modes=args.n_modes,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    drop_first=args.drop_first_mode,
                )
                print(
                    f"mvmd-transform window={window:.1f} subject={subject:02d} seconds={time.perf_counter() - t0:.2f}",
                    flush=True,
                )
            sa_data = None
            if "SA-MVMD-TRCA" in methods:
                t0 = time.perf_counter()
                sa_data = transform_subject(
                    data,
                    refs=assist_refs,
                    alpha=args.alpha,
                    n_modes=args.n_modes,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    drop_first=args.drop_first_mode,
                )
                print(
                    f"sa-transform window={window:.1f} subject={subject:02d} seconds={time.perf_counter() - t0:.2f}",
                    flush=True,
                )
            for block in blocks:
                block_idx = block - 1
                test_y = np.arange(spec.classes, dtype=np.int64)
                if mvmd_cca is not None:
                    test_x = data[:, block_idx]
                    t0 = time.perf_counter()
                    pred, scores = mvmd_cca.predict(test_x)
                    seconds = time.perf_counter() - t0
                    acc = accuracy_score(test_y, pred)
                    itr = itr_bits_per_minute(float(acc), spec.classes, window + spec.cue_seconds)
                    subject_block_rows.append(
                        {
                            "method": "MVMD-CCA",
                            "window": window,
                            "subject": subject,
                            "block": block,
                            "accuracy": float(acc),
                            "itr": float(itr),
                            "samples": spec.classes,
                            "seconds": seconds,
                        }
                    )
                    for true_label, pred_label in zip(test_y, pred):
                        prediction_rows.append(
                            {
                                "method": "MVMD-CCA",
                                "window": window,
                                "subject": subject,
                                "block": block,
                                "true": int(true_label),
                                "pred": int(pred_label),
                                "score_true": float(scores[int(true_label), int(true_label)]),
                                "score_pred": float(scores[int(true_label), int(pred_label)]),
                            }
                        )
                    print(
                        f"mvmd-cca window={window:.1f} subject={subject:02d} block={block} "
                        f"acc={acc:.3f} seconds={seconds:.2f}",
                        flush=True,
                    )

                if mvmd_trca_data is not None:
                    train_blocks = [idx for idx in range(spec.blocks) if idx != block_idx]
                    train_x = mvmd_trca_data[:, train_blocks]
                    test_x = mvmd_trca_data[:, block_idx]
                    t0 = time.perf_counter()
                    model = SimpleTRCA.fit(train_x, ensemble=False)
                    pred, scores = model.predict(test_x)
                    seconds = time.perf_counter() - t0
                    acc = accuracy_score(test_y, pred)
                    itr = itr_bits_per_minute(float(acc), spec.classes, window + spec.cue_seconds)
                    subject_block_rows.append(
                        {
                            "method": "MVMD-TRCA",
                            "window": window,
                            "subject": subject,
                            "block": block,
                            "accuracy": float(acc),
                            "itr": float(itr),
                            "samples": spec.classes,
                            "seconds": seconds,
                        }
                    )
                    for true_label, pred_label in zip(test_y, pred):
                        prediction_rows.append(
                            {
                                "method": "MVMD-TRCA",
                                "window": window,
                                "subject": subject,
                                "block": block,
                                "true": int(true_label),
                                "pred": int(pred_label),
                                "score_true": float(scores[int(true_label), int(true_label)]),
                                "score_pred": float(scores[int(true_label), int(pred_label)]),
                            }
                        )
                    print(
                        f"mvmd-trca window={window:.1f} subject={subject:02d} block={block} "
                        f"acc={acc:.3f} seconds={seconds:.2f}",
                        flush=True,
                    )

                if sa_data is not None:
                    train_blocks = [idx for idx in range(spec.blocks) if idx != block_idx]
                    train_x = sa_data[:, train_blocks]
                    test_x = sa_data[:, block_idx]
                    t0 = time.perf_counter()
                    model = SimpleTRCA.fit(train_x, ensemble=False)
                    pred, scores = model.predict(test_x)
                    seconds = time.perf_counter() - t0
                    acc = accuracy_score(test_y, pred)
                    itr = itr_bits_per_minute(float(acc), spec.classes, window + spec.cue_seconds)
                    subject_block_rows.append(
                        {
                            "method": "SA-MVMD-TRCA",
                            "window": window,
                            "subject": subject,
                            "block": block,
                            "accuracy": float(acc),
                            "itr": float(itr),
                            "samples": spec.classes,
                            "seconds": seconds,
                        }
                    )
                    for true_label, pred_label in zip(test_y, pred):
                        prediction_rows.append(
                            {
                                "method": "SA-MVMD-TRCA",
                                "window": window,
                                "subject": subject,
                                "block": block,
                                "true": int(true_label),
                                "pred": int(pred_label),
                                "score_true": float(scores[int(true_label), int(true_label)]),
                                "score_pred": float(scores[int(true_label), int(pred_label)]),
                            }
                        )
                    print(
                        f"sa-mvmd-trca window={window:.1f} subject={subject:02d} block={block} "
                        f"acc={acc:.3f} seconds={seconds:.2f}",
                        flush=True,
                    )

    write_csv(
        args.output_dir / "subject_block.csv",
        subject_block_rows,
        ["method", "window", "subject", "block", "accuracy", "itr", "samples", "seconds"],
    )
    write_csv(
        args.output_dir / "predictions.csv",
        prediction_rows,
        ["method", "window", "subject", "block", "true", "pred", "score_true", "score_pred"],
    )

    for method in methods:
        rows = [row for row in prediction_rows if row["method"] == method]
        if not rows:
            continue
        y_true = np.asarray([row["true"] for row in rows], dtype=np.int64)
        y_pred = np.asarray([row["pred"] for row in rows], dtype=np.int64)
        np.save(args.output_dir / f"confusion_{method.lower().replace('-', '_')}.npy", confusion_matrix(y_true, y_pred, labels=np.arange(spec.classes)))

    manifest["seconds"] = time.perf_counter() - started
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
