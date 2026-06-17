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
from vep_arena.methods.cca import CCAClassifier, FBCCAClassifier


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT / "cca_fbcca")
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--blocks", default="1-6")
    parser.add_argument("--windows", default="default")
    parser.add_argument("--methods", default="CCA,FBCCA")
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--n-fbs", type=int, default=5)
    args = parser.parse_args()

    spec = BenchmarkSpec()
    subjects = parse_range(args.subjects)
    blocks = parse_range(args.blocks)
    windows = parse_windows(args.windows)
    methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subject_block_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    manifest: dict[str, object] = {
        "methods": methods,
        "dataset": "Benchmark",
        "protocol": "subject-specific leave-one-block-out",
        "subjects": subjects,
        "blocks": blocks,
        "windows": windows,
        "harmonics": args.harmonics,
        "n_fbs": args.n_fbs,
        "preprocess": {
            "CCA": "raw cropped 9ch trial, frequency-only sine/cosine references",
            "FBCCA": "raw cropped 9ch trial, Nakanishi/Chen filterbank, frequency-only sine/cosine references",
        },
        "reference_code": "D:/ProjData/_reference/TRCA-SSVEP/src/test_fbcca.m",
    }

    for window in windows:
        classifiers = {}
        if "CCA" in methods:
            classifiers["CCA"] = CCAClassifier(window=window, harmonics=args.harmonics, spec=spec)
        if "FBCCA" in methods:
            classifiers["FBCCA"] = FBCCAClassifier(window=window, harmonics=args.harmonics, n_fbs=args.n_fbs, spec=spec)
        for subject in subjects:
            data = load_subject_trials(args.data_root, subject, window, spec=spec)
            for block in blocks:
                block_idx = block - 1
                test_x = data[:, block_idx]
                test_y = np.arange(spec.classes, dtype=np.int64)
                for method, classifier in classifiers.items():
                    t0 = time.perf_counter()
                    pred, scores = classifier.predict(test_x)
                    seconds = time.perf_counter() - t0
                    acc = accuracy_score(test_y, pred)
                    itr = itr_bits_per_minute(float(acc), spec.classes, window + spec.cue_seconds)
                    subject_block_rows.append(
                        {
                            "method": method,
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
                                "method": method,
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
                        f"{method.lower()} window={window:.1f} subject={subject:02d} "
                        f"block={block} acc={acc:.3f} seconds={seconds:.2f}",
                        flush=True,
                    )

    with (args.output_dir / "subject_block.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "window", "subject", "block", "accuracy", "itr", "samples", "seconds"],
        )
        writer.writeheader()
        writer.writerows(subject_block_rows)

    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "window", "subject", "block", "true", "pred", "score_true", "score_pred"],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    for method in methods:
        rows = [row for row in prediction_rows if row["method"] == method]
        y_true = np.asarray([row["true"] for row in rows], dtype=np.int64)
        y_pred = np.asarray([row["pred"] for row in rows], dtype=np.int64)
        np.save(args.output_dir / f"confusion_{method.lower()}.npy", confusion_matrix(y_true, y_pred, labels=np.arange(spec.classes)))

    manifest["seconds"] = time.perf_counter() - started
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
