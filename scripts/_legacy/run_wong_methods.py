# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Run Wong-style multi-stimulus SSVEP method adapters.
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_traditional_benchmark import plot_outputs, summarize, write_report
from vep_arena.config import DATA_ROOT, RUN_ROOT, BenchmarkSpec
from vep_arena.data.epochs import CanonicalEpochStore, EpochRequest, epoch_fingerprint
from vep_arena.data.presets import benchmark_9ch_default
from vep_arena.metrics import itr_bits_per_minute
from vep_arena.methods.multistimulus import MSCCA, MSETRCA


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


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_test_arrays(epochs: np.ndarray, block_idx: int, spec: BenchmarkSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_blocks = [idx for idx in range(spec.blocks) if idx != block_idx]
    train_x = epochs[:, train_blocks].reshape(spec.classes * len(train_blocks), *epochs.shape[2:])
    train_y = np.repeat(np.arange(spec.classes, dtype=np.int64), len(train_blocks))
    test_x = epochs[:, block_idx]
    test_y = np.arange(spec.classes, dtype=np.int64)
    return train_x, train_y, test_x, test_y


def signed_square(scores: np.ndarray) -> np.ndarray:
    return np.sign(scores) * scores * scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT / "wong_methods")
    parser.add_argument("--subjects", default="1")
    parser.add_argument("--blocks", default="1")
    parser.add_argument("--windows", default="1.0")
    parser.add_argument("--methods", default="MSCCA,MSETRCA,MSCCA+MSETRCA")
    parser.add_argument("--n-fbs", type=int, default=5)
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--mscca-neighbors", type=int, default=12)
    parser.add_argument("--msetrca-neighbors", type=int, default=2)
    parser.add_argument("--cache-window", type=float, default=None)
    args = parser.parse_args()

    spec = BenchmarkSpec()
    preset = benchmark_9ch_default(args.data_root)
    store = CanonicalEpochStore()
    subjects = parse_range(args.subjects)
    blocks = parse_range(args.blocks)
    windows = parse_windows(args.windows)
    methods = [item.strip().upper() for item in args.methods.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trial_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    cache_window = args.cache_window if args.cache_window is not None else max(windows)

    for window in windows:
        for subject in subjects:
            request = EpochRequest(
                preset=preset,
                subject=subject,
                window=window,
                kind="filterbank",
                n_fbs=args.n_fbs,
                cache_window=cache_window,
            )
            epochs = store.load_or_create(request)
            for block in blocks:
                block_idx = block - 1
                train_x, train_y, test_x, test_y = train_test_arrays(epochs, block_idx, spec)
                scores_by_method: dict[str, np.ndarray] = {}
                for method in methods:
                    if method == "MSCCA+MSETRCA":
                        continue
                    t0 = time.perf_counter()
                    if method == "MSCCA":
                        model = MSCCA(
                            window,
                            n_neighbor=args.mscca_neighbors,
                            harmonics=args.harmonics,
                            n_fbs=args.n_fbs,
                            spec=spec,
                        ).fit(train_x, train_y)
                    elif method == "MSETRCA":
                        model = MSETRCA(n_neighbor=args.msetrca_neighbors, n_fbs=args.n_fbs).fit(train_x, train_y)
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    pred, scores = model.predict(test_x)
                    seconds = time.perf_counter() - t0
                    scores_by_method[method] = scores
                    acc = accuracy_score(test_y, pred)
                    itr = itr_bits_per_minute(float(acc), spec.classes, window + spec.cue_seconds)
                    trial_rows.append(
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
                    runtime_rows.append(
                        {
                            "method": method,
                            "window": window,
                            "subject": subject,
                            "block": block,
                            "seconds": seconds,
                        }
                    )
                    for true_label, pred_label in zip(test_y, pred):
                        pred_rows.append(
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
                    print(f"{method} window={window:.1f} subject={subject:02d} block={block} acc={acc:.3f}", flush=True)

                if "MSCCA+MSETRCA" in methods:
                    if "MSCCA" not in scores_by_method or "MSETRCA" not in scores_by_method:
                        raise ValueError("MSCCA+MSETRCA requires both MSCCA and MSETRCA in --methods.")
                    scores = signed_square(scores_by_method["MSCCA"]) + signed_square(scores_by_method["MSETRCA"])
                    pred = np.argmax(scores, axis=1)
                    acc = accuracy_score(test_y, pred)
                    itr = itr_bits_per_minute(float(acc), spec.classes, window + spec.cue_seconds)
                    trial_rows.append(
                        {
                            "method": "MSCCA+MSETRCA",
                            "window": window,
                            "subject": subject,
                            "block": block,
                            "accuracy": float(acc),
                            "itr": float(itr),
                            "samples": spec.classes,
                            "seconds": 0.0,
                        }
                    )
                    for true_label, pred_label in zip(test_y, pred):
                        pred_rows.append(
                            {
                                "method": "MSCCA+MSETRCA",
                                "window": window,
                                "subject": subject,
                                "block": block,
                                "true": int(true_label),
                                "pred": int(pred_label),
                                "score_true": float(scores[int(true_label), int(true_label)]),
                                "score_pred": float(scores[int(true_label), int(pred_label)]),
                            }
                        )
                    print(f"MSCCA+MSETRCA window={window:.1f} subject={subject:02d} block={block} acc={acc:.3f}", flush=True)

    write_csv(
        args.output_dir / "trials.csv",
        trial_rows,
        ["method", "window", "subject", "block", "accuracy", "itr", "samples", "seconds"],
    )
    write_csv(
        args.output_dir / "predictions.csv",
        pred_rows,
        ["method", "window", "subject", "block", "true", "pred", "score_true", "score_pred"],
    )
    write_csv(
        args.output_dir / "runtime.csv",
        runtime_rows,
        ["method", "window", "subject", "block", "seconds"],
    )
    trials = pd.DataFrame(trial_rows)
    preds = pd.DataFrame(pred_rows)
    summary, subject_df, block_df = summarize(trials, spec)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    subject_df.to_csv(args.output_dir / "subject.csv", index=False)
    block_df.to_csv(args.output_dir / "block.csv", index=False)
    labels = np.arange(spec.classes, dtype=np.int64)
    for method in sorted(trials["method"].unique()):
        rows = preds[preds["method"] == method]
        np.save(args.output_dir / f"confusion_{method.lower().replace('+', '_plus_')}.npy", confusion_matrix(rows["true"], rows["pred"], labels=labels))
    plot_outputs(summary, subject_df, block_df, args.output_dir)
    write_report(summary, args.output_dir)
    manifest = {
        "methods": methods,
        "dataset_preset": preset.manifest(),
        "epoch_fingerprint": epoch_fingerprint(
            EpochRequest(preset=preset, subject=subjects[0], window=windows[0], kind="filterbank", n_fbs=args.n_fbs, cache_window=cache_window)
        ),
        "subjects": subjects,
        "blocks": blocks,
        "windows": windows,
        "n_fbs": args.n_fbs,
        "harmonics": args.harmonics,
        "mscca_neighbors": args.mscca_neighbors,
        "msetrca_neighbors": args.msetrca_neighbors,
        "source_references": [
            "https://github.com/edwin465/SSVEP-tlCCA",
            "https://github.com/edwin465/SSVEP-OACCA",
            "https://github.com/pikipity/SSVEP-Analysis-Toolbox",
            "Wong et al. JNE 2020 DOI 10.1088/1741-2552/ab2373",
        ],
        "seconds": time.perf_counter() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
