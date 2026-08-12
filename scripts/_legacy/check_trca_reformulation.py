# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import PROJECT_ROOT
from vep_arena.data.epochs import CanonicalEpochStore, EpochRequest
from vep_arena.data.presets import benchmark_9ch_default
from vep_arena.methods.trca_core import trca_filter, trca_filter_original_pairwise


def aligned_delta(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.dot(a, b)) < 0:
        b = -b
    return float(np.max(np.abs(a - b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--fb", type=int, default=0)
    parser.add_argument("--epoch-cache", type=Path, default=PROJECT_ROOT / "runs" / "canonical_epochs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "trca_reformulation_check")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preset = benchmark_9ch_default()
    store = CanonicalEpochStore(args.epoch_cache)
    epochs = store.load_or_create(
        EpochRequest(preset=preset, subject=args.subject, window=args.window, kind="filterbank", n_fbs=5)
    )
    trials = epochs[args.target, :, args.fb]

    original_times = []
    reformulated_times = []
    original_filter = None
    reformulated_filter = None
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        original_filter = trca_filter_original_pairwise(trials)
        original_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        reformulated_filter = trca_filter(trials)
        reformulated_times.append(time.perf_counter() - t0)

    assert original_filter is not None and reformulated_filter is not None
    rows = [
        {
            "implementation": "original_pairwise",
            "mean_seconds": float(np.mean(original_times)),
            "median_seconds": float(np.median(original_times)),
            "std_seconds": float(np.std(original_times)),
            "repeats": args.repeats,
        },
        {
            "implementation": "reformulated",
            "mean_seconds": float(np.mean(reformulated_times)),
            "median_seconds": float(np.median(reformulated_times)),
            "std_seconds": float(np.std(reformulated_times)),
            "repeats": args.repeats,
        },
    ]
    pd.DataFrame(rows).to_csv(args.output_dir / "runtime.csv", index=False)
    manifest = {
        "task_name": "trca_reformulation_check",
        "source": "Chiang et al., Reformulating Task-Related Component Analysis for Reducing its Computational Complexity",
        "subject": args.subject,
        "window": args.window,
        "target": args.target,
        "filterbank_index": args.fb,
        "trials_shape": list(trials.shape),
        "max_abs_filter_delta_after_sign_alignment": aligned_delta(original_filter, reformulated_filter),
        "speedup_mean": rows[0]["mean_seconds"] / rows[1]["mean_seconds"] if rows[1]["mean_seconds"] > 0 else None,
        "files": ["runtime.csv", "manifest.json"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
