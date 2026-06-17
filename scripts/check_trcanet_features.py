from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import BENCHMARK_CHANNELS_9, DATA_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import load_subject_trials
from vep_arena.methods.trcanet import fit_trca_filters, project_with_trca_filters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--window", type=float, default=0.4)
    args = parser.parse_args()

    spec = BenchmarkSpec()
    data = load_subject_trials(args.data_root, args.subject, args.window, BENCHMARK_CHANNELS_9, spec)
    # load_subject_trials returns classes x blocks x channels x samples.
    data = np.transpose(data, (0, 2, 3, 1))
    block_idx = args.block - 1
    train_blocks = [idx for idx in range(spec.blocks) if idx != block_idx]
    train = data[:, :, :, train_blocks]
    test = data[:, :, :, block_idx : block_idx + 1]
    weights = fit_trca_filters(train, spec.sampling_rate, n_fbs=3)
    train_features = project_with_trca_filters(train, weights, spec.sampling_rate)
    test_features = project_with_trca_filters(test, weights, spec.sampling_rate)
    print(
        {
            "subject": args.subject,
            "block": args.block,
            "window": args.window,
            "weights": tuple(weights.shape),
            "train_features": tuple(train_features.shape),
            "test_features": tuple(test_features.shape),
        }
    )


if __name__ == "__main__":
    main()
