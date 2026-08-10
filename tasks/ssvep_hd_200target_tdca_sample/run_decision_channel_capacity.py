"""Decision channel capacity (Blahut-Arimoto) for HD200 offline TDCA grid.

Equivalent analysis to benchmark_decision_channel_capacity but for HD200:
- 14 subjects × {40,80,120,160,200} targets × {9,21,32,66} channels × {100-500ms} windows
- Input: pre-computed confusion matrices (counts) as .npy
- Output: per-condition C_BA + aggregate CSV
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.channel.capacity import (
    BAResult,
    capacity_ba,
    capacity_c0,
    capacity_c1,
    capacity_c2_closed,
    conditional_entropy_rows,
    mutual_info_uniform,
)
from vep_arena.channel.confusion import normalize_confusion

TASK_ROOT = Path(__file__).resolve().parent
CONFUSIONS_DIR = TASK_ROOT / "results" / "offline_tdca_grid" / "confusions"
OUTPUT_DIR = TASK_ROOT / "results" / "offline_tdca_grid" / "decision_channel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [40, 80, 120, 160, 200]
CHANNELS = [9, 21, 32, 66]
WINDOWS_MS = [100, 200, 300, 400, 500]
ALPHA = 0.0


def discover_subjects():
    """Find all subject IDs from confusion matrix filenames."""
    subjects = set()
    for f in CONFUSIONS_DIR.glob("confusion_S*_*.npy"):
        parts = f.stem.split("_")
        sub = parts[1]
        subjects.add(sub)
    return sorted(subjects, key=lambda s: int(s[1:]))


def analyze_one(counts: np.ndarray, M: int) -> dict:
    """Compute all capacity metrics from a confusion count matrix."""
    P = normalize_confusion(counts, alpha=ALPHA)
    accuracy = float(np.trace(counts)) / float(counts.sum()) if counts.sum() > 0 else 0.0
    samples = int(counts.sum())

    c0 = capacity_c0(M)
    c1 = capacity_c1(M, accuracy)

    closed = capacity_c2_closed(P)

    ba: BAResult = capacity_ba(P)

    i_uniform = mutual_info_uniform(P)
    h_rows = conditional_entropy_rows(P)
    h_y_given_x_uniform = float(np.mean(h_rows))

    delta_asm = ba.capacity - i_uniform if ba.capacity > i_uniform else 0.0

    q = ba.q
    q_min = float(q.min())
    q_max = float(q.max())
    q_support = int(np.sum(q > 1e-3))

    return {
        "samples": samples,
        "accuracy": accuracy,
        "c0": c0,
        "c1": c1,
        "c2_closed": closed.capacity,
        "c2_valid": closed.valid,
        "c2_condition": closed.condition,
        "c2_reason": closed.reason,
        "c_ba": ba.capacity,
        "ba_iterations": ba.iterations,
        "ba_gap": ba.gap,
        "ba_converged": ba.converged,
        "i_uniform": i_uniform,
        "h_y_given_x_uniform": h_y_given_x_uniform,
        "delta_asm": delta_asm,
        "c_ba_minus_c1": ba.capacity - c1,
        "c_ba_minus_i_uniform": ba.capacity - i_uniform,
        "q_min": q_min,
        "q_max": q_max,
        "q_support_1e3": q_support,
    }


def main():
    subjects = discover_subjects()
    print(f"Discovered {len(subjects)} subjects: {subjects}")
    print(f"Grid: {len(TARGETS)} targets × {len(CHANNELS)} channels × {len(WINDOWS_MS)} windows")
    total = len(subjects) * len(TARGETS) * len(CHANNELS) * len(WINDOWS_MS)
    print(f"Total conditions: {total}")

    records = []
    errors = []
    t0 = time.time()
    done = 0

    for sub in subjects:
        for n_targets in TARGETS:
            for n_ch in CHANNELS:
                for w_ms in WINDOWS_MS:
                    fname = f"confusion_{sub}_{n_targets}target_{n_ch}ch_w{w_ms}.npy"
                    fpath = CONFUSIONS_DIR / fname
                    if not fpath.exists():
                        errors.append({
                            "subject": sub, "targets": n_targets,
                            "channels": n_ch, "window_ms": w_ms,
                            "error": "file_not_found",
                        })
                        continue

                    try:
                        counts = np.load(fpath)
                        if counts.shape[0] != n_targets or counts.shape[1] != n_targets:
                            errors.append({
                                "subject": sub, "targets": n_targets,
                                "channels": n_ch, "window_ms": w_ms,
                                "error": f"shape_mismatch_{counts.shape}",
                            })
                            continue

                        result = analyze_one(counts, n_targets)
                        result["subject"] = sub
                        result["targets"] = n_targets
                        result["channels"] = n_ch
                        result["window_ms"] = w_ms
                        result["window"] = w_ms / 1000.0
                        records.append(result)

                    except Exception as e:
                        errors.append({
                            "subject": sub, "targets": n_targets,
                            "channels": n_ch, "window_ms": w_ms,
                            "error": str(e)[:200],
                        })

                    done += 1
                    if done % 100 == 0:
                        elapsed = time.time() - t0
                        rate = done / elapsed
                        eta = (total - done) / rate if rate > 0 else 0
                        print(f"  [{done}/{total}] {elapsed:.1f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nCompleted {done}/{total} in {elapsed:.1f}s")
    print(f"  Success: {len(records)}, Errors: {len(errors)}")

    # Save per-subject results
    df = pd.DataFrame(records)
    cols_front = ["subject", "targets", "channels", "window_ms", "window",
                  "samples", "accuracy", "c0", "c1", "c_ba", "ba_iterations",
                  "ba_gap", "ba_converged", "i_uniform"]
    cols_rest = [c for c in df.columns if c not in cols_front]
    df = df[cols_front + cols_rest]
    df.to_csv(OUTPUT_DIR / "capacity_by_subject_targets_channels_window.csv", index=False)
    print(f"\n  Per-subject CSV: {len(df)} rows")

    # Aggregate (pooled across subjects)
    if not df.empty:
        agg = df.groupby(["targets", "channels", "window_ms", "window"]).agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            c_ba_mean=("c_ba", "mean"),
            c_ba_std=("c_ba", "std"),
            c_ba_max=("c_ba", "max"),
            i_uniform_mean=("i_uniform", "mean"),
            ba_converged_all=("ba_converged", "all"),
            n_subjects=("subject", "count"),
        ).reset_index()
        agg["c0"] = agg["targets"].apply(lambda m: capacity_c0(m))
        agg.to_csv(OUTPUT_DIR / "capacity_aggregate.csv", index=False)
        print(f"  Aggregate CSV: {len(agg)} rows")

    # Save errors
    if errors:
        df_err = pd.DataFrame(errors)
        df_err.to_csv(OUTPUT_DIR / "errors.csv", index=False)
        print(f"  Errors CSV: {len(df_err)} rows")

    # Summary table
    print("\n" + "=" * 100)
    print("HD200 Decision Channel Capacity Summary (TDCA, offline grid)")
    print("=" * 100)
    print(f"{'Targets':>8s} {'Ch':>4s} {'Win(ms)':>7s} {'Acc%':>7s} {'C0':>6s} "
          f"{'C_BA':>7s} {'η%':>6s} {'I_uni':>7s} {'Δ_asm':>7s} {'Conv':>5s}")
    print("-" * 100)

    if not df.empty:
        for n_targets in TARGETS:
            c0 = capacity_c0(n_targets)
            for n_ch in [66]:
                for w_ms in WINDOWS_MS:
                    sub = df[(df["targets"] == n_targets) &
                             (df["channels"] == n_ch) &
                             (df["window_ms"] == w_ms)]
                    if sub.empty:
                        continue
                    acc = sub["accuracy"].mean() * 100
                    cba = sub["c_ba"].mean()
                    eta = cba / c0 * 100
                    iu = sub["i_uniform"].mean()
                    delta = sub["delta_asm"].mean()
                    conv = sub["ba_converged"].all()
                    print(f"{n_targets:>8d} {n_ch:>4d} {w_ms:>7d} {acc:>7.1f} {c0:>6.2f} "
                          f"{cba:>7.3f} {eta:>6.1f} {iu:>7.3f} {delta:>7.4f} {'Y' if conv else 'N':>5s}")
            print()

    print("=" * 100)
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
