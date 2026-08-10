from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vep_arena.channel.asymmetry import asymmetry_score
from vep_arena.channel.capacity import (
    capacity_ba,
    capacity_c0,
    capacity_c1,
    capacity_c2_closed,
    conditional_entropy_rows,
    mutual_info_uniform,
)
from vep_arena.channel.class_selection import gain_decomposition, greedy_codebook_pruning
from vep_arena.channel.confusion import confusion_counts, normalize_confusion
from vep_arena.config import BENCHMARK_FREQS, BenchmarkSpec


TASK = Path(__file__).resolve().parent
RESULTS = TASK / "results"
ANALYSIS = RESULTS / "analysis"


def safe_key(method: str, window: float, subject: int | str) -> str:
    w = f"{float(window):.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{method.lower()}_w{w}_s{subject}"


def join_values(values: list[int] | list[float], fmt: str = "{}") -> str:
    return " ".join(fmt.format(x) for x in values)


def subset_frequency_summary(subset: tuple[int, ...]) -> dict[str, object]:
    freqs = [float(BENCHMARK_FREQS[int(idx)]) for idx in subset]
    sorted_freqs = sorted(freqs)
    gaps = np.diff(sorted_freqs) if len(sorted_freqs) > 1 else np.asarray([], dtype=float)
    return {
        "best_subset": join_values([int(x) for x in subset]),
        "best_frequencies_hz": join_values(freqs, "{:.1f}"),
        "best_min_frequency_hz": float(min(freqs)) if freqs else np.nan,
        "best_max_frequency_hz": float(max(freqs)) if freqs else np.nan,
        "best_frequency_span_hz": float(max(freqs) - min(freqs)) if freqs else np.nan,
        "best_mean_gap_hz": float(np.mean(gaps)) if gaps.size else np.nan,
        "best_min_gap_hz": float(np.min(gaps)) if gaps.size else np.nan,
    }


def analyze_group(counts: np.ndarray, alpha: float) -> tuple[dict[str, object], np.ndarray]:
    classes = counts.shape[0]
    total = float(np.sum(counts))
    correct = float(np.trace(counts))
    accuracy = correct / total if total else 0.0
    P = normalize_confusion(counts, alpha=alpha)
    ba = capacity_ba(P)
    closed = capacity_c2_closed(P)
    i_uniform = mutual_info_uniform(P)
    h_rows = conditional_entropy_rows(P)
    c1 = capacity_c1(classes, accuracy)
    row = {
        "samples": int(total),
        "accuracy": float(accuracy),
        "c0": capacity_c0(classes),
        "c1": float(c1),
        "c2_closed": float(closed.capacity) if np.isfinite(closed.capacity) else np.nan,
        "c2_valid": bool(closed.valid),
        "c2_condition": float(closed.condition),
        "c2_reason": closed.reason,
        "c_ba": float(ba.capacity),
        "ba_iterations": int(ba.iterations),
        "ba_gap": float(ba.gap),
        "ba_converged": bool(ba.converged),
        "i_uniform": float(i_uniform),
        "h_y_given_x_uniform": float(np.mean(h_rows)),
        "delta_asm": float(asymmetry_score(P)),
        "c_ba_minus_c1": float(ba.capacity - c1),
        "c_ba_minus_i_uniform": float(ba.capacity - i_uniform),
        "q_min": float(np.min(ba.q)),
        "q_max": float(np.max(ba.q)),
        "q_support_1e3": int(np.sum(ba.q >= 1e-3)),
    }
    return row, ba.q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=RESULTS / "input" / "predictions.csv")
    parser.add_argument("--classes", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--pruning-alpha", type=float, default=0.0)
    parser.add_argument("--pruning-starts", type=int, default=1)
    parser.add_argument("--pruning-max-size", type=int, default=None)
    parser.add_argument("--pruning-ba-tol", type=float, default=1e-5)
    parser.add_argument("--skip-pruning", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS)
    args = parser.parse_args()

    analysis_dir = args.output_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(args.predictions)
    required = {"method", "window", "subject", "true", "pred"}
    missing = sorted(required - set(preds.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    counts_payload: dict[str, np.ndarray] = {}
    aggregate_payload: dict[str, np.ndarray] = {}
    index_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    q_rows: list[dict[str, object]] = []
    spec = BenchmarkSpec(classes=args.classes)

    grouped = preds.groupby(["method", "window", "subject"], sort=True)
    for (method, window, subject), group in grouped:
        counts = confusion_counts(group["true"], group["pred"], args.classes)
        key = safe_key(str(method), float(window), int(subject))
        counts_payload[key] = counts
        index_rows.append({"key": key, "method": method, "window": float(window), "subject": int(subject)})
        row, q = analyze_group(counts, args.alpha)
        row.update({"method": method, "window": float(window), "subject": int(subject)})
        capacity_rows.append(row)
        for class_idx, prob in enumerate(q):
            q_rows.append(
                {
                    "method": method,
                    "window": float(window),
                    "subject": int(subject),
                    "class": class_idx,
                    "frequency_hz": BENCHMARK_FREQS[class_idx],
                    "q_star": float(prob),
                }
            )

    aggregate_rows: list[dict[str, object]] = []
    for (method, window), group in preds.groupby(["method", "window"], sort=True):
        counts = confusion_counts(group["true"], group["pred"], args.classes)
        key = safe_key(str(method), float(window), "all")
        aggregate_payload[key] = counts
        row, q = analyze_group(counts, args.alpha)
        row.update({"method": method, "window": float(window), "subject": "all"})
        aggregate_rows.append(row)

    np.savez_compressed(analysis_dir / "confusion_counts_subject_method_window.npz", **counts_payload)
    np.savez_compressed(analysis_dir / "confusion_counts_method_window_aggregate.npz", **aggregate_payload)
    pd.DataFrame(index_rows).to_csv(analysis_dir / "confusion_index.csv", index=False)
    pd.DataFrame(capacity_rows).sort_values(["method", "window", "subject"]).to_csv(
        analysis_dir / "capacity_by_subject_method_window.csv", index=False
    )
    pd.DataFrame(aggregate_rows).sort_values(["method", "window"]).to_csv(
        analysis_dir / "capacity_by_method_window_aggregate.csv", index=False
    )
    pd.DataFrame(q_rows).sort_values(["method", "window", "subject", "class"]).to_csv(
        analysis_dir / "qstar_by_class.csv", index=False
    )

    if not args.skip_pruning:
        pruning_rows: list[dict[str, object]] = []
        path_rows: list[dict[str, object]] = []
        best_rows: list[dict[str, object]] = []
        for key, counts in aggregate_payload.items():
            parts = [row for row in aggregate_rows if safe_key(str(row["method"]), float(row["window"]), "all") == key]
            if not parts:
                continue
            method = str(parts[0]["method"])
            window = float(parts[0]["window"])
            selected = greedy_codebook_pruning(
                counts,
                alpha=args.pruning_alpha,
                max_size=args.pruning_max_size,
                seed_classes="top_diag",
                n_starts=args.pruning_starts,
                scoring="ba",
                ba_tol=args.pruning_ba_tol,
            )
            gains = gain_decomposition(counts, alpha=args.pruning_alpha, selected=selected)
            best_step = max(selected.path, key=lambda step: step.capacity)
            freq_summary = subset_frequency_summary(best_step.subset)
            pruning_rows.append({"method": method, "window": window, **gains, **freq_summary})
            best_rows.append(
                {
                    "method": method,
                    "window": window,
                    "seed_class": selected.seed_class,
                    "best_size": best_step.size,
                    "best_added_class": best_step.added_class,
                    "best_capacity_ba": best_step.capacity,
                    "best_c1": best_step.c1,
                    "best_i_uniform": best_step.i_uniform,
                    "best_c2_closed": best_step.c2_closed,
                    "best_c2_valid": best_step.c2_valid,
                    "best_accuracy": best_step.accuracy,
                    "best_retained_mass": best_step.retained_mass,
                    "best_ba_minus_c1": best_step.ba_minus_c1,
                    "best_ba_minus_uniform": best_step.ba_minus_uniform,
                    **freq_summary,
                }
            )
            for step in selected.path:
                subset = [int(x) for x in step.subset]
                freqs = [float(BENCHMARK_FREQS[idx]) for idx in subset]
                path_rows.append(
                    {
                        "method": method,
                        "window": window,
                        "seed_class": selected.seed_class,
                        "size": step.size,
                        "added_class": step.added_class,
                        "capacity": step.capacity,
                        "capacity_ba": step.capacity,
                        "c1": step.c1,
                        "i_uniform": step.i_uniform,
                        "c2_closed": step.c2_closed,
                        "c2_valid": step.c2_valid,
                        "c2_condition": step.c2_condition,
                        "c2_reason": step.c2_reason,
                        "accuracy": step.accuracy,
                        "retained_mass": step.retained_mass,
                        "ba_minus_c1": step.ba_minus_c1,
                        "ba_minus_uniform": step.ba_minus_uniform,
                        "q_min": step.q_min,
                        "q_max": step.q_max,
                        "q_support_1e3": step.q_support_1e3,
                        "subset": join_values(subset),
                        "frequencies_hz": join_values(freqs, "{:.1f}"),
                        "rate_bits_per_min": float(60.0 * step.capacity / window),
                    }
                )
        pd.DataFrame(pruning_rows).sort_values(["method", "window"]).to_csv(
            analysis_dir / "codebook_pruning_by_method_window.csv", index=False
        )
        path_df = pd.DataFrame(path_rows).sort_values(["method", "window", "size"])
        path_df.to_csv(
            analysis_dir / "codebook_pruning_path.csv", index=False
        )
        path_df.to_csv(analysis_dir / "costa_selection_path.csv", index=False)
        pd.DataFrame(best_rows).sort_values(["method", "window"]).to_csv(
            analysis_dir / "costa_best_codebooks.csv", index=False
        )

    manifest = {
        "predictions": str(args.predictions),
        "rows": int(len(preds)),
        "classes": args.classes,
        "subjects": sorted(int(x) for x in preds["subject"].unique()),
        "methods": sorted(str(x) for x in preds["method"].unique()),
        "windows": sorted(float(x) for x in preds["window"].unique()),
        "alpha": args.alpha,
        "pruning_alpha": args.pruning_alpha,
        "pruning_starts": args.pruning_starts,
        "pruning_max_size": args.pruning_max_size,
        "pruning_ba_tol": args.pruning_ba_tol,
        "frequencies_hz": list(BENCHMARK_FREQS[: spec.classes]),
    }
    (analysis_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(analysis_dir)


if __name__ == "__main__":
    main()
