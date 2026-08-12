# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Import external DNN SSVEP aggregate results into the Arena result schema.
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_traditional_benchmark import plot_outputs, summarize
from vep_arena.config import RESULT_ROOT, BenchmarkSpec
from vep_arena.metrics import itr_bits_per_minute


REQUIRED_COLUMNS = {"signal_length", "block", "subject", "accuracy", "samples", "seconds"}


def import_subject_block(source_csv: Path, method: str, spec: BenchmarkSpec) -> pd.DataFrame:
    source = pd.read_csv(source_csv)
    missing = REQUIRED_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError(f"{source_csv} is missing required columns: {sorted(missing)}")

    trials = source.rename(columns={"signal_length": "window"}).copy()
    trials.insert(0, "method", method)
    trials["window"] = trials["window"].astype(float)
    trials["block"] = trials["block"].astype(int)
    trials["subject"] = trials["subject"].astype(int)
    trials["accuracy"] = trials["accuracy"].astype(float)
    trials["samples"] = trials["samples"].astype(int)
    trials["seconds"] = trials["seconds"].astype(float)
    trials["itr"] = [
        itr_bits_per_minute(float(row.accuracy), spec.classes, float(row.window) + spec.cue_seconds)
        for row in trials.itertuples(index=False)
    ]
    return trials[["method", "window", "subject", "block", "accuracy", "itr", "samples", "seconds"]]


def write_report(out_dir: Path, summary: pd.DataFrame, source_dir: Path, source_csv: Path) -> None:
    lines = [
        "# DNN External Baseline Import",
        "",
        "Scope: imported aggregate DNN results from the sibling PyTorch reproduction project.",
        "",
        "Protocol: Tsinghua Benchmark 9-channel, leave-one-block-out aggregate rows.",
        "Preprocessing: DNN project canonical crop with 3 Chebyshev-I order-2 subbands; no Arena filterbank retraining was run by this importer.",
        "",
        "## Window Summary",
        "",
        "| Window | Accuracy | ITR | Subjects |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in summary.sort_values("window").itertuples(index=False):
        lines.append(f"| {row.window:.1f}s | {row.accuracy:.4f} | {row.itr:.2f} | {row.subjects} |")
    lines.extend(
        [
            "",
            "## Source",
            "",
            f"- Source directory: `{source_dir}`",
            f"- Source table: `{source_csv}`",
            "- Trial-level predictions are not imported because the clean result table contains subject/block aggregates only.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("D:/ProjData/proj_python/dnn_ssvep_pytorch/results_clean"),
        help="Directory containing DNN clean CSV result tables.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "dnn_import_existing")
    parser.add_argument("--method", default="DNN")
    args = parser.parse_args()

    spec = BenchmarkSpec()
    source_csv = args.source_dir / "subject_block_results.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trials = import_subject_block(source_csv, args.method, spec)
    summary, subject_df, block_df = summarize(trials, spec)

    trials.to_csv(args.output_dir / "trials.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    subject_df.to_csv(args.output_dir / "subject.csv", index=False)
    block_df.to_csv(args.output_dir / "block.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": args.method,
                "window": row.window,
                "stage": "external_import",
                "seconds": row.seconds,
            }
            for row in trials.groupby(["method", "window"], as_index=False)["seconds"].sum().itertuples(index=False)
        ]
    ).to_csv(args.output_dir / "runtime.csv", index=False)

    plot_outputs(summary, subject_df, block_df, args.output_dir)
    write_report(args.output_dir, summary, args.source_dir, source_csv)

    manifest = {
        "task_name": args.output_dir.name,
        "method": args.method,
        "status": "complete",
        "imported_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_project": "D:/ProjData/proj_python/dnn_ssvep_pytorch",
        "source_dir": str(args.source_dir),
        "source_csv": str(source_csv),
        "source_scope": "Benchmark 9-channel DNN reproduction aggregate results.",
        "protocol": "subject-specific leave-one-block-out aggregate import",
        "preprocessing": {
            "cue_seconds": spec.cue_seconds,
            "visual_latency_seconds": spec.latency_seconds,
            "crop_start_seconds": spec.cue_seconds + spec.latency_seconds,
            "channels": "Benchmark 9ch: Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, O2",
            "filterbank": "DNN project: 3 Chebyshev-I order-2 subbands, [8*i, 90] Hz.",
            "notch": "None observed in dnn_ssvep_pytorch preprocessing.",
            "arena_filterbank_retrained": False,
        },
        "limitations": [
            "This import uses aggregate subject/block accuracies only.",
            "Trial-level predictions, confusion matrices, and Arena-trained checkpoints are not available from results_clean.",
            "Changing Arena preprocessing requires retraining the DNN rather than reusing these imported scores.",
        ],
        "rows_written": int(len(trials)),
        "windows_written": sorted(float(x) for x in trials["window"].unique()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
