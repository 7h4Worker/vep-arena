# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_traditional_benchmark import plot_outputs, summarize, write_report
from vep_arena.config import PROJECT_ROOT, BenchmarkSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traditional", type=Path, default=PROJECT_ROOT / "results" / "traditional_9ch_w02_2s_from_full")
    parser.add_argument("--tdca", type=Path, default=PROJECT_ROOT / "results" / "tdca_9ch_w02_2s")
    parser.add_argument("--task-name", default="traditional_tdca_9ch_w02_2s")
    args = parser.parse_args()

    out = PROJECT_ROOT / "results" / args.task_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    trials = pd.concat(
        [pd.read_csv(args.traditional / "trials.csv"), pd.read_csv(args.tdca / "trials.csv")],
        ignore_index=True,
    )
    predictions = pd.concat(
        [pd.read_csv(args.traditional / "predictions.csv"), pd.read_csv(args.tdca / "predictions.csv")],
        ignore_index=True,
    )
    runtime_frames = []
    for source in [args.traditional, args.tdca]:
        path = source / "runtime.csv"
        if path.exists():
            runtime_frames.append(pd.read_csv(path))
    runtime = pd.concat(runtime_frames, ignore_index=True) if runtime_frames else pd.DataFrame()

    spec = BenchmarkSpec()
    summary, subject, block = summarize(trials, spec)
    trials.to_csv(out / "trials.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    runtime.to_csv(out / "runtime.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    subject.to_csv(out / "subject.csv", index=False)
    block.to_csv(out / "block.csv", index=False)
    plot_outputs(summary, subject, block, out)
    write_report(summary, out)

    manifest = {
        "task_name": args.task_name,
        "operation": "combine_existing_results_and_redraw",
        "sources": {
            "traditional": str(args.traditional),
            "tdca": str(args.tdca),
        },
        "methods": sorted(str(x) for x in trials["method"].unique()),
        "windows": sorted(float(x) for x in trials["window"].unique()),
        "rows_written": int(len(trials)),
        "prediction_rows_written": int(len(predictions)),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
