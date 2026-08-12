# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_traditional_benchmark import plot_outputs, summarize, write_report
from vep_arena.config import PROJECT_ROOT, BenchmarkSpec


def copy_filtered_csv(src: Path, dst: Path, max_window: float) -> pd.DataFrame:
    df = pd.read_csv(src)
    if "window" in df.columns:
        df = df[df["window"] <= max_window + 1e-9].copy()
    df.to_csv(dst, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "results" / "traditional_9ch_w02_4s_full")
    parser.add_argument("--task-name", default="traditional_9ch_w02_2s_from_full")
    parser.add_argument("--max-window", type=float, default=2.0)
    args = parser.parse_args()

    out = PROJECT_ROOT / "results" / args.task_name
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"
    if figdir.exists():
        shutil.rmtree(figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    trials = copy_filtered_csv(args.source / "trials.csv", out / "trials.csv", args.max_window)
    preds = copy_filtered_csv(args.source / "predictions.csv", out / "predictions.csv", args.max_window)
    if (args.source / "runtime.csv").exists():
        copy_filtered_csv(args.source / "runtime.csv", out / "runtime.csv", args.max_window)

    spec = BenchmarkSpec()
    summary, subject, block = summarize(trials, spec)
    summary.to_csv(out / "summary.csv", index=False)
    subject.to_csv(out / "subject.csv", index=False)
    block.to_csv(out / "block.csv", index=False)
    plot_outputs(summary, subject, block, out)
    write_report(summary, out)

    source_manifest = {}
    source_manifest_path = args.source / "manifest.json"
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "task_name": args.task_name,
        "source": str(args.source),
        "source_task_name": source_manifest.get("task_name"),
        "operation": "subset_existing_results_and_redraw",
        "max_window": args.max_window,
        "windows": sorted(float(x) for x in trials["window"].unique()),
        "methods": sorted(str(x) for x in trials["method"].unique()),
        "rows_written": int(len(trials)),
        "prediction_rows_written": int(len(preds)),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
