# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-18
# Last updated: 2026-06-18
# Description: Run Arena-side DNN global/fine-tuned evaluations across windows.
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import RESULT_ROOT


def parse_windows(text: str) -> list[float]:
    if ":" in text:
        start, step, stop = [float(x) for x in text.split(":", 2)]
        out = []
        current = start
        while current <= stop + 1e-9:
            out.append(round(current, 10))
            current += step
        return out
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def complete_run(path: Path) -> bool:
    manifest = path / "manifest.json"
    summary = path / "summary.csv"
    if not manifest.exists() or not summary.exists():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("status") == "complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "dnn_w02_2s_arena")
    parser.add_argument("--dnn-python", type=Path, default=Path("D:/ProjData/proj_python/dnn_ssvep_pytorch/.venv/Scripts/python.exe"))
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--blocks", default="1-6")
    parser.add_argument("--windows", default="0.2:0.1:2.0")
    parser.add_argument("--finetune-epochs", type=int, default=1000)
    parser.add_argument("--train-global-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--force-train-all", action="store_true")
    parser.add_argument("--save-global-models", action="store_true")
    parser.add_argument("--save-finetuned-models", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = parse_windows(args.windows)
    manifest = {
        "task_name": args.output_dir.name,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "windows": windows,
        "subjects": args.subjects,
        "blocks": args.blocks,
        "finetune_epochs": args.finetune_epochs,
        "train_global_epochs": args.train_global_epochs,
        "runs": [],
    }
    manifest_path = args.output_dir / "sweep_manifest.json"
    if manifest_path.exists() and args.resume:
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runs"] = old.get("runs", [])
            manifest["started_at_utc"] = old.get("started_at_utc", manifest["started_at_utc"])
        except json.JSONDecodeError:
            pass

    for window in windows:
        run_dir = args.output_dir / f"window_{window:g}s"
        log_path = args.output_dir / f"window_{window:g}s.log"
        if args.resume and complete_run(run_dir):
            print(f"skip_complete={run_dir}", flush=True)
            manifest["runs"].append({"window": window, "run_dir": str(run_dir), "skipped": True})
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            continue
        cmd = [
            str(args.dnn_python),
            str(project_root / "scripts" / "evaluate_dnn_checkpoints.py"),
            "--subjects",
            args.subjects,
            "--blocks",
            args.blocks,
            "--window",
            str(window),
            "--finetune-epochs",
            str(args.finetune_epochs),
            "--train-global-epochs",
            str(args.train_global_epochs),
            "--batch-size",
            str(args.batch_size),
            "--eval-every",
            str(args.eval_every),
            "--device",
            args.device,
            "--output-dir",
            str(run_dir),
        ]
        if args.force_train_all:
            cmd.append("--force-train")
        if args.save_global_models:
            cmd.append("--save-global-models")
        if args.save_finetuned_models:
            cmd.append("--save-finetuned-models")
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=project_root, stdout=log, stderr=subprocess.STDOUT)
        record = {
            "window": window,
            "run_dir": str(run_dir),
            "log": str(log_path),
            "returncode": proc.returncode,
            "seconds": time.perf_counter() - started,
        }
        manifest["runs"].append(record)
        manifest["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    rows = []
    for path in sorted(args.output_dir.glob("window_*s/summary.csv")):
        rows.append(pd.read_csv(path))
    if rows:
        summary = pd.concat(rows, ignore_index=True).sort_values(["method", "window"])
        summary.to_csv(args.output_dir / "summary.csv", index=False)
    manifest["ended_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
