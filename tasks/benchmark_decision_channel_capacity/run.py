from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[1]
PYTHON = PROJECT / ".venv" / "Scripts" / "python.exe"
RUNNER = PROJECT / "scripts" / "run_traditional_benchmark.py"
RESULTS = TASK / "results"
INPUT = RESULTS / "input"


def run_command(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT, check=True)


def copy_runner_outputs(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("predictions.csv", "trials.csv", "summary.csv", "subject.csv", "block.csv", "runtime.csv", "manifest.json"):
        src = source / name
        if src.exists():
            shutil.copy2(src, dest / name)
    fig_src = source / "figures"
    if fig_src.exists():
        fig_dest = dest / "runner_figures"
        if fig_dest.exists():
            shutil.rmtree(fig_dest)
        shutil.copytree(fig_src, fig_dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a small 2-subject smoke sweep.")
    parser.add_argument("--reuse-existing", action="store_true", help="Only copy an existing runner output into this task.")
    parser.add_argument("--source-result", type=Path, default=None, help="Existing runner result directory to copy.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--windows", default=None)
    parser.add_argument("--methods", default="CCA,FBCCA,ECCA,TRCA,ETRCA")
    parser.add_argument("--task-name", default=None)
    args = parser.parse_args()

    subjects = args.subjects or ("1-2" if args.smoke else "1-35")
    windows = args.windows or ("0.2,0.5,1.0" if args.smoke else "0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    task_name = args.task_name or ("benchmark_decision_channel_capacity_smoke" if args.smoke else "benchmark_decision_channel_capacity_full")
    source = args.source_result or (PROJECT / "results" / task_name)

    if not args.reuse_existing:
        python = PYTHON if PYTHON.exists() else Path(sys.executable)
        cmd = [
            str(python),
            str(RUNNER),
            "--task-name",
            task_name,
            "--subjects",
            subjects,
            "--windows",
            windows,
            "--methods",
            args.methods,
            "--workers",
            str(args.workers),
            "--resume",
        ]
        run_command(cmd)
    if not (source / "predictions.csv").exists():
        raise FileNotFoundError(f"Missing predictions.csv under {source}")
    copy_runner_outputs(source, INPUT)
    print(INPUT)


if __name__ == "__main__":
    main()
