from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[2]
PYTHON = PROJECT / ".venv" / "Scripts" / "python.exe"
RUNNER = PROJECT / "scripts" / "run_traditional_benchmark.py"
DATA_ROOT = Path(
    os.environ.get(
        "SSVEP_BENCHMARK_ROOT",
        r"D:\ProjData\datasets\ssvep_benchmark",
    )
)
OUTPUT = TASK / "results" / "runner_full"


def main() -> None:
    python = PYTHON if PYTHON.exists() else Path(sys.executable)
    cmd = [
        str(python),
        str(RUNNER),
        "--data-root",
        str(DATA_ROOT),
        "--task-name",
        "BL01_ssvep_benchmark",
        "--channel-preset",
        "benchmark_9ch",
        "--output-dir",
        str(OUTPUT),
        "--subjects",
        "1-35",
        "--blocks",
        "1-6",
        "--windows",
        "default",
        "--methods",
        "CCA,FBCCA,ECCA,TRCA,ETRCA,SSCOR,ESSCOR",
        "--n-fbs",
        "5",
        "--harmonics",
        "5",
        "--workers",
        "6",
        "--resume",
    ]
    subprocess.run(cmd, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
