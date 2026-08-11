from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[2]
PYTHON = PROJECT / ".venv" / "Scripts" / "python.exe"
RUNNER = PROJECT / "scripts" / "run_toolbox_ssvep.py"
DATA_ROOT = Path(os.environ.get("SSVEP_BETA_ROOT", r"D:\ProjData\datasets\ssvep_beta"))
OUTPUT = TASK / "results" / "runner_full"


def main() -> None:
    python = PYTHON if PYTHON.exists() else Path(sys.executable)
    cmd = [
        str(python),
        str(RUNNER),
        "--dataset",
        "beta",
        "--root",
        str(DATA_ROOT),
        "--task-name",
        "beta_ssvep_9ch_baselines",
        "--output-dir",
        str(OUTPUT),
        "--subjects",
        "1-70",
        "--blocks",
        "1-4",
        "--targets",
        "0-39",
        "--channels",
        "occipital_9ch",
        "--windows",
        "0.2:0.2:2.0",
        "--methods",
        "CCA,FBCCA,TRCA,ETRCA,TDCA",
        "--n-bands",
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

