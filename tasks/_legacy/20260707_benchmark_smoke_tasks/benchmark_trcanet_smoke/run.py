"""Minimal TRCANet smoke test — Arena-native training + evaluation.

Validates the full chain: data loading -> TRCA spatial filter fitting ->
projection -> TRCANet CNN -> global training -> metrics -> artifact output.

Scope: 2 subjects, 1 block, window=1.0s, 10 global epochs.
Should finish in under 2 minutes on any CUDA GPU.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[1]
PYTHON = PROJECT / ".venv" / "Scripts" / "python.exe"
SCRIPT = PROJECT / "scripts" / "evaluate_trcanet.py"
OUTPUT = TASK / "results" / "trcanet_smoke"


def main() -> None:
    python = PYTHON if PYTHON.exists() else Path(sys.executable)
    cmd = [
        str(python),
        str(SCRIPT),
        "--subjects", "1-2",
        "--blocks", "1",
        "--window", "1.0",
        "--global-epochs", "10",
        "--batch-size", "64",
        "--device", "auto",
        "--save-global-models",
        "--output-dir", str(OUTPUT),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
