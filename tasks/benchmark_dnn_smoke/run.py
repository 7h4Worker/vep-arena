"""Minimal DNN smoke test — Arena-native training + evaluation.

Validates the full chain: data loading -> Arena nn.DNNSsvep model ->
global training -> per-subject fine-tuning -> metrics -> artifact output.

Scope: 2 subjects, 1 block, 1 window (1.0 s), 10 global epochs, 5 finetune
epochs.  Should finish in under 2 minutes on any CUDA GPU.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[1]
PYTHON = PROJECT / ".venv" / "Scripts" / "python.exe"
SCRIPT = PROJECT / "scripts" / "evaluate_dnn_checkpoints.py"
OUTPUT = TASK / "results" / "dnn_smoke"


def main() -> None:
    python = PYTHON if PYTHON.exists() else Path(sys.executable)
    cmd = [
        str(python),
        str(SCRIPT),
        "--subjects", "1-2",
        "--blocks", "1",
        "--window", "1.0",
        "--train-global-epochs", "10",
        "--finetune-epochs", "5",
        "--force-train",
        "--save-global-models",
        "--batch-size", "64",
        "--device", "auto",
        "--output-dir", str(OUTPUT),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
