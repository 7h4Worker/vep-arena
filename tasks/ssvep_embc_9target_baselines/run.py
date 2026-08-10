from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tasks.ssvep_embc_jbhi_shared import run_cli


if __name__ == "__main__":
    run_cli("embc9", default_subjects="S04,S06", default_windows="0.25,0.5,1.0,2.0", default_methods="TRCA")
