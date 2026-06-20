# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-20
# Last updated: 2026-06-20
# Description: Check the local MNE/PsychoPy environment used by VEP Arena QA tools.
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


DEFAULT_NEURO_PYTHON = Path("D:/ProjData/envs/erp_ssvep_lab/python.exe")


def module_version(name: str) -> str | None:
    if importlib.util.find_spec(name) is None:
        return None
    module = __import__(name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    modules = ["mne", "psychopy", "numpy", "scipy", "sklearn", "matplotlib"]
    result = {
        "python": sys.executable,
        "recommended_neuro_python": str(DEFAULT_NEURO_PYTHON),
        "recommended_exists": DEFAULT_NEURO_PYTHON.exists(),
        "modules": {name: module_version(name) for name in modules},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    missing = [name for name in ("mne", "psychopy") if result["modules"][name] is None]
    if missing:
        raise SystemExit(
            "缺少神经科学 QA 依赖: "
            + ", ".join(missing)
            + f"\n建议使用: {DEFAULT_NEURO_PYTHON}"
        )


if __name__ == "__main__":
    main()

