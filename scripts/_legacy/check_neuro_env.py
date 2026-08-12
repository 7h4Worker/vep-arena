# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-20
# Last updated: 2026-06-20
# Description: Check the local neuro dependencies available to the active Arena Python.
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


LEGACY_PSYCHOPY_PYTHON = Path("D:/ProjData/envs/erp_ssvep_lab/python.exe")


def module_version(name: str) -> str | None:
    if importlib.util.find_spec(name) is None:
        return None
    module = __import__(name)
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    required = ["mne", "numpy", "scipy", "sklearn", "matplotlib"]
    optional = ["psychopy"]
    modules = required + optional
    result = {
        "python": sys.executable,
        "legacy_psychopy_python": str(LEGACY_PSYCHOPY_PYTHON),
        "legacy_psychopy_python_exists": LEGACY_PSYCHOPY_PYTHON.exists(),
        "required": required,
        "optional": optional,
        "modules": {name: module_version(name) for name in modules},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    missing = [name for name in required if result["modules"][name] is None]
    if missing:
        raise SystemExit(
            "缺少神经科学 QA 依赖: "
            + ", ".join(missing)
            + "\n请运行: uv sync --extra neuro 或 uv sync --extra all"
        )

    optional_missing = [name for name in optional if result["modules"][name] is None]
    if optional_missing:
        print(
            "可选在线实验依赖未安装: "
            + ", ".join(optional_missing)
            + f"\n如需 PsychoPy/在线实验，可使用历史环境: {LEGACY_PSYCHOPY_PYTHON}"
        )


if __name__ == "__main__":
    main()
