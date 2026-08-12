from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_ROOT.parents[1]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from common import DEFAULT_CACHE_ROOT, DEFAULT_RESULT_ROOT, atomic_json, validate_subject_cache


LOCK_PATH = DEFAULT_RESULT_ROOT / "overnight.lock"
STATE_PATH = DEFAULT_RESULT_ROOT / "overnight_state.json"
LOG_PATH = DEFAULT_RESULT_ROOT / "overnight.log"
CONFIG_ORDER = ("full64", "posterior21", "posterior32", "wholehead32")
METHOD_ORDER = ("CCA", "FBCCA", "TRCA", "ETRCA", "ECCA")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def update_state(**values: object) -> None:
    state = load_json(STATE_PATH)
    state.update(values)
    state["updated_at_utc"] = utc_now()
    atomic_json(STATE_PATH, state)


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock() -> bool:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        existing = load_json(LOCK_PATH)
        existing_pid = int(existing.get("pid", -1))
        if pid_is_running(existing_pid):
            log(f"Another multichannel runner is active (pid={existing_pid}); exiting without duplication.")
            return False
        LOCK_PATH.unlink(missing_ok=True)
    descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at_utc": utc_now()}, handle, indent=2)
    return True


def release_lock() -> None:
    try:
        if int(load_json(LOCK_PATH).get("pid", -1)) == os.getpid():
            LOCK_PATH.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


def prevent_system_sleep(enabled: bool) -> None:
    if os.name != "nt":
        return
    continuous = 0x80000000
    system_required = 0x00000001
    flags = continuous | system_required if enabled else continuous
    if ctypes.windll.kernel32.SetThreadExecutionState(flags) == 0:
        raise OSError("SetThreadExecutionState failed")


def progress_snapshot() -> dict[str, int]:
    return {
        "cache_subjects": sum(validate_subject_cache(DEFAULT_CACHE_ROOT, subject) for subject in range(1, 36)),
        "confusion_files": sum(1 for _ in (DEFAULT_RESULT_ROOT / "confusions").glob("*.npy")),
    }


def run_step(name: str, command: list[str], retry_delays: list[int]) -> None:
    attempts = len(retry_delays) + 1
    for attempt in range(1, attempts + 1):
        update_state(
            status="running",
            current_step=name,
            attempt=attempt,
            command=command,
            progress=progress_snapshot(),
        )
        log(f"START {name} (attempt {attempt}/{attempts})")
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode == 0:
            snapshot = progress_snapshot()
            log(f"DONE {name}; progress={json.dumps(snapshot)}")
            update_state(last_completed_step=name, last_returncode=0, progress=snapshot)
            return
        log(f"FAIL {name}: exit={completed.returncode}")
        update_state(last_returncode=completed.returncode, progress=progress_snapshot())
        if attempt < attempts:
            delay = retry_delays[attempt - 1]
            log(f"Retrying {name} in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"{name} failed after {attempts} attempts; inspect {LOG_PATH}")


def runner_command(python: Path, workers: int, chunk_size: int, config: str, method: str) -> list[str]:
    return [
        str(python),
        str(TASK_ROOT / "run.py"),
        "--result-root",
        str(DEFAULT_RESULT_ROOT),
        "--cache-root",
        str(DEFAULT_CACHE_ROOT),
        "sweep",
        "--configs",
        config,
        "--methods",
        method,
        "--workers",
        str(workers),
        "--chunk-size",
        str(chunk_size),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliable Benchmark multichannel overnight runner")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--retry-delays", default="60,180,600")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        payload = load_json(STATE_PATH)
        payload["lock"] = load_json(LOCK_PATH)
        payload["progress_now"] = progress_snapshot()
        print(json.dumps(payload, indent=2))
        return
    if min(args.workers, args.cache_workers, args.chunk_size) < 1:
        raise ValueError("workers, cache-workers, and chunk-size must be positive")
    retry_delays = [int(value) for value in args.retry_delays.split(",") if value.strip()]
    if any(value < 0 for value in retry_delays):
        raise ValueError("Retry delays must be non-negative")

    python = Path(sys.executable).resolve()
    if not acquire_lock():
        return
    prevent_system_sleep(True)
    update_state(
        status="running",
        pid=os.getpid(),
        started_at_utc=utc_now(),
        experiment_order={"methods": list(METHOD_ORDER), "configs": list(CONFIG_ORDER)},
        progress=progress_snapshot(),
    )
    try:
        run_step(
            "extract existing occipital9 confusions",
            [str(python), str(TASK_ROOT / "run.py"), "extract-9"],
            retry_delays,
        )
        run_step(
            "prepare fixed5 full64 caches",
            [
                str(python),
                str(TASK_ROOT / "prepare_cache.py"),
                "--workers",
                str(args.cache_workers),
            ],
            retry_delays,
        )
        for method in METHOD_ORDER:
            for config in CONFIG_ORDER:
                run_step(
                    f"sweep {method} {config}",
                    runner_command(python, args.workers, args.chunk_size, config, method),
                    retry_delays,
                )
        run_step(
            "Blahut-Arimoto analysis",
            [
                str(python),
                str(TASK_ROOT / "analyze.py"),
                "--workers",
                str(args.workers),
            ],
            retry_delays,
        )
        run_step(
            "final validation",
            [
                str(python),
                str(TASK_ROOT / "run.py"),
                "validate",
                "--require-analysis",
            ],
            retry_delays,
        )
        update_state(
            status="complete",
            current_step=None,
            completed_at_utc=utc_now(),
            progress=progress_snapshot(),
        )
        log("PIPELINE COMPLETE")
    except BaseException as error:
        update_state(status="failed", error_type=type(error).__name__, error=str(error), progress=progress_snapshot())
        log(f"PIPELINE FAILED: {type(error).__name__}: {error}")
        raise
    finally:
        prevent_system_sleep(False)
        release_lock()


if __name__ == "__main__":
    main()
