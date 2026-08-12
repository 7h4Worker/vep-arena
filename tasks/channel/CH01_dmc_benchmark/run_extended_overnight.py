"""Run the complete five-method extended Benchmark pipeline with retries.

The stage sweeps are split into small window batches.  ``run_extended.py``
still checkpoints one subject at a time, so an interruption loses at most one
subject from the active batch and a restart skips every complete cell.
"""
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
EXTENDED_ROOT = TASK_ROOT / "results" / "extended"
LOCK_PATH = EXTENDED_ROOT / "overnight_full5.lock"
STATE_PATH = EXTENDED_ROOT / "overnight_full5_state.json"
LOG_PATH = EXTENDED_ROOT / "overnight_full5.log"

ALL_METHODS = ("CCA", "FBCCA", "ECCA", "TRCA", "ETRCA")
MISSING_METHODS = ("FBCCA", "ECCA", "TRCA")
ROWS_PER_CELL = 6 * 40


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_state() -> dict[str, object]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def update_state(**values: object) -> None:
    state = load_state()
    state.update(values)
    state["updated_at_utc"] = utc_now()
    atomic_json(state, STATE_PATH)


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
        try:
            existing = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            existing_pid = int(existing.get("pid", -1))
        except (OSError, ValueError, json.JSONDecodeError):
            existing_pid = -1
        if pid_is_running(existing_pid):
            log(f"Another overnight runner is active (pid={existing_pid}); exiting without duplication.")
            return False
        LOCK_PATH.unlink(missing_ok=True)
    descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at_utc": utc_now()}, handle, indent=2)
    return True


def release_lock() -> None:
    try:
        if LOCK_PATH.exists():
            payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            if int(payload.get("pid", -1)) == os.getpid():
                LOCK_PATH.unlink()
    except (OSError, ValueError, json.JSONDecodeError):
        pass


def prevent_system_sleep(enabled: bool) -> None:
    if os.name != "nt":
        return
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    flags = es_continuous | es_system_required if enabled else es_continuous
    result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
    if result == 0:
        raise OSError("SetThreadExecutionState failed")


def window_text(windows: list[float]) -> str:
    return ",".join(f"{window:.4f}".rstrip("0").rstrip(".") for window in windows)


def chunks(values: list[float], size: int) -> list[list[float]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def run_step(name: str, command: list[str], retry_delays: list[int]) -> None:
    attempts = len(retry_delays) + 1
    for attempt in range(1, attempts + 1):
        update_state(status="running", current_step=name, attempt=attempt, command=command)
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
            log(f"DONE {name}")
            update_state(last_completed_step=name, last_returncode=0)
            return
        log(f"FAIL {name}: exit={completed.returncode}")
        update_state(last_returncode=completed.returncode)
        if attempt < attempts:
            delay = retry_delays[attempt - 1]
            log(f"Retrying {name} in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"{name} failed after {attempts} attempts; inspect {LOG_PATH}")


def stage_windows(stage: str) -> list[float]:
    if stage == "coarse":
        return [round(samples / 250, 4) for samples in range(25, 1251, 25)]
    return [round(samples / 250, 4) for samples in range(25, 102, 2)]


def runner_command(
    python: Path,
    *,
    stage: str,
    methods: tuple[str, ...],
    workers: int,
    windows: list[float] | None = None,
    subjects: str | None = None,
    output_name: str | None = None,
) -> list[str]:
    command = [
        str(python),
        str(TASK_ROOT / "run_extended.py"),
        "--stage",
        stage,
        "--methods",
        ",".join(methods),
        "--workers",
        str(workers),
        "--resume",
    ]
    if windows is not None:
        command.extend(["--windows", window_text(windows)])
    if subjects is not None:
        command.extend(["--subjects", subjects])
    if output_name is not None:
        command.extend(["--output-name", output_name])
    return command


def smoke_steps(python: Path) -> list[tuple[str, list[str]]]:
    coarse_name = "protocol_fixed5_missing3_smoke"
    fine_name = "protocol_fixed5_missing3_fine_smoke"
    combined = EXTENDED_ROOT / "protocol_combined_missing3_smoke"
    return [
        (
            "smoke coarse missing3",
            runner_command(
                python,
                stage="coarse",
                methods=MISSING_METHODS,
                workers=1,
                windows=[0.1, 0.2, 1.0, 5.0],
                subjects="1",
                output_name=coarse_name,
            ),
        ),
        (
            "smoke fine missing3",
            runner_command(
                python,
                stage="fine",
                methods=MISSING_METHODS,
                workers=1,
                windows=[0.1],
                subjects="1",
                output_name=fine_name,
            ),
        ),
        (
            "smoke combine missing3",
            [
                str(python),
                str(TASK_ROOT / "combine_extended.py"),
                "--coarse",
                str(EXTENDED_ROOT / coarse_name),
                "--fine",
                str(EXTENDED_ROOT / fine_name),
                "--methods",
                ",".join(MISSING_METHODS),
                "--output-dir",
                str(combined),
                "--skip-plots",
            ],
        ),
    ]


def validate_final() -> dict[str, object]:
    expected = {
        "coarse_0.1s": (50, 35 * 50 * len(ALL_METHODS) * ROWS_PER_CELL),
        "fine_8ms": (39, 35 * 39 * len(ALL_METHODS) * ROWS_PER_CELL),
    }
    validation: dict[str, object] = {}
    for name, (window_count, expected_rows) in expected.items():
        manifest_path = EXTENDED_ROOT / name / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"{name} manifest is not complete")
        if set(manifest.get("methods_present", [])) != set(ALL_METHODS):
            raise ValueError(f"{name} does not contain all five methods")
        if len(manifest.get("windows_requested", [])) != window_count:
            raise ValueError(f"{name} manifest does not cover the full window grid")
        if int(manifest.get("prediction_rows_total", -1)) != expected_rows:
            raise ValueError(f"{name} row count does not match {expected_rows}")
        validation[name] = {"windows": window_count, "prediction_rows": expected_rows}

    combined_manifest = json.loads((EXTENDED_ROOT / "combined" / "manifest.json").read_text(encoding="utf-8"))
    expected_combined_rows = 35 * 87 * len(ALL_METHODS) * ROWS_PER_CELL
    if combined_manifest.get("status") != "complete":
        raise ValueError("combined manifest is not complete")
    if set(combined_manifest.get("methods", [])) != set(ALL_METHODS):
        raise ValueError("combined manifest does not contain all five methods")
    if int(combined_manifest.get("prediction_rows", -1)) != expected_combined_rows:
        raise ValueError("combined row count is incomplete")

    analysis_dir = EXTENDED_ROOT / "combined" / "analysis"
    aggregate_rows = sum(1 for _ in (analysis_dir / "capacity_by_method_window_aggregate.csv").open(encoding="utf-8")) - 1
    subject_rows = sum(1 for _ in (analysis_dir / "capacity_by_subject_method_window.csv").open(encoding="utf-8")) - 1
    if aggregate_rows != len(ALL_METHODS) * 87:
        raise ValueError(f"aggregate capacity rows={aggregate_rows}, expected={len(ALL_METHODS) * 87}")
    if subject_rows != len(ALL_METHODS) * 87 * 35:
        raise ValueError(f"subject capacity rows={subject_rows}, expected={len(ALL_METHODS) * 87 * 35}")
    expected_figures = {
        "fig01_capacity_ladder",
        "fig02_method_capacity_compare",
        "fig03_mean_confusion_heatmap",
        "fig04_error_vs_frequency_distance",
        "fig05_optimal_input_distribution",
        "fig06_asymmetry_vs_capacity_gain",
        "fig07_c2_vs_ba_consistency",
        "fig08_cba_minus_c1_distribution",
        "fig09_itr_time_tradeoff",
    }
    figure_names = {path.stem for path in (EXTENDED_ROOT / "combined" / "figures").glob("*.png")}
    missing_figures = sorted(expected_figures - figure_names)
    if missing_figures:
        raise ValueError(f"missing extended figures: {', '.join(missing_figures)}")
    validation["combined"] = {
        "windows": 87,
        "prediction_rows": expected_combined_rows,
        "aggregate_capacity_rows": aggregate_rows,
        "subject_capacity_rows": subject_rows,
    }
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliable overnight five-method extended Benchmark runner")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--retry-delays", default="60,180,600")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(load_state(), indent=2))
        return
    if args.workers < 1 or args.chunk_size < 1:
        raise ValueError("--workers and --chunk-size must be positive")
    retry_delays = [int(value) for value in args.retry_delays.split(",") if value.strip()]
    if any(value < 0 for value in retry_delays):
        raise ValueError("Retry delays must be non-negative")
    python = Path(sys.executable).resolve()
    if not acquire_lock():
        return
    prevent_system_sleep(True)
    update_state(status="running", pid=os.getpid(), started_at_utc=utc_now())
    try:
        if not args.skip_smoke:
            for name, command in smoke_steps(python):
                run_step(name, command, retry_delays)

        # Fine first: it is much faster and exposes subject-specific failures early.
        for stage in ("fine", "coarse"):
            for index, window_batch in enumerate(chunks(stage_windows(stage), args.chunk_size), start=1):
                run_step(
                    f"{stage} batch {index}",
                    runner_command(
                        python,
                        stage=stage,
                        methods=ALL_METHODS,
                        workers=args.workers,
                        windows=window_batch,
                    ),
                    retry_delays,
                )
            run_step(
                f"{stage} finalize full manifest",
                runner_command(python, stage=stage, methods=ALL_METHODS, workers=args.workers),
                retry_delays,
            )

        run_step(
            "combine analyze and plot all five methods",
            [
                str(python),
                str(TASK_ROOT / "combine_extended.py"),
                "--methods",
                ",".join(ALL_METHODS),
            ],
            retry_delays,
        )
        run_step(
            "refresh information accumulation figures",
            [str(python), str(TASK_ROOT.parent / "information_accumulation_rate" / "plot_info_accumulation.py")],
            retry_delays,
        )
        validation = validate_final()
        update_state(status="complete", completed_at_utc=utc_now(), validation=validation, current_step=None)
        log(f"PIPELINE COMPLETE: {json.dumps(validation, ensure_ascii=False)}")
    except BaseException as error:
        update_state(status="failed", error_type=type(error).__name__, error=str(error))
        log(f"PIPELINE FAILED: {type(error).__name__}: {error}")
        raise
    finally:
        prevent_system_sleep(False)
        release_lock()


if __name__ == "__main__":
    main()
