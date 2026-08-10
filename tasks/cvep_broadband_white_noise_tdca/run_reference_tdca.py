from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.broadband_white_noise import (  # noqa: E402
    BROADBAND_WN_CLASSES,
    BROADBAND_WN_SAMPLING_RATE,
    broadband_wn_root,
    channel_indices,
    load_subject_sessions,
    parse_subjects,
)
from vep_arena.metrics import itr_bits_per_minute  # noqa: E402


TASK_DIR = Path(__file__).resolve().parent


def parse_windows(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_sessions(text: str) -> set[int] | None:
    text = text.strip().lower()
    if text in {"all", "*"}:
        return None
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def install_statsmodels_ttest_shim() -> None:
    """Provide the only statsmodels symbol used by the Zenodo reference code."""

    def ttest_ind(x1, x2, alternative: str = "two-sided", **_: object):
        result = stats.ttest_ind(x1, x2, equal_var=True, alternative=alternative)
        return float(result.statistic), float(result.pvalue), None

    statsmodels_mod = types.ModuleType("statsmodels")
    stats_mod = types.ModuleType("statsmodels.stats")
    weightstats_mod = types.ModuleType("statsmodels.stats.weightstats")
    weightstats_mod.ttest_ind = ttest_ind
    stats_mod.weightstats = weightstats_mod
    statsmodels_mod.stats = stats_mod
    sys.modules.setdefault("statsmodels", statsmodels_mod)
    sys.modules.setdefault("statsmodels.stats", stats_mod)
    sys.modules.setdefault("statsmodels.stats.weightstats", weightstats_mod)


def load_reference_tdca(root: Path):
    install_statsmodels_ttest_shim()
    path = root / "extracted" / "core" / "spatialFilters.py"
    spec = importlib.util.spec_from_file_location("zenodo8300517_spatialFilters", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Zenodo spatialFilters.py from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TDCA


def append_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def grouped_repetition_view(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.unique(y)
    grouped_x = np.stack([x[y == cls] for cls in classes])
    grouped_y = np.stack([y[y == cls] for cls in classes])
    return (
        np.transpose(grouped_x, axes=(1, 0, 2, 3)),
        np.transpose(grouped_y, axes=(1, 0)),
        classes,
    )


def run_session(
    tdca_cls,
    session,
    channel_set: str,
    windows: list[float],
    n_band: int,
    lag: float,
    max_folds: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    picks = channel_indices(session.channels, channel_set)
    x = session.x[:, picks]
    y = session.y
    by_rep_x, by_rep_y, classes = grouped_repetition_view(x, y)
    n_reps = by_rep_x.shape[0]
    folds = list(range(n_reps))
    if max_folds > 0:
        folds = folds[:max_folds]

    trial_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for fold in folds:
        train_reps = [idx for idx in range(n_reps) if idx != fold]
        train_x = np.concatenate(by_rep_x[train_reps], axis=0)
        train_y = np.concatenate(by_rep_y[train_reps], axis=0)
        test_x = by_rep_x[fold]
        test_y = by_rep_y[fold]
        for window in windows:
            model = tdca_cls(
                winLEN=window,
                lag=lag,
                srate=BROADBAND_WN_SAMPLING_RATE,
                montage=BROADBAND_WN_CLASSES,
                n_band=n_band,
            )
            started = time.perf_counter()
            model.fit(train_x, train_y)
            fit_done = time.perf_counter()
            pred = model.predict(test_x)
            pred_done = time.perf_counter()
            accuracy = float(np.mean(pred == test_y))
            trial_rows.append(
                {
                    "method": "ZENODO_TDCA",
                    "subject": session.subject,
                    "session": session.session,
                    "tag": session.tag,
                    "window": window,
                    "lag": lag,
                    "n_band": n_band,
                    "fold": fold + 1,
                    "folds_run": len(folds),
                    "folds_total": n_reps,
                    "classes": len(classes),
                    "channels": len(picks),
                    "accuracy": accuracy,
                    "itr_bpm": float(itr_bits_per_minute(accuracy, len(classes), window + 0.5)),
                    "seconds": pred_done - started,
                    "status": "complete",
                    "reason": "",
                }
            )
            runtime_rows.extend(
                [
                    {
                        "method": "ZENODO_TDCA",
                        "subject": session.subject,
                        "session": session.session,
                        "window": window,
                        "fold": fold + 1,
                        "stage": "fit",
                        "seconds": fit_done - started,
                    },
                    {
                        "method": "ZENODO_TDCA",
                        "subject": session.subject,
                        "session": session.session,
                        "window": window,
                        "fold": fold + 1,
                        "stage": "predict",
                        "seconds": pred_done - fit_done,
                    },
                ]
            )
            rho = getattr(model, "rho", None)
            for true_label, pred_label in zip(test_y, pred):
                true_idx = int(np.where(classes == true_label)[0][0])
                pred_idx = int(np.where(classes == pred_label)[0][0])
                row = {
                    "method": "ZENODO_TDCA",
                    "subject": session.subject,
                    "session": session.session,
                    "window": window,
                    "fold": fold + 1,
                    "true": int(true_label),
                    "pred": int(pred_label),
                    "correct": int(true_label == pred_label),
                }
                if rho is not None:
                    sample_idx = int(np.where(test_y == true_label)[0][0])
                    row["score_true"] = float(rho[sample_idx, true_idx])
                    row["score_pred"] = float(rho[sample_idx, pred_idx])
                pred_rows.append(row)
    return trial_rows, pred_rows, runtime_rows


def rebuild_summary(out_dir: Path) -> pd.DataFrame:
    trials_path = out_dir / "trials.csv"
    if not trials_path.exists():
        return pd.DataFrame()
    trials = pd.read_csv(trials_path)
    if trials.empty:
        return pd.DataFrame()
    summary = (
        trials.groupby(["method", "window"], as_index=False)
        .agg(
            folds=("fold", "count"),
            subjects=("subject", "nunique"),
            accuracy=("accuracy", "mean"),
            itr_bpm=("itr_bpm", "mean"),
            seconds=("seconds", "sum"),
        )
        .sort_values(["window"])
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Zenodo 8300517 reference TDCA implementation.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--subjects", default="1")
    parser.add_argument("--sessions", default="1")
    parser.add_argument("--windows", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--channels", default="paper21")
    parser.add_argument("--n-band", type=int, default=5)
    parser.add_argument("--lag", type=float, default=0.14)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--output", type=Path, default=TASK_DIR / "results" / "reference_tdca_smoke")
    args = parser.parse_args()

    root = broadband_wn_root(args.root)
    out_dir = args.output if args.output.is_absolute() else TASK_DIR / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    tdca_cls = load_reference_tdca(root)
    subjects = parse_subjects(args.subjects)
    sessions = parse_sessions(args.sessions)
    windows = parse_windows(args.windows)

    trial_fields = [
        "method",
        "subject",
        "session",
        "tag",
        "window",
        "lag",
        "n_band",
        "fold",
        "folds_run",
        "folds_total",
        "classes",
        "channels",
        "accuracy",
        "itr_bpm",
        "seconds",
        "status",
        "reason",
    ]
    pred_fields = ["method", "subject", "session", "window", "fold", "true", "pred", "correct", "score_true", "score_pred"]
    runtime_fields = ["method", "subject", "session", "window", "fold", "stage", "seconds"]

    for subject in subjects:
        for session in load_subject_sessions(root=root, subject=subject):
            if sessions is not None and session.session not in sessions:
                continue
            trial_rows, pred_rows, runtime_rows = run_session(
                tdca_cls=tdca_cls,
                session=session,
                channel_set=args.channels,
                windows=windows,
                n_band=args.n_band,
                lag=args.lag,
                max_folds=args.max_folds,
            )
            append_csv(out_dir / "trials.csv", trial_rows, trial_fields)
            append_csv(out_dir / "predictions.csv", pred_rows, pred_fields)
            append_csv(out_dir / "runtime.csv", runtime_rows, runtime_fields)
            rebuild_summary(out_dir)
            print(f"done S{subject} session={session.session} rows={len(trial_rows)}", flush=True)

    summary = rebuild_summary(out_dir)
    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": "Broadband White Noise BCI",
        "zenodo": "https://zenodo.org/records/8300517",
        "implementation": "Zenodo core/spatialFilters.py TDCA with statsmodels ttest_ind shim",
        "dataset_root": str(root),
        "subjects": subjects,
        "sessions": sorted(sessions) if sessions is not None else "all",
        "windows": windows,
        "channels": args.channels,
        "n_band": args.n_band,
        "lag": args.lag,
        "max_folds": args.max_folds,
        "outputs": {
            "trials": "trials.csv",
            "predictions": "predictions.csv",
            "runtime": "runtime.csv",
            "summary": "summary.csv",
        },
        "summary_rows": int(len(summary)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
