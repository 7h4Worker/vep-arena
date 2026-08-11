from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.broadband_white_noise import (  # noqa: E402
    BROADBAND_WN_CLASSES,
    BROADBAND_WN_SAMPLING_RATE,
    available_subjects,
    broadband_wn_root,
    channel_indices,
    load_subject_sessions,
    parse_subjects,
    read_stimulus,
)
from vep_arena.metrics import itr_bits_per_minute  # noqa: E402
from vep_arena.methods.traditional import TRCA  # noqa: E402


TASK_DIR = Path(__file__).resolve().parent


def parse_windows(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_sessions(text: str) -> set[int] | None:
    text = text.strip().lower()
    if text in {"all", "*"}:
        return None
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def plot_qa(session, picks: list[int], stimulus: np.ndarray, output: Path) -> None:
    x = session.x[0, picks[:3]]
    time = np.arange(x.shape[-1]) / BROADBAND_WN_SAMPLING_RATE
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for idx, pick in enumerate(picks[:3]):
        axes[0].plot(time, x[idx], lw=0.8, label=session.channels[pick])
    axes[0].set_title(f"Broadband WN S{session.subject} session {session.session} EEG check")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[1].plot(np.arange(stimulus.shape[-1]), stimulus[0], lw=1.0)
    axes[1].set_title("WN stimulus code check, target 1")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Intensity")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_trca_smoke(session, picks: list[int], windows: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = session.x[:, picks]
    y = session.y - 1
    classes = np.unique(y)
    grouped_x = np.stack([x[y == cls] for cls in classes], axis=0)
    repetitions = grouped_x.shape[1]
    trial_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    for window in windows:
        samples = int(round(window * BROADBAND_WN_SAMPLING_RATE))
        start = int(round(0.14 * BROADBAND_WN_SAMPLING_RATE))
        stop = start + samples
        if stop > x.shape[-1]:
            continue
        cropped = grouped_x[:, :, :, start:stop]
        for test_rep in range(repetitions):
            train_reps = [idx for idx in range(repetitions) if idx != test_rep]
            train_x = cropped[:, train_reps].reshape(-1, 1, len(picks), samples)
            train_y = np.repeat(classes, len(train_reps))
            test_x = cropped[:, test_rep][:, None]
            model = TRCA(n_fbs=1, ensemble=True)
            model.fit(train_x, train_y)
            pred, scores = model.predict(test_x)
            acc = float(np.mean(pred == classes))
            trial_rows.append(
                {
                    "method": "ETRCA_RAW",
                    "subject": session.subject,
                    "session": session.session,
                    "tag": session.tag,
                    "window": window,
                    "repetition": test_rep + 1,
                    "classes": len(classes),
                    "channels": len(picks),
                    "accuracy": acc,
                    "itr_bpm": float(itr_bits_per_minute(acc, len(classes), window + 0.5)),
                }
            )
            for true_label, pred_label in zip(classes, pred):
                pred_rows.append(
                    {
                        "method": "ETRCA_RAW",
                        "subject": session.subject,
                        "session": session.session,
                        "window": window,
                        "repetition": test_rep + 1,
                        "true": int(true_label) + 1,
                        "pred": int(pred_label) + 1,
                        "correct": int(pred_label == true_label),
                        "score_true": float(scores[int(true_label), int(true_label)]),
                        "score_pred": float(scores[int(true_label), int(pred_label)]),
                    }
                )
    return pd.DataFrame(trial_rows), pd.DataFrame(pred_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke runner for Zenodo 8300517 Broadband White Noise BCI.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--subjects", default="1")
    parser.add_argument("--sessions", default="1")
    parser.add_argument("--windows", default="0.1,0.2")
    parser.add_argument("--channels", default="paper21")
    parser.add_argument("--output", type=Path, default=TASK_DIR / "results" / "smoke_inventory")
    parser.add_argument("--run-trca-smoke", action="store_true")
    args = parser.parse_args()

    root = broadband_wn_root(args.root)
    output = args.output if args.output.is_absolute() else TASK_DIR / args.output
    output.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    wanted_sessions = parse_sessions(args.sessions)
    windows = parse_windows(args.windows)
    stimulus = read_stimulus(root)

    session_rows: list[dict[str, object]] = []
    trial_count_rows: list[dict[str, object]] = []
    first_session = None
    first_picks: list[int] | None = None
    trca_trials: list[pd.DataFrame] = []
    trca_preds: list[pd.DataFrame] = []
    for subject in subjects:
        for session in load_subject_sessions(root=root, subject=subject):
            if wanted_sessions is not None and session.session not in wanted_sessions:
                continue
            picks = channel_indices(session.channels, args.channels)
            if first_session is None:
                first_session = session
                first_picks = picks
            session_rows.append(
                {
                    "subject": subject,
                    "session": session.session,
                    "tag": session.tag,
                    "file": str(session.path.relative_to(root)),
                    "trials": int(session.x.shape[0]),
                    "channels_total": int(session.x.shape[1]),
                    "samples": int(session.x.shape[2]),
                    "selected_channels": len(picks),
                    "classes": int(np.unique(session.y).size),
                    "repetitions_per_class_min": int(pd.Series(session.y).value_counts().min()),
                    "repetitions_per_class_max": int(pd.Series(session.y).value_counts().max()),
                }
            )
            counts = pd.Series(session.y).value_counts().sort_index()
            for label, count in counts.items():
                trial_count_rows.append(
                    {
                        "subject": subject,
                        "session": session.session,
                        "tag": session.tag,
                        "target_id": int(label),
                        "trials": int(count),
                    }
                )
            if args.run_trca_smoke:
                trials, preds = run_trca_smoke(session, picks, windows)
                trca_trials.append(trials)
                trca_preds.append(preds)

    sessions_df = pd.DataFrame(session_rows)
    trial_counts_df = pd.DataFrame(trial_count_rows)
    sessions_df.to_csv(output / "sessions.csv", index=False)
    trial_counts_df.to_csv(output / "trial_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "tag": "WN",
                "shape_0": int(stimulus.shape[0]),
                "shape_1": int(stimulus.shape[1]),
                "min": float(np.min(stimulus)),
                "max": float(np.max(stimulus)),
                "mean": float(np.mean(stimulus)),
            }
        ]
    ).to_csv(output / "stimulus_summary.csv", index=False)
    if first_session is not None and first_picks is not None:
        plot_qa(first_session, first_picks, stimulus, output / "broadband_wn_smoke_time_stimulus.png")

    trca_summary_rows = 0
    trca_pred_rows = 0
    if trca_trials:
        trca_trials_df = pd.concat(trca_trials, axis=0, ignore_index=True)
        trca_preds_df = pd.concat(trca_preds, axis=0, ignore_index=True)
        trca_trials_df.to_csv(output / "trca_smoke_trials.csv", index=False)
        trca_preds_df.to_csv(output / "trca_smoke_predictions.csv", index=False)
        trca_summary = (
            trca_trials_df.groupby(["method", "window"], as_index=False)
            .agg(folds=("repetition", "count"), accuracy=("accuracy", "mean"), itr_bpm=("itr_bpm", "mean"))
            .sort_values(["window"])
        )
        trca_summary.to_csv(output / "trca_smoke_summary.csv", index=False)
        trca_summary_rows = int(len(trca_summary))
        trca_pred_rows = int(len(trca_preds_df))

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": "Broadband White Noise BCI",
        "zenodo": "https://zenodo.org/records/8300517",
        "dataset_root": str(root),
        "available_subjects": list(available_subjects(root)),
        "subjects": subjects,
        "sessions": sorted(wanted_sessions) if wanted_sessions is not None else "all",
        "sampling_rate": BROADBAND_WN_SAMPLING_RATE,
        "classes": BROADBAND_WN_CLASSES,
        "windows": windows,
        "trca_smoke": bool(args.run_trca_smoke),
        "outputs": {
            "sessions": "sessions.csv",
            "trial_counts": "trial_counts.csv",
            "stimulus_summary": "stimulus_summary.csv",
            "qa_figure": "broadband_wn_smoke_time_stimulus.png",
            "trca_smoke_summary": "trca_smoke_summary.csv" if args.run_trca_smoke else None,
            "trca_smoke_predictions": "trca_smoke_predictions.csv" if args.run_trca_smoke else None,
        },
        "rows": {
            "sessions": int(len(sessions_df)),
            "trial_counts": int(len(trial_counts_df)),
            "trca_summary": trca_summary_rows,
            "trca_predictions": trca_pred_rows,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
