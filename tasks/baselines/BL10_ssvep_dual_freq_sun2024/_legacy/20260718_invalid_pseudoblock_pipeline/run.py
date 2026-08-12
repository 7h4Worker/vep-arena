from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.dual_frequency import (  # noqa: E402
    SUN2024_OCCIPITAL9,
    SUN2024_SPLITS,
    iter_sun_cnt_files,
    pair_sun_trial_events,
    read_sun_codebook,
    sun_stimulus_codebook,
    sun2024_root,
    sun_annotation_rows,
)
from vep_arena.methods.traditional import CCA, multi_frequency_reference_signals  # noqa: E402


def parse_subjects(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(item) for item in part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    return values


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def trigger_kind(trigger: int | None) -> str:
    if trigger is None:
        return "unknown"
    if trigger == 253:
        return "stimulus_end"
    if trigger == 254:
        return "block_end"
    return "stimulus_start"


def parse_windows(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_filter(text: str) -> tuple[float, float] | None:
    text = text.strip()
    if not text or text.lower() in {"none", "off"}:
        return None
    lo, hi = (float(item) for item in text.split(",", 1))
    return lo, hi


def plot_time_frequency(record, events: list[dict[str, object]], output: Path, seconds: float) -> None:
    import mne

    raw = mne.io.read_raw_cnt(record.path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    start_event = next((row for row in events if row["event_kind"] == "stimulus_start"), None)
    start = int(start_event["sample"]) if start_event is not None else 0
    stop = min(raw.n_times, start + int(round(seconds * sfreq)))
    picks = [ch for ch in SUN2024_OCCIPITAL9 if ch in raw.ch_names]
    if not picks:
        picks = raw.ch_names[:3]
    data = raw.get_data(picks=picks, start=start, stop=stop)
    time = np.arange(data.shape[1]) / sfreq

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for idx, ch in enumerate(picks[:3]):
        axes[0].plot(time, data[idx] * 1e6, lw=0.8, label=ch)
    axes[0].set_title(f"Sun2024 {record.split} sub-{record.subject:02d} time-domain check")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].legend(loc="upper right")

    for idx, ch in enumerate(picks[:3]):
        freqs, psd = signal.welch(data[idx], fs=sfreq, nperseg=min(data.shape[1], int(sfreq * 2)))
        keep = (freqs >= 1.0) & (freqs <= 60.0)
        axes[1].plot(freqs[keep], psd[keep], lw=0.9, label=ch)
    axes[1].set_title("Welch PSD check")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power")
    axes[1].legend(loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_confusion(rows: pd.DataFrame, output: Path, n_targets: int, title: str) -> None:
    confusion = np.zeros((n_targets, n_targets), dtype=np.int64)
    for item in rows.itertuples(index=False):
        confusion[int(item.target_id) - 1, int(item.predicted_target_id) - 1] += 1
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(confusion, cmap="Blues", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted target")
    ax.set_ylabel("True target")
    ticks = np.arange(n_targets)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if n_targets <= 40:
        labels = [str(idx + 1) for idx in range(n_targets)]
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_cca_smoke(
    root: Path,
    output: Path,
    subjects: list[int],
    split: str,
    windows: list[float],
    onset_shift: float,
    max_trials: int | None,
    harmonics: int,
    bandpass: tuple[float, float] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import mne

    codebook = sun_stimulus_codebook(root=root, kind=split)
    target_freqs = tuple(
        (float(row["freq_for_left_eye"]), float(row["freq_for_right_eye"]))
        for row in codebook
    )
    n_targets = len(target_freqs)
    records = iter_sun_cnt_files(root=root, splits=[split], subjects=subjects)
    trial_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for record in records:
        raw = mne.io.read_raw_cnt(record.path, preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        picks = [ch for ch in SUN2024_OCCIPITAL9 if ch in raw.ch_names]
        if not picks:
            raise ValueError(f"No requested occipital channels found in {record.path}")
        paired = pair_sun_trial_events(sun_annotation_rows(record.path), n_targets=n_targets)
        if max_trials is not None:
            paired = paired[:max_trials]
        for window in windows:
            samples = int(round(window * sfreq))
            epochs: list[np.ndarray] = []
            labels: list[int] = []
            metadata: list[object] = []
            for trial in paired:
                start = trial.start_sample + int(round(onset_shift * sfreq))
                stop = start + samples
                if stop > raw.n_times:
                    continue
                data = raw.get_data(picks=picks, start=start, stop=stop)
                if data.shape[-1] != samples:
                    continue
                if bandpass is not None:
                    sos = signal.butter(
                        4,
                        [bandpass[0], bandpass[1]],
                        btype="bandpass",
                        fs=sfreq,
                        output="sos",
                    )
                    data = signal.sosfiltfilt(sos, data, axis=-1)
                epochs.append(data)
                labels.append(trial.target_id - 1)
                metadata.append(trial)
            if not epochs:
                continue
            x = np.asarray(epochs, dtype=np.float64)[:, None, :, :]
            y = np.asarray(labels, dtype=np.int64)
            references = multi_frequency_reference_signals(
                target_frequencies=target_freqs,
                samples=samples,
                fs=int(round(sfreq)),
                harmonics=harmonics,
            )
            model = CCA(window=window, harmonics=harmonics, references=references)
            pred, scores = model.predict(x)
            correct = pred == y
            accuracy = float(np.mean(correct))
            summary_rows.append(
                {
                    "method": "CCA",
                    "split": split,
                    "subject": record.subject,
                    "window": window,
                    "onset_shift": onset_shift,
                    "channels": ",".join(picks),
                    "harmonics": harmonics,
                    "bandpass": "" if bandpass is None else f"{bandpass[0]:g}-{bandpass[1]:g}",
                    "trials": int(len(y)),
                    "accuracy": accuracy,
                }
            )
            for idx, trial in enumerate(metadata):
                true_label = int(y[idx])
                pred_label = int(pred[idx])
                trial_rows.append(
                    {
                        "method": "CCA",
                        "split": split,
                        "subject": record.subject,
                        "window": window,
                        "onset_shift": onset_shift,
                        "block": trial.block,
                        "trial_in_block": trial.trial_in_block,
                        "trial_index": trial.trial_index,
                        "target_id": true_label + 1,
                        "predicted_target_id": pred_label + 1,
                        "correct": bool(correct[idx]),
                        "score_true": float(scores[idx, true_label]),
                        "score_pred": float(scores[idx, pred_label]),
                        "start_seconds": trial.start_seconds,
                        "end_seconds": trial.end_seconds,
                        "duration_seconds": trial.duration_seconds,
                    }
                )

    trials_df = pd.DataFrame(trial_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not trials_df.empty:
        trials_df.to_csv(output / "cca_smoke_trials.csv", index=False)
        summary_df.to_csv(output / "cca_smoke_summary.csv", index=False)
        best_row = summary_df.sort_values(["accuracy", "window"], ascending=[False, False]).iloc[0]
        best_window = float(best_row["window"])
        subset = trials_df[trials_df["window"] == best_window]
        plot_confusion(
            subset,
            output / "sun2024_cca_smoke_confusion.png",
            n_targets=n_targets,
            title=f"Sun2024 {split} CCA smoke confusion, w={best_window:g}s",
        )
    return trials_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke inventory for Sun2024 efficient dual-frequency SSVEP data.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "smoke_inventory")
    parser.add_argument("--subjects", default="1")
    parser.add_argument("--splits", default="one_target,forty_targets")
    parser.add_argument("--plot-seconds", type=float, default=6.0)
    parser.add_argument("--run-cca-smoke", action="store_true")
    parser.add_argument("--classification-split", default="forty_targets")
    parser.add_argument("--classification-windows", default="1.0,2.0")
    parser.add_argument("--onset-shift", type=float, default=0.14)
    parser.add_argument("--max-classification-trials", type=int, default=120)
    parser.add_argument("--harmonics", type=int, default=5)
    parser.add_argument("--bandpass", default="6,90")
    args = parser.parse_args()

    root = sun2024_root(args.root)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    splits = parse_csv(args.splits)
    unknown_splits = sorted(set(splits) - set(SUN2024_SPLITS))
    if unknown_splits:
        raise SystemExit(f"Unknown Sun2024 split(s): {unknown_splits}")

    codebook_one = pd.DataFrame(read_sun_codebook(root=root, kind="one_target"))
    codebook_forty = pd.DataFrame(read_sun_codebook(root=root, kind="forty_targets"))
    codebook_one.to_csv(output / "codebook_one_target.csv", index=False)
    codebook_forty.to_csv(output / "codebook_forty_targets.csv", index=False)

    records = iter_sun_cnt_files(root=root, splits=splits, subjects=subjects)
    if not records:
        raise SystemExit("No Sun2024 .cnt files matched the requested smoke scope.")

    file_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    first_record = None
    first_events: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for record in records:
        import mne

        raw = mne.io.read_raw_cnt(record.path, preload=False, verbose="ERROR")
        annotations = sun_annotation_rows(record.path)
        stimulus_codebook = sun_stimulus_codebook(root=root, kind=record.split)
        codebook_by_trigger = {int(row["trigger_num"]): row for row in stimulus_codebook}
        paired = pair_sun_trial_events(annotations, n_targets=len(stimulus_codebook))
        for trial in paired:
            code = codebook_by_trigger.get(trial.target_id, {})
            paired_rows.append(
                {
                    "dataset": "sun2024_efficient_dual_frequency",
                    "split": record.split,
                    "subject": record.subject,
                    "file": str(record.path.relative_to(root)),
                    "trial_index": trial.trial_index,
                    "block": trial.block,
                    "trial_in_block": trial.trial_in_block,
                    "target_id": trial.target_id,
                    "freq_for_left_eye": code.get("freq_for_left_eye"),
                    "freq_for_right_eye": code.get("freq_for_right_eye"),
                    "start_sample": trial.start_sample,
                    "end_sample": trial.end_sample,
                    "start_seconds": trial.start_seconds,
                    "end_seconds": trial.end_seconds,
                    "duration_seconds": trial.duration_seconds,
                }
            )
        for row in annotations:
            event_kind = trigger_kind(row["trigger_num"])
            event_row = {
                "dataset": "sun2024_efficient_dual_frequency",
                "split": record.split,
                "subject": record.subject,
                "file": str(record.path.relative_to(root)),
                **row,
                "event_kind": event_kind,
            }
            event_rows.append(event_row)
        if first_record is None:
            first_record = record
            first_events = event_rows[-len(annotations) :]
        counts = Counter(row["event_kind"] for row in event_rows if row["file"] == str(record.path.relative_to(root)))
        file_rows.append(
            {
                "dataset": "sun2024_efficient_dual_frequency",
                "split": record.split,
                "subject": record.subject,
                "file": str(record.path.relative_to(root)),
                "size_bytes": record.path.stat().st_size,
                "sampling_rate": float(raw.info["sfreq"]),
                "channels": len(raw.ch_names),
                "samples": int(raw.n_times),
                "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
                "events": len(annotations),
                "stimulus_start_events": int(counts.get("stimulus_start", 0)),
                "stimulus_end_events": int(counts.get("stimulus_end", 0)),
                "block_end_events": int(counts.get("block_end", 0)),
            }
        )

    files_df = pd.DataFrame(file_rows)
    events_df = pd.DataFrame(event_rows)
    paired_df = pd.DataFrame(paired_rows)
    summary_df = (
        files_df.groupby(["split"], as_index=False)
        .agg(
            subjects=("subject", "nunique"),
            files=("file", "count"),
            events=("events", "sum"),
            stimulus_start_events=("stimulus_start_events", "sum"),
            stimulus_end_events=("stimulus_end_events", "sum"),
            block_end_events=("block_end_events", "sum"),
            duration_seconds=("duration_seconds", "sum"),
        )
        .sort_values(["split"])
    )

    files_df.to_csv(output / "files.csv", index=False)
    events_df.to_csv(output / "events.csv", index=False)
    paired_df.to_csv(output / "paired_trials.csv", index=False)
    summary_df.to_csv(output / "summary.csv", index=False)
    plot_time_frequency(first_record, first_events, output / "sun2024_smoke_time_frequency.png", args.plot_seconds)

    cca_trials_df = pd.DataFrame()
    cca_summary_df = pd.DataFrame()
    if args.run_cca_smoke:
        cca_trials_df, cca_summary_df = run_cca_smoke(
            root=root,
            output=output,
            subjects=subjects,
            split=args.classification_split,
            windows=parse_windows(args.classification_windows),
            onset_shift=args.onset_shift,
            max_trials=args.max_classification_trials,
            harmonics=args.harmonics,
            bandpass=parse_filter(args.bandpass),
        )

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "subjects": subjects,
        "splits": splits,
        "files": len(files_df),
        "events": len(events_df),
        "paired_trials": len(paired_df),
        "codebook_one_target_rows": len(codebook_one),
        "codebook_forty_targets_rows": len(codebook_forty),
        "cca_smoke": {
            "enabled": bool(args.run_cca_smoke),
            "trials": int(len(cca_trials_df)),
            "summary_rows": int(len(cca_summary_df)),
            "classification_split": args.classification_split if args.run_cca_smoke else None,
            "classification_windows": parse_windows(args.classification_windows) if args.run_cca_smoke else [],
        },
        "outputs": {
            "files": "files.csv",
            "events": "events.csv",
            "paired_trials": "paired_trials.csv",
            "summary": "summary.csv",
            "codebook_one_target": "codebook_one_target.csv",
            "codebook_forty_targets": "codebook_forty_targets.csv",
            "qa_figure": "sun2024_smoke_time_frequency.png",
            "cca_smoke_trials": "cca_smoke_trials.csv" if args.run_cca_smoke else None,
            "cca_smoke_summary": "cca_smoke_summary.csv" if args.run_cca_smoke else None,
            "cca_smoke_confusion": "sun2024_cca_smoke_confusion.png" if args.run_cca_smoke else None,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
