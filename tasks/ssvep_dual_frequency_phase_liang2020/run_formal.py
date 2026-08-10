from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy import signal
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.dual_frequency import iter_liang_mat_files, load_liang_mat  # noqa: E402
from vep_arena.methods.traditional import TRCA  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_BANDS = (
    (8.0, 80.0),
    (16.0, 80.0),
    (24.0, 80.0),
    (32.0, 80.0),
    (40.0, 80.0),
)
DEFAULT_WINDOWS = {
    "exp1": (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0),
    "exp2": (0.2, 0.4, 0.6, 0.8, 1.0, 1.2),
    "exp3": (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0),
    "exp4": (0.2, 0.4, 0.6, 0.8, 1.0),
}
CONDITION_LABELS = {
    ("exp1", 1): "method1_fast_gradient_descent",
    ("exp1", 2): "method2_fast_optimization",
    ("exp1", 3): "method3_global_search",
    ("exp2", 1): "method1_fast_gradient_descent",
    ("exp2", 2): "method4_random_worst",
    ("exp2", 3): "method4_random_median",
    ("exp2", 4): "method4_random_best",
    ("exp3", 1): "longest_frequency_distance_zero_phase",
    ("exp3", 2): "method1_frequency_pairs_zero_phase",
    ("exp3", 3): "method1_dual_frequency_phase",
    ("exp4", 1): "method1_dual_frequency_phase",
    ("exp4", 2): "single_frequency_phase_jfpm",
}


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_methods(text: str) -> list[str]:
    methods = [item.strip().upper() for item in text.split(",") if item.strip()]
    valid = {"TRCA", "ETRCA"}
    unknown = sorted(set(methods) - valid)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return methods


def parse_subjects(text: str, available: list[int]) -> list[int]:
    if text.strip().lower() == "all":
        return available
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = part.split("-", 1)
            values.extend(range(int(start), int(stop) + 1))
        else:
            values.append(int(part))
    return sorted(dict.fromkeys(values))


def parse_windows(text: str, experiments: list[str]) -> dict[str, tuple[float, ...]]:
    if text.strip().lower() == "paper":
        return {experiment: DEFAULT_WINDOWS[experiment] for experiment in experiments}
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    return {experiment: values for experiment in experiments}


def method_model(method: str, n_bands: int) -> TRCA:
    if method == "TRCA":
        return TRCA(n_fbs=n_bands, ensemble=False)
    if method == "ETRCA":
        return TRCA(n_fbs=n_bands, ensemble=True)
    raise ValueError(method)


def itr_bits_per_min(classes: int, accuracy: float, selection_time: float) -> float:
    p = min(max(float(accuracy), 1e-12), 1.0 - 1e-12)
    n = float(classes)
    bits = math.log2(n) + p * math.log2(p) + (1.0 - p) * math.log2((1.0 - p) / (n - 1.0))
    return bits * 60.0 / selection_time


def filterband(data: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    high = min(high, fs / 2.0 - 1.0)
    sos = signal.butter(4, (low, high), btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data, axis=1)


def file_epochs(path: Path, *, window: float, onset_shift: float, n_bands: int) -> tuple[np.ndarray, np.ndarray]:
    epoch = load_liang_mat(path)
    fs = 1000.0
    start = int(round(onset_shift * fs))
    samples = int(round(window * fs))
    stop = start + samples
    if stop > epoch.x.shape[1]:
        raise ValueError(f"Window {window}s plus onset {onset_shift}s exceeds {path} samples {epoch.x.shape[1]}")
    bands = DEFAULT_BANDS[:n_bands]
    x = np.zeros((epoch.n_trials, n_bands, epoch.n_channels, samples), dtype=np.float64)
    for band_idx, (low, high) in enumerate(bands):
        filtered = filterband(epoch.x, fs, low, high)
        x[:, band_idx] = np.transpose(filtered[:, start:stop, :], (2, 0, 1))
    y = np.ravel(epoch.labels).astype(np.int64) - 1
    return x, y


def load_group_epochs(records, *, window: float, onset_shift: float, n_bands: int):
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    blocks: list[np.ndarray] = []
    runs: list[np.ndarray] = []
    for record in sorted(records, key=lambda item: item.run):
        x, y = file_epochs(record.path, window=window, onset_shift=onset_shift, n_bands=n_bands)
        xs.append(x)
        ys.append(y)
        blocks.append(np.full(y.shape, int(record.run), dtype=np.int64))
        runs.append(np.full(y.shape, int(record.run), dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(blocks), np.concatenate(runs)


def split_masks(experiment: str, blocks: np.ndarray, *, exp4_split: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if experiment == "exp4" and exp4_split == "paper":
        train = blocks <= 6
        test = (blocks >= 7) & (blocks <= 9)
        return [("paper_train1-6_test7-9", train, test)]
    out = []
    for block in sorted(np.unique(blocks).tolist()):
        test = blocks == block
        train = ~test
        out.append((f"leave_block_{int(block)}", train, test))
    return out


def valid_split(y: np.ndarray, train: np.ndarray, test: np.ndarray) -> bool:
    if not np.any(train) or not np.any(test):
        return False
    train_classes = set(np.unique(y[train]).tolist())
    test_classes = set(np.unique(y[test]).tolist())
    return test_classes.issubset(train_classes)


def write_confusion(path: Path, y_true: list[int], y_pred: list[int], classes: int, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(classes)))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    image = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    step = 1 if classes <= 12 else 5
    ax.set_xticks(np.arange(0, classes, step))
    ax.set_yticks(np.arange(0, classes, step))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    experiments = parse_csv(args.experiments)
    methods = parse_methods(args.methods)
    all_records = iter_liang_mat_files(experiments=experiments)
    available_subjects = sorted({record.subject for record in all_records})
    subjects = parse_subjects(args.subjects, available_subjects)
    subject_set = set(subjects)
    records = [record for record in all_records if record.subject in subject_set]
    windows_by_exp = parse_windows(args.windows, experiments)

    groups: dict[tuple[str, int, int], list[object]] = defaultdict(list)
    for record in records:
        groups[(record.experiment, record.subject, record.condition)].append(record)

    output = Path(args.output)
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    manifest = {
        "task": "ssvep_dual_frequency_phase_liang2020",
        "subjects": subjects,
        "experiments": experiments,
        "methods": methods,
        "windows": {key: list(value) for key, value in windows_by_exp.items()},
        "n_bands": args.n_bands,
        "bands": DEFAULT_BANDS[: args.n_bands],
        "onset_shift": args.onset_shift,
        "exp4_split": args.exp4_split,
        "condition_labels": {f"{exp}_condition_{cond}": label for (exp, cond), label in CONDITION_LABELS.items()},
        "source_note": "TRCA/eTRCA template classification; TDCA/reference classifiers require digitized stimulus codebooks.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    trial_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    aggregate: dict[tuple[str, int, str, float, str], tuple[list[int], list[int], int]] = {}

    for key in sorted(groups):
        experiment, subject, condition = key
        condition_label = CONDITION_LABELS.get((experiment, condition), f"condition_{condition}")
        group_records = sorted(groups[key], key=lambda item: item.run)
        for window in windows_by_exp[experiment]:
            print(
                f"[Liang2020] {experiment} subject={subject:02d} condition={condition} "
                f"window={window:.1f}s load/filter",
                flush=True,
            )
            x, y, blocks, runs = load_group_epochs(
                group_records,
                window=window,
                onset_shift=args.onset_shift,
                n_bands=args.n_bands,
            )
            classes = int(np.max(y)) + 1
            splits = split_masks(experiment, blocks, exp4_split=args.exp4_split)

            for method in methods:
                true_all: list[int] = []
                pred_all: list[int] = []
                used_splits = 0
                train_trials_total = 0
                test_trials_total = 0
                for split_name, train_mask, test_mask in splits:
                    if not valid_split(y, train_mask, test_mask):
                        continue
                    model = method_model(method, args.n_bands)
                    model.fit(x[train_mask], y[train_mask])
                    pred, scores = model.predict(x[test_mask])
                    true = y[test_mask]
                    test_indices = np.flatnonzero(test_mask)
                    used_splits += 1
                    train_trials_total += int(np.sum(train_mask))
                    test_trials_total += int(np.sum(test_mask))
                    for local_idx, trial_idx in enumerate(test_indices):
                        trial_rows.append(
                            {
                                "experiment": experiment,
                                "subject": subject,
                                "condition": condition,
                                "condition_label": condition_label,
                                "window": window,
                                "method": method,
                                "split": split_name,
                                "run": int(runs[trial_idx]),
                                "target_id": int(true[local_idx] + 1),
                                "predicted_id": int(pred[local_idx] + 1),
                                "correct": int(pred[local_idx] == true[local_idx]),
                                "score_true": float(scores[local_idx, true[local_idx]]),
                                "score_pred": float(scores[local_idx, pred[local_idx]]),
                            }
                        )
                    true_all.extend(true.tolist())
                    pred_all.extend(pred.tolist())

                if not true_all:
                    continue
                accuracy = float(np.mean(np.asarray(true_all) == np.asarray(pred_all)))
                summary_rows.append(
                    {
                        "experiment": experiment,
                        "subject": subject,
                        "condition": condition,
                        "condition_label": condition_label,
                        "window": window,
                        "method": method,
                        "classes": classes,
                        "splits": used_splits,
                        "train_trials_total": train_trials_total,
                        "test_trials": test_trials_total,
                        "accuracy": accuracy,
                        "itr_bits_per_min": itr_bits_per_min(classes, accuracy, window + args.search_time),
                    }
                )
                agg_key = (experiment, condition, method, window, condition_label)
                if agg_key not in aggregate:
                    aggregate[agg_key] = ([], [], classes)
                aggregate[agg_key][0].extend(true_all)
                aggregate[agg_key][1].extend(pred_all)
                print(
                    f"[Liang2020] {experiment} subject={subject:02d} condition={condition} "
                    f"window={window:.1f}s method={method} acc={accuracy:.4f}",
                    flush=True,
                )

    if not summary_rows:
        raise RuntimeError("No Liang2020 summary rows were produced.")
    with (output / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trial_rows)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_windows = {float(value) for value in parse_csv(args.plot_windows)}
    for (experiment, condition, method, window, condition_label), (true, pred, classes) in aggregate.items():
        if window not in plot_windows:
            continue
        if experiment not in {"exp3", "exp4"}:
            continue
        figure_name = f"confusion_{experiment}_condition{condition}_{method.lower()}_{window:.1f}s.png"
        write_confusion(
            figures / figure_name,
            true,
            pred,
            classes,
            f"Liang2020 {experiment} C{condition} {method} {window:.1f}s",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Formal Liang2020 TRCA/eTRCA classification runner.")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--experiments", default="exp1,exp2,exp3,exp4")
    parser.add_argument("--methods", default="TRCA,ETRCA")
    parser.add_argument("--windows", default="paper", help="'paper' or comma-separated windows.")
    parser.add_argument("--n-bands", type=int, default=5, choices=range(1, len(DEFAULT_BANDS) + 1))
    parser.add_argument("--onset-shift", type=float, default=0.14)
    parser.add_argument("--search-time", type=float, default=0.5)
    parser.add_argument("--exp4-split", choices=["paper", "leave-one-block"], default="paper")
    parser.add_argument("--plot-windows", default="1.0")
    parser.add_argument("--output", default="tasks/ssvep_dual_frequency_phase_liang2020/results/formal_full")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
