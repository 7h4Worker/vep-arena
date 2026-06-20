# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-18
# Last updated: 2026-06-18
# Description: Evaluate saved DNN checkpoints through the Arena data path.
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import cheby1, sosfiltfilt
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import BENCHMARK_CHANNELS_9, DATA_ROOT, RESULT_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import load_subject_raw
from vep_arena.metrics import itr_bits_per_minute


def parse_range(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        elif part:
            values.append(int(part))
    return values


def make_dnn_filter_bank(subbands: int, fs: int) -> list[np.ndarray]:
    return [
        cheby1(N=2, rp=1, Wn=[8 * idx, 90], btype="bandpass", fs=fs, output="sos")
        for idx in range(1, subbands + 1)
    ]


def preprocess_subject_for_dnn(
    data_root: Path,
    subject: int,
    window: float,
    channels: tuple[int, ...],
    spec: BenchmarkSpec,
    subbands: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = load_subject_raw(data_root, subject)
    channel_idx = np.asarray(channels, dtype=np.int64) - 1
    sample_window = spec.sample_slice(window)
    sub = raw[channel_idx, sample_window, :, :]
    filters = make_dnn_filter_bank(subbands, spec.sampling_rate)
    samples = spec.sample_length(window)
    x = np.zeros((spec.classes * spec.blocks, subbands, len(channels), samples), dtype=np.float32)
    y = np.zeros((spec.classes * spec.blocks,), dtype=np.int64)
    block_numbers = np.zeros((spec.classes * spec.blocks,), dtype=np.int64)
    row = 0
    for block in range(spec.blocks):
        for cls in range(spec.classes):
            trial = sub[:, :, cls, block]
            for fb, sos in enumerate(filters):
                x[row, fb] = sosfiltfilt(sos, trial, axis=-1).astype(np.float32)
            y[row] = cls
            block_numbers[row] = block
            row += 1
    return x, y, block_numbers


def load_split_from_arena(
    data_root: Path,
    subjects: list[int],
    test_block: int,
    window: float,
    channels: tuple[int, ...],
    spec: BenchmarkSpec,
    subbands: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_xs: list[np.ndarray] = []
    train_ys: list[np.ndarray] = []
    test_xs: list[np.ndarray] = []
    test_ys: list[np.ndarray] = []
    train_subjects: list[np.ndarray] = []
    test_subjects: list[np.ndarray] = []
    block_idx = test_block - 1
    for subject in subjects:
        x, y, block_numbers = preprocess_subject_for_dnn(data_root, subject, window, channels, spec, subbands)
        test_mask = block_numbers == block_idx
        train_xs.append(x[~test_mask])
        train_ys.append(y[~test_mask])
        test_xs.append(x[test_mask])
        test_ys.append(y[test_mask])
        train_subjects.append(np.full(int((~test_mask).sum()), subject, dtype=np.int64))
        test_subjects.append(np.full(int(test_mask.sum()), subject, dtype=np.int64))
    return (
        np.concatenate(train_xs),
        np.concatenate(train_ys),
        np.concatenate(test_xs),
        np.concatenate(test_ys),
        np.concatenate(train_subjects),
        np.concatenate(test_subjects),
    )


def registry_rows(registry_path: Path, window: float, blocks: list[int]) -> list[dict[str, str]]:
    with registry_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selected = [
        row
        for row in rows
        if round(float(row["signal_length"]), 6) == round(window, 6) and int(row["block"]) in blocks
    ]
    if not selected:
        raise ValueError(f"No checkpoint rows found for window={window:g}, blocks={blocks}")
    return sorted(selected, key=lambda row: int(row["block"]))


def checkpoint_tasks(registry_path: Path, window: float, blocks: list[int]) -> list[dict[str, object]]:
    if not registry_path.exists():
        return [{"block": block, "model_path": None, "source": "train"} for block in blocks]
    with registry_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_block = {
        int(row["block"]): row
        for row in rows
        if round(float(row["signal_length"]), 6) == round(window, 6) and int(row["block"]) in blocks
    }
    tasks: list[dict[str, object]] = []
    for block in blocks:
        row = by_block.get(block)
        tasks.append(
            {
                "block": block,
                "model_path": Path(str(row["model_path"])) if row else None,
                "source": "registry" if row else "train",
            }
        )
    return tasks


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sem(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def train_epoch(model, loader, optimizer, device) -> float:
    import torch

    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


def train_global_model(
    model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    args: argparse.Namespace,
    device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epoch_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, args.train_global_epochs + 1):
        epoch_start = time.perf_counter()
        loss = train_epoch(model, train_loader, optimizer, device)
        acc = float("nan")
        if epoch == 1 or epoch == args.train_global_epochs or epoch % max(1, args.eval_every) == 0:
            pred = evaluate_batches(model, test_x, args.batch_size, device)
            acc = float(accuracy_score(test_y, pred))
            print(
                f"DNN-global-trained w={args.window:g} block={args._current_block} "
                f"epoch={epoch} loss={loss:.6f} acc={acc:.4f}",
                flush=True,
            )
        epoch_rows.append(
            {
                "method": "DNN-global-trained",
                "window": args.window,
                "block": args._current_block,
                "subject": "__all__",
                "stage": "global_train_epoch",
                "epoch": epoch,
                "loss": float(loss),
                "test_acc": acc,
                "seconds": time.perf_counter() - epoch_start,
            }
        )
    return {"epochs": args.train_global_epochs, "seconds": time.perf_counter() - started}, epoch_rows


def evaluate_batches(model, x: np.ndarray, batch_size: int, device) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dummy_y = torch.zeros((len(x),), dtype=torch.long)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), dummy_y), batch_size=batch_size, shuffle=False)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def append_prediction_rows(
    rows: list[dict[str, object]],
    method: str,
    window: float,
    block: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
) -> None:
    for true_label, pred_label, subject in zip(y_true, y_pred, subjects):
        rows.append(
            {
                "method": method,
                "window": window,
                "subject": int(subject),
                "block": block,
                "true": int(true_label),
                "pred": int(pred_label),
            }
        )


def append_trial_rows(
    rows: list[dict[str, object]],
    method: str,
    window: float,
    block: int,
    subjects: list[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_subjects: np.ndarray,
    spec: BenchmarkSpec,
    model_path: Path,
    seconds: float,
) -> None:
    acc = accuracy_score(y_true, y_pred)
    itr = itr_bits_per_minute(float(acc), spec.classes, window + spec.cue_seconds)
    rows.append(
        {
            "method": method,
            "window": window,
            "subject": "__all__",
            "block": block,
            "accuracy": float(acc),
            "itr": float(itr),
            "samples": int(len(y_true)),
            "seconds": seconds,
            "model_path": str(model_path),
        }
    )
    for subject in subjects:
        mask = test_subjects == subject
        subject_acc = accuracy_score(y_true[mask], y_pred[mask])
        subject_itr = itr_bits_per_minute(float(subject_acc), spec.classes, window + spec.cue_seconds)
        rows.append(
            {
                "method": method,
                "window": window,
                "subject": subject,
                "block": block,
                "accuracy": float(subject_acc),
                "itr": float(subject_itr),
                "samples": int(mask.sum()),
                "seconds": 0.0,
                "model_path": str(model_path),
            }
        )


def summarize_trials(trial_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    subject_trials = [row for row in trial_rows if row["subject"] != "__all__"]
    summary_rows: list[dict[str, object]] = []
    keys = sorted({(str(row["method"]), float(row["window"])) for row in subject_trials})
    for method, window in keys:
        group = [row for row in subject_trials if row["method"] == method and float(row["window"]) == window]
        accuracies = [float(row["accuracy"]) for row in group]
        itrs = [float(row["itr"]) for row in group]
        summary_rows.append(
            {
                "method": method,
                "window": window,
                "accuracy": float(np.mean(accuracies)),
                "accuracy_sem": sem(accuracies),
                "itr": float(np.mean(itrs)),
                "itr_sem": sem(itrs),
                "subjects": len({int(row["subject"]) for row in group}),
                "samples": int(sum(int(row["samples"]) for row in group)),
            }
        )
    return summary_rows


def finetune_from_checkpoint(
    model_cls,
    global_state: dict[str, object],
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_subjects: np.ndarray,
    test_x: np.ndarray,
    test_subjects: np.ndarray,
    subjects: list[int],
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    device,
    model_path: Path,
    method: str = "DNN-finetuned-pt",
) -> tuple[np.ndarray, list[dict[str, object]]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    y_pred = np.empty((len(test_x),), dtype=np.int64)
    runtime_rows: list[dict[str, object]] = []
    save_dir = args.output_dir / "finetuned_models"
    if args.save_finetuned_models:
        save_dir.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        subject_start = time.perf_counter()
        local_model = model_cls(
            channels=len(BENCHMARK_CHANNELS_9),
            samples=spec.sample_length(args.window),
            subbands=args.subbands,
            classes=spec.classes,
            dropout=args.finetune_dropout,
            final_dropout=args.final_dropout,
        ).to(device)
        local_model.load_state_dict(global_state)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_mask = train_subjects == subject
        test_mask = test_subjects == subject
        local_train = DataLoader(
            TensorDataset(torch.from_numpy(train_x[train_mask]), torch.from_numpy(train_y[train_mask])),
            batch_size=min(args.batch_size, int(train_mask.sum())),
            shuffle=True,
        )
        for _ in range(args.finetune_epochs):
            train_epoch(local_model, local_train, optimizer, device)
        y_pred[test_mask] = evaluate_batches(local_model, test_x[test_mask], args.batch_size, device)
        subject_seconds = time.perf_counter() - subject_start
        runtime_rows.append(
            {
                "method": method,
                "window": args.window,
                "block": args._current_block,
                "subject": subject,
                "stage": "subject_finetune_inference",
                "seconds": subject_seconds,
            }
        )
        if args.save_finetuned_models:
            stem = f"dnn_s{subject:02d}_b{args._current_block}_w{args.window:g}s_finetuned.pt"
            torch.save(local_model.state_dict(), save_dir / stem)
        print(
            f"{method} w={args.window:g} block={args._current_block} "
            f"subject={subject} seconds={subject_seconds:.2f}",
            flush=True,
        )
    return y_pred, runtime_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--dnn-project-root", type=Path, default=Path("D:/ProjData/proj_python/dnn_ssvep_pytorch"))
    parser.add_argument("--registry", type=Path, default=Path("D:/ProjData/proj_python/dnn_ssvep_pytorch/results_clean/model_registry.csv"))
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "dnn_checkpoint_eval")
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--blocks", default="1-6")
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--subbands", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--finetune-dropout", type=float, default=0.6)
    parser.add_argument("--final-dropout", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--save-finetuned-models", action="store_true")
    parser.add_argument("--train-global-epochs", type=int, default=0)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-global-models", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.dnn_project_root))
    import torch
    from dnn_ssvep.model import DNNSsvep

    spec = BenchmarkSpec()
    subjects = parse_range(args.subjects)
    blocks = parse_range(args.blocks)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_model_dir = args.output_dir / "global_models"
    if args.save_global_models:
        global_model_dir.mkdir(parents=True, exist_ok=True)

    trial_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for task in checkpoint_tasks(args.registry, args.window, blocks):
        block = int(task["block"])
        model_path = task["model_path"]
        use_registry = (not args.force_train) and model_path is not None
        if use_registry and not Path(model_path).exists():
            raise FileNotFoundError(model_path)
        if not use_registry and args.train_global_epochs <= 0:
            raise ValueError(
                f"No saved checkpoint for window={args.window:g}, block={block}. "
                "Pass --train-global-epochs N to train it in Arena."
            )
        split_start = time.perf_counter()
        train_x, train_y, test_x, test_y, train_subjects, test_subjects = load_split_from_arena(
            args.data_root,
            subjects,
            block,
            args.window,
            BENCHMARK_CHANNELS_9,
            spec,
            args.subbands,
        )
        load_seconds = time.perf_counter() - split_start
        model = DNNSsvep(
            channels=len(BENCHMARK_CHANNELS_9),
            samples=spec.sample_length(args.window),
            subbands=args.subbands,
            classes=spec.classes,
            dropout=args.dropout,
            final_dropout=args.final_dropout,
        ).to(device)
        args._current_block = block
        if use_registry:
            model_path = Path(model_path)
            global_state = torch.load(model_path, map_location=device)
            model.load_state_dict(global_state)
            global_method = "DNN-global-pt"
            finetune_method = "DNN-finetuned-pt"
        else:
            train_info, block_epoch_rows = train_global_model(model, train_x, train_y, test_x, test_y, args, device)
            epoch_rows.extend(block_epoch_rows)
            runtime_rows.append(
                {
                    "method": "DNN-global-trained",
                    "window": args.window,
                    "block": block,
                    "subject": "__all__",
                    "stage": "global_train",
                    "seconds": float(train_info["seconds"]),
                }
            )
            if args.save_global_models:
                model_path = global_model_dir / f"dnn_global_b{block}_w{args.window:g}s.pt"
                torch.save(model.state_dict(), model_path)
            else:
                model_path = Path("")
            global_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            global_method = "DNN-global-trained"
            finetune_method = "DNN-finetuned-trained"
        infer_start = time.perf_counter()
        pred = evaluate_batches(model, test_x, args.batch_size, device)
        infer_seconds = time.perf_counter() - infer_start
        append_trial_rows(
            trial_rows,
            global_method,
            args.window,
            block,
            subjects,
            test_y,
            pred,
            test_subjects,
            spec,
            model_path,
            load_seconds + infer_seconds,
        )
        runtime_rows.extend(
            [
                {"method": global_method, "window": args.window, "block": block, "subject": "__all__", "stage": "arena_dnn_preprocess", "seconds": load_seconds},
                {"method": global_method, "window": args.window, "block": block, "subject": "__all__", "stage": "global_inference", "seconds": infer_seconds},
            ]
        )
        append_prediction_rows(pred_rows, global_method, args.window, block, test_y, pred, test_subjects)
        acc = accuracy_score(test_y, pred)
        print(f"{global_method} w={args.window:g} block={block} acc={acc:.4f}", flush=True)

        if args.finetune_epochs > 0:
            ft_start = time.perf_counter()
            ft_method = finetune_method
            ft_pred, ft_runtime = finetune_from_checkpoint(
                DNNSsvep,
                global_state,
                train_x,
                train_y,
                train_subjects,
                test_x,
                test_subjects,
                subjects,
                args,
                spec,
                device,
                model_path,
                ft_method,
            )
            ft_seconds = time.perf_counter() - ft_start
            append_trial_rows(
                trial_rows,
                ft_method,
                args.window,
                block,
                subjects,
                test_y,
                ft_pred,
                test_subjects,
                spec,
                model_path,
                ft_seconds,
            )
            for runtime_row in ft_runtime:
                runtime_row["method"] = ft_method
            append_prediction_rows(pred_rows, ft_method, args.window, block, test_y, ft_pred, test_subjects)
            runtime_rows.extend(ft_runtime)
            ft_acc = accuracy_score(test_y, ft_pred)
            print(f"{ft_method} w={args.window:g} block={block} acc={ft_acc:.4f}", flush=True)

    block_summary = [row for row in trial_rows if row["subject"] == "__all__"]
    summary_rows = summarize_trials(trial_rows)
    write_csv(
        args.output_dir / "trials.csv",
        trial_rows,
        ["method", "window", "subject", "block", "accuracy", "itr", "samples", "seconds", "model_path"],
    )
    write_csv(args.output_dir / "summary.csv", summary_rows, ["method", "window", "accuracy", "accuracy_sem", "itr", "itr_sem", "subjects", "samples"])
    write_csv(
        args.output_dir / "block.csv",
        block_summary,
        ["method", "window", "subject", "block", "accuracy", "itr", "samples", "seconds", "model_path"],
    )
    write_csv(args.output_dir / "predictions.csv", pred_rows, ["method", "window", "subject", "block", "true", "pred"])
    write_csv(args.output_dir / "runtime.csv", runtime_rows, ["method", "window", "block", "subject", "stage", "seconds"])
    if epoch_rows:
        write_csv(
            args.output_dir / "epoch_history.csv",
            epoch_rows,
            ["method", "window", "block", "subject", "stage", "epoch", "loss", "test_acc", "seconds"],
        )
    for method in sorted({str(row["method"]) for row in pred_rows}):
        method_rows = [row for row in pred_rows if row["method"] == method]
        true_values = [int(row["true"]) for row in method_rows]
        pred_values = [int(row["pred"]) for row in method_rows]
        safe_name = method.lower().replace("-", "_")
        np.save(args.output_dir / f"confusion_{safe_name}.npy", confusion_matrix(true_values, pred_values, labels=np.arange(spec.classes)))
    manifest = {
        "task_name": args.output_dir.name,
        "methods": sorted({str(row["method"]) for row in trial_rows}),
        "status": "complete",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "Arena-side DNN preprocessing plus saved global checkpoint inference and optional subject fine-tuning.",
        "window": args.window,
        "subjects": subjects,
        "blocks": blocks,
        "finetune_epochs": args.finetune_epochs,
        "train_global_epochs": args.train_global_epochs,
        "force_train": args.force_train,
        "dropout": args.dropout,
        "finetune_dropout": args.finetune_dropout,
        "final_dropout": args.final_dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "save_finetuned_models": args.save_finetuned_models,
        "save_global_models": args.save_global_models,
        "device": str(device),
        "torch": torch.__version__,
        "dnn_project_root": str(args.dnn_project_root),
        "registry": str(args.registry),
        "preprocessing": {
            "crop_start_seconds": spec.cue_seconds + spec.latency_seconds,
            "subbands": args.subbands,
            "filterbank": "Chebyshev-I order-2 bandpass filters, [8*i, 90] Hz, scipy sosfiltfilt.",
            "notch": "None.",
            "channels": list(BENCHMARK_CHANNELS_9),
        },
        "limitations": [
            "Saved checkpoints are global stage-1 models only.",
            "Historical imported DNN scores include subject-specific fine-tuning that was not saved in these .pt files.",
        ],
        "seconds": time.perf_counter() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
