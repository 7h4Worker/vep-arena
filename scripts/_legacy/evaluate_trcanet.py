from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vep_arena.config import BENCHMARK_CHANNELS_9, DATA_ROOT, RESULT_ROOT, BenchmarkSpec
from vep_arena.data.benchmark import load_subject_raw
from vep_arena.methods.trcanet import fit_trca_filters, project_with_trca_filters
from vep_arena.metrics import itr_bits_per_minute, sem

TRCANET_SUBBANDS = 3


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


def load_subject_trials_for_trca(
    data_root: Path,
    subject: int,
    window: float,
    channels: tuple[int, ...],
    spec: BenchmarkSpec,
) -> np.ndarray:
    """Return classes x channels x samples x blocks (TRCA convention)."""
    raw = load_subject_raw(data_root, subject)
    channel_idx = np.asarray(channels, dtype=np.int64) - 1
    data = raw[channel_idx, spec.sample_slice(window), :, :]
    return np.transpose(data, (2, 0, 1, 3)).copy()


def build_features(
    data_root: Path,
    subjects: list[int],
    test_block: int,
    window: float,
    channels: tuple[int, ...],
    spec: BenchmarkSpec,
    n_fbs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    train_xs: list[np.ndarray] = []
    train_ys: list[np.ndarray] = []
    test_xs: list[np.ndarray] = []
    test_ys: list[np.ndarray] = []
    train_subjects: list[np.ndarray] = []
    test_subjects: list[np.ndarray] = []
    block_idx = test_block - 1
    feature_seconds = 0.0
    for subject in subjects:
        t0 = time.perf_counter()
        data = load_subject_trials_for_trca(data_root, subject, window, channels, spec)
        train_blocks = [b for b in range(spec.blocks) if b != block_idx]
        train_raw = data[:, :, :, train_blocks]
        test_raw = data[:, :, :, block_idx : block_idx + 1]

        weights = fit_trca_filters(train_raw, spec.sampling_rate, n_fbs)
        train_x = project_with_trca_filters(train_raw, weights, spec.sampling_rate)
        test_x = project_with_trca_filters(test_raw, weights, spec.sampling_rate)

        train_y = np.tile(np.arange(spec.classes, dtype=np.int64), len(train_blocks))
        test_y = np.arange(spec.classes, dtype=np.int64)

        train_xs.append(train_x)
        train_ys.append(train_y)
        test_xs.append(test_x)
        test_ys.append(test_y)
        train_subjects.append(np.full(len(train_y), subject, dtype=np.int64))
        test_subjects.append(np.full(len(test_y), subject, dtype=np.int64))

        elapsed = time.perf_counter() - t0
        feature_seconds += elapsed
        print(f"features subject={subject} seconds={elapsed:.2f}", flush=True)

    return (
        np.concatenate(train_xs),
        np.concatenate(train_ys),
        np.concatenate(test_xs),
        np.concatenate(test_ys),
        np.concatenate(train_subjects),
        np.concatenate(test_subjects),
        feature_seconds,
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
            }
        )


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
                "accuracy_sem": sem(np.asarray(accuracies)),
                "itr": float(np.mean(itrs)),
                "itr_sem": sem(np.asarray(itrs)),
                "subjects": len({int(row["subject"]) for row in group}),
                "samples": int(sum(int(row["samples"]) for row in group)),
            }
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT / "trcanet_eval")
    parser.add_argument("--subjects", default="1-35")
    parser.add_argument("--blocks", default="1-6")
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--subbands", type=int, default=TRCANET_SUBBANDS)
    parser.add_argument("--global-epochs", type=int, default=500)
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--finetune-dropout", type=float, default=0.6)
    parser.add_argument("--final-dropout", type=float, default=0.95)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-global-models", action="store_true")
    parser.add_argument("--save-finetuned-models", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from vep_arena.nn.trcanet import TRCANet

    spec = BenchmarkSpec()
    subjects = parse_range(args.subjects)
    blocks = parse_range(args.blocks)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_model_dir = args.output_dir / "global_models"
    if args.save_global_models:
        global_model_dir.mkdir(parents=True, exist_ok=True)

    trial_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    samples = spec.sample_length(args.window)

    for block in blocks:
        print(f"\n=== TRCANet block {block} ===", flush=True)
        feat_start = time.perf_counter()
        train_x, train_y, test_x, test_y, train_subjects, test_subjects, feature_seconds = build_features(
            args.data_root, subjects, block, args.window, BENCHMARK_CHANNELS_9, spec, args.subbands,
        )
        runtime_rows.append(
            {"method": "TRCANet-global", "window": args.window, "block": block,
             "subject": "__all__", "stage": "trca_features", "seconds": feature_seconds}
        )
        print(f"features: train={train_x.shape}, test={test_x.shape}, {feature_seconds:.1f}s", flush=True)

        model = TRCANet(
            filters=spec.classes,
            samples=samples,
            subbands=args.subbands,
            classes=spec.classes,
            dropout=args.dropout,
            final_dropout=args.final_dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(train_x.astype(np.float32)),
                torch.from_numpy(train_y.astype(np.int64)),
            ),
            batch_size=args.batch_size, shuffle=True,
        )

        train_start = time.perf_counter()
        for epoch in range(1, args.global_epochs + 1):
            epoch_start = time.perf_counter()
            loss = train_epoch(model, train_loader, optimizer, device)
            acc = float("nan")
            should_eval = epoch == 1 or epoch == args.global_epochs or epoch % max(1, args.eval_every) == 0
            if should_eval:
                pred = evaluate_batches(model, test_x.astype(np.float32), args.batch_size, device)
                acc = float(accuracy_score(test_y, pred))
                print(f"TRCANet block={block} epoch={epoch} loss={loss:.6f} acc={acc:.4f}", flush=True)
            epoch_rows.append(
                {
                    "method": "TRCANet-global",
                    "window": args.window,
                    "block": block,
                    "subject": "__all__",
                    "stage": "global_train_epoch",
                    "epoch": epoch,
                    "loss": float(loss),
                    "test_acc": acc,
                    "seconds": time.perf_counter() - epoch_start,
                }
            )
        train_seconds = time.perf_counter() - train_start
        runtime_rows.append(
            {"method": "TRCANet-global", "window": args.window, "block": block,
             "subject": "__all__", "stage": "global_train", "seconds": train_seconds}
        )

        if args.save_global_models:
            model_path = global_model_dir / f"trcanet_global_b{block}_w{args.window:g}s.pt"
            torch.save(model.state_dict(), model_path)

        global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        infer_start = time.perf_counter()
        pred = evaluate_batches(model, test_x.astype(np.float32), args.batch_size, device)
        infer_seconds = time.perf_counter() - infer_start
        runtime_rows.append(
            {"method": "TRCANet-global", "window": args.window, "block": block,
             "subject": "__all__", "stage": "global_inference", "seconds": infer_seconds}
        )
        append_trial_rows(trial_rows, "TRCANet-global", args.window, block,
                          subjects, test_y, pred, test_subjects, spec,
                          feature_seconds + train_seconds + infer_seconds)
        append_prediction_rows(pred_rows, "TRCANet-global", args.window, block, test_y, pred, test_subjects)
        acc = accuracy_score(test_y, pred)
        print(f"TRCANet-global block={block} acc={acc:.4f}", flush=True)

        if args.finetune_epochs > 0:
            ft_start = time.perf_counter()
            ft_pred = np.empty_like(pred)
            for subject in subjects:
                subject_start = time.perf_counter()
                local_model = TRCANet(
                    filters=spec.classes,
                    samples=samples,
                    subbands=args.subbands,
                    classes=spec.classes,
                    dropout=args.finetune_dropout,
                    final_dropout=args.final_dropout,
                ).to(device)
                local_model.load_state_dict(global_state)
                local_optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                train_mask = train_subjects == subject
                test_mask = test_subjects == subject
                local_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(train_x[train_mask].astype(np.float32)),
                        torch.from_numpy(train_y[train_mask].astype(np.int64)),
                    ),
                    batch_size=min(args.batch_size, int(train_mask.sum())), shuffle=True,
                )
                for _ in range(args.finetune_epochs):
                    train_epoch(local_model, local_loader, local_optimizer, device)
                ft_pred[test_mask] = evaluate_batches(
                    local_model, test_x[test_mask].astype(np.float32), args.batch_size, device,
                )
                subject_seconds = time.perf_counter() - subject_start
                runtime_rows.append(
                    {"method": "TRCANet-finetuned", "window": args.window, "block": block,
                     "subject": subject, "stage": "subject_finetune_inference", "seconds": subject_seconds}
                )
                if args.save_finetuned_models:
                    save_dir = args.output_dir / "finetuned_models"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    stem = f"trcanet_s{subject:02d}_b{block}_w{args.window:g}s_finetuned.pt"
                    torch.save(local_model.state_dict(), save_dir / stem)
                print(f"TRCANet-finetuned block={block} subject={subject} {subject_seconds:.2f}s", flush=True)
            ft_seconds = time.perf_counter() - ft_start
            append_trial_rows(trial_rows, "TRCANet-finetuned", args.window, block,
                              subjects, test_y, ft_pred, test_subjects, spec, ft_seconds)
            append_prediction_rows(pred_rows, "TRCANet-finetuned", args.window, block, test_y, ft_pred, test_subjects)
            ft_acc = accuracy_score(test_y, ft_pred)
            print(f"TRCANet-finetuned block={block} acc={ft_acc:.4f}", flush=True)

    block_summary = [row for row in trial_rows if row["subject"] == "__all__"]
    summary_rows = summarize_trials(trial_rows)
    write_csv(
        args.output_dir / "trials.csv", trial_rows,
        ["method", "window", "subject", "block", "accuracy", "itr", "samples", "seconds"],
    )
    write_csv(
        args.output_dir / "summary.csv", summary_rows,
        ["method", "window", "accuracy", "accuracy_sem", "itr", "itr_sem", "subjects", "samples"],
    )
    write_csv(
        args.output_dir / "block.csv", block_summary,
        ["method", "window", "subject", "block", "accuracy", "itr", "samples", "seconds"],
    )
    write_csv(
        args.output_dir / "predictions.csv", pred_rows,
        ["method", "window", "subject", "block", "true", "pred"],
    )
    write_csv(
        args.output_dir / "runtime.csv", runtime_rows,
        ["method", "window", "block", "subject", "stage", "seconds"],
    )
    if epoch_rows:
        write_csv(
            args.output_dir / "epoch_history.csv", epoch_rows,
            ["method", "window", "block", "subject", "stage", "epoch", "loss", "test_acc", "seconds"],
        )
    for method in sorted({str(row["method"]) for row in pred_rows}):
        method_rows = [row for row in pred_rows if row["method"] == method]
        true_values = [int(row["true"]) for row in method_rows]
        pred_values = [int(row["pred"]) for row in method_rows]
        safe_name = method.lower().replace("-", "_")
        np.save(
            args.output_dir / f"confusion_{safe_name}.npy",
            confusion_matrix(true_values, pred_values, labels=np.arange(spec.classes)),
        )

    manifest = {
        "task_name": args.output_dir.name,
        "methods": sorted({str(row["method"]) for row in trial_rows}),
        "status": "complete",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "Arena-native TRCANet: per-subject TRCA spatial filter fitting + CNN classification.",
        "window": args.window,
        "subjects": subjects,
        "blocks": blocks,
        "subbands": args.subbands,
        "global_epochs": args.global_epochs,
        "finetune_epochs": args.finetune_epochs,
        "dropout": args.dropout,
        "finetune_dropout": args.finetune_dropout,
        "final_dropout": args.final_dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "save_global_models": args.save_global_models,
        "save_finetuned_models": args.save_finetuned_models,
        "device": str(device),
        "torch": torch.__version__,
        "preprocessing": {
            "crop_start_seconds": spec.cue_seconds + spec.latency_seconds,
            "trca_filterbank": "Nakanishi-style Chebyshev-I bandpass, 3 subbands.",
            "trca_projection": "Per-subject, per-class TRCA spatial filters, top eigenvector.",
            "channels": list(BENCHMARK_CHANNELS_9),
        },
        "model": {
            "name": "TRCANet",
            "input_shape": f"(batch, {args.subbands}, {spec.classes}, {samples})",
        },
        "seconds": time.perf_counter() - started,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
