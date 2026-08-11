from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TASK_DIR / "results"
CONFUSION_DIR = RESULTS_DIR / "confusions"


def read_summary() -> list[dict[str, str]]:
    with (RESULTS_DIR / "online_table2_reproduction.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_top_errors(rows: list[dict[str, str]]) -> None:
    out_rows = []
    for row in rows:
        subject = row["subject"]
        targets = int(row["targets"])
        matrix = np.load(CONFUSION_DIR / f"confusion_{subject}_{targets}target.npy")
        errors = matrix.copy()
        np.fill_diagonal(errors, 0)
        idx = np.argwhere(errors > 0)
        ranked = sorted(idx, key=lambda p: int(errors[p[0], p[1]]), reverse=True)
        for true_idx, pred_idx in ranked[:20]:
            out_rows.append(
                {
                    "subject": subject,
                    "targets": targets,
                    "true": int(true_idx) + 1,
                    "pred": int(pred_idx) + 1,
                    "count": int(errors[true_idx, pred_idx]),
                }
            )
    with (RESULTS_DIR / "online_table2_top_errors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "targets", "true", "pred", "count"])
        writer.writeheader()
        writer.writerows(out_rows)


def main() -> None:
    rows = read_summary()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4), constrained_layout=True)
    for ax, row in zip(axes.ravel(), rows):
        subject = row["subject"]
        targets = int(row["targets"])
        matrix = np.load(CONFUSION_DIR / f"confusion_{subject}_{targets}target.npy").astype(np.float32)
        row_sum = matrix.sum(axis=1, keepdims=True)
        norm = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
        ax.imshow(norm, vmin=0, vmax=1, cmap="magma", interpolation="nearest")
        ax.set_title(f"{subject} ({targets}, {row['window_ms']} ms)")
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        if targets > 80:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ticks = np.linspace(0, targets - 1, 5, dtype=int)
            ax.set_xticks(ticks, ticks + 1)
            ax.set_yticks(ticks, ticks + 1)
    out = RESULTS_DIR / "online_table2_confusions.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    write_top_errors(rows)
    print(out)
    print(RESULTS_DIR / "online_table2_top_errors.csv")


if __name__ == "__main__":
    main()
