from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mean_sem(frame: pd.DataFrame, group: list[str], value: str) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(group):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = part[value].to_numpy(dtype=float)
        sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        rows.append({**dict(zip(group, keys)), "mean": float(np.mean(vals)), "sem": sem, "n": int(vals.size)})
    return pd.DataFrame(rows)


def save_line_with_sem(
    data: pd.DataFrame,
    out: Path,
    *,
    x: str,
    y: str,
    label: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.errorbar(data[x], data[y], yerr=data.get("sem"), marker="o", capsize=3, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def save_heatmap(matrix: np.ndarray, out: Path, *, title: str, xlabel: str, ylabel: str, colorbar: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.2, 6.8))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(label=colorbar)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
