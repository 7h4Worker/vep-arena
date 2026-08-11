from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


TASK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TASK_DIR / "results"


def read_rows() -> list[dict[str, str]]:
    path = RESULTS_DIR / "online_table2_reproduction.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = read_rows()
    subjects = [row["subject"] for row in rows]
    local_itr = [float(row["itr_bpm"]) for row in rows]
    paper_itr = [float(row["paper_itr_bpm"]) for row in rows]
    delta_trials = [int(row["delta_correct_trials"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    x = list(range(len(subjects)))
    width = 0.38

    axes[0].bar([i - width / 2 for i in x], paper_itr, width=width, label="paper", color="#6877b8")
    axes[0].bar([i + width / 2 for i in x], local_itr, width=width, label="local", color="#2e8b72")
    axes[0].set_xticks(x, subjects)
    axes[0].set_ylabel("ITR (bpm)")
    axes[0].set_title("Online Table 2 TDCA Reproduction")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    colors = ["#2e8b72" if v >= 0 else "#b85757" for v in delta_trials]
    axes[1].bar(x, delta_trials, color=colors)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(x, subjects)
    axes[1].set_ylabel("local - paper correct trials")
    axes[1].grid(axis="y", alpha=0.25)

    out = RESULTS_DIR / "online_table2_reproduction.png"
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
