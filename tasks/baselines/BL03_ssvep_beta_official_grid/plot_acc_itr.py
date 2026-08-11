from __future__ import annotations

from pathlib import Path

from vep_arena.plots.comparison_curves import AccItrPlotSpec, load_summaries, plot_acc_itr


TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[2]
TASK_RESULTS = TASK / "results"
SOURCES = [
    PROJECT / "results" / "beta_toolbox_official_grid_cca_fbcca_trca_tdca" / "summary.csv",
    PROJECT / "results" / "beta_toolbox_official_grid_etrca" / "summary.csv",
]


def main() -> None:
    summary = load_summaries(SOURCES)
    plot_acc_itr(
        summary,
        TASK_RESULTS,
        AccItrPlotSpec(
            title="BETA SSVEP: Arena Toolbox-Preprocessed Baselines",
            accuracy_title="BETA 9ch Official Grid Accuracy",
            itr_title="BETA 9ch Official Grid ITR",
            footnote="Protocol: subject-specific leave-one-block-out; 70 subjects x 4 blocks; ITR uses current Arena denominator: window + break.",
            y_accuracy=(0.0, 0.9),
            y_itr=(0.0, 185.0),
        ),
    )


if __name__ == "__main__":
    main()

