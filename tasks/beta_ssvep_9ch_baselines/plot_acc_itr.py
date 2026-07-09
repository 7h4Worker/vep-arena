from __future__ import annotations

from pathlib import Path

from vep_arena.plots.comparison_curves import AccItrPlotSpec, load_summaries, plot_acc_itr


TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[1]
TASK_RESULTS = TASK / "results"
RUNNER_SUMMARY = TASK_RESULTS / "runner_full" / "summary.csv"
LEGACY_SOURCES = [
    PROJECT / "results" / "beta_toolbox_9ch_w02_20s_cca_fbcca_trca_tdca" / "summary.csv",
    PROJECT / "results" / "beta_toolbox_9ch_w02_20s_etrca" / "summary.csv",
]


def main() -> None:
    sources = [RUNNER_SUMMARY] if RUNNER_SUMMARY.exists() else LEGACY_SOURCES
    summary = load_summaries(sources)
    plot_acc_itr(
        summary,
        TASK_RESULTS,
        AccItrPlotSpec(
            title="BETA SSVEP: Arena Toolbox-Preprocessed Baselines",
            accuracy_title="BETA 9ch Accuracy (0.2-2.0s)",
            itr_title="BETA 9ch ITR (0.2-2.0s)",
            footnote="Protocol: subject-specific leave-one-block-out; 70 subjects x 4 blocks; ITR uses current Arena denominator: window + break.",
        ),
    )


if __name__ == "__main__":
    main()

