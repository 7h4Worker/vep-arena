from __future__ import annotations

from pathlib import Path

from vep_arena.plots.comparison_curves import AccItrPlotSpec, load_summaries, plot_acc_itr


PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "results"
OUT = RESULTS / "beta_toolbox_9ch_w02_20s_compare"
SOURCES = [
    RESULTS / "beta_toolbox_9ch_w02_20s_cca_fbcca_trca_tdca" / "summary.csv",
    RESULTS / "beta_toolbox_9ch_w02_20s_etrca" / "summary.csv",
]


def main() -> None:
    summary = load_summaries(SOURCES)
    plot_acc_itr(
        summary,
        OUT,
        AccItrPlotSpec(
            title="BETA SSVEP: Arena Toolbox-Preprocessed Baselines",
            accuracy_title="BETA 9ch Accuracy (0.2-2.0s)",
            itr_title="BETA 9ch ITR (0.2-2.0s)",
            footnote="Protocol: subject-specific leave-one-block-out; 70 subjects x 4 blocks; ITR uses current Arena denominator: window + break.",
        ),
    )


if __name__ == "__main__":
    main()
