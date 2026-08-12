from __future__ import annotations

from pathlib import Path

from vep_arena.plots.comparison_curves import AccItrPlotSpec, load_summaries, plot_acc_itr


PROJECT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT / "results"
OUT = RESULTS / "beta_toolbox_official_grid_compare"
SOURCES = [
    RESULTS / "beta_toolbox_official_grid_cca_fbcca_trca_tdca" / "summary.csv",
    RESULTS / "beta_toolbox_official_grid_etrca" / "summary.csv",
]


def main() -> None:
    summary = load_summaries(SOURCES)
    plot_acc_itr(
        summary,
        OUT,
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
