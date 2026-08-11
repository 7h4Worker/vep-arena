from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent

DATASETS = {
    "EMBC9": "BS01_embc_9t",
    "JBHI16": "BS02_16t",
    "JBHI35 canonical": "BS03_35t",
}
FAMILIES = {
    "ETRCA": "TRCA",
    "SPECTRAL_TDCA": "TDCA",
    "EBPRCA": "bPRCA",
    "EFUSIONCA": "FusionCA",
}
COLORS = {"TRCA": "#2457A6", "TDCA": "#1B8A6B", "bPRCA": "#D1831F", "FusionCA": "#B23A48"}


def _load_complete_manifest(path: Path, expected_units: int | None = None) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or int(manifest.get("error_count", 0)) != 0:
        raise ValueError(f"Incomplete source manifest: {path}")
    if expected_units is not None and int(manifest.get("completed_units", -1)) != expected_units:
        raise ValueError(f"Unexpected unit count in {path}")
    return manifest


def _canonical_matrix() -> tuple[pd.DataFrame, list[str]]:
    tdca_root = TASK / "results" / "BS04_codebook" / "tdca_full"
    _load_complete_manifest(tdca_root / "manifest.json", 230)
    tdca = pd.read_csv(tdca_root / "summary.csv")
    frames: list[pd.DataFrame] = []
    sources = [str((tdca_root / "manifest.json").relative_to(PROJECT_ROOT))]
    tdca_dataset_names = {"EMBC9": "embc9", "JBHI16": "jbhi16", "JBHI35 canonical": "jbhi35"}

    for display_name, task_name in DATASETS.items():
        baseline_root = TASK / "results" / task_name / "full"
        manifest = _load_complete_manifest(baseline_root / "manifest.json")
        sources.append(str((baseline_root / "manifest.json").relative_to(PROJECT_ROOT)))
        baseline = pd.read_csv(baseline_root / "summary.csv")
        selected_baseline = baseline[baseline["method"].isin(("ETRCA", "EBPRCA", "EFUSIONCA"))].copy()
        selected_tdca = tdca[
            (tdca["dataset"] == tdca_dataset_names[display_name]) & (tdca["method"] == "SPECTRAL_TDCA")
        ].copy()
        keep = ["method", "window_seconds", "accuracy", "accuracy_sem", "subjects", "predictions"]
        if "predictions" not in selected_baseline:
            targets = int(manifest["targets"])
            blocks = int(manifest["blocks"])
            selected_baseline["predictions"] = selected_baseline["subjects"] * targets * blocks
        combined = pd.concat([selected_baseline[keep], selected_tdca[keep]], ignore_index=True)
        expected_methods = set(FAMILIES)
        if set(combined["method"].unique()) != expected_methods:
            raise ValueError(f"Missing receiver family for {display_name}")
        combined.insert(0, "dataset", display_name)
        frames.append(combined)
    matrix = pd.concat(frames, ignore_index=True)
    matrix.insert(2, "family", matrix["method"].map(FAMILIES))
    return matrix.sort_values(["dataset", "family", "window_seconds"]), sources


def _historical_five() -> tuple[pd.DataFrame, list[str]]:
    task_root = TASK / "results" / "BS03_35t"
    trca_root = task_root / "five_subject_trca_reproduction_20260729"
    tdca_root = task_root / "five_subject_tdca_full_20260729"
    periodic_root = task_root / "five_subject_periodic_receivers_full_20260729"
    trca_manifest = _load_complete_manifest(trca_root / "manifest.json", 20)
    if trca_manifest.get("reproduction_pass") is not True:
        raise ValueError("Historical five-subject eTRCA source did not pass MATLAB count matching.")
    _load_complete_manifest(tdca_root / "manifest.json", 50)
    _load_complete_manifest(periodic_root / "manifest.json", 200)

    trca_subject = pd.read_csv(trca_root / "subject.csv")
    exact = trca_subject[trca_subject["variant"] == "legacy_matlab_signed"]
    trca = pd.DataFrame(
        [
            {
                "method": "ETRCA",
                "window_seconds": 2.0,
                "accuracy": exact["accuracy"].mean(),
                "accuracy_sem": exact["accuracy"].std(ddof=1) / np.sqrt(len(exact)),
                "itr_bpm": exact["itr_bpm"].mean(),
                "subjects": exact["subject"].nunique(),
                "predictions": exact["prediction_count"].sum(),
                "evidence": "exact package MATLAB 2 s signed eTRCA count match",
            }
        ]
    )
    tdca = pd.read_csv(tdca_root / "summary.csv")
    tdca = tdca[tdca["method"] == "SPECTRAL_TDCA"].copy()
    tdca["evidence"] = "Arena extension; fixed whole-task TDCA parameters"
    periodic = pd.read_csv(periodic_root / "summary.csv")
    periodic = periodic[periodic["method"].isin(("EBPRCA", "EFUSIONCA"))].copy()
    periodic["evidence"] = "Arena extension; active 11-15 Hz periodic units"
    columns = ["method", "window_seconds", "accuracy", "accuracy_sem", "itr_bpm", "subjects", "predictions", "evidence"]
    combined = pd.concat([trca[columns], tdca[columns], periodic[columns]], ignore_index=True)
    combined.insert(0, "dataset", "JBHI35 historical five")
    combined.insert(2, "family", combined["method"].map(FAMILIES))
    return combined.sort_values(["family", "window_seconds"]), [
        str((trca_root / "manifest.json").relative_to(PROJECT_ROOT)),
        str((tdca_root / "manifest.json").relative_to(PROJECT_ROOT)),
        str((periodic_root / "manifest.json").relative_to(PROJECT_ROOT)),
    ]


def _plot_canonical(matrix: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.35), sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        subset = matrix[matrix["dataset"] == dataset]
        for family, group in subset.groupby("family", sort=False):
            group = group.sort_values("window_seconds")
            axis.errorbar(
                group["window_seconds"], group["accuracy"] * 100, yerr=group["accuracy_sem"] * 100,
                marker="o", markersize=4, linewidth=1.6, capsize=2, color=COLORS[family], label=family,
            )
        axis.set_title(dataset)
        axis.set_xlabel("Window (s)")
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_two_second(matrix: pd.DataFrame, historical: pd.DataFrame, output: Path) -> None:
    canonical = matrix[np.isclose(matrix["window_seconds"], 2.0)].copy()
    historical_2s = historical[np.isclose(historical["window_seconds"], 2.0)].copy()
    combined = pd.concat([canonical, historical_2s], ignore_index=True)
    datasets = list(DATASETS) + ["JBHI35 historical five"]
    families = list(COLORS)
    x = np.arange(len(datasets), dtype=float)
    width = 0.19
    fig, axis = plt.subplots(figsize=(10.2, 5.0), constrained_layout=True)
    for index, family in enumerate(families):
        values = []
        errors = []
        for dataset in datasets:
            row = combined[(combined["dataset"] == dataset) & (combined["family"] == family)]
            if len(row) != 1:
                raise ValueError(f"Expected one 2 s row for {dataset}/{family}")
            values.append(float(row.iloc[0]["accuracy"]) * 100)
            errors.append(float(row.iloc[0]["accuracy_sem"]) * 100)
        axis.bar(x + (index - 1.5) * width, values, width, yerr=errors, capsize=2, color=COLORS[family], label=family)
    axis.set_xticks(x, ["EMBC\n9 targets", "JBHI\n16 targets", "JBHI canonical\n35 targets", "JBHI historical\n35 targets"])
    axis.set(ylabel="Accuracy at 2 s (%)", ylim=(0, 100))
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncol=4, frameon=False, loc="upper center")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_historical(historical: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    for family, group in historical.groupby("family", sort=False):
        group = group.sort_values("window_seconds")
        kwargs = {"linestyle": "none", "markersize": 8} if family == "TRCA" else {"linestyle": "-", "markersize": 4}
        axis.errorbar(
            group["window_seconds"], group["accuracy"] * 100, yerr=group["accuracy_sem"] * 100,
            marker="o", linewidth=1.6, capsize=2, color=COLORS[family], label=family, **kwargs,
        )
    axis.set(xlabel="Window (s)", ylabel="Accuracy (%)", ylim=(0, 100), title="JBHI35 reviewed historical five-subject set")
    axis.grid(alpha=0.22)
    axis.legend(ncol=4, frameon=False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    output = TASK / "results" / "BS06_receiver" / "full_20260729"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    matrix, canonical_sources = _canonical_matrix()
    historical, historical_sources = _historical_five()
    matrix.to_csv(output / "canonical_matrix.csv", index=False)
    historical.to_csv(output / "historical_five_matrix.csv", index=False)
    two_second = pd.concat(
        [matrix[np.isclose(matrix["window_seconds"], 2.0)], historical[np.isclose(historical["window_seconds"], 2.0)]],
        ignore_index=True,
    )
    two_second.to_csv(output / "two_second_matrix.csv", index=False)
    _plot_canonical(matrix, figures / "canonical_accuracy_window.png")
    _plot_two_second(matrix, historical, figures / "two_second_accuracy.png")
    _plot_historical(historical, figures / "historical_five_accuracy_window.png")
    manifest = {
        "status": "complete",
        "evidence_role": "unified view of completed Arena evaluations; canonical and historical-five JBHI35 cohorts remain separate",
        "families": FAMILIES,
        "canonical_sources": canonical_sources,
        "historical_five_sources": historical_sources,
        "canonical_rows": len(matrix),
        "historical_five_rows": len(historical),
        "two_second_rows": len(two_second),
        "historical_trca_curve_available": False,
        "historical_trca_point": "2 s exact package MATLAB signed eTRCA count match only",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_executable": sys.executable,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "canonical_rows": len(matrix), "historical_five_rows": len(historical), "two_second_rows": len(two_second)}, indent=2))


if __name__ == "__main__":
    main()
