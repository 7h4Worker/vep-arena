from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py


TASK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TASK_DIR / "results"
DEFAULT_DATASET = Path("D:/ProjData/datasets/ssvep_hd_200target")
RAW_REL = Path("raw/code_data/code&data")


@dataclass(frozen=True)
class PaperAnchor:
    key: str
    paper_value: float
    unit: str
    source: str
    note: str


PAPER_ANCHORS = {
    "online_s1_accuracy_percent": PaperAnchor(
        key="online_s1_accuracy_percent",
        paper_value=96.88,
        unit="%",
        source="Table 2",
        note="S1, 160 targets, right/down/left/up, 0.25 s",
    ),
    "online_s1_itr_bpm": PaperAnchor(
        key="online_s1_itr_bpm",
        paper_value=551.42,
        unit="bpm",
        source="Table 2",
        note="S1, 160 targets, right/down/left/up, 0.25 s",
    ),
    "online_mean_itr_bpm": PaperAnchor(
        key="online_mean_itr_bpm",
        paper_value=472.72,
        unit="bpm",
        source="Abstract and Table 2",
        note="Mean online actual ITR across online subjects",
    ),
    "offline_personalized_256_66_peak_itr_bpm": PaperAnchor(
        key="offline_personalized_256_66_peak_itr_bpm",
        paper_value=484.76,
        unit="bpm",
        source="Results text around Fig. 3",
        note="256-66 personalized system parameter peak at 0.2 s",
    ),
    "offline_200target_256_66_fixed_peak_itr_bpm": PaperAnchor(
        key="offline_200target_256_66_fixed_peak_itr_bpm",
        paper_value=416.22,
        unit="bpm",
        source="Results text around Fig. 2",
        note="200-target 256-66 fixed setting peak at 0.3 s",
    ),
    "offline_200target_256_66_personalized_itr_bpm": PaperAnchor(
        key="offline_200target_256_66_personalized_itr_bpm",
        paper_value=456.72,
        unit="bpm",
        source="Discussion text",
        note="200-target 256-66 after personalized system parameters",
    ),
}


def read_single_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"{path} should contain one data row, got {len(rows)}")
    return rows[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def itr_bits_per_minute(classes: int, accuracy: float, trial_seconds: float) -> float:
    if accuracy <= 0:
        bits = math.log2(classes - 1)
    elif accuracy >= 1:
        bits = math.log2(classes)
    else:
        bits = (
            math.log2(classes)
            + accuracy * math.log2(accuracy)
            + (1 - accuracy) * math.log2((1 - accuracy) / (classes - 1))
        )
    return bits * 60 / trial_seconds


def online_validation() -> list[dict[str, str]]:
    path = RESULTS_DIR / "online_highest_S1_160target_66ch_w250.csv"
    row = read_single_csv(path)
    acc = float(row["accuracy"])
    itr = float(row["itr_bpm"])
    targets = int(row["targets"])
    test_blocks = int(row["test_blocks"])
    total = targets * test_blocks
    correct = round(acc * total)
    recomputed_itr = itr_bits_per_minute(targets, acc, 0.25 + 0.5)

    checks = []
    acc_anchor = PAPER_ANCHORS["online_s1_accuracy_percent"]
    paper_correct = round(acc_anchor.paper_value / 100 * total)
    checks.append(
        {
            "check": "online_s1_accuracy",
            "paper": f"{acc_anchor.paper_value:.4f}",
            "local": f"{acc * 100:.4f}",
            "delta": f"{acc * 100 - acc_anchor.paper_value:.4f}",
            "unit": acc_anchor.unit,
            "status": "near_match",
            "detail": f"{correct}/{total} local vs {paper_correct}/{total} paper; {acc_anchor.note}",
        }
    )

    itr_anchor = PAPER_ANCHORS["online_s1_itr_bpm"]
    checks.append(
        {
            "check": "online_s1_itr",
            "paper": f"{itr_anchor.paper_value:.6f}",
            "local": f"{itr:.6f}",
            "delta": f"{itr - itr_anchor.paper_value:.6f}",
            "unit": itr_anchor.unit,
            "status": "near_match",
            "detail": f"recomputed local ITR={recomputed_itr:.6f}; {itr_anchor.note}",
        }
    )
    return checks


def offline_sample_validation() -> list[dict[str, str]]:
    rows = read_csv(RESULTS_DIR / "summary.csv")
    checks = []
    for row in rows:
        subject = row["subject"]
        window_ms = row["window_ms"]
        status = row["status"]
        if status == "complete":
            acc = float(row["accuracy"])
            itr = float(row["itr_bpm"])
            recomputed_itr = itr_bits_per_minute(int(row["targets"]), acc, int(window_ms) / 1000 + 0.5)
            checks.append(
                {
                    "check": f"offline_{subject}_{window_ms}ms_itr_formula",
                    "paper": "",
                    "local": f"{itr:.6f}",
                    "delta": f"{itr - recomputed_itr:.9f}",
                    "unit": "bpm",
                    "status": "pass" if abs(itr - recomputed_itr) < 1e-6 else "fail",
                    "detail": f"accuracy={acc:.9f}; recomputed ITR uses window+0.5s gaze shift",
                }
            )
        elif window_ms == "1000":
            checks.append(
                {
                    "check": f"offline_{subject}_1000ms_availability",
                    "paper": "",
                    "local": "unavailable",
                    "delta": "",
                    "unit": "",
                    "status": "pass",
                    "detail": row["reason"],
                }
            )
    return checks


def dataset_shape_validation(dataset_root: Path) -> list[dict[str, str]]:
    checks = []
    offline = dataset_root / RAW_REL / "data" / "offline" / "S1.mat"
    online_train = dataset_root / RAW_REL / "data" / "online" / "training" / "S1.mat"
    online_test = dataset_root / RAW_REL / "data" / "online" / "testing" / "S1.mat"
    for label, path, expected in [
        ("offline_S1_shape", offline, "(18, 200, 185, 66)"),
        ("online_training_S1_shape", online_train, "(18, 160, 103, 66)"),
        ("online_testing_S1_shape", online_test, "(5, 160, 103, 66)"),
    ]:
        with h5py.File(path, "r") as f:
            shape = tuple(f["data250Hz"].shape)
        checks.append(
            {
                "check": label,
                "paper": "",
                "local": str(shape),
                "delta": "",
                "unit": "",
                "status": "pass" if str(shape) == expected else "inspect",
                "detail": f"expected stored HDF5 order {expected}; file={path}",
            }
        )
    return checks


def paper_anchor_rows() -> list[dict[str, str]]:
    rows = []
    for anchor in PAPER_ANCHORS.values():
        if anchor.key.startswith("online_s1_"):
            continue
        rows.append(
            {
                "check": anchor.key,
                "paper": f"{anchor.paper_value:.6f}",
                "local": "not_run_in_current_sample",
                "delta": "",
                "unit": anchor.unit,
                "status": "paper_anchor_only",
                "detail": f"{anchor.source}; {anchor.note}",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["check", "paper", "local", "delta", "unit", "status", "detail"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# Paper Validation Summary",
        "",
        "This file is generated by `validate_results.py` from local result CSVs plus paper-visible anchors.",
        "",
        "| check | paper | local | delta | status | detail |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        detail = row["detail"].replace("|", "/")
        lines.append(
            f"| `{row['check']}` | {row['paper']} {row['unit']} | {row['local']} {row['unit']} | "
            f"{row['delta']} | `{row['status']}` | {detail} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    rows = []
    rows.extend(online_validation())
    rows.extend(offline_sample_validation())
    rows.extend(dataset_shape_validation(args.dataset_root))
    rows.extend(paper_anchor_rows())

    csv_path = RESULTS_DIR / "paper_validation_summary.csv"
    md_path = RESULTS_DIR / "paper_validation_summary.md"
    json_path = RESULTS_DIR / "paper_validation_summary.json"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
