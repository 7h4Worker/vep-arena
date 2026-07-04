from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def has_confusion(path: Path) -> bool:
    if any(path.glob("confusion*.npy")):
        return True
    if (path / "confusions").exists() and any((path / "confusions").glob("confusion*.npy")):
        return True
    if (path / "figures").exists() and any((path / "figures").glob("confusion*.npy")):
        return True
    return False


def classify(path: Path) -> dict[str, object]:
    files = {p.name for p in path.glob("*") if p.is_file()}
    summary = "summary.csv" in files or any(path.glob("*summary*.csv"))
    trials = "trials.csv" in files
    predictions = "predictions.csv" in files or any(path.glob("*predictions.csv"))
    manifest = "manifest.json" in files or any(path.glob("*manifest.json"))
    confusion = has_confusion(path)
    runtime = "runtime.csv" in files
    block = "block.csv" in files
    subject = "subject.csv" in files
    manifest_status = ""
    for manifest_path in list(path.glob("*manifest.json")) + [path / "manifest.json"]:
        if manifest_path.exists():
            try:
                manifest_status = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("status", ""))
                if manifest_status:
                    break
            except Exception:
                pass

    if manifest_status == "partial":
        status = "partial"
    elif predictions and confusion and (summary or trials):
        status = "complete"
    elif summary and not predictions:
        status = "aggregate_only"
    elif predictions and not confusion:
        status = "missing_confusion"
    elif trials and not predictions:
        status = "missing_predictions"
    else:
        status = "partial_or_auxiliary"

    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "status": status,
        "summary": int(summary),
        "trials": int(trials),
        "predictions": int(predictions),
        "confusion": int(confusion),
        "manifest": int(manifest),
        "runtime": int(runtime),
        "subject": int(subject),
        "block": int(block),
    }


def candidate_dirs(root: Path) -> list[Path]:
    dirs = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if any(path.glob("*.csv")) or any(path.glob("*.json")) or any(path.glob("*.npy")):
            dirs.append(path)
    return sorted(set(dirs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "artifact_audit.csv")
    args = parser.parse_args()

    roots = [PROJECT_ROOT / "results", PROJECT_ROOT / "tasks"]
    rows = []
    for root in roots:
        if root.exists():
            rows.extend(classify(path) for path in candidate_dirs(root))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "status", "summary", "trials", "predictions", "confusion", "manifest", "runtime", "subject", "block"]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"wrote {args.out}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
