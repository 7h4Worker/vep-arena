from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vep_arena.data.binocular import BINOCULAR_AR_ROOT


def md5_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_epoch_files(manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rows = []
    for item in manifest["epoch"]["files"]:
        name = str(item["name"])
        if name == "Description.zip" or name.startswith("sub-"):
            rows.append(item)
    return rows


def selected_files(files: Iterable[dict[str, object]], subjects: set[int] | None) -> list[dict[str, object]]:
    if subjects is None:
        return list(files)
    rows = []
    for item in files:
        name = str(item["name"])
        if name == "Description.zip":
            rows.append(item)
            continue
        try:
            subject = int(name.removeprefix("sub-").removesuffix(".zip"))
        except ValueError:
            continue
        if subject in subjects:
            rows.append(item)
    return rows


def parse_subjects(value: str | None) -> set[int] | None:
    if not value:
        return None
    subjects: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            subjects.update(range(lo, hi + 1))
        else:
            subjects.add(int(part))
    return subjects


def validate_and_promote_part(part: Path, target: Path, expected_size: int, expected_md5: str) -> dict[str, object] | None:
    if not part.exists() or part.stat().st_size != expected_size:
        return None
    if expected_md5:
        actual_md5 = md5_file(part)
        if actual_md5 != expected_md5:
            return {
                "name": target.name,
                "status": "md5_mismatch",
                "bytes": part.stat().st_size,
                "expected_md5": expected_md5,
                "actual_md5": actual_md5,
            }
    if target.exists():
        target.unlink()
    part.replace(target)
    return {"name": target.name, "status": "downloaded", "bytes": expected_size}


def download_one_attempt(item: dict[str, object], out_dir: Path, timeout: float) -> dict[str, object]:
    name = str(item["name"])
    expected_size = int(item["size"])
    expected_md5 = str(item.get("supplied_md5") or "")
    url = str(item["download_url"])
    target = out_dir / name
    part = target.with_suffix(target.suffix + ".part")

    if target.exists() and target.stat().st_size == expected_size:
        actual_md5 = md5_file(target) if expected_md5 else ""
        if not expected_md5 or actual_md5 == expected_md5:
            return {"name": name, "status": "complete", "bytes": expected_size}

    out_dir.mkdir(parents=True, exist_ok=True)
    promoted = validate_and_promote_part(part, target, expected_size, expected_md5)
    if promoted is not None:
        return promoted
    if part.exists() and part.stat().st_size > expected_size:
        part.unlink()

    downloaded = part.stat().st_size if part.exists() else 0
    headers = {}
    mode = "wb"
    if downloaded > 0 and downloaded < expected_size:
        headers["Range"] = f"bytes={downloaded}-"
        mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        if response.status_code == 200 and downloaded > 0:
            downloaded = 0
            mode = "wb"
        elif response.status_code not in {200, 206}:
            response.raise_for_status()
        with part.open(mode + "") as f:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)

    promoted = validate_and_promote_part(part, target, expected_size, expected_md5)
    if promoted is not None:
        return promoted

    actual_size = part.stat().st_size
    if actual_size != expected_size:
        return {
            "name": name,
            "status": "incomplete",
            "bytes": actual_size,
            "expected": expected_size,
        }

    return {"name": name, "status": "downloaded", "bytes": expected_size}


def download_one(item: dict[str, object], out_dir: Path, timeout: float, retries: int) -> dict[str, object]:
    name = str(item["name"])
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            result = download_one_attempt(item, out_dir, timeout)
            if result["status"] == "incomplete" and attempt < retries:
                time.sleep(min(20.0, 2.0 * attempt))
                continue
            return result
        except Exception as exc:  # noqa: BLE001 - keep batch downloads moving and report per-file failures.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(20.0, 2.0 * attempt))
                continue
    return {"name": name, "status": "failed", "error": last_error, "bytes": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download binocular AR epoch figshare files with size/md5 validation.")
    parser.add_argument("--root", type=Path, default=BINOCULAR_AR_ROOT)
    parser.add_argument("--subjects", help="Subject list, e.g. 1,3,8-10. Description.zip is always included.")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Limit selected file count for testing.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = args.root / "metadata" / "ssvep_binocular_ar_manifest.json"
    out_dir = args.root / "epoch"
    files = selected_files(load_epoch_files(manifest_path), parse_subjects(args.subjects))
    if args.limit > 0:
        files = files[: args.limit]

    pending = []
    for item in files:
        target = out_dir / str(item["name"])
        expected_size = int(item["size"])
        actual_size = target.stat().st_size if target.exists() else 0
        if actual_size == expected_size:
            continue
        pending.append(item)

    total_bytes = sum(int(item["size"]) for item in pending)
    print(f"selected={len(files)} pending={len(pending)} pending_gb={total_bytes / (1024**3):.2f}")
    for item in pending:
        print(f"pending {item['name']} {int(item['size']) / (1024**2):.1f} MiB")
    if args.dry_run or not pending:
        return

    workers = max(1, min(args.workers, len(pending)))
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, item, out_dir, args.timeout, args.retries) for item in pending]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {"name": "<unknown>", "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"{result['status']} {result['name']} {result.get('bytes', 0)}")
            if result["status"] not in {"complete", "downloaded"}:
                failures.append(result)

    if failures:
        print(json.dumps(failures, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
