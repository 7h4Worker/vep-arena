from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import loadmat


OCCIPITAL9 = ("Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2")

JBHI35_HISTORICAL5_SUBJECT_IDS = ("S01", "S03", "S04", "S05", "S06")
JBHI35_HISTORICAL5_PACKAGE_NAME = "JBHI35_5Subject_Reproduction_Package_20260728"
JBHI35_HISTORICAL5_PACKAGE_CONFIG_KEY = "ssvep_jbhi_35target_five_subject_reproduction_private"
JBHI35_HISTORICAL5_SUBJECT_MANIFEST_CONFIG_KEY = "ssvep_jbhi_35target_five_subject_manifest_private"

EMBC9_TARGET_FREQUENCIES = (
    (0.0, 9.5),
    (8.5, 8.5),
    (9.5, 0.0),
    (9.5, 8.5),
    (0.0, 0.0),
    (8.5, 9.5),
    (8.5, 0.0),
    (9.5, 9.5),
    (0.0, 8.5),
)

JBHI16_TARGET_FREQUENCIES = (
    (0.0, 13.0),
    (11.0, 12.0),
    (12.0, 11.0),
    (13.0, 0.0),
    (12.0, 12.0),
    (13.0, 11.0),
    (0.0, 0.0),
    (11.0, 13.0),
    (11.0, 0.0),
    (0.0, 12.0),
    (12.0, 13.0),
    (0.0, 11.0),
    (13.0, 13.0),
    (12.0, 0.0),
    (11.0, 11.0),
    (13.0, 12.0),
)

JBHI35_HISTORICAL5_TARGET_FREQUENCIES = (
    (13.0, 15.0), (0.0, 14.0), (11.0, 15.0), (14.0, 12.0), (15.0, 11.0),
    (11.0, 12.0), (12.0, 12.0), (0.0, 13.0), (12.0, 14.0), (13.0, 0.0),
    (14.0, 13.0), (15.0, 14.0), (11.0, 11.0), (0.0, 15.0), (14.0, 11.0),
    (12.0, 11.0), (15.0, 12.0), (13.0, 13.0), (12.0, 0.0), (13.0, 11.0),
    (13.0, 14.0), (0.0, 11.0), (14.0, 0.0), (15.0, 15.0), (14.0, 15.0),
    (15.0, 0.0), (0.0, 12.0), (12.0, 13.0), (11.0, 0.0), (13.0, 12.0),
    (11.0, 13.0), (11.0, 14.0), (15.0, 13.0), (14.0, 14.0), (12.0, 15.0),
)


@dataclass(frozen=True)
class PrivateSSVEPSpec:
    dataset: str
    config_key: str
    targets: int
    blocks: int
    native_channels: int
    native_samples: int
    stored_sampling_rate: int
    analysis_sampling_rate: int
    stimulation_seconds: float
    cue_seconds: float
    stored_epoch_start_seconds: float
    itr_shift_seconds: float
    paper_role: str
    cv_protocol: str
    preprocessing_profile: dict[str, object]
    receiver_filterbank: dict[str, object]


@dataclass(frozen=True)
class PrivateSSVEPSubject:
    dataset: str
    subject_id: str
    x: np.ndarray
    sampling_rate: int
    channels: tuple[str, ...]
    targets: tuple[int, ...]
    blocks: tuple[int, ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class HistoricalJBHI35SubjectRecord:
    """Anonymous binding between one reviewed subject and one package MAT file."""

    subject_id: str
    relative_path: str
    expected_sha256: str
    matlab_signed_accuracy: float | None


@dataclass(frozen=True)
class HistoricalJBHI35Package:
    """Validated local view of the final five-subject historical package."""

    root: Path
    subject_manifest_path: Path
    package_sha_manifest_path: Path
    package_sha_manifest_sha256: str
    subject_manifest_sha256: str
    package_sha_entries_verified: int
    subjects: tuple[HistoricalJBHI35SubjectRecord, ...]


@dataclass(frozen=True)
class HistoricalJBHI35StaticValidation:
    """Safe, anonymous contract summary for the final historical package."""

    dataset: str
    subjects: tuple[str, ...]
    package_sha_entries_verified: int
    native_shape: tuple[int, int, int, int]
    canonical_shape: tuple[int, int, int, int]
    sampling_rate: int
    channels: tuple[str, ...]
    model_label_range: tuple[int, int]
    source_axis_order: str
    target_block_order: str
    matlab_reference_kind: str


SPECS = {
    "embc9": PrivateSSVEPSpec(
        dataset="embc9",
        config_key="ssvep_embc_9target_private",
        targets=9,
        blocks=20,
        native_channels=13,
        native_samples=4000,
        stored_sampling_rate=1000,
        analysis_sampling_rate=250,
        stimulation_seconds=4.0,
        cue_seconds=2.0,
        stored_epoch_start_seconds=0.13,
        itr_shift_seconds=2.13,
        paper_role="published",
        cv_protocol="subject-specific 20-fold leave-one-block-out; paper says leave-one-out and legacy MATLAB code verifies block-level folds",
        preprocessing_profile={
            "status": "paper-verified, transfer-note-verified, and legacy-code-verified",
            "reference": "Cz",
            "ground": "FPz",
            "epoch_seconds": [0.13, 4.13],
            "notch_hz": [[48.0, 52.0]],
            "eeglab_fir_bandpass_hz": [5.0, 100.0],
            "analysis_resample_hz": 250,
            "analysis_resample_method": "direct 4x decimation matching legacy MATLAB 1:4:end",
        },
        receiver_filterbank={
            "enabled": True,
            "kind": "dynamic Chebyshev Type I bandpass",
            "subbands_hz": [[6.0, 90.0], [14.0, 90.0], [22.0, 90.0]],
            "stopbands_hz": [[4.0, 100.0], [10.0, 100.0], [16.0, 100.0]],
            "passband_order_db": 3.0,
            "stopband_attenuation_db": 40.0,
            "ripple_db": 0.5,
            "provenance": "legacy MATLAB paper-result code; Arena applies bands independently and does not reproduce its training-only cascade",
        },
    ),
    "jbhi16": PrivateSSVEPSpec(
        dataset="jbhi16",
        config_key="ssvep_jbhi_16target_private",
        targets=16,
        blocks=6,
        native_channels=9,
        native_samples=4001,
        stored_sampling_rate=1000,
        analysis_sampling_rate=1000,
        stimulation_seconds=4.0,
        cue_seconds=2.0,
        stored_epoch_start_seconds=0.13,
        itr_shift_seconds=0.5,
        paper_role="final manuscript",
        cv_protocol="subject-specific six-fold leave-one-block-out",
        preprocessing_profile={
            "status": "manuscript-verified; one supplied EEGLAB history records additional operator steps and is not generalized to all subjects",
            "reference": "Cz",
            "ground": "FPz",
            "epoch_seconds": [0.13, 4.13],
            "notch_hz": [[48.0, 52.0], [98.0, 102.0]],
            "eeglab_fir_bandpass_hz": [4.0, 100.0],
            "analysis_resample_hz": None,
        },
        receiver_filterbank={
            "enabled": True,
            "kind": "Chebyshev Type I bandpass",
            "subbands_hz": [[6.0, 90.0], [14.0, 90.0], [22.0, 90.0], [30.0, 90.0], [38.0, 90.0]],
            "order": 6,
            "ripple_db": 0.5,
            "provenance": "manuscript specifies fixed order and passband rule; ripple and B=5 are explicit Arena smoke assumptions",
        },
    ),
    "jbhi35": PrivateSSVEPSpec(
        dataset="jbhi35",
        config_key="ssvep_jbhi_35target_private",
        targets=35,
        blocks=6,
        native_channels=9,
        native_samples=2000,
        stored_sampling_rate=1000,
        analysis_sampling_rate=1000,
        stimulation_seconds=2.0,
        cue_seconds=2.0,
        stored_epoch_start_seconds=0.13,
        itr_shift_seconds=0.5,
        paper_role="unpublished extension",
        cv_protocol="subject-specific six-fold leave-one-block-out diagnostic protocol",
        preprocessing_profile={
            "status": "operator-confirmed JBHI manual EEGLAB profile; no per-file history supplied",
            "reference": "Cz",
            "ground": "FPz",
            "epoch_seconds": [0.13, 2.13],
            "notch_hz": [[48.0, 52.0], [98.0, 102.0]],
            "eeglab_fir_bandpass_hz": [4.0, 100.0],
            "analysis_resample_hz": None,
        },
        receiver_filterbank={
            "enabled": True,
            "kind": "Chebyshev Type I bandpass",
            "subbands_hz": [[6.0, 90.0], [14.0, 90.0], [22.0, 90.0], [30.0, 90.0], [38.0, 90.0]],
            "order": 6,
            "ripple_db": 0.5,
            "provenance": "Arena diagnostic extension of the JBHI receiver; not a paper-verified 35-target algorithm",
        },
    ),
    "jbhi35_historical5": PrivateSSVEPSpec(
        dataset="jbhi35_historical5",
        config_key=JBHI35_HISTORICAL5_PACKAGE_CONFIG_KEY,
        targets=35,
        blocks=6,
        native_channels=9,
        native_samples=2000,
        stored_sampling_rate=1000,
        analysis_sampling_rate=1000,
        stimulation_seconds=2.0,
        cue_seconds=2.0,
        stored_epoch_start_seconds=0.13,
        itr_shift_seconds=0.5,
        paper_role="historical-reproduction package; Arena receiver results are extensions",
        cv_protocol="subject-specific six-fold leave-one-block-out",
        preprocessing_profile={
            "status": "package-validated historical input; no per-file acquisition identity is exposed",
            "reference": "Cz",
            "ground": "FPz",
            "epoch_seconds": [0.13, 2.13],
            "notch_hz": [[48.0, 52.0], [98.0, 102.0]],
            "eeglab_fir_bandpass_hz": [4.0, 100.0],
            "analysis_resample_hz": None,
        },
        receiver_filterbank={
            "enabled": True,
            "kind": "Chebyshev Type I bandpass",
            "subbands_hz": [[6.0, 90.0], [14.0, 90.0], [22.0, 90.0], [30.0, 90.0], [38.0, 90.0]],
            "order": 6,
            "ripple_db": 0.5,
            "provenance": "Arena extension filter bank; distinct from the package MATLAB reference implementation",
        },
    ),
}


def dataset_spec(dataset: str) -> PrivateSSVEPSpec:
    try:
        return SPECS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown private SSVEP dataset: {dataset}") from exc


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "datasets" / "local_paths.json"


def _configured_dataset_paths(config_path: Path | str | None = None) -> dict[str, str]:
    path = Path(config_path) if config_path is not None else _default_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset path config is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError(f"Dataset path config has a non-object datasets field: {path}")
    return {str(key): str(value) for key, value in datasets.items()}


def resolve_dataset_root(
    dataset: str,
    root: Path | str | None = None,
    config_path: Path | str | None = None,
) -> Path:
    if root is not None:
        resolved = Path(root)
    else:
        spec = dataset_spec(dataset)
        configured = _configured_dataset_paths(config_path).get(spec.config_key)
        if not configured:
            raise KeyError(f"Dataset key {spec.config_key!r} is missing from the local path config.")
        resolved = Path(str(configured))
    if not resolved.is_dir():
        raise FileNotFoundError(f"Configured {dataset} dataset directory does not exist: {resolved}")
    return resolved


def _mat_files(dataset: str, root: Path) -> tuple[Path, ...]:
    if dataset == "jbhi35":
        files = tuple(sorted(root.glob("S[0-9][0-9].mat"), key=lambda path: path.name.lower()))
        expected_names = tuple(f"S{index:02d}.mat" for index in range(1, len(files) + 1))
        if tuple(path.name for path in files) != expected_names:
            raise ValueError(
                "JBHI35 canonical files must use a contiguous anonymous S01.mat-SNN.mat sequence."
            )
    else:
        files = tuple(sorted(root.glob("*.mat"), key=lambda path: path.name.lower()))
    if not files:
        raise FileNotFoundError(f"No MAT files found for {dataset} in the configured directory.")
    return files


def available_subject_ids(
    dataset: str,
    root: Path | str | None = None,
    config_path: Path | str | None = None,
) -> tuple[str, ...]:
    if dataset == "jbhi35_historical5":
        return available_jbhi35_historical5_subject_ids(package_root=root, config_path=config_path)
    resolved = resolve_dataset_root(dataset, root, config_path)
    return tuple(f"S{index:02d}" for index, _ in enumerate(_mat_files(dataset, resolved), start=1))


def _subject_path(dataset: str, subject_id: str, root: Path) -> Path:
    normalized = subject_id.strip().upper()
    if not normalized.startswith("S") or not normalized[1:].isdigit():
        raise ValueError(f"Subject must use an anonymous SNN id, got {subject_id!r}.")
    index = int(normalized[1:])
    files = _mat_files(dataset, root)
    if index < 1 or index > len(files):
        raise ValueError(f"Anonymous subject {normalized} is outside the available range S01-S{len(files):02d}.")
    return files[index - 1]


def _validate_common(x: np.ndarray, spec: PrivateSSVEPSpec, subject_id: str) -> np.ndarray:
    expected = (spec.targets, spec.blocks, 9, spec.native_samples)
    if x.shape != expected:
        raise ValueError(f"{spec.dataset} {subject_id} has shape {x.shape}; expected {expected}.")
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError(f"{spec.dataset} {subject_id} data is not numeric: {x.dtype}.")
    if not np.isfinite(x).all():
        raise ValueError(f"{spec.dataset} {subject_id} contains NaN or Inf.")
    return np.asarray(x, dtype=np.float64)


def _matlab_strings(values: np.ndarray) -> tuple[str, ...]:
    strings: list[str] = []
    for value in np.asarray(values).reshape(-1):
        parts = np.asarray(value).reshape(-1)
        strings.append("".join(str(part) for part in parts).strip())
    return tuple(strings)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_package_relative_path(value: str) -> str:
    normalized = str(value).strip().replace("\\", "/")
    parts = normalized.split("/")
    if (
        not normalized
        or normalized.startswith("/")
        or ":" in parts[0]
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("Invalid package-relative path.")
    return "/".join(parts)


def _package_file(package_root: Path, relative_path: str) -> Path:
    normalized = _normalized_package_relative_path(relative_path)
    root = package_root.resolve()
    path = (root / Path(*normalized.split("/"))).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Package-relative path escapes the package root.") from exc
    return path


def _verify_jbhi35_historical_package_sha_manifest(package_root: Path) -> tuple[Path, dict[str, str]]:
    manifest_path = package_root / "manifests" / "PACKAGE_SHA256.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("Final JBHI35 package SHA manifest is missing.")
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"relative_path", "size_bytes", "sha256"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError("Final JBHI35 package SHA manifest has unexpected columns.")
        rows = list(reader)
    if len(rows) != 51:
        raise ValueError(f"Final JBHI35 package SHA manifest has {len(rows)} entries; expected 51.")

    hashes: dict[str, str] = {}
    for row in rows:
        relative_path = _normalized_package_relative_path(row["relative_path"])
        if relative_path in hashes:
            raise ValueError("Final JBHI35 package SHA manifest has duplicate relative paths.")
        expected_hash = str(row["sha256"]).strip().lower()
        if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
            raise ValueError("Final JBHI35 package SHA manifest has an invalid SHA-256 value.")
        try:
            expected_size = int(row["size_bytes"])
        except ValueError as exc:
            raise ValueError("Final JBHI35 package SHA manifest has an invalid file size.") from exc
        path = _package_file(package_root, relative_path)
        if not path.is_file():
            raise FileNotFoundError("A file bound by the final JBHI35 package SHA manifest is missing.")
        if path.stat().st_size != expected_size:
            raise ValueError("A file bound by the final JBHI35 package SHA manifest has an unexpected size.")
        if _sha256(path) != expected_hash:
            raise ValueError("A file bound by the final JBHI35 package SHA manifest has an unexpected SHA-256.")
        hashes[relative_path] = expected_hash
    return manifest_path, hashes


def _read_jbhi35_historical_subject_records(subject_manifest_path: Path) -> tuple[HistoricalJBHI35SubjectRecord, ...]:
    if not subject_manifest_path.is_file():
        raise FileNotFoundError("Final JBHI35 private subject manifest is missing.")
    with subject_manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"subject_id", "relative_path", "expected_sha256", "matlab_signed_accuracy"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError("Final JBHI35 private subject manifest has unexpected columns.")
        rows = list(reader)
    if len(rows) != len(JBHI35_HISTORICAL5_SUBJECT_IDS):
        raise ValueError("Final JBHI35 private subject manifest must contain exactly five subjects.")

    records: list[HistoricalJBHI35SubjectRecord] = []
    for row in rows:
        subject_id = str(row["subject_id"]).strip().upper()
        relative_path = _normalized_package_relative_path(row["relative_path"])
        expected_hash = str(row["expected_sha256"]).strip().lower()
        if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
            raise ValueError("Final JBHI35 private subject manifest has an invalid SHA-256 value.")
        try:
            matlab_signed_accuracy = float(row["matlab_signed_accuracy"])
        except ValueError as exc:
            raise ValueError("Final JBHI35 private subject manifest has an invalid MATLAB reference accuracy.") from exc
        if not 0.0 <= matlab_signed_accuracy <= 1.0:
            raise ValueError("Final JBHI35 private subject manifest has an out-of-range MATLAB reference accuracy.")
        records.append(
            HistoricalJBHI35SubjectRecord(
                subject_id=subject_id,
                relative_path=relative_path,
                expected_sha256=expected_hash,
                matlab_signed_accuracy=matlab_signed_accuracy,
            )
        )
    if tuple(record.subject_id for record in records) != JBHI35_HISTORICAL5_SUBJECT_IDS:
        raise ValueError("Final JBHI35 private subject manifest has an unexpected anonymous subject sequence.")
    return tuple(records)


def _read_jbhi35_historical_mapping_paths(package_root: Path) -> set[str]:
    mapping_path = package_root / "manifests" / "data_mapping.csv"
    if not mapping_path.is_file():
        raise FileNotFoundError("Final JBHI35 package data mapping is missing.")
    with mapping_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "output_mat" not in reader.fieldnames:
            raise ValueError("Final JBHI35 package data mapping has unexpected columns.")
        rows = list(reader)
    mapped_paths: set[str] = set()
    for row in rows:
        output_mat = _normalized_package_relative_path(row["output_mat"])
        relative_path = output_mat if output_mat.startswith("data/") else f"data/{output_mat}"
        mapped_paths.add(relative_path)
    if len(mapped_paths) != len(JBHI35_HISTORICAL5_SUBJECT_IDS):
        raise ValueError("Final JBHI35 package data mapping must bind exactly five output MAT files.")
    return mapped_paths


def resolve_jbhi35_historical5_package(
    package_root: Path | str | None = None,
    subject_manifest_path: Path | str | None = None,
    config_path: Path | str | None = None,
) -> HistoricalJBHI35Package:
    """Resolve and integrity-check the final historical package through local paths only."""

    configured_paths = _configured_dataset_paths(config_path)
    configured_root = configured_paths.get(JBHI35_HISTORICAL5_PACKAGE_CONFIG_KEY)
    if package_root is None and not configured_root:
        raise KeyError(f"Dataset key {JBHI35_HISTORICAL5_PACKAGE_CONFIG_KEY!r} is missing from the local path config.")
    root = Path(package_root if package_root is not None else configured_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Configured final JBHI35 historical package directory does not exist.")
    if root.name != JBHI35_HISTORICAL5_PACKAGE_NAME:
        raise ValueError("Only the final JBHI35 five-subject package is accepted; the obsolete handoff is rejected.")

    configured_manifest = configured_paths.get(JBHI35_HISTORICAL5_SUBJECT_MANIFEST_CONFIG_KEY)
    if subject_manifest_path is None and not configured_manifest:
        raise KeyError(
            f"Dataset key {JBHI35_HISTORICAL5_SUBJECT_MANIFEST_CONFIG_KEY!r} is missing from the local path config."
        )
    manifest = Path(subject_manifest_path if subject_manifest_path is not None else configured_manifest).resolve()
    if not manifest.is_file():
        raise FileNotFoundError("Configured final JBHI35 private subject manifest does not exist.")

    package_sha_manifest_path, package_hashes = _verify_jbhi35_historical_package_sha_manifest(root)
    subject_records = _read_jbhi35_historical_subject_records(manifest)
    mapped_paths = _read_jbhi35_historical_mapping_paths(root)
    subject_paths = {record.relative_path for record in subject_records}
    if subject_paths != mapped_paths:
        raise ValueError("Final JBHI35 private subject manifest and package data mapping disagree.")
    for record in subject_records:
        _package_file(root, record.relative_path)
        if package_hashes.get(record.relative_path) != record.expected_sha256:
            raise ValueError("Final JBHI35 private subject manifest SHA binding disagrees with the package SHA manifest.")

    return HistoricalJBHI35Package(
        root=root,
        subject_manifest_path=manifest,
        package_sha_manifest_path=package_sha_manifest_path,
        package_sha_manifest_sha256=_sha256(package_sha_manifest_path),
        subject_manifest_sha256=_sha256(manifest),
        package_sha_entries_verified=len(package_hashes),
        subjects=subject_records,
    )


def available_jbhi35_historical5_subject_ids(
    package: HistoricalJBHI35Package | None = None,
    package_root: Path | str | None = None,
    subject_manifest_path: Path | str | None = None,
    config_path: Path | str | None = None,
) -> tuple[str, ...]:
    resolved = package or resolve_jbhi35_historical5_package(
        package_root=package_root,
        subject_manifest_path=subject_manifest_path,
        config_path=config_path,
    )
    return tuple(record.subject_id for record in resolved.subjects)


def _jbhi35_historical_subject_record(
    package: HistoricalJBHI35Package,
    subject_id: str,
) -> HistoricalJBHI35SubjectRecord:
    normalized = subject_id.strip().upper()
    for record in package.subjects:
        if record.subject_id == normalized:
            return record
    raise ValueError(f"Anonymous historical subject {normalized!r} is not present in the final five-subject package.")


def load_jbhi35_historical5_subject(
    package: HistoricalJBHI35Package,
    subject_id: str,
) -> PrivateSSVEPSubject:
    """Load one SHA-bound anonymous subject as target x block x channel x sample."""

    record = _jbhi35_historical_subject_record(package, subject_id)
    path = _package_file(package.root, record.relative_path)
    if _sha256(path) != record.expected_sha256:
        raise ValueError("Final JBHI35 historical subject file no longer matches its verified SHA-256 binding.")
    payload = loadmat(
        path,
        variable_names=[
            "data",
            "Fs",
            "num_channels",
            "num_samples",
            "num_targets",
            "num_blocks",
            "channel_list",
            "target_labels",
        ],
    )
    required_fields = {
        "data",
        "Fs",
        "num_channels",
        "num_samples",
        "num_targets",
        "num_blocks",
        "channel_list",
        "target_labels",
    }
    missing_fields = sorted(required_fields.difference(payload))
    if missing_fields:
        raise ValueError("Final JBHI35 historical subject MAT file is missing required schema fields.")
    data = np.asarray(payload["data"])
    if data.shape != (9, 2000, 35, 6):
        raise ValueError(f"{record.subject_id} historical input shape is {data.shape}; expected (9, 2000, 35, 6).")
    if data.dtype != np.float64:
        raise TypeError(f"{record.subject_id} historical input dtype is {data.dtype}; expected float64.")
    expected_counts = {"num_channels": 9, "num_samples": 2000, "num_targets": 35, "num_blocks": 6}
    for key, expected in expected_counts.items():
        if int(np.asarray(payload[key]).squeeze()) != expected:
            raise ValueError(f"{record.subject_id} has inconsistent {key}.")
    sampling_rate = int(np.asarray(payload["Fs"]).squeeze())
    if sampling_rate != 1000:
        raise ValueError(f"{record.subject_id} sampling rate is {sampling_rate}; expected 1000.")
    channels = tuple(value.upper() for value in _matlab_strings(payload["channel_list"]))
    if channels != tuple(value.upper() for value in OCCIPITAL9):
        raise ValueError(f"{record.subject_id} channel order does not match the required nine-channel montage.")
    target_labels = np.asarray(payload["target_labels"]).reshape(-1).astype(np.int64)
    if not np.array_equal(target_labels, np.arange(1, 36, dtype=np.int64)):
        raise ValueError(f"{record.subject_id} target_labels must be exactly MATLAB labels 1-35.")

    spec = dataset_spec("jbhi35_historical5")
    x = _validate_common(np.transpose(data, (2, 3, 0, 1)), spec, record.subject_id)
    return PrivateSSVEPSubject(
        dataset="jbhi35_historical5",
        subject_id=record.subject_id,
        x=x,
        sampling_rate=sampling_rate,
        channels=OCCIPITAL9,
        targets=tuple(range(spec.targets)),
        blocks=tuple(range(spec.blocks)),
        metadata={
            "native_shape": [9, 2000, 35, 6],
            "canonical_shape": list(x.shape),
            "source_axis_order": "channel x sample x target x block",
            "source_schema": "final_historical_package_4d_v1",
            "channel_metadata_status": "file-verified",
            "label_order_status": "MATLAB target axis 1-35 converted to Python labels 0-34 without reordering",
            "target_block_order_status": "source target and block axes preserved",
            "model_label_base": 0,
            "identity_metadata_exposed": False,
        },
    )


def validate_jbhi35_historical5_contract(
    package: HistoricalJBHI35Package | None = None,
    package_root: Path | str | None = None,
    subject_manifest_path: Path | str | None = None,
    config_path: Path | str | None = None,
) -> HistoricalJBHI35StaticValidation:
    """Read and validate every final-five input without running a receiver.

    The return value deliberately contains only the anonymous, schema-level
    contract needed by Arena callers. It does not expose filenames, source
    identities, trial predictions, or any experimental result.
    """

    resolved_package = package or resolve_jbhi35_historical5_package(
        package_root=package_root,
        subject_manifest_path=subject_manifest_path,
        config_path=config_path,
    )
    expected_native_shape = (9, 2000, 35, 6)
    expected_canonical_shape = (35, 6, 9, 2000)
    expected_targets = tuple(range(35))
    expected_blocks = tuple(range(6))
    for record in resolved_package.subjects:
        subject = load_jbhi35_historical5_subject(resolved_package, record.subject_id)
        if subject.x.shape != expected_canonical_shape:
            raise ValueError("Final JBHI35 historical subject has an unexpected canonical shape.")
        if subject.sampling_rate != 1000 or subject.channels != OCCIPITAL9:
            raise ValueError("Final JBHI35 historical subject violates the shared sampling or channel contract.")
        if subject.targets != expected_targets or subject.blocks != expected_blocks:
            raise ValueError("Final JBHI35 historical subject violates the zero-based target or block contract.")

    return HistoricalJBHI35StaticValidation(
        dataset="jbhi35_historical5",
        subjects=tuple(record.subject_id for record in resolved_package.subjects),
        package_sha_entries_verified=resolved_package.package_sha_entries_verified,
        native_shape=expected_native_shape,
        canonical_shape=expected_canonical_shape,
        sampling_rate=1000,
        channels=OCCIPITAL9,
        model_label_range=(0, 34),
        source_axis_order="channel x sample x target x block",
        target_block_order="source target and block axes preserved",
        matlab_reference_kind="aggregate_correct_counts_only",
    )


def load_jbhi35_target_frequencies(root: Path | str | None = None) -> tuple[tuple[float, float], ...]:
    resolved = resolve_dataset_root("jbhi35", root)
    mapping_path = resolved / "target_mapping_35.csv"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"JBHI35 target mapping is missing: {mapping_path}")
    with mapping_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected_labels = list(range(1, 36))
    labels = [int(row["target_label_1based"]) for row in rows]
    if labels != expected_labels:
        raise ValueError("JBHI35 target mapping labels must be ordered 1-35.")
    zero_based_labels = [int(row["target_label_0based"]) for row in rows]
    if zero_based_labels != list(range(35)):
        raise ValueError("JBHI35 target mapping Python labels must be ordered 0-34.")
    frequencies: list[tuple[float, float]] = []
    for row in rows:
        pair = (float(row["left_freq_hz"]), float(row["right_freq_hz"]))
        if not any(frequency > 0 for frequency in pair):
            raise ValueError(f"JBHI35 target {row['target_label_1based']} has no active frequency.")
        frequencies.append(pair)
    if len(set(frequencies)) != 35:
        raise ValueError("JBHI35 left/right target frequency pairs must be unique.")
    return tuple(frequencies)


def load_target_frequency_pairs(
    dataset: str,
    root: Path | str | None = None,
) -> tuple[tuple[float, float], ...]:
    if dataset == "embc9":
        return EMBC9_TARGET_FREQUENCIES
    if dataset == "jbhi16":
        return JBHI16_TARGET_FREQUENCIES
    if dataset == "jbhi35":
        return load_jbhi35_target_frequencies(root)
    if dataset == "jbhi35_historical5":
        return JBHI35_HISTORICAL5_TARGET_FREQUENCIES
    raise ValueError(f"Unknown private SSVEP dataset: {dataset}")


def load_subject(
    dataset: str,
    subject_id: str,
    root: Path | str | None = None,
    config_path: Path | str | None = None,
) -> PrivateSSVEPSubject:
    if dataset == "jbhi35_historical5":
        package = resolve_jbhi35_historical5_package(package_root=root, config_path=config_path)
        return load_jbhi35_historical5_subject(package, subject_id)
    spec = dataset_spec(dataset)
    resolved = resolve_dataset_root(dataset, root, config_path)
    path = _subject_path(dataset, subject_id, resolved)
    normalized = subject_id.strip().upper()

    if dataset == "embc9":
        payload = loadmat(path, variable_names=["data", "Fs", "num_targets"])
        data = np.asarray(payload["data"])
        native_expected = (13, 4000, 9, 20)
        if data.shape != native_expected:
            raise ValueError(f"embc9 {normalized} has native shape {data.shape}; expected {native_expected}.")
        fs = int(np.asarray(payload["Fs"]).squeeze())
        x = np.transpose(data[-9:, :, :, :], (2, 3, 0, 1))
        native_shape = list(data.shape)
        channel_status = "inferred from supplied preprocessing documentation; source MAT omits channel_list"
        label_status = "target axis 1-9 and block axis 1-20; codebook mapping is not embedded"
    elif dataset == "jbhi16":
        payload = loadmat(path, variable_names=["eegdata", "sample_rate", "channel_list", "label_list"])
        eeg = np.asarray(payload["eegdata"])
        labels = np.asarray(payload["label_list"]).reshape(-1).astype(np.int64)
        if eeg.shape != (9, 4001, 96):
            raise ValueError(f"jbhi16 {normalized} has native shape {eeg.shape}; expected (9, 4001, 96).")
        expected_labels = np.tile(np.arange(1, 17, dtype=np.int64), 6)
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f"jbhi16 {normalized} labels are not six ordered 1-16 blocks.")
        fs = int(np.asarray(payload["sample_rate"]).squeeze())
        grouped = np.stack([np.transpose(eeg[:, :, labels == target], (2, 0, 1)) for target in range(1, 17)])
        x = grouped
        native_shape = list(eeg.shape)
        channel_status = "file-verified"
        label_status = "file-verified repeated 1-16 sequence for six blocks"
    else:
        payload = loadmat(
            path,
            variable_names=[
                "data",
                "Fs",
                "num_channels",
                "num_samples",
                "num_targets",
                "num_blocks",
                "eegdata",
                "sample_rate",
                "channel_list",
                "label_list",
            ],
        )
        if "data" in payload:
            data = np.asarray(payload["data"])
            if data.shape != (9, 2000, 35, 6):
                raise ValueError(
                    f"jbhi35 {normalized} canonical shape is {data.shape}; expected (9, 2000, 35, 6)."
                )
            expected_counts = {
                "num_channels": 9,
                "num_samples": 2000,
                "num_targets": 35,
                "num_blocks": 6,
            }
            for key, expected in expected_counts.items():
                if key in payload and int(np.asarray(payload[key]).squeeze()) != expected:
                    raise ValueError(f"jbhi35 {normalized} {key} does not match canonical data shape.")
            x = np.transpose(data, (2, 3, 0, 1))
            fs = int(np.asarray(payload["Fs"]).squeeze())
            native_shape = list(data.shape)
            channel_status = "canonical lineage-verified occipital montage"
            label_status = "canonical target axis converted from MATLAB 1-35 to Python 0-34"
            source_schema = "canonical_4d_v1"
        elif "eegdata" in payload:
            eeg = np.asarray(payload["eegdata"])
            if eeg.ndim != 3 or eeg.shape[0] != 9 or eeg.shape[1] not in (2000, 2001) or eeg.shape[2] != 210:
                raise ValueError(
                    f"jbhi35 {normalized} has native shape {eeg.shape}; expected (9, 2000/2001, 210)."
                )
            labels = np.asarray(payload["label_list"]).reshape(-1).astype(np.int64)
            if labels.shape != (210,):
                raise ValueError(f"jbhi35 {normalized} has {labels.size} labels; expected 210.")
            blocks: list[np.ndarray] = []
            expected_targets = np.arange(1, 36, dtype=np.int64)
            for block_index in range(6):
                block_slice = slice(block_index * 35, (block_index + 1) * 35)
                block_labels = labels[block_slice]
                if not np.array_equal(np.sort(block_labels), expected_targets):
                    raise ValueError(
                        f"jbhi35 {normalized} block {block_index + 1} is not a permutation of labels 1-35."
                    )
                block_eeg = eeg[:, :2000, block_slice]
                order = np.argsort(block_labels)
                blocks.append(np.transpose(block_eeg[:, :, order], (2, 0, 1)))
            x = np.stack(blocks, axis=1)
            fs = int(np.asarray(payload["sample_rate"]).squeeze())
            channels = tuple(name.upper() for name in _matlab_strings(payload["channel_list"]))
            if channels != tuple(name.upper() for name in OCCIPITAL9):
                raise ValueError(f"jbhi35 {normalized} channel order does not match the required occipital montage.")
            native_shape = list(eeg.shape)
            channel_status = "file-verified"
            label_status = "file-verified; each consecutive 35-trial block contains labels 1-35 exactly once"
            source_schema = "labeled_session_3d"
        else:
            raise ValueError(f"jbhi35 {normalized} contains neither canonical data nor labeled eegdata.")
        target_frequencies = load_jbhi35_target_frequencies(resolved)

    if fs != spec.stored_sampling_rate:
        raise ValueError(f"{dataset} {normalized} has sampling rate {fs}; expected {spec.stored_sampling_rate}.")
    x = _validate_common(x, spec, normalized)
    return PrivateSSVEPSubject(
        dataset=dataset,
        subject_id=normalized,
        x=x,
        sampling_rate=fs,
        channels=OCCIPITAL9,
        targets=tuple(range(spec.targets)),
        blocks=tuple(range(spec.blocks)),
        metadata={
            "native_shape": native_shape,
            "canonical_shape": list(x.shape),
            "channel_metadata_status": channel_status,
            "label_order_status": label_status,
            "target_codebook_status": "file-verified" if dataset == "jbhi35" else "not embedded in source MAT",
            "target_frequency_pair_count": len(target_frequencies) if dataset == "jbhi35" else None,
            "source_schema": source_schema if dataset == "jbhi35" else None,
            "model_label_base": 0,
            "identity_metadata_exposed": False,
        },
    )


@lru_cache(maxsize=None)
def _embc_filter_coefficients(sampling_rate: int, band_index: int) -> tuple[np.ndarray, np.ndarray]:
    passbands = (6.0, 14.0, 22.0)
    stopbands = (4.0, 10.0, 16.0)
    order, critical = signal.cheb1ord(
        [passbands[band_index], 90.0],
        [stopbands[band_index], 100.0],
        3.0,
        40.0,
        fs=sampling_rate,
    )
    return signal.cheby1(order, 0.5, critical, btype="bandpass", fs=sampling_rate)


def _embc_filterbank(x: np.ndarray, sampling_rate: int, n_bands: int) -> np.ndarray:
    if n_bands < 1 or n_bands > 3:
        raise ValueError("EMBC legacy-code receiver filter bank supports 1-3 bands.")
    bands = []
    for band_index in range(n_bands):
        numerator, denominator = _embc_filter_coefficients(sampling_rate, band_index)
        bands.append(signal.filtfilt(numerator, denominator, x, axis=-1))
    return np.stack(bands, axis=2)


def jbhi_receiver_filter_band(x: np.ndarray, sampling_rate: int, band_index: int) -> np.ndarray:
    """Apply one shared Arena JBHI receiver subband without changing tensor axes."""

    if band_index < 0 or band_index >= 5:
        raise ValueError("JBHI receiver filter-bank index must be in [0, 4].")
    low = 6.0 + 8.0 * band_index
    sos = signal.cheby1(6, 0.5, [low, 90.0], btype="bandpass", fs=sampling_rate, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)


def _jbhi_filterbank(x: np.ndarray, sampling_rate: int, n_bands: int) -> np.ndarray:
    if n_bands < 1 or n_bands > 5:
        raise ValueError("JBHI receiver filter bank supports 1-5 bands.")
    return np.stack(
        [jbhi_receiver_filter_band(x, sampling_rate, band_index) for band_index in range(n_bands)],
        axis=2,
    )


def analysis_epochs(
    subject: PrivateSSVEPSubject,
    window_seconds: float,
    n_bands: int | None = None,
) -> np.ndarray:
    spec = dataset_spec(subject.dataset)
    if window_seconds <= 0 or window_seconds > spec.stimulation_seconds:
        raise ValueError(
            f"Window {window_seconds:g}s is outside (0, {spec.stimulation_seconds:g}] for {subject.dataset}."
        )
    if subject.dataset == "embc9":
        selected_bands = 3 if n_bands is None else n_bands
        decimation = subject.sampling_rate // spec.analysis_sampling_rate
        if decimation * spec.analysis_sampling_rate != subject.sampling_rate:
            raise ValueError("EMBC stored and analysis sampling rates do not define an integer decimation.")
        resampled = subject.x[..., ::decimation]
        samples = int(round(window_seconds * spec.analysis_sampling_rate))
        return _embc_filterbank(resampled[..., :samples], spec.analysis_sampling_rate, selected_bands)
    bands = _jbhi_filterbank(subject.x, subject.sampling_rate, 5 if n_bands is None else n_bands)
    samples = int(round(window_seconds * spec.analysis_sampling_rate))
    return bands[..., :samples]
