from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import h5py
import numpy as np
import scipy.io as sio


LIANG2020_KEY = "ssvep_dual_frequency_phase_liang2020"
LIANG2020_ROOT = Path("D:/ProjData/datasets/ssvep_dual_frequency_phase_liang2020")
LIANG2020_SAMPLING_RATE = 1000
LIANG2020_CHANNELS = ("Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2")

# Liang et al. (2020), Figure 1(a), read row-major from targets 1-40.
# Each tuple is (frequency_1_hz, frequency_2_hz, shared_initial_phase_in_pi).
# The public MAT files contain target IDs but not this stimulus codebook.
LIANG2020_METHOD1_DUAL_FREQUENCY_PHASE_CODEBOOK = (
    (8.2, 9.0, 0.0), (9.0, 9.8, 1.0), (9.8, 11.4, 0.0), (10.6, 11.4, 1.0),
    (10.6, 15.4, 1.0), (11.4, 15.4, 0.0), (12.2, 16.2, 1.5), (13.8, 15.4, 1.5),
    (8.2, 9.8, 0.0), (9.0, 10.6, 0.0), (9.8, 13.8, 0.5), (10.6, 12.2, 1.0),
    (11.4, 12.2, 0.5), (12.2, 13.0, 0.0), (13.0, 13.8, 0.0), (13.8, 16.2, 1.0),
    (8.2, 13.0, 0.0), (9.0, 11.4, 0.5), (9.8, 14.6, 0.5), (10.6, 13.0, 0.5),
    (11.4, 13.0, 1.0), (12.2, 13.8, 0.0), (13.0, 14.6, 1.0), (14.6, 15.4, 1.5),
    (8.2, 15.4, 0.5), (9.0, 16.2, 0.0), (9.8, 15.4, 0.5), (10.6, 13.8, 1.5),
    (11.4, 13.8, 1.0), (12.2, 14.6, 1.0), (13.0, 16.2, 0.5), (14.6, 16.2, 0.0),
    (8.2, 16.2, 0.0), (9.8, 10.6, 1.0), (9.8, 16.2, 1.0), (10.6, 14.6, 1.5),
    (11.4, 14.6, 0.0), (12.2, 15.4, 1.0), (13.8, 14.6, 0.0), (15.4, 16.2, 0.0),
)

SUN2024_KEY = "ssvep_efficient_dual_frequency_sun2024"
SUN2024_ROOT = Path("D:/ProjData/datasets/ssvep_efficient_dual_frequency_sun2024")
SUN2024_SAMPLING_RATE = 1000
SUN2024_OCCIPITAL9 = ("Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2")
SUN2024_SPLITS = {
    "one_target": "Data of 1Target",
    "forty_targets": "Data of 40targets offline",
}

# Sun et al. (2024), Figure 2(III), row-major target order. The public
# forty-target spreadsheet contains the frequencies but omits these phases.
SUN2024_FORTY_TARGET_PHASES_PI = (
    (0.15, 0.24), (0.28, -0.27), (0.71, -0.21), (0.59, -0.72),
    (-0.21, 0.71), (0.24, 0.15), (0.17, -0.55), (0.87, -0.32),
    (0.31, 0.88), (-0.28, -0.24), (0.08, -0.96), (-0.15, -0.30),
    (0.23, 0.16), (0.42, 0.06), (-0.72, 0.59), (0.33, -0.21),
    (-0.32, 0.87), (0.07, -0.21), (-0.74, -0.32), (0.97, -0.27),
    (-0.21, 0.07), (0.73, 0.39), (-0.55, 0.17), (0.39, 0.73),
    (0.02, 0.61), (0.12, 0.31), (0.61, 0.02), (-0.96, 0.08),
    (-0.30, -0.15), (-0.27, 0.97), (0.40, -0.82), (-0.24, -0.28),
    (0.16, 0.23), (0.88, 0.31), (-0.82, 0.40), (-0.27, 0.28),
    (0.06, 0.42), (0.31, 0.12), (-0.21, 0.33), (-0.32, -0.74),
)

_XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass(frozen=True)
class LiangMatRecord:
    path: Path
    experiment: str
    batch: str
    subject: int
    condition: int
    condition_name: str
    run: int


@dataclass(frozen=True)
class LiangEpochFile:
    x: np.ndarray
    labels: np.ndarray
    engine: str

    @property
    def n_channels(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_trials(self) -> int:
        return int(self.x.shape[2])


@dataclass(frozen=True)
class SunCntRecord:
    path: Path
    split: str
    split_dir: str
    subject: int


@dataclass(frozen=True)
class SunTrialEvent:
    trial_index: int
    block: int
    trial_in_block: int
    target_id: int
    start_sample: int
    end_sample: int | None
    start_seconds: float
    end_seconds: float | None
    duration_seconds: float | None


@dataclass(frozen=True)
class SunTargetRepeatTrial:
    """A Sun2024 stimulus event indexed by occurrence within its target.

    The public CNT annotations do not provide a usable block boundary for the
    forty-target offline files. Repetition is therefore the only split unit
    inferred from the recorded trigger stream.
    """

    trial_index: int
    target_id: int
    repetition: int
    start_sample: int
    end_sample: int | None
    start_seconds: float
    end_seconds: float | None
    duration_seconds: float | None


def resolve_dataset_root(key: str, default: Path) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "configs" / "datasets" / "local_paths.json"
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        datasets = config.get("datasets", {})
        if key in datasets:
            return Path(str(datasets[key]))
    return default


def liang2020_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else resolve_dataset_root(LIANG2020_KEY, LIANG2020_ROOT)


def sun2024_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else resolve_dataset_root(SUN2024_KEY, SUN2024_ROOT)


def _natural_key(path: Path | str) -> tuple[object, ...]:
    text = str(path)
    parts = re.split(r"(\d+)", text)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def _parse_subject(name: str) -> int:
    match = re.search(r"sub(?:ject)?\s*0*(\d+)", name, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"0*(\d+)", name)
    if not match:
        raise ValueError(f"Could not parse subject from {name!r}")
    return int(match.group(1))


def _parse_condition(name: str) -> int:
    match = re.search(r"condition\s*0*(\d+)", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse condition from {name!r}")
    return int(match.group(1))


def iter_liang_mat_files(
    root: Path | str | None = None,
    experiments: Iterable[str] | None = None,
    subjects: Iterable[int] | None = None,
) -> list[LiangMatRecord]:
    base = liang2020_root(root) / "extracted"
    wanted_experiments = {item.lower() for item in experiments} if experiments is not None else None
    wanted_subjects = {int(item) for item in subjects} if subjects is not None else None
    records: list[LiangMatRecord] = []
    for path in sorted(base.rglob("*.mat"), key=_natural_key):
        if "__MACOSX" in path.parts:
            continue
        rel = path.relative_to(base).parts
        if len(rel) < 4:
            continue
        batch, subject_name, condition_name, file_name = rel[:4]
        experiment = batch.split("-", 1)[0]
        subject = _parse_subject(subject_name)
        condition = _parse_condition(condition_name)
        run = int(Path(file_name).stem)
        if wanted_experiments is not None and experiment.lower() not in wanted_experiments:
            continue
        if wanted_subjects is not None and subject not in wanted_subjects:
            continue
        records.append(
            LiangMatRecord(
                path=path,
                experiment=experiment,
                batch=batch,
                subject=subject,
                condition=condition,
                condition_name=condition_name,
                run=run,
            )
        )
    return records


def load_liang_mat(path: Path | str) -> LiangEpochFile:
    path = Path(path)
    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        x = np.asarray(data["trialDataSet"], dtype=np.float64)
        labels = np.ravel(np.asarray(data["successTrialTypeId"])).astype(np.int64)
        engine = "scipy"
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            x = np.asarray(handle["trialDataSet"], dtype=np.float64)
            labels = np.ravel(np.asarray(handle["successTrialTypeId"])).astype(np.int64)
        engine = "h5py"

    if x.ndim != 3:
        raise ValueError(f"Expected 3D trialDataSet in {path}, got shape {x.shape}")
    if x.shape[0] == len(LIANG2020_CHANNELS):
        canonical = x
    elif x.shape[-1] == len(LIANG2020_CHANNELS):
        canonical = np.transpose(x, (2, 1, 0))
    else:
        raise ValueError(f"Could not orient Liang2020 data with shape {x.shape}")
    if canonical.shape[2] != labels.size:
        raise ValueError(f"Label count {labels.size} does not match trials {canonical.shape[2]} in {path}")
    return LiangEpochFile(x=canonical, labels=labels, engine=engine)


def _xlsx_col_row(ref: str) -> tuple[int, int]:
    col_text = "".join(ch for ch in ref if ch.isalpha())
    row_text = "".join(ch for ch in ref if ch.isdigit())
    col = 0
    for ch in col_text:
        col = col * 26 + ord(ch.upper()) - ord("A") + 1
    return int(row_text), col


def read_xlsx_first_sheet(path: Path | str) -> list[list[str]]:
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", _XLSX_NS):
                shared.append("".join(text.text or "" for text in item.findall(".//a:t", _XLSX_NS)))

        sheet_names = sorted(
            name for name in zf.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not sheet_names:
            raise ValueError(f"No worksheet found in {path}")
        sheet_root = ET.fromstring(zf.read(sheet_names[0]))
        cells: dict[tuple[int, int], str] = {}
        max_row = 0
        max_col = 0
        for cell in sheet_root.findall(".//a:c", _XLSX_NS):
            ref = cell.attrib.get("r")
            if ref is None:
                continue
            row, col = _xlsx_col_row(ref)
            max_row = max(max_row, row)
            max_col = max(max_col, col)
            value_node = cell.find("a:v", _XLSX_NS)
            value = "" if value_node is None or value_node.text is None else value_node.text
            if cell.attrib.get("t") == "s" and value:
                value = shared[int(value)]
            elif cell.attrib.get("t") == "inlineStr":
                value = "".join(text.text or "" for text in cell.findall(".//a:t", _XLSX_NS))
            cells[(row, col)] = value
    return [[cells.get((row, col), "") for col in range(1, max_col + 1)] for row in range(1, max_row + 1)]


def _to_int(value: str) -> int | None:
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def _to_float(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_sun_codebook(root: Path | str | None = None, kind: str = "forty_targets") -> list[dict[str, object]]:
    base = sun2024_root(root) / "extracted"
    file_name = "code of 1Target.xlsx" if kind == "one_target" else "code of 40Targets.xlsx"
    rows = read_xlsx_first_sheet(base / file_name)
    out: list[dict[str, object]] = []
    for row in rows[1:]:
        padded = [*row, "", "", "", ""]
        trigger = _to_int(padded[0])
        if trigger is None:
            continue
        out.append(
            {
                "trigger_num": trigger,
                "freq_for_left_eye": _to_float(padded[1]),
                "freq_for_right_eye": _to_float(padded[2]),
                "meaning": str(padded[3]).strip(),
            }
        )
    return out


def sun_stimulus_codebook(root: Path | str | None = None, kind: str = "forty_targets") -> list[dict[str, object]]:
    rows = []
    for row in read_sun_codebook(root=root, kind=kind):
        left = row["freq_for_left_eye"]
        right = row["freq_for_right_eye"]
        if left is None or right is None:
            continue
        if kind == "forty_targets":
            target_index = int(row["trigger_num"]) - 1
            phase_left, phase_right = SUN2024_FORTY_TARGET_PHASES_PI[target_index]
            row = {
                **row,
                "phase_pi_for_left_eye": phase_left,
                "phase_pi_for_right_eye": phase_right,
            }
        rows.append(row)
    return sorted(rows, key=lambda item: int(item["trigger_num"]))


def iter_sun_cnt_files(
    root: Path | str | None = None,
    splits: Iterable[str] | None = None,
    subjects: Iterable[int] | None = None,
) -> list[SunCntRecord]:
    base = sun2024_root(root) / "extracted"
    wanted_splits = set(splits) if splits is not None else set(SUN2024_SPLITS)
    wanted_subjects = {int(item) for item in subjects} if subjects is not None else None
    records: list[SunCntRecord] = []
    for split, split_dir in SUN2024_SPLITS.items():
        if split not in wanted_splits:
            continue
        for path in sorted((base / split_dir).glob("Subject *.cnt"), key=_natural_key):
            subject = _parse_subject(path.stem)
            if wanted_subjects is not None and subject not in wanted_subjects:
                continue
            records.append(SunCntRecord(path=path, split=split, split_dir=split_dir, subject=subject))
    return records


def sun_annotation_rows(path: Path | str) -> list[dict[str, object]]:
    import mne

    raw = mne.io.read_raw_cnt(Path(path), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    rows: list[dict[str, object]] = []
    for idx, annotation in enumerate(raw.annotations):
        description = str(annotation["description"])
        trigger = _to_int(description)
        onset = float(annotation["onset"])
        rows.append(
            {
                "event_index": idx,
                "sample": int(round(onset * sfreq)),
                "onset_seconds": onset,
                "duration_seconds": float(annotation["duration"]),
                "description": description,
                "trigger_num": trigger,
            }
        )
    return rows


def pair_sun_trial_events(annotation_rows: list[dict[str, object]], n_targets: int) -> list[SunTrialEvent]:
    trials: list[SunTrialEvent] = []
    for idx, row in enumerate(annotation_rows):
        trigger = row["trigger_num"]
        if trigger is None or trigger < 1 or trigger > n_targets:
            continue
        end_row = None
        for candidate in annotation_rows[idx + 1 :]:
            candidate_trigger = candidate["trigger_num"]
            if candidate_trigger == 253:
                end_row = candidate
                break
            if candidate_trigger is not None and 1 <= candidate_trigger <= n_targets:
                break
        trial_index = len(trials)
        block = trial_index // n_targets + 1
        trial_in_block = trial_index % n_targets + 1
        end_sample = int(end_row["sample"]) if end_row is not None else None
        end_seconds = float(end_row["onset_seconds"]) if end_row is not None else None
        start_seconds = float(row["onset_seconds"])
        duration_seconds = None if end_seconds is None else end_seconds - start_seconds
        trials.append(
            SunTrialEvent(
                trial_index=trial_index,
                block=block,
                trial_in_block=trial_in_block,
                target_id=int(trigger),
                start_sample=int(row["sample"]),
                end_sample=end_sample,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                duration_seconds=duration_seconds,
            )
        )
    return trials


def sun_target_repeat_trials(annotation_rows: list[dict[str, object]], n_targets: int) -> list[SunTargetRepeatTrial]:
    """Pair start/end markers and index each target by its recorded repetition.

    This deliberately does not assign artificial blocks based on event order.
    """

    repetitions: dict[int, int] = {}
    trials: list[SunTargetRepeatTrial] = []
    for idx, row in enumerate(annotation_rows):
        trigger = row["trigger_num"]
        if trigger is None or trigger < 1 or trigger > n_targets:
            continue
        end_row = None
        for candidate in annotation_rows[idx + 1 :]:
            candidate_trigger = candidate["trigger_num"]
            if candidate_trigger == 253:
                end_row = candidate
                break
            if candidate_trigger is not None and 1 <= candidate_trigger <= n_targets:
                break
        target_id = int(trigger)
        repetition = repetitions.get(target_id, 0) + 1
        repetitions[target_id] = repetition
        end_sample = int(end_row["sample"]) if end_row is not None else None
        end_seconds = float(end_row["onset_seconds"]) if end_row is not None else None
        start_seconds = float(row["onset_seconds"])
        trials.append(
            SunTargetRepeatTrial(
                trial_index=len(trials),
                target_id=target_id,
                repetition=repetition,
                start_sample=int(row["sample"]),
                end_sample=end_sample,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                duration_seconds=None if end_seconds is None else end_seconds - start_seconds,
            )
        )
    return trials
