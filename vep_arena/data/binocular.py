"""Shared loaders for two public binocular SSVEP datasets.

Dual-Alpha
    Sun Y et al., GigaScience 2024, doi:10.1093/gigascience/giae041;
    GigaDB dataset doi:10.5524/102557.
Binocular AR
    Ke Y et al., Scientific Data 2025,
    doi:10.1038/s41597-025-05696-0; Figshare article 26768287.

The constants below are local defaults only. Public task entry points expose a
--root option, and no raw EEG paths or files are committed.
"""

from __future__ import annotations

import csv
import json
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import signal


DUAL_ALPHA_ROOT = Path("D:/ProjData/datasets/ssvep_dual_alpha_gigadb_102557")
BINOCULAR_AR_ROOT = Path("D:/ProjData/datasets/ssvep_binocular_ar")

DUAL_ALPHA_CLASSES = 40
DUAL_ALPHA_BLOCKS = 5
DUAL_ALPHA_SAMPLING_RATE = 250
DUAL_ALPHA_ITR_SHIFT_SECONDS = 0.5
DUAL_ALPHA_PARADIGMS = {
    "Checkerboard_Arrangment": {
        "file": "Checkerboard_Arrangment.tar.gz",
        "code_section": "Checkerboard Arrangement",
        "display": "Checkerboard Arrangement",
    },
    "Binocular_Vision": {
        "file": "Binocular_Vision.tar.gz",
        "code_section": "Binocular Vision",
        "display": "Binocular Vision",
    },
    "Binocular-Swap_Vision": {
        "file": "Binocular-Swap_Vision.tar.gz",
        "code_section": "Binocular-Swap Vision",
        "display": "Binocular-Swap Vision",
    },
}

DUAL_ALPHA_OCCIPITAL9 = ("Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2")
AR_PAPER_OCCIPITAL10 = ("PO7", "PO8", "PO5", "PO4", "PO3", "POz", "PO6", "O1", "Oz", "O2")
AR_EXTENDED_OCCIPITAL13 = ("P7", "P3", "Pz", "P4", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2")
AR_OCCIPITAL = AR_PAPER_OCCIPITAL10
AR_TASKS = ("LF", "MF", "SFSP", "SFDP", "DFSP", "DFDP", "DFDP1", "DFDP3", "DFDP5")
AR_SESSIONS = ("ses-01", "ses-02")
AR_SAMPLING_RATE = 1024
AR_VISUAL_LATENCY_SECONDS = 0.14
AR_GAZE_SHIFT_SECONDS = 1.0
AR_CLASSES = 8


@dataclass(frozen=True)
class DualFrequencyCodebook:
    dataset: str
    paradigm: str
    freq1: tuple[float, ...]
    freq2: tuple[float, ...]
    phase1: tuple[float, ...] | None = None
    phase2: tuple[float, ...] | None = None

    @property
    def n_targets(self) -> int:
        return len(self.freq1)

    def target_freqs(self, zero_based_label: int) -> tuple[float, ...]:
        return (self.freq1[zero_based_label], self.freq2[zero_based_label])


@dataclass(frozen=True)
class DatasetFileStatus:
    subject: int | None
    name: str
    path: str
    expected_size: int | None
    actual_size: int
    complete: bool


@dataclass(frozen=True)
class EpochSample:
    dataset: str
    paradigm: str
    subject: int
    sampling_rate: int
    channels: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    time: np.ndarray
    target_freqs: tuple[tuple[float, ...], ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class DualAlphaEpochBlock:
    dataset: str
    paradigm: str
    subject: int
    sampling_rate: int
    channels: tuple[str, ...]
    x: np.ndarray
    target_freqs: tuple[tuple[float, ...], ...]
    block_keys: tuple[str, ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class AREpochBlock:
    dataset: str
    task: str
    subject: int
    sampling_rate: int
    channels: tuple[str, ...]
    x: np.ndarray
    target_freqs: tuple[tuple[float, ...], ...]
    target_phases: tuple[tuple[float, ...], ...]
    block_keys: tuple[str, ...]
    metadata: dict[str, object]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_float_list(value: object) -> tuple[float, ...]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ()
    parts = re.split(r"\s*/\s*|\s*,\s*", text)
    out: list[float] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return tuple(out)


def _natural_subject_key(name: str) -> int:
    match = re.search(r"Subject(\d+)", name)
    return int(match.group(1)) if match else 0


def _dual_alpha_tar_path(root: Path, paradigm: str) -> Path:
    try:
        file_name = DUAL_ALPHA_PARADIGMS[paradigm]["file"]
    except KeyError as exc:
        raise ValueError(f"Unknown Dual-Alpha paradigm: {paradigm}") from exc
    return root / "raw" / str(file_name)


def parse_dual_alpha_codebooks(root: Path = DUAL_ALPHA_ROOT) -> dict[str, DualFrequencyCodebook]:
    path = root / "source_code" / "Stimulate_Code.txt"
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    codebooks: dict[str, DualFrequencyCodebook] = {}
    section_to_paradigm = {
        str(meta["code_section"]): paradigm for paradigm, meta in DUAL_ALPHA_PARADIGMS.items()
    }
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line in section_to_paradigm:
            paradigm = section_to_paradigm[line]
            freq1 = tuple(float(x) for x in lines[idx + 2].split("\t") if x)
            freq2 = tuple(float(x) for x in lines[idx + 4].split("\t") if x)
            codebooks[paradigm] = DualFrequencyCodebook(
                dataset="dual_alpha",
                paradigm=paradigm,
                freq1=freq1,
                freq2=freq2,
            )
            idx += 5
        idx += 1
    return codebooks


def list_dual_alpha_files(root: Path = DUAL_ALPHA_ROOT) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for paradigm in DUAL_ALPHA_PARADIGMS:
        tar_path = _dual_alpha_tar_path(root, paradigm)
        row: dict[str, object] = {
            "dataset": "dual_alpha",
            "paradigm": paradigm,
            "path": str(tar_path),
            "size": tar_path.stat().st_size if tar_path.exists() else 0,
            "exists": tar_path.exists(),
        }
        if tar_path.exists() and tar_path.stat().st_size > 0:
            with tarfile.open(tar_path, "r:gz") as tf:
                names = tf.getnames()
                csv_names = [name for name in names if name.endswith(".csv")]
                txt_names = [name for name in names if name.endswith(".txt")]
                row.update(
                    {
                        "csv_subjects": len(csv_names),
                        "txt_subjects": len(txt_names),
                        "first_csv": sorted(csv_names, key=_natural_subject_key)[0],
                    }
                )
        rows.append(row)
    return rows


def load_dual_alpha_subject(
    root: Path,
    paradigm: str,
    subject: int,
    channels: Iterable[str] | None = None,
) -> pd.DataFrame:
    tar_path = _dual_alpha_tar_path(root, paradigm)
    member = f"./Subject{subject}.csv"
    with tarfile.open(tar_path, "r:gz") as tf:
        try:
            file_obj = tf.extractfile(member)
        except KeyError:
            members = {Path(name).name: name for name in tf.getnames()}
            file_obj = tf.extractfile(members[f"Subject{subject}.csv"])
        if file_obj is None:
            raise FileNotFoundError(f"{member} not found in {tar_path}")
        df = pd.read_csv(file_obj)
    df = df.rename(columns={"Unnamed: 0": "sample_index"})
    if channels is None:
        return df
    keep = ["sample_index", "time", "condition", "epoch", *list(channels)]
    missing = [col for col in keep if col not in df.columns]
    if missing:
        raise ValueError(f"Missing Dual-Alpha channels/columns: {missing}")
    return df[keep]


def available_dual_alpha_subjects(
    root: Path = DUAL_ALPHA_ROOT,
    paradigm: str = "Binocular_Vision",
) -> tuple[int, ...]:
    tar_path = _dual_alpha_tar_path(root, paradigm)
    with tarfile.open(tar_path, "r:gz") as tf:
        subjects = []
        for name in tf.getnames():
            if not name.endswith(".csv"):
                continue
            match = re.search(r"Subject(\d+)\.csv$", name)
            if match:
                subjects.append(int(match.group(1)))
    return tuple(sorted(subjects))


def _dual_alpha_channel_columns(df: pd.DataFrame) -> tuple[str, ...]:
    metadata = {"sample_index", "Unnamed: 0", "time", "condition", "epoch"}
    return tuple(str(col) for col in df.columns if str(col) not in metadata and not str(col).startswith("Unnamed"))


def dual_alpha_channel_names(
    root: Path = DUAL_ALPHA_ROOT,
    paradigm: str = "Binocular_Vision",
    subject: int | None = None,
) -> tuple[str, ...]:
    if subject is None:
        subject = available_dual_alpha_subjects(root, paradigm)[0]
    df = load_dual_alpha_subject(root, paradigm, subject, channels=None)
    return _dual_alpha_channel_columns(df)


def dual_alpha_channels_for_set(
    root: Path = DUAL_ALPHA_ROOT,
    paradigm: str = "Binocular_Vision",
    channel_set: str = "official",
    subject: int | None = None,
) -> tuple[str, ...]:
    channel_set = channel_set.lower()
    all_channels = dual_alpha_channel_names(root, paradigm, subject)
    if channel_set in {"official", "all"}:
        return all_channels
    if channel_set in {"occipital9", "9ch"}:
        available = {name.upper(): name for name in all_channels}
        missing = [name for name in DUAL_ALPHA_OCCIPITAL9 if name.upper() not in available]
        if missing:
            raise ValueError(f"Missing Dual-Alpha occipital9 channels for {paradigm}: {missing}")
        return tuple(available[name.upper()] for name in DUAL_ALPHA_OCCIPITAL9)
    raise ValueError(f"Unsupported Dual-Alpha channel set: {channel_set}")


def load_dual_alpha_epochs(
    root: Path = DUAL_ALPHA_ROOT,
    paradigm: str = "Binocular_Vision",
    subject: int = 1,
    window_seconds: float = 2.0,
    channels: Iterable[str] | None = None,
    channel_set: str = "official",
) -> DualAlphaEpochBlock:
    """Load Dual-Alpha subject data as classes x blocks x channels x samples.

    The public Dual-Alpha CSV files are already epoched. The official scripts
    ignore the absolute `time` values and crop the MNE EpochsArray from 0 to
    `t_max`; this loader follows that convention by taking the first samples of
    every epoch.
    """

    if channels is None:
        channels = dual_alpha_channels_for_set(root, paradigm, channel_set, subject)
    channel_names = tuple(channels)
    df = load_dual_alpha_subject(root, paradigm, subject, channels=channel_names)
    samples = int(round(float(window_seconds) * DUAL_ALPHA_SAMPLING_RATE))
    if samples <= 0:
        raise ValueError(f"Invalid Dual-Alpha window_seconds: {window_seconds}")

    class_blocks: dict[int, list[np.ndarray]] = {label: [] for label in range(DUAL_ALPHA_CLASSES)}
    block_keys_by_label: dict[int, list[str]] = {label: [] for label in range(DUAL_ALPHA_CLASSES)}
    for epoch_id, epoch_df in df.groupby("epoch", sort=True):
        label = int(epoch_df["condition"].iloc[0]) - 1
        if not 0 <= label < DUAL_ALPHA_CLASSES:
            raise ValueError(f"Invalid Dual-Alpha condition {label + 1} in subject {subject}, {paradigm}.")
        values = epoch_df.loc[:, channel_names].to_numpy(dtype=np.float32).T
        if values.shape[1] < samples:
            raise ValueError(
                f"Epoch {epoch_id} has {values.shape[1]} samples, requested {samples} "
                f"for subject {subject}, {paradigm}."
            )
        class_blocks[label].append(values[:, :samples])
        block_keys_by_label[label].append(f"epoch{int(epoch_id):03d}")

    missing = [label + 1 for label, blocks in class_blocks.items() if len(blocks) != DUAL_ALPHA_BLOCKS]
    if missing:
        raise ValueError(f"Dual-Alpha subject {subject}, {paradigm} does not have {DUAL_ALPHA_BLOCKS} blocks for labels {missing}")

    x = np.empty((DUAL_ALPHA_CLASSES, DUAL_ALPHA_BLOCKS, len(channel_names), samples), dtype=np.float32)
    for label in range(DUAL_ALPHA_CLASSES):
        x[label] = np.stack(class_blocks[label], axis=0)
    codebook = parse_dual_alpha_codebooks(root)[paradigm]
    target_freqs = tuple(codebook.target_freqs(label) for label in range(DUAL_ALPHA_CLASSES))
    block_keys = tuple(f"block{idx + 1:02d}" for idx in range(DUAL_ALPHA_BLOCKS))
    return DualAlphaEpochBlock(
        dataset="dual_alpha",
        paradigm=paradigm,
        subject=subject,
        sampling_rate=DUAL_ALPHA_SAMPLING_RATE,
        channels=channel_names,
        x=x,
        target_freqs=target_freqs,
        block_keys=block_keys,
        metadata={
            "source": str(_dual_alpha_tar_path(root, paradigm)),
            "window_seconds": float(window_seconds),
            "samples": samples,
            "channel_set": channel_set,
            "epoch_keys_by_label": {str(label + 1): block_keys_by_label[label] for label in range(DUAL_ALPHA_CLASSES)},
            "preprocessing": "official CSV epochs; crop first window samples; method-specific filterbank applied after crop",
            "time_min": float(df["time"].min()),
            "time_max": float(df["time"].max()),
        },
    )


def dual_alpha_trca_filterbank_specs() -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return (
        ((6.0, 90.0), (4.0, 100.0)),
        ((14.0, 90.0), (10.0, 100.0)),
        ((22.0, 90.0), (16.0, 100.0)),
        ((30.0, 90.0), (24.0, 100.0)),
        ((38.0, 90.0), (32.0, 100.0)),
        ((46.0, 90.0), (40.0, 100.0)),
        ((54.0, 90.0), (48.0, 100.0)),
    )


def dual_alpha_fbdcca_filterbank_bands() -> tuple[tuple[float, float], ...]:
    return (
        (5.0, 95.0),
        (12.0, 95.0),
        (19.0, 95.0),
        (27.0, 95.0),
        (35.0, 95.0),
        (43.0, 95.0),
        (51.0, 95.0),
        (59.0, 95.0),
        (66.0, 95.0),
        (75.0, 95.0),
    )


def _sosfiltfilt_safe(sos: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    samples = x.shape[axis]
    if samples <= 3:
        return signal.sosfilt(sos, x, axis=axis)
    padlen = min(samples - 1, 3 * (2 * sos.shape[0] + 1))
    return signal.sosfiltfilt(sos, x, axis=axis, padlen=padlen)


def _filtfilt_safe(b: np.ndarray, a: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    samples = x.shape[axis]
    if samples <= 3:
        return signal.lfilter(b, a, x, axis=axis)
    padlen = min(samples - 1, 3 * (max(len(a), len(b)) - 1))
    return signal.filtfilt(b, a, x, axis=axis, padlen=padlen)


def dual_alpha_apply_trca_filterbank(
    epochs: np.ndarray,
    n_fbs: int = 7,
    fs: int = DUAL_ALPHA_SAMPLING_RATE,
) -> np.ndarray:
    """Return Dual-Alpha TRCA epochs as classes x blocks x subbands x channels x samples."""

    specs = dual_alpha_trca_filterbank_specs()[:n_fbs]
    classes, blocks, channels, samples = epochs.shape
    out = np.empty((classes, blocks, len(specs), channels, samples), dtype=np.float32)
    for fb_idx, (passband, stopband) in enumerate(specs):
        order, wn = signal.cheb1ord(passband, stopband, gpass=3, gstop=40, fs=fs)
        sos = signal.cheby1(order, 0.5, wn, btype="bandpass", fs=fs, output="sos")
        filtered = _sosfiltfilt_safe(sos, np.asarray(epochs, dtype=np.float64), axis=-1)
        out[:, :, fb_idx] = filtered.astype(np.float32, copy=False)
    return out


def dual_alpha_apply_fbdcca_filterbank(
    epochs: np.ndarray,
    n_fbs: int = 5,
    fs: int = DUAL_ALPHA_SAMPLING_RATE,
    backend: str = "mne-fir",
    n_jobs: int | str | None = 1,
) -> np.ndarray:
    """Return Dual-Alpha FBDCCA epochs as classes x blocks x subbands x channels x samples.

    The official Dual-Alpha FBDCCA code calls MNE FIR filters. The default
    therefore requires MNE and follows that path. The SciPy backend is kept only
    for explicitly named legacy/diagnostic runs.
    """

    if backend not in {"mne-fir", "scipy-fir-legacy"}:
        raise ValueError(f"Unsupported FBDCCA filter backend: {backend}")
    bands = dual_alpha_fbdcca_filterbank_bands()[:n_fbs]
    classes, blocks, channels, samples = epochs.shape
    out = np.empty((classes, blocks, len(bands), channels, samples), dtype=np.float32)
    data64 = np.asarray(epochs, dtype=np.float64)
    for fb_idx, (low, high) in enumerate(bands):
        high = min(high, fs / 2.0 - 1.0)
        if backend == "mne-fir":
            try:
                from mne.filter import filter_data
            except ImportError as exc:
                raise ImportError(
                    "Dual-Alpha official FBDCCA requires MNE for backend='mne-fir'. "
                    "Install mne in the active Arena environment or pass "
                    "backend='scipy-fir-legacy' for an explicitly approximate legacy run."
                ) from exc
            flat = data64.reshape(classes * blocks * channels, samples)
            filtered = filter_data(
                flat,
                sfreq=float(fs),
                l_freq=float(low),
                h_freq=float(high),
                method="fir",
                phase="zero",
                fir_design="firwin",
                n_jobs=n_jobs,
                verbose=False,
            ).reshape(classes, blocks, channels, samples)
        elif samples < 9:
            sos = signal.butter(4, (low, high), btype="bandpass", fs=fs, output="sos")
            filtered = _sosfiltfilt_safe(sos, data64, axis=-1)
        else:
            max_taps = samples - 1 if samples % 2 == 0 else samples
            numtaps = max(9, min(101, max_taps))
            if numtaps % 2 == 0:
                numtaps -= 1
            b = signal.firwin(numtaps, (low, high), pass_zero=False, fs=fs)
            filtered = _filtfilt_safe(b, np.asarray([1.0]), data64, axis=-1)
        out[:, :, fb_idx] = filtered.astype(np.float32, copy=False)
    return out


def dual_alpha_sample(
    root: Path = DUAL_ALPHA_ROOT,
    paradigm: str = "Binocular_Vision",
    subject: int = 1,
    channels: Iterable[str] | None = DUAL_ALPHA_OCCIPITAL9,
    max_epochs: int = 40,
    window_seconds: float | None = 2.0,
) -> EpochSample:
    df = load_dual_alpha_subject(root, paradigm, subject, channels)
    channel_names = tuple(col for col in df.columns if col not in {"sample_index", "time", "condition", "epoch"})
    groups = list(df.groupby("epoch", sort=True))
    if not groups:
        raise ValueError(f"No epochs found for Dual-Alpha subject {subject}, {paradigm}.")

    first_times = groups[0][1]["time"].to_numpy(dtype=float)
    fs = 250
    n_samples = len(first_times)
    if window_seconds is not None:
        n_samples = min(n_samples, int(round(fs * window_seconds)))
    x = np.empty((min(max_epochs, len(groups)), len(channel_names), n_samples), dtype=np.float32)
    y = np.empty((x.shape[0],), dtype=np.int64)
    time = first_times[:n_samples] - first_times[0]
    for out_idx, (_, epoch_df) in enumerate(groups[: x.shape[0]]):
        x[out_idx] = epoch_df.loc[:, channel_names].to_numpy(dtype=np.float32).T[:, :n_samples]
        y[out_idx] = int(epoch_df["condition"].iloc[0]) - 1

    codebooks = parse_dual_alpha_codebooks(root)
    codebook = codebooks[paradigm]
    target_freqs = tuple(codebook.target_freqs(int(label)) for label in y)
    return EpochSample(
        dataset="dual_alpha",
        paradigm=paradigm,
        subject=subject,
        sampling_rate=fs,
        channels=channel_names,
        x=x,
        y=y,
        time=np.asarray(time, dtype=np.float64),
        target_freqs=target_freqs,
        metadata={
            "source": str(_dual_alpha_tar_path(root, paradigm)),
            "n_total_epochs": len(groups),
            "time_min": float(df["time"].min()),
            "time_max": float(df["time"].max()),
            "codebook_targets": codebook.n_targets,
        },
    )


def ar_epoch_file_status(root: Path = BINOCULAR_AR_ROOT) -> list[DatasetFileStatus]:
    manifest = _read_json(root / "metadata" / "ssvep_binocular_ar_manifest.json")
    expected = {
        item["name"]: int(item["size"])
        for item in manifest["epoch"]["files"]
        if str(item["name"]).startswith("sub-")
    }
    rows: list[DatasetFileStatus] = []
    for name, expected_size in sorted(expected.items()):
        path = root / "epoch" / name
        actual = path.stat().st_size if path.exists() else 0
        match = re.search(r"sub-(\d+)", name)
        rows.append(
            DatasetFileStatus(
                subject=int(match.group(1)) if match else None,
                name=name,
                path=str(path),
                expected_size=expected_size,
                actual_size=actual,
                complete=actual == expected_size,
            )
        )
    return rows


def available_ar_subjects(root: Path = BINOCULAR_AR_ROOT) -> tuple[int, ...]:
    return tuple(status.subject for status in ar_epoch_file_status(root) if status.complete and status.subject is not None)


def _ar_zip_path(root: Path, subject: int) -> Path:
    path = root / "epoch" / f"sub-{subject:03d}.zip"
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty AR epoch zip: {path}")
    return path


def _ar_task_prefix(zip_file: zipfile.ZipFile, subject: int, session: str, task: str) -> str:
    prefix = f"{session}/eeg/sub-{subject:03d}_{session}_task-{task}"
    if f"{prefix}_events.tsv" not in zip_file.namelist():
        raise FileNotFoundError(f"Task {task!r} session {session!r} not found for sub-{subject:03d}.")
    return prefix


def list_ar_tasks(root: Path = BINOCULAR_AR_ROOT, subject: int = 8) -> list[dict[str, object]]:
    zip_path = _ar_zip_path(root, subject)
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            match = re.search(r"(ses-\d+)/eeg/sub-\d+_(ses-\d+)_task-(.+?)_events.tsv$", name)
            if not match:
                continue
            session = match.group(1)
            task = match.group(3)
            prefix = _ar_task_prefix(zf, subject, session, task)
            events = pd.read_csv(zf.open(f"{prefix}_events.tsv"), sep="\t")
            info = json.loads(zf.read(f"{prefix}_eeg.json").decode("utf-8-sig"))
            rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "events": int(len(events)),
                    "sampling_rate": int(info["SamplingFrequency"]),
                    "channels": int(info["EEGChannelCount"]),
                    "duration_seconds": float(events["duration"].median()) if len(events) else np.nan,
                    "first_frequency": str(events["stim_frequency"].iloc[0]) if len(events) else "",
                }
            )
    return rows


def load_ar_epoch_sample(
    root: Path = BINOCULAR_AR_ROOT,
    subject: int = 8,
    task: str = "DFDP",
    session: str = "ses-01",
    event_index: int = 0,
    window_seconds: float = 3.0,
    channels: Iterable[str] | None = AR_OCCIPITAL,
) -> EpochSample:
    zip_path = _ar_zip_path(root, subject)
    with zipfile.ZipFile(zip_path) as zf:
        prefix = _ar_task_prefix(zf, subject, session, task)
        events = pd.read_csv(zf.open(f"{prefix}_events.tsv"), sep="\t")
        channel_table = pd.read_csv(zf.open(f"{prefix}_channels.tsv"), sep="\t")
        info = json.loads(zf.read(f"{prefix}_eeg.json").decode("utf-8-sig"))
        fs = int(info["SamplingFrequency"])
        n_ch = int(info["EEGChannelCount"])
        fdt_bytes = zf.read(f"{prefix}_eeg.fdt")
    raw = np.frombuffer(fdt_bytes, dtype="<f4")
    n_frames = raw.size // n_ch
    data = raw[: n_frames * n_ch].reshape(n_frames, n_ch).T
    all_channels = tuple(str(name) for name in channel_table["name"].tolist())
    if channels is None:
        keep_idx = tuple(range(n_ch))
    else:
        requested = tuple(channels)
        channel_lookup = {name.upper(): idx for idx, name in enumerate(all_channels)}
        missing = [name for name in requested if name.upper() not in channel_lookup]
        if missing:
            raise ValueError(f"Missing AR channels: {missing}")
        keep_idx = tuple(channel_lookup[name.upper()] for name in requested)
    event = events.iloc[event_index]
    start = max(0, int(float(event["onset"])))
    n_samples = min(int(round(window_seconds * fs)), data.shape[1] - start)
    if n_samples <= 0:
        raise ValueError(f"Invalid AR sample window for onset {start} in {prefix}.")
    x = data[np.asarray(keep_idx), start : start + n_samples][None, :, :].astype(np.float32, copy=False)
    time = np.arange(n_samples, dtype=np.float64) / fs
    target_freqs = (_parse_float_list(event["stim_frequency"]),)
    phase_values = _parse_float_list(event["stim_phase"])
    return EpochSample(
        dataset="binocular_ar",
        paradigm=task,
        subject=subject,
        sampling_rate=fs,
        channels=tuple(all_channels[idx] for idx in keep_idx),
        x=x,
        y=np.asarray([int(event["value"]) - 1], dtype=np.int64),
        time=time,
        target_freqs=target_freqs,
        metadata={
            "source": str(zip_path),
            "session": session,
            "task": task,
            "event_index": int(event_index),
            "event_trial": int(event["trial"]),
            "event_value": int(event["value"]),
            "event_onset_samples": int(start),
            "event_duration_seconds": float(event["duration"]),
            "event_stim_frequency": str(event["stim_frequency"]),
            "event_stim_phase": str(event["stim_phase"]),
            "phase_values": phase_values,
            "n_frames": int(n_frames),
            "all_channels": all_channels,
        },
    )


def ar_task_n_fbs(task: str) -> int:
    return 3 if task.upper() == "MF" else 5


def ar_task_stimuli_id(task: str) -> int:
    task_map = {
        "LF": 1,
        "MF": 2,
        "SFSP": 3,
        "SFDP": 4,
        "DFSP": 5,
        "DFDP": 6,
        "DFDP1": 7,
        "DFDP3": 8,
        "DFDP5": 9,
    }
    try:
        return task_map[task.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown binocular AR task: {task}") from exc


def ar_filterbank_passbands(task: str, n_fbs: int | None = None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    stimuli = ar_task_stimuli_id(task)
    if stimuli == 2:
        passband = (21.0, 44.0, 67.0)
        stopband = (19.0, 40.0, 61.0)
    else:
        passband = (6.0, 14.0, 22.0, 30.0, 38.0, 46.0, 54.0, 62.0, 70.0, 78.0)
        stopband = (4.0, 10.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0)
    count = n_fbs if n_fbs is not None else ar_task_n_fbs(task)
    return passband[:count], stopband[:count]


def ar_preprocess_continuous(data: np.ndarray, fs: int = AR_SAMPLING_RATE) -> np.ndarray:
    """Approximate the paper's continuous EEGLAB preprocessing in SciPy."""

    x = np.asarray(data, dtype=np.float64)
    x = x - np.mean(x, axis=-1, keepdims=True)
    sos_notch = signal.butter(4, (49.0, 51.0), btype="bandstop", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos_notch, x, axis=-1)
    sos_bp = signal.butter(4, (5.0, 95.0), btype="bandpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos_bp, x, axis=-1)
    return x


def ar_apply_filterbank(epochs: np.ndarray, task: str, n_fbs: int | None = None, fs: int = AR_SAMPLING_RATE) -> np.ndarray:
    """Return epochs as classes x blocks x subbands x channels x samples."""

    passband, _ = ar_filterbank_passbands(task, n_fbs)
    classes, blocks, channels, samples = epochs.shape
    out = np.empty((classes, blocks, len(passband), channels, samples), dtype=np.float32)
    nyq = fs / 2.0
    for fb_idx, low in enumerate(passband):
        b, a = signal.cheby1(6, 0.5, (low / nyq, 90.0 / nyq), btype="bandpass")
        for cls in range(classes):
            out[cls, :, fb_idx] = signal.filtfilt(
                b,
                a,
                epochs[cls],
                axis=-1,
                padtype="odd",
                padlen=3 * (max(len(b), len(a)) - 1),
            ).astype(np.float32, copy=False)
    return out


def _ar_read_continuous(
    zf: zipfile.ZipFile,
    prefix: str,
    channels: Iterable[str] | None,
) -> tuple[np.ndarray, tuple[str, ...], pd.DataFrame, int]:
    events = pd.read_csv(zf.open(f"{prefix}_events.tsv"), sep="\t")
    channel_table = pd.read_csv(zf.open(f"{prefix}_channels.tsv"), sep="\t")
    info = json.loads(zf.read(f"{prefix}_eeg.json").decode("utf-8-sig"))
    fs = int(info["SamplingFrequency"])
    n_ch = int(info["EEGChannelCount"])
    raw = np.frombuffer(zf.read(f"{prefix}_eeg.fdt"), dtype="<f4")
    n_frames = raw.size // n_ch
    data = raw[: n_frames * n_ch].reshape(n_frames, n_ch).T
    all_channels = tuple(str(name) for name in channel_table["name"].tolist())
    if channels is None:
        keep_idx = tuple(range(n_ch))
    else:
        requested = tuple(channels)
        lookup = {name.upper(): idx for idx, name in enumerate(all_channels)}
        missing = [name for name in requested if name.upper() not in lookup]
        if missing:
            raise ValueError(f"Missing AR channels: {missing}")
        keep_idx = tuple(lookup[name.upper()] for name in requested)
    return data[np.asarray(keep_idx)], tuple(all_channels[idx] for idx in keep_idx), events, fs


def load_ar_task_epochs(
    root: Path = BINOCULAR_AR_ROOT,
    subject: int = 1,
    task: str = "LF",
    window_seconds: float = 1.0,
    channels: Iterable[str] | None = AR_OCCIPITAL,
    sessions: Iterable[str] = AR_SESSIONS,
    n_fbs: int | None = None,
    latency_seconds: float = AR_VISUAL_LATENCY_SECONDS,
    filterbank: bool = True,
) -> AREpochBlock:
    """Load a paper-aligned binocular AR task as classes x blocks epochs.

    The official classification scripts crop `[-0.5, 3.14]` epochs with
    `len_delay_s = 0.5 + 0.14`, which is equivalent to starting 0.14 s after
    flicker onset in the event table. Events are ordered as 10 blocks of 8
    targets per session.
    """

    task = task.upper()
    zip_path = _ar_zip_path(root, subject)
    fs_expected: int | None = None
    channel_names: tuple[str, ...] | None = None
    blocks: list[np.ndarray] = []
    block_keys: list[str] = []
    target_freqs: dict[int, tuple[float, ...]] = {}
    target_phases: dict[int, tuple[float, ...]] = {}
    samples = int(round(window_seconds * AR_SAMPLING_RATE))

    with zipfile.ZipFile(zip_path) as zf:
        for session in sessions:
            prefix = _ar_task_prefix(zf, subject, session, task)
            data, these_channels, events, fs = _ar_read_continuous(zf, prefix, channels)
            if fs_expected is None:
                fs_expected = fs
                samples = int(round(window_seconds * fs))
            elif fs != fs_expected:
                raise ValueError(f"Inconsistent sampling rate for sub-{subject:03d} {task}: {fs} != {fs_expected}")
            if channel_names is None:
                channel_names = these_channels
            elif channel_names != these_channels:
                raise ValueError(f"Inconsistent channels for sub-{subject:03d} {task} {session}.")

            preprocessed = ar_preprocess_continuous(data, fs)
            events = events.sort_values("trial").reset_index(drop=True)
            if len(events) % AR_CLASSES != 0:
                raise ValueError(f"Event count for sub-{subject:03d} {task} {session} is not divisible by {AR_CLASSES}.")
            n_blocks = len(events) // AR_CLASSES
            for block_idx in range(n_blocks):
                block = np.zeros((AR_CLASSES, len(channel_names), samples), dtype=np.float32)
                rows = events.iloc[block_idx * AR_CLASSES : (block_idx + 1) * AR_CLASSES]
                seen: set[int] = set()
                for _, event in rows.iterrows():
                    label = int(event["value"]) - 1
                    if not 0 <= label < AR_CLASSES:
                        raise ValueError(f"Invalid label {label + 1} in sub-{subject:03d} {task} {session}.")
                    start = int(float(event["onset"])) + int(round(latency_seconds * fs))
                    stop = start + samples
                    if stop > preprocessed.shape[1]:
                        raise ValueError(f"Window exceeds data length for sub-{subject:03d} {task} {session}.")
                    block[label] = preprocessed[:, start:stop]
                    target_freqs.setdefault(label, _parse_float_list(event["stim_frequency"]))
                    target_phases.setdefault(label, _parse_float_list(event["stim_phase"]))
                    seen.add(label)
                if seen != set(range(AR_CLASSES)):
                    raise ValueError(f"Block {block_idx + 1} in sub-{subject:03d} {task} {session} is incomplete.")
                blocks.append(block)
                block_keys.append(f"{session}_b{block_idx + 1:02d}")

    if fs_expected is None or channel_names is None or not blocks:
        raise ValueError(f"No epochs loaded for sub-{subject:03d} {task}.")
    raw_epochs = np.stack(blocks, axis=1)
    x = ar_apply_filterbank(raw_epochs, task=task, n_fbs=n_fbs, fs=fs_expected) if filterbank else raw_epochs[:, :, None]
    ordered_freqs = tuple(target_freqs[idx] for idx in range(AR_CLASSES))
    ordered_phases = tuple(target_phases[idx] for idx in range(AR_CLASSES))
    return AREpochBlock(
        dataset="binocular_ar",
        task=task,
        subject=subject,
        sampling_rate=fs_expected,
        channels=channel_names,
        x=x.astype(np.float32, copy=False),
        target_freqs=ordered_freqs,
        target_phases=ordered_phases,
        block_keys=tuple(block_keys),
        metadata={
            "source": str(zip_path),
            "sessions": list(sessions),
            "window_seconds": float(window_seconds),
            "latency_seconds": float(latency_seconds),
            "gaze_shift_seconds": AR_GAZE_SHIFT_SECONDS,
            "n_fbs": int(x.shape[2]),
            "preprocessing": "scipy butter bandstop 49-51 Hz + butter bandpass 5-95 Hz; event onset + 0.14s crop",
            "filterbank": "official code style cheby1 order=6 ripple=0.5 with task-specific passbands",
        },
    )


def describe_epoch_sample(sample: EpochSample) -> dict[str, object]:
    x = np.asarray(sample.x, dtype=np.float64)
    return {
        "dataset": sample.dataset,
        "paradigm": sample.paradigm,
        "subject": sample.subject,
        "sampling_rate": sample.sampling_rate,
        "trials": int(x.shape[0]),
        "channels": int(x.shape[1]),
        "samples": int(x.shape[2]),
        "seconds": float(x.shape[2] / sample.sampling_rate),
        "channel_names": list(sample.channels),
        "label_min": int(np.min(sample.y)),
        "label_max": int(np.max(sample.y)),
        "target_freqs_first": list(sample.target_freqs[0]) if sample.target_freqs else [],
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "ptp": float(np.ptp(x)),
        "rms": float(np.sqrt(np.mean(x**2))),
        "metadata": sample.metadata,
    }


def preprocess_for_smoke(
    x: np.ndarray,
    fs: int,
    notch_hz: float | None = 50.0,
    bandpass: tuple[float, float] | None = (5.0, 90.0),
) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    y = y - np.mean(y, axis=-1, keepdims=True)
    if notch_hz is not None and notch_hz < fs / 2:
        b, a = signal.iirnotch(notch_hz, Q=30.0, fs=fs)
        y = signal.filtfilt(b, a, y, axis=-1)
    if bandpass is not None:
        lo, hi = bandpass
        hi = min(hi, fs / 2 - 1.0)
        if lo > 0 and hi > lo:
            sos = signal.butter(4, (lo, hi), btype="bandpass", fs=fs, output="sos")
            y = signal.sosfiltfilt(sos, y, axis=-1)
    return y.astype(np.float32, copy=False)


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
