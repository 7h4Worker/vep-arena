from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat, whosmat


TASK = Path(__file__).resolve().parent
PROJECT = TASK.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from audit_jbhi35_subject_quality import (  # noqa: E402
    exact_log_power,
    expected_frequency_effect,
    expected_frequency_itpc_oz,
    leave_one_block_accuracy,
    unordered_code_accuracy,
)
from vep_arena.data.embc_jbhi import (  # noqa: E402
    OCCIPITAL9,
    PrivateSSVEPSubject,
    analysis_epochs,
    available_subject_ids,
    load_subject,
    load_target_frequency_pairs,
    resolve_dataset_root,
)
from vep_arena.methods.traditional import TRCA  # noqa: E402


METHOD_FIELDS = {
    "ETRCA": "trca_ens",
    "EBPRCA": "fusion_ens",
    "EFUSIONCA": "fusion_trca_ens",
}
AUDIT_WINDOWS = (0.4, 1.0, 2.0)
EMBC9_WINDOWS = tuple(round(0.2 * index, 1) for index in range(1, 21))
EMBC9_COMMON_WINDOWS = (1.0, 2.0, 3.0, 4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit anonymous Arena results against the private legacy backup.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs" / "datasets" / "local_paths.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TASK / "results" / "legacy_backup_audit",
    )
    parser.add_argument("--force", action="store_true", help="Recompute the legacy 35-target receiver audit.")
    return parser.parse_args()


def configured_path(config_path: Path, key: str) -> Path:
    payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    value = payload.get("datasets", {}).get(key)
    if not value:
        raise KeyError(f"{key} is missing from {config_path}")
    root = Path(str(value))
    if not root.is_dir():
        raise FileNotFoundError(f"Configured path does not exist for {key}: {root}")
    return root


def array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def matlab_strings(values: object) -> tuple[str, ...]:
    output = []
    for value in np.atleast_1d(values).reshape(-1):
        if isinstance(value, str):
            output.append(value.strip())
        else:
            output.append("".join(np.asarray(value).astype(str).reshape(-1)).strip())
    return tuple(output)


def grouped_16(path: Path) -> np.ndarray:
    payload = loadmat(path, variable_names=["eegdata", "label_list"])
    if "eegdata" not in payload or "label_list" not in payload:
        raise ValueError("not a labeled 16-target source")
    eeg = np.asarray(payload["eegdata"])
    labels = np.asarray(payload["label_list"]).reshape(-1).astype(np.int64)
    if eeg.shape != (9, 4001, 96) or not np.array_equal(labels, np.tile(np.arange(1, 17), 6)):
        raise ValueError("not a canonical labeled 16-target source")
    return np.stack([np.transpose(eeg[:, :, labels == target], (2, 0, 1)) for target in range(1, 17)])


def grouped_35(path: Path) -> np.ndarray:
    payload = loadmat(path, variable_names=["eegdata", "label_list", "channel_list"], simplify_cells=True)
    if not {"eegdata", "label_list", "channel_list"}.issubset(payload):
        raise ValueError("not a labeled 35-target source")
    eeg = np.asarray(payload["eegdata"], dtype=np.float64)
    labels = np.asarray(payload["label_list"]).reshape(-1).astype(np.int64)
    channels = tuple(name.upper() for name in matlab_strings(payload["channel_list"]))
    if eeg.ndim != 3 or eeg.shape[1] not in (2000, 2001) or eeg.shape[2] != 210 or labels.shape != (210,):
        raise ValueError("not a canonical labeled 35-target source")
    picks = [channels.index(name.upper()) for name in OCCIPITAL9]
    eeg = eeg[picks, :2000]
    blocks = []
    for block in range(6):
        block_slice = slice(block * 35, (block + 1) * 35)
        block_labels = labels[block_slice]
        if not np.array_equal(np.sort(block_labels), np.arange(1, 36)):
            raise ValueError(f"block {block + 1} is not a permutation of labels 1-35")
        order = np.argsort(block_labels)
        blocks.append(np.transpose(eeg[:, :, block_slice][:, :, order], (2, 0, 1)))
    return np.stack(blocks, axis=1)


def current_subject_hashes(dataset: str) -> dict[str, str]:
    root = resolve_dataset_root(dataset)
    return {
        array_digest(load_subject(dataset, subject_id, root).x): subject_id
        for subject_id in available_subject_ids(dataset, root)
    }


def grouped_9(path: Path) -> np.ndarray:
    payload = loadmat(path, variable_names=["eegdata", "label_list"])
    if "eegdata" not in payload or "label_list" not in payload:
        raise ValueError("not a labeled 9-target source")
    eeg = np.asarray(payload["eegdata"], dtype=np.float64)
    labels = np.asarray(payload["label_list"]).reshape(-1).astype(np.int64)
    if eeg.shape != (13, 4000, 180) or labels.shape != (180,):
        raise ValueError("not a canonical labeled 9-target source")
    counts = np.bincount(labels, minlength=10)[1:]
    if not np.array_equal(counts, np.full(9, 20)):
        raise ValueError("9-target labels do not contain 20 trials per class")
    return np.stack(
        [np.transpose(eeg[-9:, :, labels == target], (2, 0, 1)) for target in range(1, 10)]
    )


def embc9_result_files(legacy_root: Path) -> dict[str, Path]:
    result_root = legacy_root / "code" / "result_figures_mat"
    suffix_pattern = re.compile(r"_(?:opE|opG|opEG|gamma)$", flags=re.IGNORECASE)
    paths = sorted(
        path
        for path in result_root.glob("result_acc_itr_9sc_*.mat")
        if not suffix_pattern.search(path.stem)
    )
    if len(paths) != 8:
        raise RuntimeError(f"Expected eight base EMBC9 result MAT files, found {len(paths)}")
    return {
        path.stem.removeprefix("result_acc_itr_9sc_").lower(): path
        for path in paths
    }


def embc9_paper_included_codes(legacy_root: Path, result_codes: set[str]) -> set[str]:
    script = legacy_root / "code" / "result_figures_mat" / "performance_figures_2.m"
    text = script.read_text(encoding="utf-8", errors="ignore")
    line = next(
        value.split("%", 1)[0]
        for value in text.splitlines()
        if value.strip().startswith("accs_mean_trca =")
    )
    included = set(re.findall(r"accs_([A-Za-z0-9]+)", line)) & result_codes
    if len(included) != 7 or len(result_codes - included) != 1:
        raise RuntimeError("The EMBC9 final plotting script does not define a seven-subject paper mean")
    return included


def audit_legacy_9(legacy_root: Path) -> pd.DataFrame:
    result_files = embc9_result_files(legacy_root)
    included_codes = embc9_paper_included_codes(legacy_root, set(result_files))
    current_by_hash = current_subject_hashes("embc9")
    matched: dict[str, tuple[str, Path]] = {}
    for path in (legacy_root / "code").rglob("*.mat"):
        try:
            metadata = {name: shape for name, shape, _ in whosmat(path)}
        except (OSError, ValueError):
            continue
        if metadata.get("eegdata") != (13, 4000, 180) or metadata.get("label_list") not in {
            (1, 180),
            (180, 1),
        }:
            continue
        values = grouped_9(path)
        subject_id = current_by_hash.get(array_digest(values))
        if subject_id is None:
            continue
        result_matches = [
            code
            for code in result_files
            if re.search(rf"(?:^|_){re.escape(code)}(?:_|$)", path.stem, flags=re.IGNORECASE)
        ]
        if len(result_matches) != 1:
            raise RuntimeError(f"Anonymous {subject_id} has {len(result_matches)} EMBC9 result matches")
        if subject_id in matched:
            raise RuntimeError(f"Anonymous {subject_id} has multiple exact legacy EMBC9 source matches")
        matched[subject_id] = (result_matches[0], result_files[result_matches[0]])
    expected_subjects = set(current_by_hash.values())
    if set(matched) != expected_subjects:
        raise RuntimeError("Not all current EMBC9 arrays have one exact labeled legacy source/result match")

    truth = np.tile(np.arange(1, 10), 20)
    rows: list[dict[str, object]] = []
    for subject_id in sorted(matched):
        code, result_path = matched[subject_id]
        payload = loadmat(
            result_path,
            variable_names=["ACCs", "ITRs", "result_label_best"],
        )
        accuracies = np.asarray(payload["ACCs"], dtype=np.float64).reshape(-1) / 100.0
        itrs = np.asarray(payload["ITRs"], dtype=np.float64).reshape(-1)
        predictions = np.asarray(payload["result_label_best"]).reshape(-1).astype(np.int64)
        if accuracies.shape != (20,) or itrs.shape != (20,) or predictions.shape != (180,):
            raise RuntimeError(f"Anonymous {subject_id} has an unexpected historical result shape")
        prediction_accuracy = float(np.mean(predictions == truth))
        matching_windows = [
            window
            for window, accuracy in zip(EMBC9_WINDOWS, accuracies)
            if np.isclose(accuracy, prediction_accuracy)
        ]
        for window, accuracy, itr in zip(EMBC9_WINDOWS, accuracies, itrs):
            rows.append(
                {
                    "subject": subject_id,
                    "paper_included": code in included_codes,
                    "window_seconds": window,
                    "legacy_accuracy": float(accuracy),
                    "legacy_itr_bpm": float(itr),
                    "stored_prediction_accuracy": prediction_accuracy,
                    "stored_prediction_matching_window_count": len(matching_windows),
                }
            )
    return pd.DataFrame(rows)


def embc9_inventory(legacy_root: Path) -> dict[str, int]:
    result_root = legacy_root / "code" / "result_figures_mat"
    accuracy_paths = list(result_root.glob("result_acc_itr_9sc_*.mat"))
    categories = {
        "base": 0,
        "opE": 0,
        "opG": 0,
        "opEG": 0,
        "gamma": 0,
    }
    for path in accuracy_paths:
        for suffix in ("opEG", "opE", "opG", "gamma"):
            if path.stem.lower().endswith(f"_{suffix.lower()}"):
                categories[suffix] += 1
                break
        else:
            categories["base"] += 1
    return {
        "result_mat_total": len(list(result_root.glob("*.mat"))),
        "accuracy_itr_base": categories["base"],
        "accuracy_itr_opE": categories["opE"],
        "accuracy_itr_opG": categories["opG"],
        "accuracy_itr_opEG": categories["opEG"],
        "accuracy_itr_gamma": categories["gamma"],
        "frequency_analysis_mat": len(list((legacy_root / "code" / "freq_analysis").glob("*.mat"))),
        "frequency_average_mat": len(
            list((legacy_root / "code" / "freq_analysis" / "freq_analysis_avg").glob("*.mat"))
        ),
        "matlab_figure_sources": len(list((legacy_root / "images" / "source").glob("*.fig"))),
        "manuscript_pdfs": len(list((legacy_root / "manuscript").glob("*.pdf"))),
    }


def jbhi_inventory(legacy_root: Path) -> dict[str, object]:
    result_root = legacy_root / "Results_mat"
    categories = {
        path.name: len(list(path.rglob("*.mat")))
        for path in sorted(result_root.iterdir(), key=lambda value: value.name.lower())
        if path.is_dir()
    }
    direct = len(list(result_root.glob("*.mat")))
    return {
        "result_mat_total": direct + sum(categories.values()),
        "direct_result_mat": direct,
        "category_counts": categories,
    }


def audit_legacy_16(legacy_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = legacy_root / "Data_mat" / "16sc"
    result_root = legacy_root / "Results_mat" / "Bfusion"
    source_by_hash: dict[str, list[Path]] = {}
    for path in data_root.glob("*.mat"):
        try:
            digest = array_digest(grouped_16(path))
        except (KeyError, ValueError):
            continue
        source_by_hash.setdefault(digest, []).append(path)

    current_predictions = pd.read_csv(
        TASK / "results" / "BS02_16t" / "full" / "predictions.csv"
    )
    rows: list[dict[str, object]] = []
    agreement_rows: list[dict[str, object]] = []
    root = resolve_dataset_root("jbhi16")
    for subject_id in available_subject_ids("jbhi16", root):
        digest = array_digest(load_subject("jbhi16", subject_id, root).x)
        matches = source_by_hash.get(digest, [])
        if len(matches) != 1:
            raise RuntimeError(f"{subject_id} has {len(matches)} exact legacy 16-target source matches")
        result_path = result_root / f"{matches[0].stem}_results_Bfusion.mat"
        if not result_path.is_file():
            raise FileNotFoundError(f"Historical final result is missing for anonymous {subject_id}")
        payload = loadmat(result_path, simplify_cells=True)
        windows = np.asarray(payload["test_lengths"], dtype=np.float64).reshape(-1)
        for method, field in METHOD_FIELDS.items():
            result = payload["results"][field]
            accuracies = np.asarray(result["acc"], dtype=np.float64).reshape(-1) / 100.0
            historical_predictions = np.atleast_1d(result["pred_labels"])
            for window_index, (window, accuracy) in enumerate(zip(windows, accuracies)):
                rows.append(
                    {
                        "subject": subject_id,
                        "method": method,
                        "window_seconds": float(window),
                        "legacy_accuracy": float(accuracy),
                    }
                )
                current = current_predictions[
                    (current_predictions["subject"] == subject_id)
                    & (current_predictions["method"] == method)
                    & np.isclose(current_predictions["window_seconds"], window)
                ].sort_values(["block", "trial_index"])
                historical = np.asarray(historical_predictions[window_index]).reshape(-1).astype(np.int64)
                if len(current) != 96 or historical.shape != (96,):
                    raise RuntimeError(f"Prediction count mismatch for {subject_id} {method} {window:g}s")
                agreement_rows.append(
                    {
                        "subject": subject_id,
                        "method": method,
                        "window_seconds": float(window),
                        "prediction_agreement": float(np.mean(current["pred"].to_numpy() == historical)),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(agreement_rows)


def parse_plot_sessions(legacy_root: Path) -> list[tuple[str, str | None]]:
    script = legacy_root / "Results_Plot" / "_old" / "plot_exp7_2.m"
    text = script.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"results_([A-Za-z0-9]+)_35sc_allmethod(?:_(\d{8}))?")
    sessions: list[tuple[str, str | None]] = []
    for code, date in pattern.findall(text):
        item = (code.lower(), date or None)
        if item not in sessions:
            sessions.append(item)
    if len(sessions) != 5:
        raise RuntimeError(f"Expected five legacy plot sessions, found {len(sessions)}")
    return sessions


def source_code(path: Path) -> str | None:
    match = re.match(r"\d{8}_([^_]+)_dual35", path.stem, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def select_plot_source(data_root: Path, code: str, date: str | None) -> Path:
    candidates = [
        path
        for path in data_root.glob(f"*_{code}_dual35*.mat")
        if "first4" not in path.stem.lower() and re.match(r"\d{8}_", path.name)
    ]
    if date is not None:
        candidates = [path for path in candidates if path.name.startswith(date)]
    if not candidates:
        raise FileNotFoundError("A legacy plot source session could not be resolved")
    return max(candidates, key=lambda path: path.name[:8])


def discover_legacy_35_sessions(legacy_root: Path) -> list[tuple[str, np.ndarray]]:
    data_root = legacy_root / "Data_mat" / "35sc"
    current_by_hash = current_subject_hashes("jbhi35")
    code_to_current: dict[str, str] = {}
    for path in data_root.glob("*.mat"):
        code = source_code(path)
        if code is None or "first4" in path.stem.lower():
            continue
        try:
            subject_id = current_by_hash.get(array_digest(grouped_35(path)))
        except (KeyError, ValueError):
            continue
        if subject_id is not None:
            code_to_current[code] = subject_id

    output = []
    for code, date in parse_plot_sessions(legacy_root):
        values = grouped_35(select_plot_source(data_root, code, date))
        subject_id = current_by_hash.get(array_digest(values))
        if subject_id is None:
            current_id = code_to_current.get(code)
            if current_id is None:
                raise RuntimeError("An alternate legacy session could not be mapped to an anonymous current subject")
            subject_id = f"{current_id}-alt"
        output.append((subject_id, values))
    if len({subject_id for subject_id, _ in output}) != 5:
        raise RuntimeError("Legacy 35-target plot sessions are not unique after anonymization")
    return output


def run_etrca(subject_id: str, values: np.ndarray) -> list[dict[str, object]]:
    subject = PrivateSSVEPSubject(
        dataset="jbhi35",
        subject_id=subject_id,
        x=values,
        sampling_rate=1000,
        channels=OCCIPITAL9,
        targets=tuple(range(35)),
        blocks=tuple(range(6)),
        metadata={},
    )
    full_epochs = analysis_epochs(subject, max(AUDIT_WINDOWS), n_bands=5)
    rows = []
    for window in AUDIT_WINDOWS:
        epochs = full_epochs[..., : int(round(window * subject.sampling_rate))]
        predictions = []
        for test_block in range(6):
            train_blocks = [block for block in range(6) if block != test_block]
            train_x = epochs[:, train_blocks].reshape(-1, 5, 9, epochs.shape[-1])
            train_y = np.repeat(np.arange(35), len(train_blocks))
            model = TRCA(n_fbs=5, ensemble=True).fit(train_x, train_y)
            predicted, _ = model.predict(epochs[:, test_block])
            predictions.extend(predicted.tolist())
        truth = np.tile(np.arange(35), 6)
        rows.append(
            {
                "session": subject_id,
                "window_seconds": window,
                "accuracy": float(np.mean(np.asarray(predictions) == truth)),
                "prediction_count": len(predictions),
            }
        )
    return rows


def audit_legacy_35(
    legacy_root: Path,
    output: Path,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = discover_legacy_35_sessions(legacy_root)
    receiver_path = output / "jbhi35_legacy_receiver.csv"
    expected_sessions = {subject_id for subject_id, _ in sessions}
    if receiver_path.is_file() and not force:
        receiver = pd.read_csv(receiver_path)
        complete = (
            set(receiver["session"]) == expected_sessions
            and set(np.round(receiver["window_seconds"], 6)) == set(AUDIT_WINDOWS)
            and len(receiver) == 5 * len(AUDIT_WINDOWS)
        )
    else:
        complete = False
    if not complete:
        receiver = pd.DataFrame(
            [row for subject_id, values in sessions for row in run_etrca(subject_id, values)]
        )

    pairs = load_target_frequency_pairs("jbhi35")
    current_root = resolve_dataset_root("jbhi35")
    receiver_lookup = receiver.set_index(["session", "window_seconds"])["accuracy"].to_dict()
    phase_rows = []
    for alternate_id, alternate_values in sessions:
        if not alternate_id.endswith("-alt"):
            continue
        current_id = alternate_id.removesuffix("-alt")
        current_values = load_subject("jbhi35", current_id, current_root).x
        for session_id, values in ((current_id, current_values), (alternate_id, alternate_values)):
            power = exact_log_power(values, 1000)
            temporal = signal.detrend(values, axis=-1, type="constant")
            if session_id.endswith("-alt"):
                accuracy = receiver_lookup[(session_id, 2.0)]
            else:
                current_subject = pd.read_csv(
                    TASK / "results" / "BS03_35t" / "full" / "subject.csv"
                )
                row = current_subject[
                    (current_subject["subject"] == session_id)
                    & (current_subject["method"] == "ETRCA")
                    & np.isclose(current_subject["window_seconds"], 2.0)
                ]
                accuracy = float(row["accuracy"].iloc[0])
            phase_rows.append(
                {
                    "session": session_id,
                    "etrca_2s_accuracy": accuracy,
                    "frequency_effect_oz_db": expected_frequency_effect(power, pairs, 7),
                    "expected_frequency_itpc_oz": expected_frequency_itpc_oz(values, 1000, pairs, 7),
                    "unordered_frequency_set_accuracy": unordered_code_accuracy(power, pairs),
                    "spectral_cross_block_label_accuracy": leave_one_block_accuracy(power),
                    "temporal_cross_block_label_accuracy": leave_one_block_accuracy(temporal),
                }
            )
    return receiver, pd.DataFrame(phase_rows)


def embc9_filter(values: np.ndarray, band_index: int) -> np.ndarray:
    passbands = (6.0, 14.0, 22.0)
    stopbands = (4.0, 10.0, 16.0)
    order, critical = signal.cheb1ord(
        [passbands[band_index], 90.0],
        [stopbands[band_index], 100.0],
        3.0,
        40.0,
        fs=250.0,
    )
    numerator, denominator = signal.cheby1(
        order,
        0.5,
        critical,
        btype="bandpass",
        fs=250.0,
    )
    return signal.filtfilt(numerator, denominator, values, axis=-1)


def embc9_filterbank(values: np.ndarray, cascade: bool) -> np.ndarray:
    output = []
    current = values
    for band_index in range(3):
        current = embc9_filter(current if cascade else values, band_index)
        output.append(current)
    return np.stack(output, axis=2)


def embc9_trca_accuracy(train_bands: np.ndarray, test_bands: np.ndarray) -> tuple[float, int]:
    labels = np.arange(9, dtype=np.int64)
    predictions = []
    for test_block in range(20):
        train_blocks = [block for block in range(20) if block != test_block]
        train_x = train_bands[:, train_blocks].reshape(-1, 3, 9, train_bands.shape[-1])
        train_y = np.repeat(labels, len(train_blocks))
        model = TRCA(n_fbs=3, ensemble=False).fit(train_x, train_y)
        predicted, _ = model.predict(test_bands[:, test_block])
        predictions.extend(predicted.tolist())
    truth = np.tile(labels, 20)
    return float(np.mean(np.asarray(predictions) == truth)), len(predictions)


def embc9_part_accuracies(filter_bands: int, windows: tuple[float, ...]) -> pd.DataFrame:
    parts = TASK / "results" / "BS01_embc_9t" / "full" / "parts"
    rows = []
    for subject_id in available_subject_ids("embc9", resolve_dataset_root("embc9")):
        for window in windows:
            window_ms = int(round(window * 1000.0))
            path = parts / f"{subject_id}_trca_w{window_ms:04d}_fb{filter_bands}_unit.json"
            if not path.is_file():
                raise FileNotFoundError(f"Anonymous historical Arena part is missing: {path.name}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "complete" or int(payload.get("prediction_count", -1)) != 180:
                raise RuntimeError(f"Anonymous Arena part is incomplete: {path.name}")
            rows.append(
                {
                    "subject": subject_id,
                    "window_seconds": window,
                    "accuracy": float(payload["accuracy"]),
                }
            )
    return pd.DataFrame(rows)


def audit_embc9_receiver_2s(historical: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_csv(
        TASK / "results" / "BS01_embc_9t" / "full" / "subject.csv"
    )
    current = current[(current["method"] == "TRCA") & np.isclose(current["window_seconds"], 2.0)]
    current_lookup = current.set_index("subject")["accuracy"].to_dict()
    single_band_lookup = embc9_part_accuracies(1, (2.0,)).set_index("subject")["accuracy"].to_dict()
    historical_2s = historical[np.isclose(historical["window_seconds"], 2.0)].set_index("subject")
    root = resolve_dataset_root("embc9")
    rows = []
    for subject_id in available_subject_ids("embc9", root):
        subject = load_subject("embc9", subject_id, root)
        direct_decimation = subject.x[..., ::4][..., :500]
        independent = embc9_filterbank(direct_decimation, cascade=False)
        cascaded_training = embc9_filterbank(direct_decimation, cascade=True)
        independent_accuracy, independent_count = embc9_trca_accuracy(independent, independent)
        asymmetric_accuracy, asymmetric_count = embc9_trca_accuracy(cascaded_training, independent)
        historical_row = historical_2s.loc[subject_id]
        if not np.isclose(independent_accuracy, current_lookup[subject_id]):
            raise RuntimeError(f"Anonymous {subject_id} three-band audit does not match the canonical run")
        rows.append(
            {
                "subject": subject_id,
                "paper_included": bool(historical_row["paper_included"]),
                "legacy_matlab_accuracy": float(historical_row["legacy_accuracy"]),
                "arena_single_band_accuracy": float(single_band_lookup[subject_id]),
                "arena_3band_accuracy": float(current_lookup[subject_id]),
                "independent_3band_accuracy": independent_accuracy,
                "legacy_asymmetric_3band_accuracy": asymmetric_accuracy,
                "independent_prediction_count": independent_count,
                "legacy_asymmetric_prediction_count": asymmetric_count,
                "decimation": "direct_stride_4",
                "filterbank_definition": "legacy_dynamic_chebyshev_3band",
            }
        )
    return pd.DataFrame(rows)


def plot_legacy_9(
    comparison: pd.DataFrame,
    receiver: pd.DataFrame,
    figures: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.35), constrained_layout=True)
    included = comparison[comparison["paper_included"]]
    means = included.groupby("window_seconds")[[
        "legacy_accuracy",
        "arena_3band_accuracy",
        "arena_single_band_accuracy",
    ]].mean() * 100.0
    axes[0].plot(means.index, means["legacy_accuracy"], "--o", color="#C0504D", label="Legacy MATLAB")
    axes[0].plot(means.index, means["arena_3band_accuracy"], "-o", color="#4472C4", label="Arena 3-band")
    axes[0].plot(
        means.index,
        means["arena_single_band_accuracy"],
        ":s",
        color="#7F7F7F",
        label="Previous Arena 1-band",
    )
    axes[0].set_xlabel("Window (s)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Paper-included 7 subjects")
    axes[0].set_xticks(EMBC9_COMMON_WINDOWS)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    positions = np.arange(len(receiver))
    width = 0.36
    axes[1].bar(
        positions - width / 2,
        100.0 * receiver["legacy_matlab_accuracy"],
        width,
        color="#C0504D",
        label="Legacy MATLAB",
    )
    axes[1].bar(
        positions + width / 2,
        100.0 * receiver["arena_3band_accuracy"],
        width,
        color="#4472C4",
        label="Arena 3-band",
    )
    labels = [
        f"{row.subject}*" if not row.paper_included else row.subject
        for row in receiver.itertuples(index=False)
    ]
    axes[1].set_xticks(positions, labels, rotation=35)
    axes[1].set_ylabel("2 s accuracy (%)")
    axes[1].set_title("Anonymous subjects (* excluded in paper mean)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    columns = (
        "arena_single_band_accuracy",
        "arena_3band_accuracy",
        "legacy_asymmetric_3band_accuracy",
        "legacy_matlab_accuracy",
    )
    names = ("Arena\n1 band", "Arena\n3 bands", "Legacy-code\nasymmetry", "Recovered\nMAT")
    paper = receiver[receiver["paper_included"]]
    all_values = [100.0 * receiver[column].mean() for column in columns]
    paper_values = [100.0 * paper[column].mean() for column in columns]
    protocol_positions = np.arange(len(columns))
    axes[2].bar(protocol_positions - width / 2, all_values, width, color="#70AD47", label="All 8")
    axes[2].bar(protocol_positions + width / 2, paper_values, width, color="#8064A2", label="Paper 7")
    axes[2].set_xticks(protocol_positions, names)
    axes[2].set_ylabel("2 s accuracy (%)")
    axes[2].set_title("Protocol decomposition")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()
    fig.suptitle("EMBC9: recovered paper results versus current Arena protocol")
    fig.savefig(figures / "embc9_legacy_vs_arena.png", dpi=210)
    plt.close(fig)


def plot_legacy_16(comparison: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharey=True, constrained_layout=True)
    for axis, method in zip(axes, METHOD_FIELDS):
        part = comparison[comparison["method"] == method]
        means = part.groupby("window_seconds")[["legacy_accuracy", "arena_accuracy"]].mean() * 100.0
        axis.plot(means.index, means["legacy_accuracy"], "--o", label="Legacy MATLAB", color="#C0504D")
        axis.plot(means.index, means["arena_accuracy"], "-o", label="Arena", color="#4472C4")
        axis.set_title(method)
        axis.set_xlabel("Window (s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(loc="lower right")
    fig.suptitle("JBHI16: exact-source legacy result comparison (13 anonymous subjects)")
    fig.savefig(figures / "jbhi16_legacy_vs_arena.png", dpi=210)
    plt.close(fig)


def plot_legacy_35(receiver: pd.DataFrame, figures: Path) -> None:
    current = pd.read_csv(
        TASK / "results" / "BS03_35t" / "full" / "subject.csv"
    )
    current = current[(current["method"] == "ETRCA") & current["window_seconds"].isin(AUDIT_WINDOWS)]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), constrained_layout=True)
    for session, part in receiver.groupby("session", sort=False):
        axes[0].plot(part["window_seconds"], 100.0 * part["accuracy"], "-o", alpha=0.72, label=session)
    old_mean = receiver.groupby("window_seconds")["accuracy"].mean()
    current_mean = current.groupby("window_seconds")["accuracy"].mean()
    axes[0].plot(old_mean.index, 100.0 * old_mean, "-o", color="black", linewidth=2.6, label="Legacy-5 mean")
    axes[0].plot(
        current_mean.index,
        100.0 * current_mean,
        "--s",
        color="#C0504D",
        linewidth=2.4,
        label="Current-6 mean",
    )
    axes[0].set_xlabel("Window (s)")
    axes[0].set_ylabel("eTRCA accuracy (%)")
    axes[0].set_title("Cohort curves")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)

    current_2s = current[np.isclose(current["window_seconds"], 2.0)].sort_values("subject")
    old_2s = receiver[np.isclose(receiver["window_seconds"], 2.0)]
    axes[1].bar(np.arange(len(current_2s)), 100.0 * current_2s["accuracy"], color="#4472C4", label="Current-6")
    offset = len(current_2s) + 1
    axes[1].bar(offset + np.arange(len(old_2s)), 100.0 * old_2s["accuracy"], color="#70AD47", label="Legacy-5")
    labels = current_2s["subject"].tolist() + [""] + old_2s["session"].tolist()
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=42, ha="right")
    axes[1].set_ylabel("2 s eTRCA accuracy (%)")
    axes[1].set_title("Session composition")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.suptitle("JBHI35: legacy paper-workflow sessions versus current canonical sessions")
    fig.savefig(figures / "jbhi35_legacy_cohort_comparison.png", dpi=210)
    plt.close(fig)


def plot_phase_audit(phase: pd.DataFrame, figures: Path) -> None:
    metrics = (
        ("etrca_2s_accuracy", "eTRCA 2 s", 100.0, "%"),
        ("frequency_effect_oz_db", "Oz frequency effect", 1.0, "dB"),
        ("expected_frequency_itpc_oz", "Oz expected-frequency ITPC", 1.0, "unit"),
        ("unordered_frequency_set_accuracy", "Frequency-set agreement", 100.0, "%"),
        ("temporal_cross_block_label_accuracy", "Temporal repeatability", 100.0, "%"),
    )
    fig, axes = plt.subplots(1, len(metrics), figsize=(15.8, 4.1), constrained_layout=True)
    colors = ["#4472C4" if not value.endswith("-alt") else "#70AD47" for value in phase["session"]]
    for axis, (column, title, scale, unit) in zip(axes, metrics):
        axis.bar(phase["session"], scale * phase[column], color=colors)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.tick_params(axis="x", rotation=38)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("JBHI35: current low sessions versus alternate legacy sessions")
    fig.savefig(figures / "jbhi35_current_vs_alternate_signal.png", dpi=210)
    plt.close(fig)


def write_report(
    output: Path,
    comparison9: pd.DataFrame,
    receiver9: pd.DataFrame,
    inventory9: dict[str, int],
    inventory_jbhi: dict[str, object],
    comparison16: pd.DataFrame,
    agreement16: pd.DataFrame,
    receiver35: pd.DataFrame,
    phase35: pd.DataFrame,
) -> None:
    lines = [
        "# EMBC/JBHI legacy backup audit",
        "",
        "All persisted tables and figures use anonymous Arena IDs only. The configured external backup was read in place.",
        "",
        "## EMBC9",
        "",
        "All eight current transfer arrays have one exact labeled legacy source match and one recovered base TRCA result MAT.",
        "The final plotting script includes seven subjects in the paper mean and excludes one anonymous subject, S02.",
        "",
        "| Cohort | Legacy 2 s | Previous Arena 1-band | Arena 3-band | Legacy-asymmetric 3-band |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, part in (("All 8", receiver9), ("Paper 7", receiver9[receiver9["paper_included"]])):
        lines.append(
            f"| {label} | {100 * part['legacy_matlab_accuracy'].mean():.4f}% | "
            f"{100 * part['arena_single_band_accuracy'].mean():.4f}% | "
            f"{100 * part['arena_3band_accuracy'].mean():.4f}% | "
            f"{100 * part['legacy_asymmetric_3band_accuracy'].mean():.4f}% |"
        )
    common = comparison9[comparison9["paper_included"]].groupby("window_seconds")[[
        "legacy_accuracy",
        "arena_3band_accuracy",
        "arena_single_band_accuracy",
    ]].mean()
    lines.extend(
        [
            "",
            f"At 4 s, the recovered paper-seven mean is {100 * common.loc[4.0, 'legacy_accuracy']:.4f}% "
            f"versus {100 * common.loc[4.0, 'arena_3band_accuracy']:.4f}% for the corrected Arena three-band run "
            f"and {100 * common.loc[4.0, 'arena_single_band_accuracy']:.4f}% for the previous single-band run.",
            "The recovered MATLAB pipeline uses direct 4x decimation and three dynamic Chebyshev receiver bands. "
            "Its training function cascades the bands while its test function applies them independently. "
            "Restoring direct decimation and three bands explains most of the single-band reproduction gap.",
            "The stored per-subject prediction vectors are not a common-window anchor: their accuracies match "
            "different or multiple curve windows across subjects, so this audit does not compare those decisions to Arena.",
            "",
            f"The old project contains {inventory9['result_mat_total']} result MAT files in the final figure folder: "
            f"{inventory9['accuracy_itr_base']} base, {inventory9['accuracy_itr_opE']} opE, "
            f"{inventory9['accuracy_itr_opG']} opG, {inventory9['accuracy_itr_opEG']} opEG, and "
            f"{inventory9['accuracy_itr_gamma']} gamma accuracy/ITR variants. It also contains "
            f"{inventory9['frequency_analysis_mat']} frequency-analysis MAT files, "
            f"{inventory9['frequency_average_mat']} averaged frequency MAT files, "
            f"{inventory9['matlab_figure_sources']} MATLAB figure sources, and "
            f"{inventory9['manuscript_pdfs']} manuscript PDFs.",
            "",
        "## JBHI16",
        "",
        "All 13 current transfer MAT arrays have one exact legacy source match and one historical final-result MAT.",
        "",
        "| Method | Window | Legacy | Arena | Delta |",
        "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHOD_FIELDS:
        part = comparison16[comparison16["method"] == method]
        means = part.groupby("window_seconds")[["legacy_accuracy", "arena_accuracy"]].mean()
        for window in (0.2, 0.4, 2.0):
            row = means.loc[window]
            lines.append(
                f"| {method} | {window:.1f} s | {100 * row['legacy_accuracy']:.4f}% | "
                f"{100 * row['arena_accuracy']:.4f}% | {100 * (row['arena_accuracy'] - row['legacy_accuracy']):+.4f} pp |"
            )
    lines.extend(
        [
            "",
            f"Mean historical/Arena hard-decision agreement across 390 subject-method-window units: "
            f"{100 * agreement16['prediction_agreement'].mean():.2f}%.",
            f"The complete published snapshot contains {inventory_jbhi['result_mat_total']} JBHI result MAT files "
            f"({inventory_jbhi['direct_result_mat']} directly under Results_mat plus categorized variants). "
            "Only protocol-resolved final Bfusion anchors are used in the numerical comparison above; parameter sweeps "
            "and TDCA/gamma variants are inventoried but are not silently promoted to paper results.",
            "",
            "## JBHI35",
            "",
            "The legacy plotting workflow names five sessions. Three overlap the current canonical set; two are alternate sessions, and the current S02 session is absent.",
            "",
            "| Session | 0.4 s | 1.0 s | 2.0 s |",
            "|---|---:|---:|---:|",
        ]
    )
    pivot = receiver35.pivot(index="session", columns="window_seconds", values="accuracy")
    for session, row in pivot.iterrows():
        lines.append(
            f"| {session} | {100 * row[0.4]:.2f}% | {100 * row[1.0]:.2f}% | {100 * row[2.0]:.2f}% |"
        )
    means = receiver35.groupby("window_seconds")["accuracy"].mean()
    lines.append(
        f"| Legacy-5 mean | {100 * means.loc[0.4]:.2f}% | {100 * means.loc[1.0]:.2f}% | {100 * means.loc[2.0]:.2f}% |"
    )
    lines.extend(
        [
            "",
            "The two alternate sessions are strong under the same standard Arena eTRCA definition. Their current counterparts retain above-chance frequency-set evidence but lose cross-block phase locking and temporal repeatability, so the near-chance current result is session-specific rather than a stable participant trait.",
            "",
            "| Session | eTRCA 2 s | Oz effect | Oz ITPC | Frequency-set | Temporal repeatability |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in phase35.iterrows():
        lines.append(
            f"| {row['session']} | {100 * row['etrca_2s_accuracy']:.2f}% | "
            f"{row['frequency_effect_oz_db']:.3f} dB | {row['expected_frequency_itpc_oz']:.3f} | "
            f"{100 * row['unordered_frequency_set_accuracy']:.2f}% | "
            f"{100 * row['temporal_cross_block_label_accuracy']:.2f}% |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    embc9_legacy_root = configured_path(args.config, "ssvep_embc_9target_legacy_private")
    jbhi_legacy_root = configured_path(args.config, "ssvep_embc_jbhi_legacy_private")
    output = args.output
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    legacy9 = audit_legacy_9(embc9_legacy_root)
    inventory9 = embc9_inventory(embc9_legacy_root)
    inventory_jbhi = jbhi_inventory(jbhi_legacy_root)
    current9 = pd.read_csv(
        TASK / "results" / "BS01_embc_9t" / "full" / "subject.csv"
    )
    current9 = current9[
        (current9["method"] == "TRCA") & current9["window_seconds"].isin(EMBC9_COMMON_WINDOWS)
    ][["subject", "window_seconds", "accuracy"]].rename(columns={"accuracy": "arena_3band_accuracy"})
    single_band9 = embc9_part_accuracies(1, EMBC9_COMMON_WINDOWS).rename(
        columns={"accuracy": "arena_single_band_accuracy"}
    )
    legacy9["window_seconds"] = legacy9["window_seconds"].round(6)
    current9["window_seconds"] = current9["window_seconds"].round(6)
    single_band9["window_seconds"] = single_band9["window_seconds"].round(6)
    comparison9 = legacy9.merge(
        current9,
        on=["subject", "window_seconds"],
        validate="one_to_one",
    ).merge(single_band9, on=["subject", "window_seconds"], validate="one_to_one")
    comparison9["delta_accuracy"] = comparison9["arena_3band_accuracy"] - comparison9["legacy_accuracy"]
    receiver9 = audit_embc9_receiver_2s(legacy9)

    legacy16, agreement16 = audit_legacy_16(jbhi_legacy_root)
    current16 = pd.read_csv(
        TASK / "results" / "BS02_16t" / "full" / "subject.csv"
    )
    current16 = current16[current16["method"].isin(METHOD_FIELDS)][
        ["subject", "method", "window_seconds", "accuracy"]
    ].rename(columns={"accuracy": "arena_accuracy"})
    legacy16["window_seconds"] = legacy16["window_seconds"].round(6)
    agreement16["window_seconds"] = agreement16["window_seconds"].round(6)
    current16["window_seconds"] = current16["window_seconds"].round(6)
    comparison16 = legacy16.merge(current16, on=["subject", "method", "window_seconds"], validate="one_to_one")
    comparison16["delta_accuracy"] = comparison16["arena_accuracy"] - comparison16["legacy_accuracy"]

    receiver35, phase35 = audit_legacy_35(jbhi_legacy_root, output, args.force)
    legacy9.to_csv(output / "embc9_legacy_trca_curves.csv", index=False)
    comparison9.to_csv(output / "embc9_legacy_comparison.csv", index=False)
    receiver9.to_csv(output / "embc9_receiver_protocol_2s.csv", index=False)
    comparison16.to_csv(output / "jbhi16_legacy_comparison.csv", index=False)
    agreement16.to_csv(output / "jbhi16_prediction_agreement.csv", index=False)
    receiver35.to_csv(output / "jbhi35_legacy_receiver.csv", index=False)
    phase35.to_csv(output / "jbhi35_current_vs_alternate_signal.csv", index=False)

    plot_legacy_9(comparison9, receiver9, figures)
    plot_legacy_16(comparison16, figures)
    plot_legacy_35(receiver35, figures)
    plot_phase_audit(phase35, figures)
    write_report(
        output,
        comparison9,
        receiver9,
        inventory9,
        inventory_jbhi,
        comparison16,
        agreement16,
        receiver35,
        phase35,
    )

    manifest = {
        "status": "complete",
        "identity_policy": "anonymous Arena IDs only",
        "external_backup_modified": False,
        "embc9_exact_source_matches": int(legacy9["subject"].nunique()),
        "embc9_legacy_curve_units": int(len(legacy9)),
        "embc9_comparison_units": int(len(comparison9)),
        "embc9_receiver_protocol_units": int(len(receiver9)),
        "embc9_inventory": inventory9,
        "jbhi_inventory": inventory_jbhi,
        "jbhi16_exact_source_matches": int(comparison16["subject"].nunique()),
        "jbhi16_comparison_units": int(len(comparison16)),
        "jbhi16_prediction_units": int(len(agreement16)),
        "jbhi35_legacy_sessions": int(receiver35["session"].nunique()),
        "jbhi35_receiver_units": int(len(receiver35)),
        "outputs": [
            "report.md",
            "embc9_legacy_trca_curves.csv",
            "embc9_legacy_comparison.csv",
            "embc9_receiver_protocol_2s.csv",
            "jbhi16_legacy_comparison.csv",
            "jbhi16_prediction_agreement.csv",
            "jbhi35_legacy_receiver.csv",
            "jbhi35_current_vs_alternate_signal.csv",
            "figures/embc9_legacy_vs_arena.png",
            "figures/jbhi16_legacy_vs_arena.png",
            "figures/jbhi35_legacy_cohort_comparison.png",
            "figures/jbhi35_current_vs_alternate_signal.png",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
