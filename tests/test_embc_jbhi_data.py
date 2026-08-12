from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from vep_arena.data.embc_jbhi import (
    JBHI35_HISTORICAL5_PACKAGE_CONFIG_KEY,
    JBHI35_HISTORICAL5_SUBJECT_IDS,
    JBHI35_HISTORICAL5_SUBJECT_MANIFEST_CONFIG_KEY,
    analysis_epochs,
    available_subject_ids,
    dataset_spec,
    load_jbhi35_historical5_subject,
    load_jbhi35_target_frequencies,
    load_subject,
    load_target_frequency_pairs,
    resolve_jbhi35_historical5_package,
    validate_jbhi35_historical5_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_historical5_fixture(
    tmp_path: Path,
    *,
    subject_ids: tuple[str, ...] = JBHI35_HISTORICAL5_SUBJECT_IDS,
    mapping_subject_ids: tuple[str, ...] = JBHI35_HISTORICAL5_SUBJECT_IDS,
    target_labels: np.ndarray | None = None,
) -> tuple[Path, Path, Path]:
    package_root = tmp_path / "JBHI35_5Subject_Reproduction_Package_20260728"
    data_dir = package_root / "data"
    manifests_dir = package_root / "manifests"
    data_dir.mkdir(parents=True)
    manifests_dir.mkdir()
    labels = np.arange(1, 36, dtype=np.int64) if target_labels is None else target_labels
    channel_list = np.asarray([("Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2")], dtype=object)

    data_paths: list[Path] = []
    for subject_index, subject_id in enumerate(JBHI35_HISTORICAL5_SUBJECT_IDS, start=1):
        data = np.zeros((9, 2000, 35, 6), dtype=np.float64)
        for target in range(35):
            for block in range(6):
                data[0, 0, target, block] = subject_index * 10000 + target * 100 + block
        path = data_dir / f"{subject_id}.mat"
        savemat(
            path,
            {
                "data": data,
                "Fs": 1000,
                "num_channels": 9,
                "num_samples": 2000,
                "num_targets": 35,
                "num_blocks": 6,
                "channel_list": channel_list,
                "target_labels": labels,
            },
            do_compression=True,
        )
        data_paths.append(path)

    private_manifest = tmp_path / "historical_subject_manifest.csv"
    subject_rows = ["subject_id,relative_path,expected_sha256,matlab_signed_accuracy"]
    for subject_id in subject_ids:
        path = data_dir / f"{subject_id}.mat"
        subject_rows.append(f"{subject_id},data/{path.name},{_sha256(path)},0.5")
    private_manifest.write_text("\n".join(subject_rows), encoding="utf-8")
    (manifests_dir / "data_mapping.csv").write_text(
        "output_mat\n" + "\n".join(f"{subject_id}.mat" for subject_id in mapping_subject_ids),
        encoding="utf-8",
    )

    extra_paths: list[Path] = []
    for index in range(45):
        path = package_root / "metadata" / f"entry_{index:02d}.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"fixture-{index}\n", encoding="ascii")
        extra_paths.append(path)
    package_files = [*data_paths, manifests_dir / "data_mapping.csv", *extra_paths]
    assert len(package_files) == 51
    manifest_rows = ["relative_path,size_bytes,sha256"]
    for path in package_files:
        relative_path = path.relative_to(package_root).as_posix()
        manifest_rows.append(f"{relative_path},{path.stat().st_size},{_sha256(path)}")
    package_sha = manifests_dir / "PACKAGE_SHA256.csv"
    package_sha.write_text("\n".join(manifest_rows), encoding="utf-8")

    config_path = tmp_path / "local_paths.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": {
                    JBHI35_HISTORICAL5_PACKAGE_CONFIG_KEY: str(package_root),
                    JBHI35_HISTORICAL5_SUBJECT_MANIFEST_CONFIG_KEY: str(private_manifest),
                }
            }
        ),
        encoding="utf-8",
    )
    return package_root, private_manifest, config_path


def test_embc_loader_uses_anonymous_ids_and_occipital_channels(tmp_path: Path) -> None:
    data = np.arange(13 * 4000 * 9 * 20, dtype=np.float64).reshape(13, 4000, 9, 20)
    savemat(tmp_path / "private_name.mat", {"data": data, "Fs": 1000, "num_targets": 9})

    assert available_subject_ids("embc9", tmp_path) == ("S01",)
    subject = load_subject("embc9", "S01", tmp_path)

    assert subject.x.shape == (9, 20, 9, 4000)
    assert subject.subject_id == "S01"
    assert subject.metadata["identity_metadata_exposed"] is False
    assert np.array_equal(subject.x[0, 0, 0], data[4, :, 0, 0])


def test_jbhi16_loader_preserves_target_block_order(tmp_path: Path) -> None:
    eeg = np.zeros((9, 4001, 96), dtype=np.float64)
    labels = np.tile(np.arange(1, 17), 6)
    for trial, label in enumerate(labels):
        eeg[:, :, trial] = label * 100 + trial // 16
    channels = np.asarray([[name] for name in ("PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2")], dtype=object).T
    savemat(
        tmp_path / "private_name.mat",
        {"eegdata": eeg, "sample_rate": 1000, "label_list": labels, "channel_list": channels},
    )

    subject = load_subject("jbhi16", "S01", tmp_path)

    assert subject.x.shape == (16, 6, 9, 4001)
    assert np.all(subject.x[4, 3] == 503)
    pairs = load_target_frequency_pairs("jbhi16")
    assert len(pairs) == 16
    assert len(set(pairs)) == 16
    assert pairs[6] == (0.0, 0.0)


def test_analysis_epochs_separates_paper_resampling_and_receiver_filterbank(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    embc = rng.normal(size=(13, 4000, 9, 20))
    savemat(tmp_path / "subject.mat", {"data": embc, "Fs": 1000, "num_targets": 9})
    embc_subject = load_subject("embc9", "S01", tmp_path)
    epochs = analysis_epochs(embc_subject, 0.4)
    first_band = analysis_epochs(embc_subject, 0.4, n_bands=1)

    assert epochs.shape == (9, 20, 3, 9, 100)
    assert first_band.shape == (9, 20, 1, 9, 100)
    assert np.allclose(epochs[:, :, :1], first_band)
    assert dataset_spec("embc9").itr_shift_seconds == 2.13
    assert dataset_spec("embc9").receiver_filterbank["enabled"] is True


def test_jbhi35_loader_groups_explicit_labels_without_identity_metadata(tmp_path: Path) -> None:
    eeg = np.zeros((9, 2001, 210), dtype=np.float64)
    labels = np.empty(210, dtype=np.int64)
    for block in range(6):
        block_labels = np.roll(np.arange(1, 36), block + 1)
        labels[block * 35 : (block + 1) * 35] = block_labels
        for offset, label in enumerate(block_labels):
            eeg[:, :, block * 35 + offset] = label * 100 + block
    channels = np.asarray([("PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2")], dtype=object)
    savemat(
        tmp_path / "S01.mat",
        {"eegdata": eeg, "sample_rate": 1000, "label_list": labels, "channel_list": channels},
    )
    mapping = ["target_label_1based,target_label_0based,grid_row_1based,grid_col_1based,left_freq_hz,right_freq_hz"]
    mapping.extend(f"{label},{label - 1},1,1,11,{label + 20}" for label in range(1, 36))
    (tmp_path / "target_mapping_35.csv").write_text("\n".join(mapping), encoding="utf-8")

    subject = load_subject("jbhi35", "S01", tmp_path)

    assert subject.x.shape == (35, 6, 9, 2000)
    assert np.all(subject.x[4, 3] == 503)
    assert subject.targets == tuple(range(35))
    assert subject.blocks == tuple(range(6))
    assert subject.metadata["model_label_base"] == 0
    assert subject.metadata["identity_metadata_exposed"] is False
    assert load_jbhi35_target_frequencies(tmp_path)[0] == (11.0, 21.0)


def test_jbhi35_loader_reads_anonymous_canonical_4d(tmp_path: Path) -> None:
    data = np.zeros((9, 2000, 35, 6), dtype=np.float64)
    for target in range(35):
        for block in range(6):
            data[:, :, target, block] = target * 100 + block
    savemat(
        tmp_path / "S01.mat",
        {
            "data": data,
            "Fs": 1000,
            "num_channels": 9,
            "num_samples": 2000,
            "num_targets": 35,
            "num_blocks": 6,
        },
    )
    mapping = ["target_label_1based,target_label_0based,grid_row_1based,grid_col_1based,left_freq_hz,right_freq_hz"]
    mapping.extend(f"{label},{label - 1},1,1,11,{label + 20}" for label in range(1, 36))
    (tmp_path / "target_mapping_35.csv").write_text("\n".join(mapping), encoding="utf-8")

    subject = load_subject("jbhi35", "S01", tmp_path)

    assert available_subject_ids("jbhi35", tmp_path) == ("S01",)
    assert subject.x.shape == (35, 6, 9, 2000)
    assert np.all(subject.x[4, 3] == 403)
    assert subject.targets == tuple(range(35))
    assert subject.metadata["source_schema"] == "canonical_4d_v1"
    assert subject.metadata["label_order_status"] == "canonical target axis converted from MATLAB 1-35 to Python 0-34"


def test_jbhi35_historical5_adapter_uses_local_config_and_preserves_axes(tmp_path: Path) -> None:
    _, _, config_path = _write_historical5_fixture(tmp_path)

    package = resolve_jbhi35_historical5_package(config_path=config_path)
    assert package.package_sha_entries_verified == 51
    assert available_subject_ids("jbhi35_historical5", config_path=config_path) == JBHI35_HISTORICAL5_SUBJECT_IDS

    subject = load_jbhi35_historical5_subject(package, "S03")
    generic_subject = load_subject("jbhi35_historical5", "S03", config_path=config_path)
    assert subject.x.shape == (35, 6, 9, 2000)
    assert np.array_equal(subject.x, generic_subject.x)
    assert subject.x[4, 3, 0, 0] == 20403
    assert subject.channels == ("Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2")
    assert subject.targets == tuple(range(35))
    assert subject.blocks == tuple(range(6))
    assert subject.metadata["model_label_base"] == 0
    assert subject.metadata["identity_metadata_exposed"] is False
    assert len(load_target_frequency_pairs("jbhi35_historical5")) == 35


def test_jbhi35_historical5_static_validation_reports_only_shared_contract(tmp_path: Path) -> None:
    _, _, config_path = _write_historical5_fixture(tmp_path)

    validation = validate_jbhi35_historical5_contract(config_path=config_path)

    assert validation.dataset == "jbhi35_historical5"
    assert validation.subjects == JBHI35_HISTORICAL5_SUBJECT_IDS
    assert validation.package_sha_entries_verified == 51
    assert validation.native_shape == (9, 2000, 35, 6)
    assert validation.canonical_shape == (35, 6, 9, 2000)
    assert validation.sampling_rate == 1000
    assert validation.channels == ("Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2")
    assert validation.model_label_range == (0, 34)
    assert validation.matlab_reference_kind == "aggregate_correct_counts_only"


def test_jbhi35_historical5_adapter_rejects_mapping_or_anonymous_sequence_mismatch(tmp_path: Path) -> None:
    _, _, mapping_config = _write_historical5_fixture(
        tmp_path / "mapping",
        mapping_subject_ids=("S01", "S03", "S04", "S05", "S02"),
    )
    with pytest.raises(ValueError, match="data mapping disagree"):
        resolve_jbhi35_historical5_package(config_path=mapping_config)

    _, private_manifest, sequence_config = _write_historical5_fixture(tmp_path / "sequence")
    private_manifest.write_text(
        private_manifest.read_text(encoding="utf-8").replace("S01,", "S02,", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected anonymous subject sequence"):
        resolve_jbhi35_historical5_package(config_path=sequence_config)


def test_jbhi35_historical5_adapter_does_not_echo_invalid_private_paths(tmp_path: Path) -> None:
    _, private_manifest, config_path = _write_historical5_fixture(tmp_path)
    private_manifest.write_text(
        private_manifest.read_text(encoding="utf-8").replace("data/S01.mat", "../identity_name.mat", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        resolve_jbhi35_historical5_package(config_path=config_path)
    assert "identity_name" not in str(exc_info.value)


def test_jbhi35_historical5_adapter_rejects_non_matlab_target_labels(tmp_path: Path) -> None:
    _, _, config_path = _write_historical5_fixture(tmp_path, target_labels=np.arange(35, dtype=np.int64))
    package = resolve_jbhi35_historical5_package(config_path=config_path)

    with pytest.raises(ValueError, match="target_labels must be exactly MATLAB labels 1-35"):
        load_jbhi35_historical5_subject(package, "S01")


def test_final_historical_task_config_matches_shared_adapter_contract() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (project_root / "tasks" / "bessvep" / "five_subject_reproduction_config_20260728.json").read_text(
            encoding="utf-8"
        )
    )

    assert tuple(config["subjects"]) == JBHI35_HISTORICAL5_SUBJECT_IDS
    assert config["dataset_config_key"] == JBHI35_HISTORICAL5_PACKAGE_CONFIG_KEY
    assert config["subject_manifest_config_key"] == JBHI35_HISTORICAL5_SUBJECT_MANIFEST_CONFIG_KEY
    assert tuple(tuple(float(value) for value in pair) for pair in config["target_frequency_pairs_hz"]) == load_target_frequency_pairs(
        "jbhi35_historical5"
    )


def test_final_historical_runners_use_shared_adapter_and_explicit_outputs() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = (
        project_root / "vep_arena" / "data" / "embc_jbhi.py",
        project_root / "tasks" / "bessvep" / "run_35t_five_subject_reproduction.py",
        project_root / "tasks" / "bessvep" / "run_35t_five_subject_tdca.py",
        project_root / "tasks" / "bessvep" / "run_35t_five_subject_periodic_receivers.py",
    )
    forbidden_tokens = (
        "D:",
        "OneDrive",
        "JBHI35_Reproducibility_Handoff_20260728",
        "_local_paths",
        "_load_subject",
        "_verify_package_manifest",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "resolve_jbhi35_historical5_package" in source
        assert "load_jbhi35_historical5_subject" in source
        assert all(token not in source for token in forbidden_tokens)
    for path in paths[1:]:
        source = path.read_text(encoding="utf-8")
        assert 'parser.add_argument("--output", type=Path, required=True' in source
