from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vep_arena.data.interface import DatasetInfo, TrialBatch


BETA_CHANNEL_NAMES = (
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6",
    "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5",
    "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPZ",
    "CP2", "CP4", "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6",
    "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
)

BETA_FREQS = (
    8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4,
    10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0, 12.2, 12.4,
    12.6, 12.8, 13.0, 13.2, 13.4, 13.6, 13.8, 14.0, 14.2, 14.4,
    14.6, 14.8, 15.0, 15.2, 15.4, 15.6, 15.8, 8.0, 8.2, 8.4,
)

BETA_PHASES_PI = (
    1.5, 0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 1.5, 0.0,
    0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0,
    1.5, 0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 1.5, 0.0,
    0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0,
)


@dataclass(frozen=True)
class BetaDataset:
    """BETA dataset placeholder following Arena's dataset contract.

    The metadata mirrors SSVEP-Analysis-Toolbox. File loading remains disabled
    until local BETA files and their exact MATLAB structure are confirmed.
    """

    root: Path
    id: str = "beta_9ch"

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            id=self.id,
            name="BETA SSVEP",
            root=self.root,
            subjects=tuple(range(1, 71)),
            blocks=tuple(range(1, 5)),
            targets=tuple(range(40)),
            channels=("PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"),
            sampling_rate=250,
            frequencies=tuple(float(x) for x in BETA_FREQS),
            phases=tuple(float(x) for x in BETA_PHASES_PI),
            prestim_seconds=0.5,
            break_seconds=0.5,
            latency_seconds=0.13,
        )

    def load_subject(self, subject: int) -> np.ndarray:
        raise NotImplementedError("BETA local file loading is pending raw file inspection.")

    def get_trials(
        self,
        subject: int,
        blocks: list[int] | tuple[int, ...],
        targets: list[int] | tuple[int, ...],
        channels: str | list[int] | tuple[int, ...],
        window: float,
        preprocess: str = "raw",
        n_bands: int = 1,
    ) -> TrialBatch:
        raise NotImplementedError("BETA trials are pending local file inspection.")
