from __future__ import annotations

from vep_arena.methods.bprca import BPRCA


class FusionCA(BPRCA):
    """Fuse periodic bPRCA streams with a full-window spatial stream."""

    name = "FUSIONCA"

    def __init__(
        self,
        frequencies: tuple[float, ...] | list[float],
        sampling_rate: int,
        n_fbs: int = 5,
        *,
        ensemble: bool = False,
    ) -> None:
        super().__init__(
            frequencies,
            sampling_rate,
            n_fbs=n_fbs,
            ensemble=ensemble,
            fusion=True,
        )
