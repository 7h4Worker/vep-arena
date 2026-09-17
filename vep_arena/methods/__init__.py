# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
"""Method adapters."""

from vep_arena.methods.gtrca import gTRCA
from vep_arena.methods.la_trca import LATRCA
from vep_arena.methods.mohp_filter import MultiObjectiveHighPassFilter
from vep_arena.methods.prca import PRCA
from vep_arena.methods.ress import RESS
from vep_arena.methods.sctrca import scTRCA
from vep_arena.methods.sinref_trca import SinusoidalReferencedTRCA
from vep_arena.methods.strca import sTRCA
from vep_arena.methods.xtrca import xTRCA

__all__ = [
    "LATRCA",
    "MultiObjectiveHighPassFilter",
    "PRCA",
    "RESS",
    "SinusoidalReferencedTRCA",
    "gTRCA",
    "sTRCA",
    "scTRCA",
    "xTRCA",
]
