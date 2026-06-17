# Author: Liu Haifeng <liuhf@728@gmail.com>
# Created: 2026-06-17
# Last updated: 2026-06-17
# Description: Part of the VEP Arena SSVEP benchmark workspace.
from __future__ import annotations

import math

import numpy as np


def itr_bits_per_minute(accuracy: float, classes: int, trial_seconds: float) -> float:
    if accuracy < 1 / classes:
        return 0.0
    if accuracy >= 1.0:
        return math.log2(classes) * (60.0 / trial_seconds)
    return (
        math.log2(classes)
        + accuracy * math.log2(accuracy)
        + (1 - accuracy) * math.log2((1 - accuracy) / (classes - 1))
    ) * (60.0 / trial_seconds)


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def cohen_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    sd = np.std(diff, ddof=1)
    if diff.size <= 1 or sd <= 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)
