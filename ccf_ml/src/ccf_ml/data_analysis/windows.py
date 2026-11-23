"""Systematic hour and age window families."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ccf_ml.data_analysis.constants import (
    AGE_WINDOW_LENGTHS,
    HOUR_WINDOW_LENGTHS,
    N_HOURS,
)


class Window(NamedTuple):
    """One contiguous interval on a circular or linear axis."""

    kind: str
    start: int
    length: int
    label: str


def hour_label(*, start: int, length: int) -> str:
    """End-exclusive circular label, for example ``[22, 04)``."""
    end = (start + length) % N_HOURS
    return f"[{start:02d}, {end:02d})"


def age_label(*, start: int, length: int) -> str:
    """End-exclusive linear label, for example ``[18, 26)``."""
    return f"[{start}, {start + length})"


def hour_windows() -> tuple[Window, ...]:
    """Every circular interval of the pre-registered lengths."""
    n_windows = N_HOURS * len(HOUR_WINDOW_LENGTHS)
    windows = [None] * n_windows
    index = 0
    for length in HOUR_WINDOW_LENGTHS:
        for start in range(N_HOURS):
            windows[index] = Window(
                kind="hour",
                start=int(start),
                length=int(length),
                label=hour_label(start=start, length=length),
            )
            index += 1
    return tuple(windows)


def age_windows(*, age_min: int, age_max: int) -> tuple[Window, ...]:
    """Every linear interval of the pre-registered lengths on ``[age_min, age_max]``."""
    if age_max < age_min:
        raise ValueError(f"age_max {age_max} is below age_min {age_min}")
    collected = [None] * (len(AGE_WINDOW_LENGTHS) * (age_max - age_min + 1))
    filled = 0
    for length in AGE_WINDOW_LENGTHS:
        last_start = age_max - length + 1
        for start in range(age_min, last_start + 1):
            collected[filled] = Window(
                kind="age",
                start=int(start),
                length=int(length),
                label=age_label(start=start, length=length),
            )
            filled += 1
    return tuple(collected[:filled])


def hour_membership(hours: np.ndarray, *, window: Window) -> np.ndarray:
    """True where ``hours`` falls in the circular window."""
    if window.kind != "hour":
        raise ValueError(f"expected hour window, got {window.kind!r}")
    shifted = np.mod(hours.astype(np.int64) - window.start, N_HOURS)
    return shifted < window.length


def age_membership(ages: np.ndarray, *, window: Window) -> np.ndarray:
    """True where ``ages`` falls in ``[start, start+length)``."""
    if window.kind != "age":
        raise ValueError(f"expected age window, got {window.kind!r}")
    values = ages.astype(np.float64)
    lo = float(window.start)
    hi = float(window.start + window.length)
    return (values >= lo) & (values < hi)
