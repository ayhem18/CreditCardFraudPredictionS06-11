"""Alarm-flag union and the alarm × target stratum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.constants import (
    ALARM_FLAG_COLUMNS,
    STRATUM_CODE_TO_NAME,
    TARGET_COLUMN,
)


def as_boolean_array(values: pd.Series, *, column: str) -> np.ndarray:
    """Return a 1-d bool array. Accepts a bool series or a {0, 1} numeric series."""
    if values.isna().any():
        raise ValueError(f"column {column!r} contains NA")
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values):
        arr = values.to_numpy()
        as_int = arr.astype(np.int64)
        if not np.all(arr == as_int):
            raise ValueError(f"column {column!r} is not integer-valued")
        if not np.all((as_int == 0) | (as_int == 1)):
            raise ValueError(f"column {column!r} is not binary")
        return as_int.astype(bool)
    raise TypeError(
        f"column {column!r} must be boolean or binary numeric, got {values.dtype}"
    )


def as_binary_label(values: pd.Series, *, column: str) -> np.ndarray:
    """Return a 1-d int64 array in ``{0, 1}``."""
    if values.isna().any():
        raise ValueError(f"column {column!r} contains NA")
    arr = values.to_numpy()
    as_int = arr.astype(np.int64)
    if not np.all(arr == as_int):
        raise ValueError(f"column {column!r} is not integer-valued")
    if not np.all((as_int == 0) | (as_int == 1)):
        raise ValueError(f"column {column!r} is not binary")
    return as_int


def as_record_ids(values: pd.Series, *, column: str) -> np.ndarray:
    """Return unique integer record ids as int64."""
    if values.isna().any():
        raise ValueError(f"column {column!r} contains NA")
    if pd.api.types.is_integer_dtype(values):
        ids = values.to_numpy(dtype=np.int64)
    elif pd.api.types.is_float_dtype(values):
        arr = values.to_numpy()
        as_int = arr.astype(np.int64)
        if not np.all(arr == as_int):
            raise ValueError(f"column {column!r} is not integer-valued")
        ids = as_int
    else:
        raise TypeError(
            f"column {column!r} must be integer ids, got {values.dtype}"
        )
    unique_ids, counts = np.unique(ids, return_counts=True)
    duplicated = unique_ids[counts > 1]
    if duplicated.size:
        raise ValueError(
            f"column {column!r} has duplicate ids, e.g. {int(duplicated[0])}"
        )
    return ids


def binary_pair_stratum(*, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Encode two aligned binary arrays as ``2 * left + right``.

    Alarm × target uses this mapping::

        0 no-alarm not-fraud
        1 no-alarm fraud
        2 alarm not-fraud
        3 alarm fraud
    """
    if left.shape != right.shape:
        raise ValueError(
            "left and right must have the same shape "
            f"(got {left.shape} and {right.shape})"
        )
    if left.ndim != 1:
        raise ValueError(f"left and right must be 1-d (got ndim={left.ndim})")
    left_bit = _as_bit(left, name="left")
    right_bit = _as_bit(right, name="right")
    return left_bit * 2 + right_bit


def alarm_flag_mask(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = ALARM_FLAG_COLUMNS,
) -> pd.Series:
    """True where at least one alarm flag is on."""
    if len(columns) == 0:
        raise ValueError("columns is empty")
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing alarm flag columns: {missing}")
    n_columns = len(columns)
    bits: list[np.ndarray | None] = [None] * n_columns
    for i in range(n_columns):
        name = columns[i]
        bits[i] = as_boolean_array(frame[name], column=name)
    filled_bits = [bit for bit in bits if bit is not None]
    if len(filled_bits) != n_columns:
        raise ValueError("failed to read every alarm flag column")
    stacked = np.stack(filled_bits, axis=1)
    return pd.Series(stacked.any(axis=1), index=frame.index, name="has_alarm_flag")


def flag_target_stratum(
    frame: pd.DataFrame,
    *,
    alarm_flag_columns: tuple[str, ...] = ALARM_FLAG_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> np.ndarray:
    """Integer stratum codes for the alarm-flag union crossed with the target."""
    if target_column not in frame.columns:
        raise ValueError(f"frame is missing target column: {target_column!r}")
    alarm = alarm_flag_mask(frame, columns=alarm_flag_columns).to_numpy()
    target = as_binary_label(frame[target_column], column=target_column)
    return binary_pair_stratum(left=alarm, right=target)


def stratum_count_names(stratum: np.ndarray) -> dict[str, int]:
    """Count rows per named alarm × target cell."""
    counts = {name: 0 for name in STRATUM_CODE_TO_NAME.values()}
    unique, occ = np.unique(stratum, return_counts=True)
    n_codes = len(unique)
    for i in range(n_codes):
        code = int(unique[i])
        name = STRATUM_CODE_TO_NAME.get(code)
        if name is None:
            raise ValueError(f"unknown stratum code: {code}")
        counts[name] = int(occ[i])
    return counts


def _as_bit(values: np.ndarray, *, name: str) -> np.ndarray:
    if values.dtype == np.bool_ or values.dtype == bool:
        return values.astype(np.int64, copy=False)
    as_int = values.astype(np.int64)
    if not np.all(values == as_int):
        raise ValueError(f"{name} is not integer-valued")
    if not np.all((as_int == 0) | (as_int == 1)):
        raise ValueError(f"{name} is not binary")
    return as_int
