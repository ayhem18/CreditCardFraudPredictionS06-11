"""Place-value digit encoding of written decimal strings (B.4)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

NUMBER_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")
NON_DIGIT_SLOT = 0
DIGIT_VOCAB_SIZE = 10


def split_written_parts(*, texts: pd.Series, column: str) -> tuple[pd.Series, pd.Series]:
    """Split written numbers into unsigned integer digits and decimal digits."""
    stripped = texts.astype(str).str.strip()
    if stripped.eq("").any() or stripped.isna().any():
        raise ValueError(f"{column}: empty or missing written values")
    valid = stripped.str.fullmatch(NUMBER_PATTERN.pattern)
    if not bool(valid.all()):
        bad_index = int((~valid).to_numpy().nonzero()[0][0])
        raise ValueError(
            f"{column}: not a plain decimal number at row {bad_index}: "
            f"{stripped.iloc[bad_index]!r}"
        )
    unsigned = stripped.str.replace(r"^-", "", n=1, regex=True)
    split = unsigned.str.split(".", n=1, expand=True)
    integer_parts = split[0]
    if 1 in split.columns:
        decimal_parts = split[1].fillna("")
    else:
        decimal_parts = pd.Series("", index=unsigned.index)
    if integer_parts.eq("").any():
        raise ValueError(f"{column}: integer part is empty")
    return integer_parts, decimal_parts


def measure_numeric_slots(*, series: pd.Series, column: str) -> tuple[int, int]:
    """Return ``(int_slots, decimal_slots)`` from pooled written strings."""
    integer_parts, decimal_parts = split_written_parts(texts=series, column=column)
    int_slots = int(integer_parts.str.len().max())
    decimal_slots = int(decimal_parts.str.len().max())
    if int_slots < 1:
        raise ValueError(f"{column}: int_slots must be at least 1, got {int_slots}")
    return int_slots, decimal_slots


def max_place_slots(*, max_int_slots: int, max_decimal_slots: int) -> int:
    """Slot table size: non-digit slot 0 plus every integer and decimal place."""
    if not isinstance(max_int_slots, int) or max_int_slots < 1:
        raise ValueError(f"max_int_slots must be a positive int, got {max_int_slots!r}")
    if not isinstance(max_decimal_slots, int) or max_decimal_slots < 0:
        raise ValueError(
            f"max_decimal_slots must be a non-negative int, got {max_decimal_slots!r}"
        )
    return 1 + max_int_slots + max_decimal_slots


def place_slot_ids_for_numeric(
    *,
    int_slots: int,
    decimal_slots: int,
    max_int_slots: int,
) -> tuple[int, ...]:
    """Place-value slot ids for one numeric column, high integer place first."""
    if int_slots < 1 or decimal_slots < 0:
        raise ValueError(
            f"int_slots must be >= 1 and decimal_slots >= 0, got "
            f"int_slots={int_slots!r} decimal_slots={decimal_slots!r}"
        )
    if int_slots > max_int_slots:
        raise ValueError(
            f"int_slots {int_slots} exceeds max_int_slots {max_int_slots}"
        )
    width = int_slots + decimal_slots
    slots = [NON_DIGIT_SLOT] * width
    for i in range(int_slots):
        exponent = int_slots - 1 - i
        slots[i] = 1 + (max_int_slots - 1 - exponent)
    for j in range(decimal_slots):
        slots[int_slots + j] = 1 + max_int_slots + j
    return tuple(slots)


def digit_codes_for_numeric(
    *,
    series: pd.Series,
    column: str,
    int_slots: int,
    decimal_slots: int,
) -> np.ndarray:
    """Zero-pad a written number to place-value width and map digits to ids 0–9."""
    integer_parts, decimal_parts = split_written_parts(texts=series, column=column)
    if bool(integer_parts.str.len().gt(int_slots).any()):
        raise ValueError(f"{column}: an integer part is longer than int_slots={int_slots}")
    if bool(decimal_parts.str.len().gt(decimal_slots).any()):
        raise ValueError(
            f"{column}: a decimal part is longer than decimal_slots={decimal_slots}"
        )
    padded = integer_parts.str.zfill(int_slots) + decimal_parts.str.ljust(
        decimal_slots, "0"
    )
    width = int_slots + decimal_slots
    return ascii_digit_codes(texts=padded, width=width, column=column)


def ascii_digit_codes(*, texts: pd.Series, width: int, column: str) -> np.ndarray:
    """Map fixed-width ASCII digit strings to an ``(n, width)`` int32 array."""
    if width < 1:
        raise ValueError(f"{column}: width must be positive, got {width}")
    arr = texts.to_numpy(dtype=f"U{width}")
    n_rows = int(arr.shape[0])
    encoded = np.char.encode(arr, encoding="ascii")
    raw = np.asarray(encoded).view(np.uint8).reshape(n_rows, width)
    codes = raw.astype(np.int32) - np.int32(ord("0"))
    if bool((codes < 0).any()) or bool((codes > 9).any()):
        raise ValueError(f"{column}: padded form contains a non-digit")
    return codes
