"""Place-value digit encoding and slot ids (B.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ccf_tabformer_dl.data_contract.written_numbers import (
    NON_DIGIT_SLOT,
    ascii_digit_codes,
    digit_codes_for_numeric,
    max_place_slots,
    place_slot_ids_for_numeric,
)


def test_place_slots_align_across_column_widths() -> None:
    max_int = 5
    amount = place_slot_ids_for_numeric(
        int_slots=4, decimal_slots=2, max_int_slots=max_int
    )
    velocity = place_slot_ids_for_numeric(
        int_slots=2, decimal_slots=1, max_int_slots=max_int
    )
    balance = place_slot_ids_for_numeric(
        int_slots=5, decimal_slots=2, max_int_slots=max_int
    )
    assert amount == (2, 3, 4, 5, 6, 7)
    assert velocity == (4, 5, 6)
    assert balance == (1, 2, 3, 4, 5, 6, 7)
    assert max_place_slots(max_int_slots=max_int, max_decimal_slots=2) == 8
    assert NON_DIGIT_SLOT == 0


def test_digit_codes_zero_pad_place_value() -> None:
    series = pd.Series(["23.56", "1.4", "2006.3"])
    codes = digit_codes_for_numeric(
        series=series,
        column="amount_usd",
        int_slots=4,
        decimal_slots=2,
    )
    expected = np.array(
        [
            [0, 0, 2, 3, 5, 6],
            [0, 0, 0, 1, 4, 0],
            [2, 0, 0, 6, 3, 0],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(codes, expected)


def test_ascii_digit_codes_rejects_non_digit() -> None:
    series = pd.Series(["12x"])
    try:
        ascii_digit_codes(texts=series, width=3, column="amount_usd")
    except ValueError as exc:
        assert "non-digit" in str(exc)
    else:
        raise AssertionError("expected ValueError")
