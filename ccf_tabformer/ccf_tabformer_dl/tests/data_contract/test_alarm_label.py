"""Equal (is_alarm, is_fraud) cell sampling for the representative subset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ccf_tabformer_dl.data_contract.columns import (
    ALARM_FLAG_COLUMNS,
    BOOLEAN_FALSE,
    BOOLEAN_TRUE,
)
from ccf_tabformer_dl.data_contract.corrected_table import (
    LABEL_MISSING,
    SPLIT_TEST,
    SPLIT_TRAIN,
    alarm_flag_union,
)
from ccf_tabformer_dl.data_contract.row_subset.preparation.alarm_label import (
    N_ALARM_LABEL_CELLS,
    sample_balanced_alarm_label_rows,
)


def test_alarm_flag_union_is_or_of_five_flags_not_new_merchant() -> None:
    n_rows = 4
    frame = pd.DataFrame(
        {
            "is_new_merchant": [BOOLEAN_TRUE] * n_rows,
            ALARM_FLAG_COLUMNS[0]: [
                BOOLEAN_TRUE,
                BOOLEAN_FALSE,
                BOOLEAN_FALSE,
                BOOLEAN_FALSE,
            ],
            ALARM_FLAG_COLUMNS[1]: [BOOLEAN_FALSE] * n_rows,
            ALARM_FLAG_COLUMNS[2]: [
                BOOLEAN_FALSE,
                BOOLEAN_TRUE,
                BOOLEAN_FALSE,
                BOOLEAN_FALSE,
            ],
            ALARM_FLAG_COLUMNS[3]: [BOOLEAN_FALSE] * n_rows,
            ALARM_FLAG_COLUMNS[4]: [BOOLEAN_FALSE] * n_rows,
        }
    )
    np.testing.assert_array_equal(
        alarm_flag_union(frame),
        np.array([True, True, False, False]),
    )


def test_balanced_sample_draws_each_cell_and_is_seeded() -> None:
    split = np.array(
        [SPLIT_TRAIN] * 24 + [SPLIT_TEST] * 4,
        dtype=np.uint8,
    )
    labels = np.array(
        [0] * 6
        + [1] * 6
        + [0] * 6
        + [1] * 6
        + [LABEL_MISSING] * 4,
        dtype=np.int8,
    )
    is_alarm = np.array(
        [False] * 12 + [True] * 12 + [False] * 4,
        dtype=bool,
    )
    first = sample_balanced_alarm_label_rows(
        split=split,
        labels=labels,
        is_alarm=is_alarm,
        n_per_cell=3,
        seed=0,
    )
    second = sample_balanced_alarm_label_rows(
        split=split,
        labels=labels,
        is_alarm=is_alarm,
        n_per_cell=3,
        seed=0,
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (N_ALARM_LABEL_CELLS * 3,)
    assert len(np.unique(first)) == int(first.shape[0])
    assert np.all(first < 24)
    sampled_alarm = is_alarm[first]
    sampled_labels = labels[first]
    for alarm in (False, True):
        for fraud in (0, 1):
            n_cell = int(
                np.sum((sampled_alarm == alarm) & (sampled_labels == fraud))
            )
            assert n_cell == 3


def test_balanced_sample_rejects_a_thin_cell() -> None:
    split = np.array([SPLIT_TRAIN] * 8, dtype=np.uint8)
    labels = np.array([0, 0, 0, 1, 0, 0, 1, 1], dtype=np.int8)
    is_alarm = np.array(
        [False, False, False, False, True, True, True, True],
        dtype=bool,
    )
    with pytest.raises(ValueError, match="cell 1"):
        sample_balanced_alarm_label_rows(
            split=split,
            labels=labels,
            is_alarm=is_alarm,
            n_per_cell=2,
            seed=0,
        )
