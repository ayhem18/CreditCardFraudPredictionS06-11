"""Split sidecar → row index for the TrainVal loaders."""

from __future__ import annotations

import numpy as np
import pytest

from ccf_tabformer_dl.data_contract.corrected_table import SPLIT_TEST, SPLIT_TRAIN
from ccf_tabformer_dl.task_engines.full_run.data import row_index_for_split


def test_row_index_for_split_keeps_train_and_test() -> None:
    split = np.array(
        [SPLIT_TRAIN, SPLIT_TRAIN, SPLIT_TEST, SPLIT_TRAIN, SPLIT_TEST],
        dtype=np.uint8,
    )
    train = row_index_for_split(split, split_value=SPLIT_TRAIN)
    test = row_index_for_split(split, split_value=SPLIT_TEST)
    assert train.tolist() == [0, 1, 3]
    assert test.tolist() == [2, 4]


def test_row_index_for_split_rejects_empty() -> None:
    split = np.array([SPLIT_TRAIN, SPLIT_TRAIN], dtype=np.uint8)
    with pytest.raises(ValueError, match="no rows"):
        row_index_for_split(split, split_value=SPLIT_TEST)
