"""Stratified train row sampling for the overfit subset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ccf_tabformer_dl.data_contract.corrected_table import (
    LABEL_MISSING,
    SPLIT_TEST,
    SPLIT_TRAIN,
    _binary_labels,
)
from ccf_tabformer_dl.data_contract.row_subset.preparation.stratified import (
    sample_stratified_train_rows,
)


def test_binary_labels_accept_float_written_zero_one() -> None:
    labels = _binary_labels(pd.Series(["0.0", "1.0", "0", "1"]), label="is_fraud")
    np.testing.assert_array_equal(labels, np.array([0, 1, 0, 1], dtype=np.int8))


def test_stratified_sample_is_train_only_and_seeded() -> None:
    split = np.array(
        [SPLIT_TRAIN] * 20 + [SPLIT_TEST] * 5,
        dtype=np.uint8,
    )
    labels = np.array(
        [0] * 16 + [1] * 4 + [LABEL_MISSING] * 5,
        dtype=np.int8,
    )
    first = sample_stratified_train_rows(
        split=split, labels=labels, n_rows=8, seed=0
    )
    second = sample_stratified_train_rows(
        split=split, labels=labels, n_rows=8, seed=0
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (8,)
    assert len(np.unique(first)) == 8
    assert np.all(first < 20)
    sampled_labels = labels[first]
    assert 0 in sampled_labels
    assert 1 in sampled_labels
