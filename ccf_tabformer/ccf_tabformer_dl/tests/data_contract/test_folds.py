"""Imported CV fold assignment: partition, n_val, no missing train ids."""

from __future__ import annotations

import numpy as np
import pytest

from ccf_tabformer_dl.data_contract.corrected_table import SPLIT_TEST, SPLIT_TRAIN
from ccf_tabformer_dl.data_contract.folds.collection import FOLD_UNASSIGNED
from ccf_tabformer_dl.data_contract.folds.preparation.imported import assign_fold_ids


def _document(*, folds: list[dict], n_rows: int, n_splits: int = 2) -> dict:
    return {
        "n_splits": n_splits,
        "n_rows": n_rows,
        "id_column": "id",
        "target_column": "is_fraud",
        "folds": folds,
    }


def test_assign_fold_ids_partitions_train_and_leaves_test() -> None:
    row_ids = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
    split = np.array(
        [SPLIT_TRAIN, SPLIT_TRAIN, SPLIT_TRAIN, SPLIT_TRAIN, SPLIT_TEST, SPLIT_TEST],
        dtype=np.uint8,
    )
    document = _document(
        n_rows=4,
        folds=[
            {"fold_id": 0, "n_train": 2, "n_val": 2, "val_ids": [10, 30]},
            {"fold_id": 1, "n_train": 2, "n_val": 2, "val_ids": [40, 20]},
        ],
    )
    fold_id = assign_fold_ids(row_ids=row_ids, split=split, document=document)
    assert fold_id.dtype == np.int8
    assert fold_id.tolist() == [0, 1, 0, 1, FOLD_UNASSIGNED, FOLD_UNASSIGNED]


def test_assign_fold_ids_rejects_missing_train_id() -> None:
    row_ids = np.array([1, 2, 3], dtype=np.int64)
    split = np.array([SPLIT_TRAIN, SPLIT_TRAIN, SPLIT_TEST], dtype=np.uint8)
    document = _document(
        n_rows=2,
        folds=[
            {"fold_id": 0, "n_train": 1, "n_val": 1, "val_ids": [1]},
            {"fold_id": 1, "n_train": 1, "n_val": 1, "val_ids": [99]},
        ],
    )
    with pytest.raises(ValueError, match="absent"):
        assign_fold_ids(row_ids=row_ids, split=split, document=document)


def test_assign_fold_ids_rejects_document_n_rows_mismatch() -> None:
    row_ids = np.array([1, 2, 3], dtype=np.int64)
    split = np.array([SPLIT_TRAIN, SPLIT_TRAIN, SPLIT_TEST], dtype=np.uint8)
    document = _document(
        n_rows=99,
        folds=[
            {"fold_id": 0, "n_train": 1, "n_val": 1, "val_ids": [1]},
            {"fold_id": 1, "n_train": 1, "n_val": 1, "val_ids": [2]},
        ],
    )
    with pytest.raises(ValueError, match="n_rows"):
        assign_fold_ids(row_ids=row_ids, split=split, document=document)


def test_assign_fold_ids_rejects_duplicate_across_folds() -> None:
    row_ids = np.array([1, 2, 3], dtype=np.int64)
    split = np.array([SPLIT_TRAIN, SPLIT_TRAIN, SPLIT_TEST], dtype=np.uint8)
    document = _document(
        n_rows=2,
        folds=[
            {"fold_id": 0, "n_train": 1, "n_val": 1, "val_ids": [1]},
            {"fold_id": 1, "n_train": 1, "n_val": 1, "val_ids": [1]},
        ],
    )
    with pytest.raises(ValueError, match="reassigns"):
        assign_fold_ids(row_ids=row_ids, split=split, document=document)
