"""Frozen folds as a sklearn ``PredefinedSplit``. Domain-free: ids only."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import PredefinedSplit

from ccf_ml.cross_validation.stratified import Fold


def val_fold_vector(
    *,
    record_ids: tuple[int, ...],
    val_ids_by_fold: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    """Integer val-fold id for each record, in ``record_ids`` order.

    Ordinary k-fold: every id appears in exactly one val list. sklearn's
    ``PredefinedSplit`` then takes train as the complement.
    """
    n_items = len(record_ids)
    if n_items == 0:
        raise ValueError("record_ids is empty")
    n_folds = len(val_ids_by_fold)
    if n_folds < 2:
        raise ValueError(f"need at least 2 folds, got {n_folds}")
    position: dict[int, int] = {}
    for i in range(n_items):
        record_id = record_ids[i]
        if type(record_id) is not int:
            raise TypeError(
                f"record id at index {i} must be an int, got {type(record_id)!r}"
            )
        if record_id in position:
            raise ValueError(f"duplicate record id: {record_id!r}")
        position[record_id] = i
    test_fold = np.full(n_items, -1, dtype=int)
    for fold_id in range(n_folds):
        val_ids = val_ids_by_fold[fold_id]
        n_val = len(val_ids)
        if n_val == 0:
            raise ValueError(f"fold {fold_id} has an empty val list")
        for j in range(n_val):
            record_id = val_ids[j]
            if record_id not in position:
                raise KeyError(
                    f"val id {record_id!r} in fold {fold_id} is not in record_ids"
                )
            index = position[record_id]
            if test_fold[index] != -1:
                raise ValueError(
                    f"record {record_id!r} is val in folds {test_fold[index]} and {fold_id}"
                )
            test_fold[index] = fold_id
    missing = int(np.sum(test_fold < 0))
    if missing:
        raise ValueError(f"{missing} records never appear in a val list")
    return test_fold


def predefined_split(
    *,
    record_ids: tuple[int, ...],
    val_ids_by_fold: tuple[tuple[int, ...], ...],
) -> PredefinedSplit:
    """sklearn splitter over frozen val membership."""
    return PredefinedSplit(
        val_fold_vector(
            record_ids=record_ids,
            val_ids_by_fold=val_ids_by_fold,
        )
    )


def predefined_split_from_folds(
    *,
    record_ids: tuple[int, ...],
    folds: tuple[Fold, ...],
) -> PredefinedSplit:
    """sklearn splitter from frozen ``Fold`` val lists, aligned to ``record_ids``."""
    n_folds = len(folds)
    val_ids_by_fold: list[tuple[int, ...]] = [()] * n_folds
    for i in range(n_folds):
        val_ids_by_fold[i] = folds[i].val_ids
    return predefined_split(
        record_ids=record_ids,
        val_ids_by_fold=tuple(val_ids_by_fold),
    )
