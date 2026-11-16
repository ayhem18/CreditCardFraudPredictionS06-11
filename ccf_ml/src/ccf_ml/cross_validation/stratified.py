"""Domain-free stratified k-fold on a precomputed stratum."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ccf_ml.cross_validation.constants import STRATUM_CODE_TO_NAME


class Fold(NamedTuple):
    """One k-fold split: train ids and val ids."""

    fold_id: int
    train_ids: tuple[int, ...]
    val_ids: tuple[int, ...]


def stratified_kfold_ids(
    *,
    record_ids: np.ndarray,
    stratum: np.ndarray,
    n_splits: int,
    random_state: int,
    shuffle: bool = True,
) -> tuple[Fold, ...]:
    """Return ``n_splits`` folds stratified on ``stratum``.

    Ids in each list are unique and sorted. Val lists partition ``record_ids``;
    each train list is the complement of that fold's val list.

    A four-level stratum (alarm × target) has the same fold composition as
    running ``StratifiedKFold`` on flagged rows and unflagged rows separately
    (each stratified on the target) and concatenating fold ``i``.
    """
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError(f"n_splits must be an int >= 2, got {n_splits!r}")
    if type(shuffle) is not bool:
        raise ValueError(f"shuffle must be a bool, got {shuffle!r}")
    if shuffle:
        if type(random_state) is not int:
            raise ValueError(
                f"random_state must be an int when shuffle is True, got {random_state!r}"
            )
        splitter_random_state = random_state
    else:
        if random_state is not None:
            raise ValueError("random_state must be None when shuffle is False")
        splitter_random_state = None
    if not isinstance(record_ids, np.ndarray) or not isinstance(stratum, np.ndarray):
        raise TypeError("record_ids and stratum must be numpy arrays")
    if record_ids.ndim != 1 or stratum.ndim != 1:
        raise ValueError("record_ids and stratum must be 1-d")
    n_items = len(record_ids)
    if n_items == 0:
        raise ValueError("record_ids is empty")
    if n_items != len(stratum):
        raise ValueError(
            "record_ids and stratum must have the same length "
            f"(got {n_items} and {len(stratum)})"
        )
    unique_ids, id_counts = np.unique(record_ids, return_counts=True)
    duplicated = unique_ids[id_counts > 1]
    if duplicated.size:
        raise ValueError(f"duplicate record id: {int(duplicated[0])}")
    unique_codes, code_counts = np.unique(stratum, return_counts=True)
    n_codes = len(unique_codes)
    for i in range(n_codes):
        if int(code_counts[i]) < n_splits:
            code = int(unique_codes[i])
            name = STRATUM_CODE_TO_NAME.get(code, str(code))
            raise ValueError(
                f"stratum {name!r} has {int(code_counts[i])} rows, "
                f"need at least n_splits={n_splits}"
            )
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=splitter_random_state,
    )
    indices = np.arange(n_items)
    split_rows = list(splitter.split(indices, stratum))
    n_folds = len(split_rows)
    if n_folds != n_splits:
        raise ValueError(
            f"StratifiedKFold returned {n_folds} folds, expected {n_splits}"
        )
    folds: list[Fold] = [Fold(fold_id=-1, train_ids=(), val_ids=())] * n_folds
    for fold_id in range(n_folds):
        train_index, val_index = split_rows[fold_id]
        train_ids = tuple(int(x) for x in np.sort(record_ids[train_index]))
        val_ids = tuple(int(x) for x in np.sort(record_ids[val_index]))
        folds[fold_id] = Fold(
            fold_id=fold_id,
            train_ids=train_ids,
            val_ids=val_ids,
        )
    filled = tuple(folds)
    require_partition(filled, record_ids=record_ids)
    return filled


def require_partition(
    folds: tuple[Fold, ...],
    *,
    record_ids: np.ndarray,
) -> None:
    """Raise unless the folds are an ordinary k-fold partition of ``record_ids``."""
    n_folds = len(folds)
    if n_folds < 2:
        raise ValueError(f"need at least 2 folds, got {n_folds}")
    universe = {int(x) for x in record_ids}
    if len(universe) != len(record_ids):
        raise ValueError("record_ids must be unique")
    val_union: set[int] = set()
    for i in range(n_folds):
        fold = folds[i]
        if fold.fold_id != i:
            raise ValueError(f"fold_id {fold.fold_id} does not match index {i}")
        train_set = set(fold.train_ids)
        val_set = set(fold.val_ids)
        if len(train_set) != len(fold.train_ids):
            raise ValueError(f"fold {i} train_ids are not unique")
        if len(val_set) != len(fold.val_ids):
            raise ValueError(f"fold {i} val_ids are not unique")
        if not val_set:
            raise ValueError(f"fold {i} has an empty val list")
        overlap = train_set.intersection(val_set)
        if overlap:
            raise ValueError(f"fold {i} has ids on both train and val")
        if train_set.union(val_set) != universe:
            raise ValueError(f"fold {i} train and val do not cover all record ids")
        leaked = val_union.intersection(val_set)
        if leaked:
            preview = sorted(leaked)[:10]
            raise ValueError(f"val ids appear in more than one fold: {preview!r}")
        val_union.update(val_set)
        if tuple(sorted(fold.train_ids)) != fold.train_ids:
            raise ValueError(f"fold {i} train_ids are not sorted")
        if tuple(sorted(fold.val_ids)) != fold.val_ids:
            raise ValueError(f"fold {i} val_ids are not sorted")
    if val_union != universe:
        raise ValueError("val ids do not partition record_ids")
