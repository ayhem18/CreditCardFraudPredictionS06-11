"""Alarm × target stratified k-fold, frozen as a package artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.alarm import (
    as_record_ids,
    flag_target_stratum,
    stratum_count_names,
)
from ccf_ml.cross_validation.constants import (
    ALARM_FLAG_COLUMNS,
    ALARM_STRATUM_NAMES,
    DEFAULT_N_SPLITS,
    DEFAULT_RANDOM_STATE,
    FLAG_STRATIFIED_CV_FILENAME,
    FLAG_STRATIFIED_SCHEME,
    FRAUD_STRATUM_NAMES,
    ID_COLUMN,
    STRATUM_CODE_TO_NAME,
    TARGET_COLUMN,
)
from ccf_ml.cross_validation.stratified import (
    Fold,
    require_partition,
    stratified_kfold_ids,
)
from ccf_ml.file_utils import get_cv_splits_dir

_DOCUMENT_KEYS = (
    "scheme",
    "n_splits",
    "random_state",
    "shuffle",
    "id_column",
    "target_column",
    "alarm_flag_columns",
    "n_rows",
    "stratum_counts",
    "folds",
)
_FOLD_KEYS = (
    "fold_id",
    "n_train",
    "n_val",
    "train_stratum_counts",
    "val_stratum_counts",
    "val_ids",
)


class FoldSummary(NamedTuple):
    """Occupancy of one fold, derived from the alarm × target stratum."""

    fold_id: int
    n_train: int
    n_val: int
    train_stratum_counts: dict[str, int]
    val_stratum_counts: dict[str, int]


class FlagStratifiedCv(NamedTuple):
    """Frozen flag × target stratified k-fold partition."""

    n_splits: int
    random_state: int
    shuffle: bool
    id_column: str
    target_column: str
    alarm_flag_columns: tuple[str, ...]
    n_rows: int
    stratum_counts: dict[str, int]
    folds: tuple[Fold, ...]
    fold_summaries: tuple[FoldSummary, ...]

    def fold(self, fold_id: int) -> Fold:
        """Return fold ``fold_id`` (``0 .. n_splits-1``)."""
        if type(fold_id) is not int:
            raise TypeError(f"fold_id must be an int, got {type(fold_id)!r}")
        n_folds = len(self.folds)
        if fold_id < 0 or fold_id >= n_folds:
            raise KeyError(f"fold_id {fold_id} is out of range 0..{n_folds - 1}")
        return self.folds[fold_id]


def flag_stratified_cv_path() -> Path:
    """Return ``<package_artifacts>/cv_splits/flag_stratified_cv.json``."""
    return get_cv_splits_dir() / FLAG_STRATIFIED_CV_FILENAME


def build_flag_stratified_cv(
    frame: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    shuffle: bool = True,
    id_column: str = ID_COLUMN,
    target_column: str = TARGET_COLUMN,
    alarm_flag_columns: tuple[str, ...] = ALARM_FLAG_COLUMNS,
) -> FlagStratifiedCv:
    """Split ``frame`` with StratifiedKFold on the alarm × target stratum."""
    if id_column not in frame.columns:
        raise ValueError(f"frame is missing id column: {id_column!r}")
    record_ids = as_record_ids(frame[id_column], column=id_column)
    stratum = flag_target_stratum(
        frame,
        alarm_flag_columns=alarm_flag_columns,
        target_column=target_column,
    )
    folds = stratified_kfold_ids(
        record_ids=record_ids,
        stratum=stratum,
        n_splits=n_splits,
        random_state=random_state,
        shuffle=shuffle,
    )
    n_items = len(record_ids)
    position = {int(record_ids[i]): i for i in range(n_items)}
    n_folds = len(folds)
    summaries: list[FoldSummary] = [
        FoldSummary(
            fold_id=-1,
            n_train=0,
            n_val=0,
            train_stratum_counts={},
            val_stratum_counts={},
        )
    ] * n_folds
    for fold_id in range(n_folds):
        fold = folds[fold_id]
        train_index = _ids_to_index(fold.train_ids, position=position)
        val_index = _ids_to_index(fold.val_ids, position=position)
        train_counts = stratum_count_names(stratum[train_index])
        val_counts = stratum_count_names(stratum[val_index])
        if sum(train_counts.values()) != len(fold.train_ids):
            raise ValueError("train stratum counts do not match n_train")
        if sum(val_counts.values()) != len(fold.val_ids):
            raise ValueError("val stratum counts do not match n_val")
        summaries[fold_id] = FoldSummary(
            fold_id=fold_id,
            n_train=len(fold.train_ids),
            n_val=len(fold.val_ids),
            train_stratum_counts=train_counts,
            val_stratum_counts=val_counts,
        )
    return FlagStratifiedCv(
        n_splits=n_splits,
        random_state=random_state,
        shuffle=shuffle,
        id_column=id_column,
        target_column=target_column,
        alarm_flag_columns=tuple(alarm_flag_columns),
        n_rows=n_items,
        stratum_counts=stratum_count_names(stratum),
        folds=folds,
        fold_summaries=tuple(summaries),
    )


def dump_flag_stratified_cv(document: FlagStratifiedCv) -> dict[str, object]:
    """JSON-ready document. Each fold stores val ids; train is the complement."""
    n_folds = len(document.folds)
    fold_docs: list[dict[str, object]] = [{}] * n_folds
    for i in range(n_folds):
        fold = document.folds[i]
        summary = document.fold_summaries[i]
        fold_docs[i] = {
            "fold_id": fold.fold_id,
            "n_train": summary.n_train,
            "n_val": summary.n_val,
            "train_stratum_counts": dict(summary.train_stratum_counts),
            "val_stratum_counts": dict(summary.val_stratum_counts),
            "val_ids": list(fold.val_ids),
        }
    return {
        "scheme": FLAG_STRATIFIED_SCHEME,
        "n_splits": document.n_splits,
        "random_state": document.random_state,
        "shuffle": document.shuffle,
        "id_column": document.id_column,
        "target_column": document.target_column,
        "alarm_flag_columns": list(document.alarm_flag_columns),
        "n_rows": document.n_rows,
        "stratum_counts": dict(document.stratum_counts),
        "folds": fold_docs,
    }


def parse_flag_stratified_cv(payload: object) -> FlagStratifiedCv:
    """Rebuild the frozen partition from a JSON object."""
    document = _require_dict(payload, name="flag_stratified_cv")
    _require_exact_keys(document, _DOCUMENT_KEYS, name="flag_stratified_cv")
    scheme = document["scheme"]
    if scheme != FLAG_STRATIFIED_SCHEME:
        raise ValueError(
            f"scheme must be {FLAG_STRATIFIED_SCHEME!r}, got {scheme!r}"
        )
    n_splits = _require_int(document["n_splits"], name="n_splits")
    random_state = _require_int(document["random_state"], name="random_state")
    shuffle = document["shuffle"]
    if type(shuffle) is not bool:
        raise TypeError(f"shuffle must be a bool, got {type(shuffle)!r}")
    id_column = _require_str(document["id_column"], name="id_column")
    target_column = _require_str(document["target_column"], name="target_column")
    alarm_flag_columns = _require_str_tuple(
        document["alarm_flag_columns"],
        name="alarm_flag_columns",
    )
    n_rows = _require_int(document["n_rows"], name="n_rows")
    stratum_counts = _require_stratum_counts(
        document["stratum_counts"],
        name="stratum_counts",
    )
    fold_payloads = document["folds"]
    if type(fold_payloads) is not list:
        raise TypeError(f"folds must be a list, got {type(fold_payloads)!r}")
    if len(fold_payloads) != n_splits:
        raise ValueError(
            f"folds has length {len(fold_payloads)}, n_splits={n_splits}"
        )
    folds: list[Fold] = [Fold(fold_id=-1, train_ids=(), val_ids=())] * n_splits
    summaries: list[FoldSummary] = [
        FoldSummary(
            fold_id=-1,
            n_train=0,
            n_val=0,
            train_stratum_counts={},
            val_stratum_counts={},
        )
    ] * n_splits
    for i in range(n_splits):
        fold_payload = _require_dict(fold_payloads[i], name=f"folds[{i}]")
        _require_exact_keys(fold_payload, _FOLD_KEYS, name=f"folds[{i}]")
        fold_id = _require_int(fold_payload["fold_id"], name=f"folds[{i}].fold_id")
        if fold_id != i:
            raise ValueError(f"folds[{i}].fold_id must be {i}, got {fold_id}")
        val_ids = _require_int_tuple(
            fold_payload["val_ids"],
            name=f"folds[{i}].val_ids",
        )
        n_train = _require_int(fold_payload["n_train"], name=f"folds[{i}].n_train")
        n_val = _require_int(fold_payload["n_val"], name=f"folds[{i}].n_val")
        if n_val != len(val_ids):
            raise ValueError(
                f"folds[{i}].n_val is {n_val}, len(val_ids) is {len(val_ids)}"
            )
        train_counts = _require_stratum_counts(
            fold_payload["train_stratum_counts"],
            name=f"folds[{i}].train_stratum_counts",
        )
        val_counts = _require_stratum_counts(
            fold_payload["val_stratum_counts"],
            name=f"folds[{i}].val_stratum_counts",
        )
        if sum(train_counts.values()) != n_train:
            raise ValueError(
                f"folds[{i}] train_stratum_counts do not sum to n_train"
            )
        if sum(val_counts.values()) != n_val:
            raise ValueError(
                f"folds[{i}] val_stratum_counts do not sum to n_val"
            )
        folds[i] = Fold(fold_id=fold_id, train_ids=(), val_ids=val_ids)
        summaries[i] = FoldSummary(
            fold_id=fold_id,
            n_train=n_train,
            n_val=n_val,
            train_stratum_counts=train_counts,
            val_stratum_counts=val_counts,
        )
    universe_set: set[int] = set()
    for i in range(n_splits):
        universe_set.update(folds[i].val_ids)
    if n_rows != len(universe_set):
        raise ValueError(
            f"n_rows is {n_rows}, union of val ids is {len(universe_set)}"
        )
    for i in range(n_splits):
        train_ids = tuple(sorted(universe_set.difference(folds[i].val_ids)))
        if summaries[i].n_train != len(train_ids):
            raise ValueError(
                f"folds[{i}].n_train is {summaries[i].n_train}, "
                f"reconstructed train has {len(train_ids)}"
            )
        folds[i] = Fold(
            fold_id=folds[i].fold_id,
            train_ids=train_ids,
            val_ids=folds[i].val_ids,
        )
    filled_folds = tuple(folds)
    universe = np.array(sorted(universe_set), dtype=np.int64)
    require_partition(filled_folds, record_ids=universe)
    return FlagStratifiedCv(
        n_splits=n_splits,
        random_state=random_state,
        shuffle=shuffle,
        id_column=id_column,
        target_column=target_column,
        alarm_flag_columns=alarm_flag_columns,
        n_rows=n_rows,
        stratum_counts=stratum_counts,
        folds=filled_folds,
        fold_summaries=tuple(summaries),
    )


def write_flag_stratified_cv(
    document: FlagStratifiedCv,
    *,
    path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the frozen partition. Refuses to replace an existing file unless asked."""
    dest = path if path is not None else flag_stratified_cv_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"cv splits already exist at {dest}. Pass overwrite=True to replace them."
        )
    text = json.dumps(
        dump_flag_stratified_cv(document),
        separators=(",", ":"),
        allow_nan=False,
    )
    dest.write_text(text, encoding="utf-8")
    return dest


def load_flag_stratified_cv(*, path: Path | None = None) -> FlagStratifiedCv:
    """Load the frozen partition from disk."""
    source = path if path is not None else flag_stratified_cv_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    return parse_flag_stratified_cv(payload)


def n_from_stratum_counts(
    counts: dict[str, int],
    *,
    names: tuple[str, ...],
) -> int:
    """Sum selected named cells of a stratum-count table."""
    total = 0
    n_names = len(names)
    for i in range(n_names):
        total += counts[names[i]]
    return total


def n_fraud_from_counts(counts: dict[str, int]) -> int:
    """Fraud occupancy from an alarm × target count table."""
    return n_from_stratum_counts(counts, names=FRAUD_STRATUM_NAMES)


def n_alarm_from_counts(counts: dict[str, int]) -> int:
    """Alarm occupancy from an alarm × target count table."""
    return n_from_stratum_counts(counts, names=ALARM_STRATUM_NAMES)


def _ids_to_index(
    record_ids: tuple[int, ...],
    *,
    position: dict[int, int],
) -> np.ndarray:
    n_ids = len(record_ids)
    index = np.zeros(n_ids, dtype=np.int64)
    for i in range(n_ids):
        index[i] = position[record_ids[i]]
    return index


def _require_dict(payload: object, *, name: str) -> dict:
    if type(payload) is not dict:
        raise TypeError(f"{name} must be a dict, got {type(payload)!r}")
    return payload


def _require_exact_keys(
    payload: dict,
    keys: tuple[str, ...],
    *,
    name: str,
) -> None:
    expected = set(keys)
    actual = set(payload)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{name} keys do not match: missing={missing!r} extra={extra!r}"
        )


def _require_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value)!r}")
    return value


def _require_str(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty str, got {value!r}")
    return value


def _require_str_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a list, got {type(value)!r}")
    n_items = len(value)
    items: list[str] = [""] * n_items
    for i in range(n_items):
        items[i] = _require_str(value[i], name=f"{name}[{i}]")
    return tuple(items)


def _require_int_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a list, got {type(value)!r}")
    n_items = len(value)
    items: list[int] = [0] * n_items
    for i in range(n_items):
        items[i] = _require_int(value[i], name=f"{name}[{i}]")
    return tuple(items)


def _require_stratum_counts(value: object, *, name: str) -> dict[str, int]:
    payload = _require_dict(value, name=name)
    expected = set(STRATUM_CODE_TO_NAME.values())
    actual = set(payload)
    if expected != actual:
        raise ValueError(
            f"{name} must have keys {sorted(expected)!r}, got {sorted(actual)!r}"
        )
    counts = {key: 0 for key in STRATUM_CODE_TO_NAME.values()}
    for key in STRATUM_CODE_TO_NAME.values():
        counts[key] = _require_int(payload[key], name=f"{name}.{key}")
        if counts[key] < 0:
            raise ValueError(f"{name}.{key} must be >= 0, got {counts[key]}")
    return counts
