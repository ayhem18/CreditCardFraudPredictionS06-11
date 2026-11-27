"""Load the working CSVs and hash them as the ``corrected_table`` identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dl_toolbox.base.hashing.identity import hash_file

from ccf_tabformer_dl.data_contract.columns import (
    ALARM_FLAG_COLUMNS,
    BOOLEAN_FALSE,
    BOOLEAN_TRUE,
    BOOLEAN_WRITTEN,
    ID_COLUMN,
    LABEL_COLUMN,
    feature_names,
)
from ccf_tabformer_dl.file_utils import get_test_csv_path, get_train_csv_path

SPLIT_TRAIN = 0
SPLIT_TEST = 1
LABEL_MISSING = -1


def corrected_table_identity() -> dict:
    """SHA-256 of the corrected train and test CSV bytes."""
    return {
        "train_sha256": hash_file(get_train_csv_path()),
        "test_sha256": hash_file(get_test_csv_path()),
    }


def load_corrected_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train and test as written strings. Feature columns must be present."""
    names = feature_names()
    train_cols = (ID_COLUMN, *names, LABEL_COLUMN)
    test_cols = (ID_COLUMN, *names)
    train = pd.read_csv(get_train_csv_path(), dtype=str, usecols=list(train_cols))
    test = pd.read_csv(get_test_csv_path(), dtype=str, usecols=list(test_cols))
    _require_columns(train, columns=train_cols, label="train")
    _require_columns(test, columns=test_cols, label="test")
    if train.empty:
        raise ValueError("corrected train CSV has no rows")
    if test.empty:
        raise ValueError("corrected test CSV has no rows")
    return train, test


def pooled_feature_series(
    *, train: pd.DataFrame, test: pd.DataFrame, column: str
) -> pd.Series:
    """Train then test written values for one feature, as a single series."""
    return pd.concat(
        (train[column], test[column]),
        axis=0,
        ignore_index=True,
    )


def pooled_row_ids(*, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    train_ids = _int64_column(train[ID_COLUMN], label=f"train {ID_COLUMN}")
    test_ids = _int64_column(test[ID_COLUMN], label=f"test {ID_COLUMN}")
    n_train = int(train_ids.shape[0])
    n_test = int(test_ids.shape[0])
    row_ids = np.empty(n_train + n_test, dtype=np.int64)
    row_ids[:n_train] = train_ids
    row_ids[n_train:] = test_ids
    return row_ids


def pooled_split(*, n_train: int, n_test: int) -> np.ndarray:
    split = np.empty(n_train + n_test, dtype=np.uint8)
    split[:n_train] = SPLIT_TRAIN
    split[n_train:] = SPLIT_TEST
    return split


def written_boolean_array(series: pd.Series, *, column: str) -> np.ndarray:
    """Parse a written ``False``/``True`` column as a 1-d bool array."""
    values = series.to_numpy()
    true_mask = values == BOOLEAN_TRUE
    false_mask = values == BOOLEAN_FALSE
    if not np.all(true_mask | false_mask):
        raise ValueError(
            f"{column} must be one of {BOOLEAN_WRITTEN}"
        )
    return true_mask


def alarm_flag_union(frame: pd.DataFrame) -> np.ndarray:
    """True where at least one EDA alarm flag is on (Cut 1 ``is_alarm``)."""
    n_flags = len(ALARM_FLAG_COLUMNS)
    n_rows = int(frame.shape[0])
    stacked = np.empty((n_rows, n_flags), dtype=bool)
    for i in range(n_flags):
        name = ALARM_FLAG_COLUMNS[i]
        if name not in frame.columns:
            raise ValueError(f"frame is missing alarm flag column {name!r}")
        stacked[:, i] = written_boolean_array(frame[name], column=name)
    return stacked.any(axis=1)


def pooled_labels(*, train: pd.DataFrame, n_test: int) -> np.ndarray:
    n_train = int(train.shape[0])
    labels = np.empty(n_train + n_test, dtype=np.int8)
    labels[:n_train] = _binary_labels(train[LABEL_COLUMN], label=f"train {LABEL_COLUMN}")
    labels[n_train:] = LABEL_MISSING
    return labels


def _int64_column(series: pd.Series, *, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="raise").to_numpy()
    as_int = values.astype(np.int64, copy=False)
    if not np.allclose(values.astype(np.float64), as_int.astype(np.float64)):
        raise ValueError(f"{label} must be whole numbers")
    return as_int


def _binary_labels(series: pd.Series, *, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="raise").to_numpy()
    as_int = values.astype(np.int8, copy=False)
    if not np.allclose(values.astype(np.float64), as_int.astype(np.float64)):
        raise ValueError(f"{label} must be whole numbers")
    unique_train = {int(v) for v in np.unique(as_int)}
    if unique_train - {0, 1}:
        raise ValueError(f"{label} must be 0/1, got {sorted(unique_train)}")
    return as_int


def _require_columns(frame: pd.DataFrame, *, columns: tuple[str, ...], label: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{label} CSV missing columns {missing}")
