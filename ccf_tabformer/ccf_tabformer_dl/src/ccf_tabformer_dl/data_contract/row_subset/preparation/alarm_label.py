"""``balanced_alarm_label``: equal train draws from each (is_alarm, is_fraud) cell."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dl_toolbox.base.hashing.identity import is_full_sha256
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import (
    one_of_message,
    parse_one_of,
    require_exact_keys,
)

from ccf_tabformer_dl.data_contract.columns import ALARM_FLAG_COLUMNS
from ccf_tabformer_dl.data_contract.corrected_table import (
    LABEL_MISSING,
    SPLIT_TRAIN,
    alarm_flag_union,
    load_corrected_frames,
)
from ccf_tabformer_dl.data_contract.inner_table.collection import InnerTableCollection
from ccf_tabformer_dl.data_contract.row_subset.collection import (
    ROW_INDEX_FILE,
    payload_meta_from_dir,
)
from ccf_tabformer_dl.data_contract.row_subset.preparation.preparation import RowSubsetPreparation
from ccf_tabformer_dl.data_contract.row_subset.preparation.stratified import SPLIT_NAME_TRAIN

BALANCED_ALARM_LABEL_STRATEGY = "balanced_alarm_label"
N_ALARM_LABEL_CELLS = 4
_ARG_KEYS = ("inner_table_hash", "n_per_cell", "seed", "split")


class BalancedAlarmLabelPreparation(RowSubsetPreparation):
    strategy = BALANCED_ALARM_LABEL_STRATEGY

    def __init__(
        self,
        *,
        inner_table_hash: str,
        n_per_cell: int,
        seed: int,
        split: str,
    ) -> None:
        digest = parse_non_empty_str(inner_table_hash, label="inner_table_hash")
        if not is_full_sha256(digest):
            raise ValueError(
                f"inner_table_hash must be a collection hash, got {inner_table_hash!r}"
            )
        if not isinstance(n_per_cell, int) or n_per_cell < 1:
            raise ValueError(f"n_per_cell must be an int >= 1, got {n_per_cell!r}")
        if not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed)!r}")
        split_name = parse_one_of(
            split,
            allowed=(SPLIT_NAME_TRAIN,),
            label="split",
            error_type=ValueError,
            error_message=one_of_message,
        )
        self._inner_table_hash = digest
        self._n_per_cell = n_per_cell
        self._seed = seed
        self._split = split_name
        parent = InnerTableCollection.load(digest)
        self._n_parent_rows = parent.n_rows()

    def generating_process_signature(self) -> dict:
        return {
            "strategy": self.strategy,
            "args": {
                "inner_table_hash": self._inner_table_hash,
                "n_per_cell": self._n_per_cell,
                "seed": self._seed,
                "split": self._split,
                "n_parent_rows": self._n_parent_rows,
                "n_cells": N_ALARM_LABEL_CELLS,
                "alarm_flag_columns": list(ALARM_FLAG_COLUMNS),
            },
        }

    def write_payload(self, *, artifacts_dir: Path) -> dict:
        parent = InnerTableCollection.load(self._inner_table_hash)
        train, _test = load_corrected_frames()
        n_train = int(train.shape[0])
        split = np.asarray(parent.split())
        labels = np.asarray(parent.labels())
        n_train_split = int(np.sum(split == SPLIT_TRAIN))
        if n_train_split != n_train:
            raise ValueError(
                f"corrected train has {n_train} rows, inner_table train split "
                f"has {n_train_split}"
            )
        if not np.all(split[:n_train] == SPLIT_TRAIN):
            raise ValueError(
                "inner_table train rows must occupy the leading block"
            )
        is_alarm = np.zeros(self._n_parent_rows, dtype=bool)
        is_alarm[:n_train] = alarm_flag_union(train)
        row_index = sample_balanced_alarm_label_rows(
            split=split,
            labels=labels,
            is_alarm=is_alarm,
            n_per_cell=self._n_per_cell,
            seed=self._seed,
        )
        np.save(artifacts_dir / ROW_INDEX_FILE, row_index.astype(np.int64, copy=False))
        return payload_meta_from_dir(
            artifacts_dir, n_parent_rows=self._n_parent_rows
        )


def sample_balanced_alarm_label_rows(
    *,
    split: np.ndarray,
    labels: np.ndarray,
    is_alarm: np.ndarray,
    n_per_cell: int,
    seed: int,
) -> np.ndarray:
    """Sample ``n_per_cell`` train rows from each (is_alarm, is_fraud) cell."""
    if split.shape != labels.shape or split.shape != is_alarm.shape:
        raise ValueError(
            "split, labels, and is_alarm must share shape, got "
            f"{split.shape}, {labels.shape}, {is_alarm.shape}"
        )
    train_pos = np.flatnonzero(split == SPLIT_TRAIN)
    train_labels = labels[train_pos]
    if np.any(train_labels == LABEL_MISSING):
        raise ValueError("train split has missing labels")
    train_alarm = is_alarm[train_pos].astype(np.int64, copy=False)
    train_fraud = train_labels.astype(np.int64, copy=False)
    stratum = train_alarm * 2 + train_fraud
    rng = np.random.default_rng(seed)
    chosen_parts = [None] * N_ALARM_LABEL_CELLS
    for cell in range(N_ALARM_LABEL_CELLS):
        pool = train_pos[stratum == cell]
        n_pool = int(pool.shape[0])
        if n_pool < n_per_cell:
            raise ValueError(
                f"cell {cell} (is_alarm={cell // 2}, is_fraud={cell % 2}) "
                f"has {n_pool} train rows, need {n_per_cell}"
            )
        chosen_parts[cell] = rng.choice(pool, size=n_per_cell, replace=False)
    chosen = np.concatenate(chosen_parts)
    rng.shuffle(chosen)
    return chosen.astype(np.int64, copy=False)


def build_balanced_alarm_label_args(args: dict) -> dict:
    require_exact_keys(args, _ARG_KEYS, label="args")
    return {
        "inner_table_hash": args["inner_table_hash"],
        "n_per_cell": args["n_per_cell"],
        "seed": args["seed"],
        "split": args["split"],
    }
