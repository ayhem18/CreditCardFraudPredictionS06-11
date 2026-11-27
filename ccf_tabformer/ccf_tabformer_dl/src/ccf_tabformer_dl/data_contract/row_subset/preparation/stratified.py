"""``stratified_by_label``: sample train rows, keeping both fraud classes."""

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

from ccf_tabformer_dl.data_contract.corrected_table import LABEL_MISSING, SPLIT_TRAIN
from ccf_tabformer_dl.data_contract.inner_table.collection import InnerTableCollection
from ccf_tabformer_dl.data_contract.row_subset.collection import (
    ROW_INDEX_FILE,
    payload_meta_from_dir,
)
from ccf_tabformer_dl.data_contract.row_subset.preparation.preparation import RowSubsetPreparation

STRATIFIED_BY_LABEL_STRATEGY = "stratified_by_label"
SPLIT_NAME_TRAIN = "train"
_ARG_KEYS = ("inner_table_hash", "n_rows", "seed", "split")


class StratifiedByLabelPreparation(RowSubsetPreparation):
    strategy = STRATIFIED_BY_LABEL_STRATEGY

    def __init__(
        self,
        *,
        inner_table_hash: str,
        n_rows: int,
        seed: int,
        split: str,
    ) -> None:
        digest = parse_non_empty_str(inner_table_hash, label="inner_table_hash")
        if not is_full_sha256(digest):
            raise ValueError(
                f"inner_table_hash must be a collection hash, got {inner_table_hash!r}"
            )
        if not isinstance(n_rows, int) or n_rows < 2:
            raise ValueError(f"n_rows must be an int >= 2, got {n_rows!r}")
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
        self._n_rows = n_rows
        self._seed = seed
        self._split = split_name
        parent = InnerTableCollection.load(digest)
        self._n_parent_rows = parent.n_rows()

    def generating_process_signature(self) -> dict:
        return {
            "strategy": self.strategy,
            "args": {
                "inner_table_hash": self._inner_table_hash,
                "n_rows": self._n_rows,
                "seed": self._seed,
                "split": self._split,
                "n_parent_rows": self._n_parent_rows,
            },
        }

    def write_payload(self, *, artifacts_dir: Path) -> dict:
        parent = InnerTableCollection.load(self._inner_table_hash)
        row_index = sample_stratified_train_rows(
            split=np.asarray(parent.split()),
            labels=np.asarray(parent.labels()),
            n_rows=self._n_rows,
            seed=self._seed,
        )
        np.save(artifacts_dir / ROW_INDEX_FILE, row_index.astype(np.int64, copy=False))
        return payload_meta_from_dir(
            artifacts_dir, n_parent_rows=self._n_parent_rows
        )


def sample_stratified_train_rows(
    *,
    split: np.ndarray,
    labels: np.ndarray,
    n_rows: int,
    seed: int,
) -> np.ndarray:
    """Sample ``n_rows`` train indices without replacement, both classes present."""
    train_pos = np.flatnonzero(split == SPLIT_TRAIN)
    if int(train_pos.shape[0]) < n_rows:
        raise ValueError(
            f"n_rows {n_rows} exceeds n_train {int(train_pos.shape[0])}"
        )
    train_labels = labels[train_pos]
    if np.any(train_labels == LABEL_MISSING):
        raise ValueError("train split has missing labels")
    pos = train_pos[train_labels == 1]
    neg = train_pos[train_labels == 0]
    if int(pos.shape[0]) < 1 or int(neg.shape[0]) < 1:
        raise ValueError("train split must contain both label classes")
    n_pos_pool = int(pos.shape[0])
    n_neg_pool = int(neg.shape[0])
    n_pos = int(round(n_rows * (n_pos_pool / (n_pos_pool + n_neg_pool))))
    n_pos = max(1, min(n_pos_pool, n_pos))
    n_neg = n_rows - n_pos
    if n_neg < 1:
        n_neg = 1
        n_pos = n_rows - 1
    if n_neg > n_neg_pool:
        n_neg = n_neg_pool
        n_pos = n_rows - n_neg
    if n_pos < 1 or n_neg < 1 or n_pos > n_pos_pool:
        raise ValueError(
            f"cannot draw stratified n_rows={n_rows} from "
            f"n_pos={n_pos_pool} n_neg={n_neg_pool}"
        )
    rng = np.random.default_rng(seed)
    chosen_pos = rng.choice(pos, size=n_pos, replace=False)
    chosen_neg = rng.choice(neg, size=n_neg, replace=False)
    chosen = np.concatenate((chosen_pos, chosen_neg))
    rng.shuffle(chosen)
    return chosen.astype(np.int64, copy=False)


def build_stratified_by_label_args(args: dict) -> dict:
    require_exact_keys(args, _ARG_KEYS, label="args")
    return {
        "inner_table_hash": args["inner_table_hash"],
        "n_rows": args["n_rows"],
        "seed": args["seed"],
        "split": args["split"],
    }
