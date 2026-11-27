"""``imported``: align a stack CV document's val_ids onto inner_table row_ids."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from dl_toolbox.base.hashing.identity import hash_file, is_full_sha256
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys

from ccf_tabformer_dl.data_contract.corrected_table import SPLIT_TRAIN
from ccf_tabformer_dl.data_contract.folds.collection import (
    FOLD_ID_FILE,
    FOLD_UNASSIGNED,
    payload_meta_from_dir,
)
from ccf_tabformer_dl.data_contract.folds.preparation.preparation import FoldsPreparation
from ccf_tabformer_dl.data_contract.inner_table.collection import InnerTableCollection
from ccf_tabformer_dl.file_utils import get_project_root, package_artifacts_dir

IMPORTED_STRATEGY = "imported"
ALARM_PROFILE_SCHEME = "alarm_profile_stratified_cv"
_ARG_KEYS = ("inner_table_hash", "scheme")
_FOLD_KEYS = ("fold_id", "n_train", "n_val", "val_ids")


class ImportedFoldsPreparation(FoldsPreparation):
    strategy = IMPORTED_STRATEGY

    def __init__(self, *, inner_table_hash: str, scheme: str) -> None:
        digest = parse_non_empty_str(inner_table_hash, label="inner_table_hash")
        if not is_full_sha256(digest):
            raise ValueError(
                f"inner_table_hash must be a collection hash, got {inner_table_hash!r}"
            )
        scheme_name = parse_non_empty_str(scheme, label="scheme")
        source = resolve_cv_document_path(scheme=scheme_name)
        parent = InnerTableCollection.load(digest)
        self._inner_table_hash = digest
        self._scheme = scheme_name
        self._source_path = source
        self._source_sha256 = hash_file(source)
        self._n_parent_rows = parent.n_rows()
        self._n_splits = n_splits_from_document(read_cv_document(source))

    def generating_process_signature(self) -> dict:
        return {
            "strategy": self.strategy,
            "args": {
                "inner_table_hash": self._inner_table_hash,
                "scheme": self._scheme,
                "source_sha256": self._source_sha256,
                "n_parent_rows": self._n_parent_rows,
                "n_splits": self._n_splits,
            },
        }

    def write_payload(self, *, artifacts_dir: Path) -> dict:
        parent = InnerTableCollection.load(self._inner_table_hash)
        document = read_cv_document(self._source_path)
        fold_id = assign_fold_ids(
            row_ids=np.asarray(parent.row_ids()),
            split=np.asarray(parent.split()),
            document=document,
        )
        np.save(artifacts_dir / FOLD_ID_FILE, fold_id)
        return payload_meta_from_dir(
            artifacts_dir,
            n_parent_rows=self._n_parent_rows,
            n_splits=self._n_splits,
        )


def resolve_cv_document_path(*, scheme: str) -> Path:
    """Packaged copy first, then the sibling ``ccf_ml`` stack artifact."""
    filename = f"{scheme}.json"
    packaged = package_artifacts_dir() / "cv_splits" / filename
    if packaged.is_file():
        return packaged
    sibling = (
        get_project_root().parent
        / "ccf_ml"
        / "src"
        / "ccf_ml"
        / "package_artifacts"
        / "cv_splits"
        / filename
    )
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        f"CV document {filename!r} not found at {packaged} or {sibling}"
    )


def read_cv_document(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"CV document must be a JSON object, got {type(payload)!r}")
    return payload


def n_splits_from_document(document: dict) -> int:
    n_splits = document.get("n_splits")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError(f"n_splits must be an int >= 2, got {n_splits!r}")
    return n_splits


def assign_fold_ids(
    *,
    row_ids: np.ndarray,
    split: np.ndarray,
    document: dict,
) -> np.ndarray:
    """Build a ``(n_rows,)`` int8 sidecar: fold id on train rows, ``-1`` on test."""
    if row_ids.ndim != 1 or split.ndim != 1 or row_ids.shape != split.shape:
        raise ValueError(
            "row_ids and split must be 1-D and the same shape, "
            f"got {row_ids.shape} and {split.shape}"
        )
    n_splits = n_splits_from_document(document)
    folds = document.get("folds")
    if not isinstance(folds, list):
        raise TypeError(f"folds must be a list, got {type(folds)!r}")
    if len(folds) != n_splits:
        raise ValueError(
            f"folds must have length n_splits={n_splits}, got {len(folds)}"
        )
    train_pos = np.flatnonzero(split == SPLIT_TRAIN)
    if int(train_pos.shape[0]) < 1:
        raise ValueError("train split is empty")
    document_n_rows = document.get("n_rows")
    if document_n_rows is not None and int(document_n_rows) != int(train_pos.shape[0]):
        raise ValueError(
            f"CV document n_rows {document_n_rows} != inner_table train "
            f"rows {int(train_pos.shape[0])}"
        )
    train_ids = np.asarray(row_ids[train_pos], dtype=np.int64)
    unique = np.unique(train_ids)
    if int(unique.shape[0]) != int(train_ids.shape[0]):
        raise ValueError("train row_ids must be unique")
    order = np.argsort(train_ids, kind="mergesort")
    sorted_ids = train_ids[order]
    sorted_pos = train_pos[order]
    fold_id = np.full(int(row_ids.shape[0]), FOLD_UNASSIGNED, dtype=np.int8)
    expected_n_val = [0] * n_splits
    for i in range(n_splits):
        fold = folds[i]
        if not isinstance(fold, dict):
            raise TypeError(f"folds[{i}] must be a JSON object, got {type(fold)!r}")
        require_exact_keys(fold, _FOLD_KEYS, label=f"folds[{i}]")
        fold_id_value = fold["fold_id"]
        if fold_id_value != i:
            raise ValueError(
                f"folds[{i}].fold_id must be {i}, got {fold_id_value!r}"
            )
        val_ids = np.asarray(fold["val_ids"], dtype=np.int64)
        if val_ids.ndim != 1:
            raise ValueError(
                f"folds[{i}].val_ids must be 1-D, got {val_ids.shape}"
            )
        n_val = int(fold["n_val"])
        if int(val_ids.shape[0]) != n_val:
            raise ValueError(
                f"folds[{i}].n_val {n_val} != len(val_ids) {int(val_ids.shape[0])}"
            )
        positions = _lookup_train_positions(
            sorted_ids=sorted_ids,
            sorted_pos=sorted_pos,
            query=val_ids,
            fold_index=i,
        )
        already = fold_id[positions]
        if np.any(already != FOLD_UNASSIGNED):
            raise ValueError(f"folds[{i}] reassigns a row already in another fold")
        fold_id[positions] = np.int8(i)
        expected_n_val[i] = n_val
    assigned = fold_id[train_pos]
    missing = int(np.sum(assigned == FOLD_UNASSIGNED))
    if missing > 0:
        raise ValueError(
            f"{missing} train rows have no fold; every train row_id must appear "
            "in exactly one folds[k].val_ids"
        )
    actual_n_val = [int(np.sum(fold_id == k)) for k in range(n_splits)]
    if actual_n_val != expected_n_val:
        raise ValueError(
            f"assigned n_val {actual_n_val} != document n_val {expected_n_val}"
        )
    return fold_id


def _lookup_train_positions(
    *,
    sorted_ids: np.ndarray,
    sorted_pos: np.ndarray,
    query: np.ndarray,
    fold_index: int,
) -> np.ndarray:
    index = np.searchsorted(sorted_ids, query)
    n = int(sorted_ids.shape[0])
    in_range = (index >= 0) & (index < n)
    if not bool(np.all(in_range)) or not np.array_equal(sorted_ids[index], query):
        raise ValueError(
            f"folds[{fold_index}].val_ids contains ids absent from train row_ids"
        )
    return sorted_pos[index]


def build_imported_args(args: dict) -> dict:
    require_exact_keys(args, _ARG_KEYS, label="args")
    return {
        "inner_table_hash": args["inner_table_hash"],
        "scheme": args["scheme"],
    }
