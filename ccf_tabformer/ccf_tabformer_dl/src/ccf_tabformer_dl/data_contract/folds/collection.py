"""Frozen fold-id sidecar aligned to an inner_table. Payload is ``fold_id.npy``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.base.hashing.identity import hash_file
from dl_toolbox.data_contract.configs import load_collection_configuration
from dl_toolbox.data_contract.names import assert_generating_process_name

from ccf_tabformer_dl.data_contract.corrected_table import SPLIT_TEST
from ccf_tabformer_dl.data_contract.inner_table.collection import InnerTableCollection
from ccf_tabformer_dl.file_utils import (
    data_for_training_dir,
    package_artifact_directory,
    project_hash_manager,
)

FOLD_ID_FILE = "fold_id.npy"
FOLD_UNASSIGNED = -1


def _collection_identity(
    *, generating_process_signature: dict, payload_meta: dict
) -> dict:
    return {
        "generating_process_signature": generating_process_signature,
        "payload_meta": payload_meta,
    }


class FoldsCollection:
    def __init__(
        self,
        *,
        generating_process_name: str,
        generating_process_signature: dict,
        payload_meta: dict,
    ) -> None:
        manager = project_hash_manager()
        name = assert_generating_process_name(
            generating_process_name, manager=manager
        )
        if not isinstance(generating_process_signature, dict):
            raise TypeError("generating_process_signature must be a dict")
        artifacts = self.artifacts_dir(name)
        n_parent = int(generating_process_signature["args"]["n_parent_rows"])
        n_splits = int(generating_process_signature["args"]["n_splits"])
        checked = payload_meta_from_dir(
            artifacts, n_parent_rows=n_parent, n_splits=n_splits
        )
        if checked != payload_meta:
            raise ValueError("folds payload_meta does not match files on disk")
        collection_hash = manager.hash_payload(
            _collection_identity(
                generating_process_signature=generating_process_signature,
                payload_meta=checked,
            )
        )
        self.generating_process_name = name
        self.generating_process_signature = generating_process_signature
        self.collection_hash = collection_hash
        self._payload_meta = checked
        self._artifacts = artifacts

    @staticmethod
    def configs_dir() -> Directory:
        return package_artifact_directory("folds/collections")

    @staticmethod
    def artifacts_dir(generating_process_name: str) -> Path:
        name = assert_generating_process_name(
            generating_process_name, manager=project_hash_manager()
        )
        return data_for_training_dir() / "folds" / name

    def fold_id(self) -> np.ndarray:
        return np.load(self._artifacts / FOLD_ID_FILE, mmap_mode="r")

    def n_rows(self) -> int:
        return int(self._payload_meta["n_rows"])

    def n_splits(self) -> int:
        return int(self._payload_meta["n_splits"])

    def parent_collection(self) -> InnerTableCollection:
        parent_hash = self.generating_process_signature["args"]["inner_table_hash"]
        return InnerTableCollection.load(parent_hash)

    def related_collections(self) -> dict:
        return {"inner_table": self.parent_collection}

    def train_row_index(self, *, held_out_fold: int) -> np.ndarray:
        self._require_fold(held_out_fold)
        fold_id = np.asarray(self.fold_id())
        return np.flatnonzero(
            (fold_id != held_out_fold) & (fold_id != FOLD_UNASSIGNED)
        ).astype(np.int64)

    def val_row_index(self, *, held_out_fold: int) -> np.ndarray:
        self._require_fold(held_out_fold)
        fold_id = np.asarray(self.fold_id())
        return np.flatnonzero(fold_id == held_out_fold).astype(np.int64)

    def test_row_index(self) -> np.ndarray:
        split = np.asarray(self.parent_collection().split())
        return np.flatnonzero(split == SPLIT_TEST).astype(np.int64)

    def to_dict(self) -> dict:
        return {
            "hash": self.collection_hash,
            "generating_process_name": self.generating_process_name,
            "generating_process_signature": self.generating_process_signature,
            "payload_meta": self._payload_meta,
        }

    def _require_fold(self, held_out_fold: int) -> None:
        n_splits = self.n_splits()
        if (
            isinstance(held_out_fold, bool)
            or not isinstance(held_out_fold, int)
            or held_out_fold < 0
            or held_out_fold >= n_splits
        ):
            raise ValueError(
                f"held_out_fold must be in [0, {n_splits}), got {held_out_fold!r}"
            )

    @classmethod
    def from_payload(cls, collection_config: dict) -> FoldsCollection:
        collection = cls(
            generating_process_name=collection_config["generating_process_name"],
            generating_process_signature=collection_config[
                "generating_process_signature"
            ],
            payload_meta=collection_config["payload_meta"],
        )
        stored_hash = collection_config["hash"]
        if stored_hash != collection.collection_hash:
            raise ValueError(
                "folds collection hash does not match identity "
                f"(stored {stored_hash}, recomputed {collection.collection_hash})"
            )
        return collection

    @classmethod
    def load(cls, identity: str) -> FoldsCollection:
        _file, payload = load_collection_configuration(
            cls.configs_dir(), identity, manager=project_hash_manager()
        )
        return cls.from_payload(payload)


def payload_meta_from_dir(
    artifacts: Path, *, n_parent_rows: int, n_splits: int
) -> dict:
    path = artifacts / FOLD_ID_FILE
    fold_id = np.load(path, mmap_mode="r")
    if fold_id.ndim != 1 or int(fold_id.shape[0]) != n_parent_rows:
        raise ValueError(
            f"fold_id shape {fold_id.shape} does not match n_parent_rows "
            f"{n_parent_rows}"
        )
    assigned = fold_id[fold_id != FOLD_UNASSIGNED]
    if int(assigned.shape[0]) < 1:
        raise ValueError("fold_id assigns no train rows")
    if int(assigned.min()) < 0 or int(assigned.max()) >= n_splits:
        raise ValueError(
            f"assigned fold ids must be in [0, {n_splits}), "
            f"got min={int(assigned.min())} max={int(assigned.max())}"
        )
    n_val = [int(np.sum(fold_id == k)) for k in range(n_splits)]
    return {
        "n_rows": n_parent_rows,
        "n_parent_rows": n_parent_rows,
        "n_splits": n_splits,
        "n_train": int(assigned.shape[0]),
        "n_test": int(np.sum(fold_id == FOLD_UNASSIGNED)),
        "n_val": n_val,
        "fold_id_sha256": hash_file(path),
    }
