"""Frozen row-index list into an inner_table. Payload is ``row_index.npy``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.base.hashing.identity import hash_file
from dl_toolbox.data_contract.configs import load_collection_configuration
from dl_toolbox.data_contract.names import assert_generating_process_name

from ccf_tabformer_dl.data_contract.inner_table.collection import InnerTableCollection
from ccf_tabformer_dl.file_utils import (
    data_for_training_dir,
    package_artifact_directory,
    project_hash_manager,
)

ROW_INDEX_FILE = "row_index.npy"


def _collection_identity(
    *, generating_process_signature: dict, payload_meta: dict
) -> dict:
    return {
        "generating_process_signature": generating_process_signature,
        "payload_meta": payload_meta,
    }


class RowSubsetCollection:
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
        checked = payload_meta_from_dir(artifacts, n_parent_rows=n_parent)
        if checked != payload_meta:
            raise ValueError(
                "row_subset payload_meta does not match files on disk"
            )
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
        return package_artifact_directory("row_subset/collections")

    @staticmethod
    def artifacts_dir(generating_process_name: str) -> Path:
        name = assert_generating_process_name(
            generating_process_name, manager=project_hash_manager()
        )
        return data_for_training_dir() / "row_subset" / name

    def row_index(self) -> np.ndarray:
        return np.load(self._artifacts / ROW_INDEX_FILE, mmap_mode="r")

    def n_rows(self) -> int:
        return int(self._payload_meta["n_rows"])

    def parent_collection(self) -> InnerTableCollection:
        parent_hash = self.generating_process_signature["args"]["inner_table_hash"]
        return InnerTableCollection.load(parent_hash)

    def related_collections(self) -> dict:
        return {"inner_table": self.parent_collection}

    def to_dict(self) -> dict:
        return {
            "hash": self.collection_hash,
            "generating_process_name": self.generating_process_name,
            "generating_process_signature": self.generating_process_signature,
            "payload_meta": self._payload_meta,
        }

    @classmethod
    def from_payload(cls, collection_config: dict) -> RowSubsetCollection:
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
                "row_subset collection hash does not match identity "
                f"(stored {stored_hash}, recomputed {collection.collection_hash})"
            )
        return collection

    @classmethod
    def load(cls, identity: str) -> RowSubsetCollection:
        _file, payload = load_collection_configuration(
            cls.configs_dir(), identity, manager=project_hash_manager()
        )
        return cls.from_payload(payload)


def payload_meta_from_dir(artifacts: Path, *, n_parent_rows: int) -> dict:
    path = artifacts / ROW_INDEX_FILE
    row_index = np.load(path, mmap_mode="r")
    if row_index.ndim != 1 or int(row_index.shape[0]) < 1:
        raise ValueError(f"row_index must be 1-D and non-empty, got {row_index.shape}")
    n_rows = int(row_index.shape[0])
    if int(row_index.min()) < 0 or int(row_index.max()) >= n_parent_rows:
        raise ValueError(
            f"row_index values must be in [0, {n_parent_rows}), "
            f"got min={int(row_index.min())} max={int(row_index.max())}"
        )
    unique = np.unique(row_index)
    if int(unique.shape[0]) != n_rows:
        raise ValueError("row_index must not contain duplicates")
    return {
        "n_rows": n_rows,
        "n_parent_rows": n_parent_rows,
        "row_index_sha256": hash_file(path),
    }
