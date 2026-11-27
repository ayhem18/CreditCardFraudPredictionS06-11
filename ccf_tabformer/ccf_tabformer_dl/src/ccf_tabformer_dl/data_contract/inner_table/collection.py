"""Frozen tokenized table. Payload is numpy under storage/data_for_training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.base.hashing.identity import hash_file
from dl_toolbox.data_contract.configs import load_collection_configuration
from dl_toolbox.data_contract.names import assert_generating_process_name

from ccf_tabformer_dl.data_contract.inner_table.layout import InnerTableLayout
from ccf_tabformer_dl.data_contract.vocabulary.collection import VocabularyCollection
from ccf_tabformer_dl.file_utils import (
    data_for_training_dir,
    package_artifact_directory,
    project_hash_manager,
)
from ccf_tabformer_dl.tabformer_components.table_spec import FrozenTableSpec

TOKEN_IDS_FILE = "token_ids.npy"
ROW_IDS_FILE = "row_ids.npy"
SPLIT_FILE = "split.npy"
LABELS_FILE = "labels.npy"
PAYLOAD_FILES = (TOKEN_IDS_FILE, ROW_IDS_FILE, SPLIT_FILE, LABELS_FILE)


def _collection_identity(
    *, generating_process_signature: dict, layout: dict, payload_meta: dict
) -> dict:
    return {
        "generating_process_signature": generating_process_signature,
        "layout": layout,
        "payload_meta": payload_meta,
    }


class InnerTableCollection:
    def __init__(
        self,
        *,
        generating_process_name: str,
        generating_process_signature: dict,
        layout: dict,
        payload_meta: dict,
    ) -> None:
        manager = project_hash_manager()
        name = assert_generating_process_name(
            generating_process_name, manager=manager
        )
        if not isinstance(generating_process_signature, dict):
            raise TypeError("generating_process_signature must be a dict")
        parsed_layout = InnerTableLayout.from_payload(layout)
        layout_document = parsed_layout.to_dict()
        artifacts = self.artifacts_dir(name)
        checked_meta = payload_meta_from_dir(
            artifacts, seq_len=parsed_layout.seq_len
        )
        if checked_meta != payload_meta:
            raise ValueError(
                "inner_table payload_meta does not match files on disk"
            )
        collection_hash = manager.hash_payload(
            _collection_identity(
                generating_process_signature=generating_process_signature,
                layout=layout_document,
                payload_meta=checked_meta,
            )
        )
        self.generating_process_name = name
        self.generating_process_signature = generating_process_signature
        self.collection_hash = collection_hash
        self._layout = parsed_layout
        self._layout_document = layout_document
        self._payload_meta = checked_meta
        self._artifacts = artifacts

    @staticmethod
    def configs_dir() -> Directory:
        return package_artifact_directory("inner_table/collections")

    @staticmethod
    def artifacts_dir(generating_process_name: str) -> Path:
        name = assert_generating_process_name(
            generating_process_name, manager=project_hash_manager()
        )
        return data_for_training_dir() / "inner_table" / name

    def layout(self) -> InnerTableLayout:
        return self._layout

    def table_spec(self) -> FrozenTableSpec:
        return self._layout.to_table_spec()

    def token_ids(self) -> np.ndarray:
        return np.load(self._artifacts / TOKEN_IDS_FILE, mmap_mode="r")

    def row_ids(self) -> np.ndarray:
        return np.load(self._artifacts / ROW_IDS_FILE, mmap_mode="r")

    def split(self) -> np.ndarray:
        return np.load(self._artifacts / SPLIT_FILE, mmap_mode="r")

    def labels(self) -> np.ndarray:
        return np.load(self._artifacts / LABELS_FILE, mmap_mode="r")

    def n_rows(self) -> int:
        return int(self._payload_meta["n_rows"])

    def parent_collection(self) -> VocabularyCollection:
        parent_hash = self.generating_process_signature["args"]["vocabulary_hash"]
        return VocabularyCollection.load(parent_hash)

    def related_collections(self) -> dict:
        return {"vocabulary": self.parent_collection}

    def to_dict(self) -> dict:
        return {
            "hash": self.collection_hash,
            "generating_process_name": self.generating_process_name,
            "generating_process_signature": self.generating_process_signature,
            "layout": self._layout_document,
            "payload_meta": self._payload_meta,
        }

    @classmethod
    def from_payload(cls, collection_config: dict) -> InnerTableCollection:
        collection = cls(
            generating_process_name=collection_config["generating_process_name"],
            generating_process_signature=collection_config[
                "generating_process_signature"
            ],
            layout=collection_config["layout"],
            payload_meta=collection_config["payload_meta"],
        )
        stored_hash = collection_config["hash"]
        if stored_hash != collection.collection_hash:
            raise ValueError(
                "inner_table collection hash does not match identity "
                f"(stored {stored_hash}, recomputed {collection.collection_hash})"
            )
        return collection

    @classmethod
    def load(cls, identity: str) -> InnerTableCollection:
        _file, payload = load_collection_configuration(
            cls.configs_dir(), identity, manager=project_hash_manager()
        )
        return cls.from_payload(payload)


def payload_meta_from_dir(artifacts: Path, *, seq_len: int) -> dict:
    token_path = artifacts / TOKEN_IDS_FILE
    row_path = artifacts / ROW_IDS_FILE
    split_path = artifacts / SPLIT_FILE
    labels_path = artifacts / LABELS_FILE
    token_ids = np.load(token_path, mmap_mode="r")
    if token_ids.ndim != 2 or int(token_ids.shape[1]) != seq_len:
        raise ValueError(
            f"token_ids shape {token_ids.shape} does not match seq_len {seq_len}"
        )
    n_rows = int(token_ids.shape[0])
    _require_length(row_path, n_rows=n_rows, label="row_ids")
    _require_length(split_path, n_rows=n_rows, label="split")
    _require_length(labels_path, n_rows=n_rows, label="labels")
    return {
        "n_rows": n_rows,
        "seq_len": seq_len,
        "token_ids_sha256": hash_file(token_path),
        "row_ids_sha256": hash_file(row_path),
        "split_sha256": hash_file(split_path),
        "labels_sha256": hash_file(labels_path),
    }


def _require_length(path: Path, *, n_rows: int, label: str) -> None:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 1 or int(array.shape[0]) != n_rows:
        raise ValueError(
            f"{label} shape {array.shape} does not match n_rows {n_rows}"
        )
