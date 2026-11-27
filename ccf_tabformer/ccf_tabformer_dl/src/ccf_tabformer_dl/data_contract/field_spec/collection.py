"""Frozen set of column specs. Identity = signature + measured members."""

from __future__ import annotations

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.data_contract.configs import load_collection_configuration
from dl_toolbox.data_contract.names import assert_generating_process_name

from ccf_tabformer_dl.data_contract.field_spec.entities import (
    BooleanColumnSpec,
    CategoricalColumnSpec,
    NumericColumnSpec,
    column_spec_from_payload,
)
from ccf_tabformer_dl.file_utils import package_artifact_directory, project_hash_manager

ColumnSpec = NumericColumnSpec | CategoricalColumnSpec | BooleanColumnSpec


def _collection_identity(*, generating_process_signature: dict, members: dict) -> dict:
    return {
        "generating_process_signature": generating_process_signature,
        "members": members,
    }


class FieldSpecCollection:
    def __init__(
        self,
        *,
        generating_process_name: str,
        generating_process_signature: dict,
        members: dict,
    ) -> None:
        manager = project_hash_manager()
        name = assert_generating_process_name(
            generating_process_name, manager=manager
        )
        if not isinstance(generating_process_signature, dict):
            raise TypeError("generating_process_signature must be a dict")
        if not isinstance(members, dict):
            raise TypeError("members must be a dict")
        column_order = _column_order(generating_process_signature)
        indexed, documents = _index_members(members, column_order=column_order)
        collection_hash = manager.hash_payload(
            _collection_identity(
                generating_process_signature=generating_process_signature,
                members=documents,
            )
        )
        self.generating_process_name = name
        self.generating_process_signature = generating_process_signature
        self.collection_hash = collection_hash
        self._members = indexed
        self._documents = documents
        self._column_order = column_order

    @staticmethod
    def configs_dir() -> Directory:
        return package_artifact_directory("field_spec/collections")

    def item(self, column: str) -> ColumnSpec:
        if column not in self._members:
            raise KeyError(
                f"no column {column!r} in collection {self.collection_hash}"
            )
        return self._members[column]

    def columns(self) -> tuple[ColumnSpec, ...]:
        n = len(self._column_order)
        specs = [None] * n
        for i in range(n):
            specs[i] = self._members[self._column_order[i]]
        return tuple(specs)

    def related_collections(self) -> dict:
        return {}

    def to_dict(self) -> dict:
        return {
            "hash": self.collection_hash,
            "generating_process_name": self.generating_process_name,
            "generating_process_signature": self.generating_process_signature,
            "members": self._documents,
        }

    @classmethod
    def from_payload(cls, collection_config: dict) -> FieldSpecCollection:
        collection = cls(
            generating_process_name=collection_config["generating_process_name"],
            generating_process_signature=collection_config[
                "generating_process_signature"
            ],
            members=collection_config["members"],
        )
        stored_hash = collection_config["hash"]
        if stored_hash != collection.collection_hash:
            raise ValueError(
                "field_spec collection hash does not match identity "
                f"(stored {stored_hash}, recomputed {collection.collection_hash})"
            )
        return collection

    @classmethod
    def load(cls, identity: str) -> FieldSpecCollection:
        _file, payload = load_collection_configuration(
            cls.configs_dir(), identity, manager=project_hash_manager()
        )
        return cls.from_payload(payload)


def _column_order(signature: dict) -> tuple[str, ...]:
    args = signature["args"]
    columns = args["columns"]
    n = len(columns)
    names = [""] * n
    seen: dict[str, None] = {}
    for i in range(n):
        name = columns[i]["name"]
        if name in seen:
            raise ValueError(f"duplicate declared column {name!r}")
        seen[name] = None
        names[i] = name
    return tuple(names)


def _index_members(
    members: dict, *, column_order: tuple[str, ...]
) -> tuple[dict[str, ColumnSpec], dict]:
    if set(members) != set(column_order):
        raise ValueError(
            "members keys must equal declared column names: "
            f"declared={list(column_order)} members={sorted(members)}"
        )
    n = len(column_order)
    indexed: dict[str, ColumnSpec] = {}
    documents: dict = {}
    for i in range(n):
        name = column_order[i]
        spec = column_spec_from_payload(members[name])
        if spec.name != name:
            raise ValueError(
                f"members[{name!r}] name is {spec.name!r}"
            )
        indexed[name] = spec
        documents[name] = spec.to_dict()
    return indexed, documents
