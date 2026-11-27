"""Frozen vocabulary inventory. Identity = signature + inventory document."""

from __future__ import annotations

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.data_contract.configs import load_collection_configuration
from dl_toolbox.data_contract.names import assert_generating_process_name

from ccf_tabformer_dl.data_contract.field_spec.collection import FieldSpecCollection
from ccf_tabformer_dl.data_contract.vocabulary.entity import VocabularyInventory
from ccf_tabformer_dl.file_utils import package_artifact_directory, project_hash_manager


def _collection_identity(*, generating_process_signature: dict, inventory: dict) -> dict:
    return {
        "generating_process_signature": generating_process_signature,
        "inventory": inventory,
    }


class VocabularyCollection:
    def __init__(
        self,
        *,
        generating_process_name: str,
        generating_process_signature: dict,
        inventory: dict,
    ) -> None:
        manager = project_hash_manager()
        name = assert_generating_process_name(
            generating_process_name, manager=manager
        )
        if not isinstance(generating_process_signature, dict):
            raise TypeError("generating_process_signature must be a dict")
        if not isinstance(inventory, dict):
            raise TypeError("inventory must be a dict")
        entity = VocabularyInventory.from_payload(inventory)
        document = entity.to_dict()
        parent_hash = generating_process_signature["args"]["field_spec_hash"]
        collection_hash = manager.hash_payload(
            _collection_identity(
                generating_process_signature=generating_process_signature,
                inventory=document,
            )
        )
        self.generating_process_name = name
        self.generating_process_signature = generating_process_signature
        self.collection_hash = collection_hash
        self._inventory = entity
        self._document = document
        self._field_spec_hash = parent_hash

    @staticmethod
    def configs_dir() -> Directory:
        return package_artifact_directory("vocabulary/collections")

    def inventory(self) -> VocabularyInventory:
        return self._inventory

    def parent_collection(self) -> FieldSpecCollection:
        return FieldSpecCollection.load(self._field_spec_hash)

    def related_collections(self) -> dict:
        return {"field_spec": self.parent_collection}

    def to_dict(self) -> dict:
        return {
            "hash": self.collection_hash,
            "generating_process_name": self.generating_process_name,
            "generating_process_signature": self.generating_process_signature,
            "inventory": self._document,
        }

    @classmethod
    def from_payload(cls, collection_config: dict) -> VocabularyCollection:
        collection = cls(
            generating_process_name=collection_config["generating_process_name"],
            generating_process_signature=collection_config[
                "generating_process_signature"
            ],
            inventory=collection_config["inventory"],
        )
        stored_hash = collection_config["hash"]
        if stored_hash != collection.collection_hash:
            raise ValueError(
                "vocabulary collection hash does not match identity "
                f"(stored {stored_hash}, recomputed {collection.collection_hash})"
            )
        return collection

    @classmethod
    def load(cls, identity: str) -> VocabularyCollection:
        _file, payload = load_collection_configuration(
            cls.configs_dir(), identity, manager=project_hash_manager()
        )
        return cls.from_payload(payload)
