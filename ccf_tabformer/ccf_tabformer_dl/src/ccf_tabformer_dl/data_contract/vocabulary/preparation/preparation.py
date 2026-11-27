"""vocabulary-preparation ABC. JSON-only. One parent field_spec collection."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dl_toolbox.base.file_management.directory import Directory

from ccf_tabformer_dl.data_contract.process_loop import execute_generating_process
from ccf_tabformer_dl.data_contract.vocabulary.collection import VocabularyCollection
from ccf_tabformer_dl.file_utils import package_artifact_directory


class VocabularyPreparation(ABC):
    strategy: str

    @staticmethod
    def process_configurations_dir() -> Directory:
        return package_artifact_directory("vocabulary/preparation")

    @abstractmethod
    def generating_process_signature(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def collect_inventory(self) -> dict:
        raise NotImplementedError

    def process(self, *, generating_process_name: str) -> VocabularyCollection:
        signature = self.generating_process_signature()

        def build_collection(name: str) -> VocabularyCollection:
            inventory = self.collect_inventory()
            return VocabularyCollection(
                generating_process_name=name,
                generating_process_signature=signature,
                inventory=inventory,
            )

        return execute_generating_process(
            generating_process_name=generating_process_name,
            signature=signature,
            process_dir=self.process_configurations_dir(),
            configs_dir=VocabularyCollection.configs_dir(),
            payload_for_name=None,
            build_collection=build_collection,
        )
