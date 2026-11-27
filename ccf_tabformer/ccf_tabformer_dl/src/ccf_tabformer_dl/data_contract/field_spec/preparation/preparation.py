"""field_spec-preparation ABC. JSON-only. Upstream is the corrected CSVs."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dl_toolbox.base.file_management.directory import Directory

from ccf_tabformer_dl.data_contract.field_spec.collection import FieldSpecCollection
from ccf_tabformer_dl.data_contract.process_loop import execute_generating_process
from ccf_tabformer_dl.file_utils import package_artifact_directory


class FieldSpecPreparation(ABC):
    strategy: str

    @staticmethod
    def process_configurations_dir() -> Directory:
        return package_artifact_directory("field_spec/preparation")

    @abstractmethod
    def generating_process_signature(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def collect_members(self) -> dict:
        raise NotImplementedError

    def process(self, *, generating_process_name: str) -> FieldSpecCollection:
        signature = self.generating_process_signature()

        def build_collection(name: str) -> FieldSpecCollection:
            members = self.collect_members()
            return FieldSpecCollection(
                generating_process_name=name,
                generating_process_signature=signature,
                members=members,
            )

        return execute_generating_process(
            generating_process_name=generating_process_name,
            signature=signature,
            process_dir=self.process_configurations_dir(),
            configs_dir=FieldSpecCollection.configs_dir(),
            payload_for_name=None,
            build_collection=build_collection,
        )
