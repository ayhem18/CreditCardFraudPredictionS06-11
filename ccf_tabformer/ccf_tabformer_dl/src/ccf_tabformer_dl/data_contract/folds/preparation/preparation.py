"""folds-preparation ABC. File-backed fold-id payload."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)

from ccf_tabformer_dl.data_contract.folds.collection import FoldsCollection
from ccf_tabformer_dl.data_contract.process_loop import execute_generating_process
from ccf_tabformer_dl.file_utils import package_artifact_directory


class FoldsPreparation(ABC):
    strategy: str

    @staticmethod
    def process_configurations_dir() -> Directory:
        return package_artifact_directory("folds/preparation")

    @abstractmethod
    def generating_process_signature(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def write_payload(self, *, artifacts_dir: Path) -> dict:
        raise NotImplementedError

    def process(self, *, generating_process_name: str) -> FoldsCollection:
        signature = self.generating_process_signature()

        def payload_for_name(name: str) -> Directory:
            return Directory.from_absolute(
                path=FoldsCollection.artifacts_dir(name),
                policy=DirectoryParsePolicy.ALLOW_MISSING,
            )

        def build_collection(name: str) -> FoldsCollection:
            artifacts = FoldsCollection.artifacts_dir(name)
            payload_meta = self.write_payload(artifacts_dir=artifacts)
            return FoldsCollection(
                generating_process_name=name,
                generating_process_signature=signature,
                payload_meta=payload_meta,
            )

        return execute_generating_process(
            generating_process_name=generating_process_name,
            signature=signature,
            process_dir=self.process_configurations_dir(),
            configs_dir=FoldsCollection.configs_dir(),
            payload_for_name=payload_for_name,
            build_collection=build_collection,
        )
