"""inner_table-preparation ABC. File-backed numpy payload."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)

from ccf_tabformer_dl.data_contract.inner_table.collection import InnerTableCollection
from ccf_tabformer_dl.data_contract.process_loop import execute_generating_process
from ccf_tabformer_dl.file_utils import package_artifact_directory


class InnerTablePreparation(ABC):
    strategy: str

    @staticmethod
    def process_configurations_dir() -> Directory:
        return package_artifact_directory("inner_table/preparation")

    @abstractmethod
    def generating_process_signature(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def write_payload(self, *, artifacts_dir: Path) -> tuple[dict, dict]:
        """Write numpy files. Return ``(layout_document, payload_meta)``."""
        raise NotImplementedError

    def process(self, *, generating_process_name: str) -> InnerTableCollection:
        signature = self.generating_process_signature()

        def payload_for_name(name: str) -> Directory:
            return Directory.from_absolute(
                path=InnerTableCollection.artifacts_dir(name),
                policy=DirectoryParsePolicy.ALLOW_MISSING,
            )

        def build_collection(name: str) -> InnerTableCollection:
            artifacts = InnerTableCollection.artifacts_dir(name)
            layout, payload_meta = self.write_payload(artifacts_dir=artifacts)
            return InnerTableCollection(
                generating_process_name=name,
                generating_process_signature=signature,
                layout=layout,
                payload_meta=payload_meta,
            )

        return execute_generating_process(
            generating_process_name=generating_process_name,
            signature=signature,
            process_dir=self.process_configurations_dir(),
            configs_dir=InnerTableCollection.configs_dir(),
            payload_for_name=payload_for_name,
            build_collection=build_collection,
        )
