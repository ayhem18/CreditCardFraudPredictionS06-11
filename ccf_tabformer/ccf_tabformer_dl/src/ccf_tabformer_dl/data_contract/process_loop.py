"""Claim, write collection, write process last. Shared by generating-process ABCs."""

from __future__ import annotations

from collections.abc import Callable

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.data_contract.configs import write_collection_configuration
from dl_toolbox.data_contract.process_records import (
    claim_generating_process,
    create_payload_directory,
    delete_incomplete_run,
    write_process_configuration,
)

from ccf_tabformer_dl.file_utils import project_hash_manager


def execute_generating_process(
    *,
    generating_process_name: str,
    signature: dict,
    process_dir: Directory,
    configs_dir: Directory,
    payload_for_name: Callable[[str], Directory] | None,
    build_collection: Callable[[str], object],
) -> object:
    manager = project_hash_manager()
    configuration_hash = manager.hash_payload(signature)
    index = manager.create_identity_index(process_dir)
    name = claim_generating_process(
        manager,
        index,
        generating_process_name=generating_process_name,
        configuration_hash=configuration_hash,
    )
    payload = None
    saved = False
    try:
        if payload_for_name is not None:
            payload = create_payload_directory(payload_for_name(name))
        collection = build_collection(name)
        write_collection_configuration(collection, manager=manager)
        write_process_configuration(
            manager,
            index,
            generating_process_name=name,
            configuration_hash=configuration_hash,
            generating_process_signature=signature,
        )
        saved = True
        return collection
    finally:
        if not saved:
            delete_incomplete_run(
                manager=manager,
                directory=index,
                generating_process_name=name,
                configs_dir=configs_dir,
                payload=payload,
            )
