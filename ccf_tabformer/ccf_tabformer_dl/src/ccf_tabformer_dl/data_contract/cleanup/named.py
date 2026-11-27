"""Delete a named freeze, and optionally whichever name already holds a process hash."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.data_contract.named_freeze import (
    process_hash_occupant,
    remove_named_freeze,
)

from ccf_tabformer_dl.data_contract.cleanup.names import generating_process_names_to_remove
from ccf_tabformer_dl.file_utils import project_hash_manager


def cleanup_named_freeze(
    *,
    kind: str,
    name: str,
    process_dir: Directory,
    collection_dir: Directory,
    payload_dir: Directory | Path | None = None,
    configuration_hash: str | None = None,
) -> None:
    manager = project_hash_manager()
    payload = _as_payload_directory(payload_dir)
    if configuration_hash is not None:
        occupant = process_hash_occupant(
            manager,
            process_dir,
            configuration_hash=configuration_hash,
        )
        if occupant is not None:
            _remove(
                kind=kind,
                name=occupant,
                process_dir=process_dir,
                collection_dir=collection_dir,
                payload=_payload_for_name(payload, occupant, name),
            )
    names = generating_process_names_to_remove(name)
    n_names = len(names)
    for i in range(n_names):
        current = names[i]
        _remove(
            kind=kind,
            name=current,
            process_dir=process_dir,
            collection_dir=collection_dir,
            payload=_payload_for_name(payload, current, name),
        )


def _as_payload_directory(
    payload_dir: Directory | Path | None,
) -> Directory | None:
    if payload_dir is None:
        return None
    if isinstance(payload_dir, Directory):
        return payload_dir
    return Directory.from_absolute(
        path=payload_dir, policy=DirectoryParsePolicy.ALLOW_MISSING
    )


def _payload_for_name(
    payload_dir: Directory | None, current: str, requested: str
) -> Directory | None:
    if payload_dir is None:
        return None
    if current == requested:
        return payload_dir
    return Directory.from_absolute(
        path=payload_dir.path.parent / current,
        policy=DirectoryParsePolicy.ALLOW_MISSING,
    )


def _remove(
    *,
    kind: str,
    name: str,
    process_dir: Directory,
    collection_dir: Directory,
    payload: Directory | None,
) -> None:
    manager = project_hash_manager()
    removed = remove_named_freeze(
        manager,
        process_dir,
        generating_process_name=name,
        configs_dir=collection_dir,
        payload=payload,
    )
    status = "cleaned" if removed else "absent"
    print(f"{status} {kind} generating_process_name={name}")
