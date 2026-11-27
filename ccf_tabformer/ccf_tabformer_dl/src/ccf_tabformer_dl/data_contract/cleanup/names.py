"""Names a --force pass should delete."""

from __future__ import annotations

from dl_toolbox.data_contract.names import assert_generating_process_name

from ccf_tabformer_dl.file_utils import project_hash_manager


def generating_process_names_to_remove(name: str) -> tuple[str, ...]:
    return (
        assert_generating_process_name(name, manager=project_hash_manager()),
    )
