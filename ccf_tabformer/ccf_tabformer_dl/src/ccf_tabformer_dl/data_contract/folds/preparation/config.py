"""Named generating-process executions for imported CV folds."""

from __future__ import annotations

from copy import deepcopy

from dl_toolbox.base.hashing.identity import is_full_sha256
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.data_contract.names import assert_generating_process_name

from ccf_tabformer_dl.data_contract.folds.preparation.imported import (
    ALARM_PROFILE_SCHEME,
    IMPORTED_STRATEGY,
)
from ccf_tabformer_dl.data_contract.folds.preparation.oracle import (
    build_folds_preparation,
)
from ccf_tabformer_dl.data_contract.folds.preparation.preparation import FoldsPreparation
from ccf_tabformer_dl.file_utils import project_hash_manager

ALARM_PROFILE_5FOLD_V1 = "alarm_profile_5fold_v1"
NAMED_FOLDS_CONFIGS = {
    ALARM_PROFILE_5FOLD_V1: {
        "strategy": IMPORTED_STRATEGY,
        "args": {"scheme": ALARM_PROFILE_SCHEME},
    },
}


def folds_config_for_generating_process_name(generating_process_name: str) -> dict:
    name = assert_generating_process_name(
        generating_process_name, manager=project_hash_manager()
    )
    if name not in NAMED_FOLDS_CONFIGS:
        legal = tuple(NAMED_FOLDS_CONFIGS)
        raise ValueError(
            f"unknown folds generating_process_name {name!r}; "
            f"legal names are {legal}"
        )
    return deepcopy(NAMED_FOLDS_CONFIGS[name])


def load_named_folds(
    *, inner_table_hash: str, generating_process_name: str
) -> FoldsPreparation:
    digest = parse_non_empty_str(inner_table_hash, label="inner_table_hash")
    if not is_full_sha256(digest):
        raise ValueError(
            f"inner_table_hash must be a collection hash, got {inner_table_hash!r}"
        )
    spec = folds_config_for_generating_process_name(generating_process_name)
    args = dict(spec["args"])
    args["inner_table_hash"] = digest
    return build_folds_preparation(strategy=spec["strategy"], args=args)
