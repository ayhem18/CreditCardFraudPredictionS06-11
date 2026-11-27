"""Named generating-process executions for row-index subsets."""

from __future__ import annotations

from copy import deepcopy

from dl_toolbox.base.hashing.identity import is_full_sha256
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.data_contract.names import assert_generating_process_name

from ccf_tabformer_dl.data_contract.row_subset.preparation.alarm_label import (
    BALANCED_ALARM_LABEL_STRATEGY,
)
from ccf_tabformer_dl.data_contract.row_subset.preparation.oracle import (
    build_row_subset_preparation,
)
from ccf_tabformer_dl.data_contract.row_subset.preparation.preparation import RowSubsetPreparation
from ccf_tabformer_dl.data_contract.row_subset.preparation.stratified import SPLIT_NAME_TRAIN
from ccf_tabformer_dl.file_utils import project_hash_manager

ALARM_LABEL_1K_SEED0 = "alarm_label_1k_seed0"
NAMED_SUBSET_CONFIGS = {
    ALARM_LABEL_1K_SEED0: {
        "strategy": BALANCED_ALARM_LABEL_STRATEGY,
        "args": {"n_per_cell": 1000, "seed": 0, "split": SPLIT_NAME_TRAIN},
    },
}


def subset_config_for_generating_process_name(generating_process_name: str) -> dict:
    name = assert_generating_process_name(
        generating_process_name, manager=project_hash_manager()
    )
    if name not in NAMED_SUBSET_CONFIGS:
        legal = tuple(NAMED_SUBSET_CONFIGS)
        raise ValueError(
            f"unknown row_subset generating_process_name {name!r}; "
            f"legal names are {legal}"
        )
    return deepcopy(NAMED_SUBSET_CONFIGS[name])


def load_named_subset(
    *, inner_table_hash: str, generating_process_name: str
) -> RowSubsetPreparation:
    digest = parse_non_empty_str(inner_table_hash, label="inner_table_hash")
    if not is_full_sha256(digest):
        raise ValueError(
            f"inner_table_hash must be a collection hash, got {inner_table_hash!r}"
        )
    spec = subset_config_for_generating_process_name(generating_process_name)
    args = dict(spec["args"])
    args["inner_table_hash"] = digest
    return build_row_subset_preparation(strategy=spec["strategy"], args=args)
