"""Named generating-process executions for fixed_width_padded."""

from __future__ import annotations

from dl_toolbox.base.hashing.identity import is_full_sha256
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str

from ccf_tabformer_dl.data_contract.inner_table.preparation.fixed_width import (
    FIXED_WIDTH_PADDED_STRATEGY,
)
from ccf_tabformer_dl.data_contract.inner_table.preparation.oracle import (
    build_inner_table_preparation,
)
from ccf_tabformer_dl.data_contract.inner_table.preparation.preparation import InnerTablePreparation

FIXED_WIDTH_V1 = "fixed_width_v1"


def load_fixed_width_v1(*, vocabulary_hash: str) -> InnerTablePreparation:
    digest = parse_non_empty_str(vocabulary_hash, label="vocabulary_hash")
    if not is_full_sha256(digest):
        raise ValueError(
            f"vocabulary_hash must be a collection hash, got {vocabulary_hash!r}"
        )
    return build_inner_table_preparation(
        strategy=FIXED_WIDTH_PADDED_STRATEGY,
        args={"vocabulary_hash": digest},
    )
