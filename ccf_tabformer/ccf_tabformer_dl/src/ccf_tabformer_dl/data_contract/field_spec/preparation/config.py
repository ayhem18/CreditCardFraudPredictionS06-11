"""Named generating-process executions for declared field_spec."""

from __future__ import annotations

from ccf_tabformer_dl.data_contract.columns import COLUMN_DECLARATIONS
from ccf_tabformer_dl.data_contract.field_spec.preparation.declared import DECLARED_STRATEGY
from ccf_tabformer_dl.data_contract.field_spec.preparation.oracle import (
    build_field_spec_preparation,
)
from ccf_tabformer_dl.data_contract.field_spec.preparation.preparation import FieldSpecPreparation

DECLARED_V1 = "declared_v1"


def load_declared_v1() -> FieldSpecPreparation:
    n = len(COLUMN_DECLARATIONS)
    columns = [None] * n
    for i in range(n):
        columns[i] = dict(COLUMN_DECLARATIONS[i])
    return build_field_spec_preparation(
        strategy=DECLARED_STRATEGY,
        args={"columns": columns},
    )
