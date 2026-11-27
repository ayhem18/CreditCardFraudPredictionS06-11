"""inner_table preparation oracle."""

from ccf_tabformer_dl.data_contract.inner_table.preparation.oracle import (
    build_inner_table_preparation,
)
from ccf_tabformer_dl.data_contract.inner_table.preparation.preparation import InnerTablePreparation

__all__ = ["InnerTablePreparation", "build_inner_table_preparation"]
