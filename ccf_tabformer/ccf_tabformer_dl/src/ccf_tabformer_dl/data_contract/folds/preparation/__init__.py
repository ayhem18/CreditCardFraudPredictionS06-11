"""folds preparation oracle."""

from ccf_tabformer_dl.data_contract.folds.preparation.oracle import (
    build_folds_preparation,
)
from ccf_tabformer_dl.data_contract.folds.preparation.preparation import (
    FoldsPreparation,
)

__all__ = ["FoldsPreparation", "build_folds_preparation"]
