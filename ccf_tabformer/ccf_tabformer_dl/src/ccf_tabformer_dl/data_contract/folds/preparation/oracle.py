"""Strategy name → folds-preparation class."""

from __future__ import annotations

from ccf_tabformer_dl.data_contract.folds.preparation.imported import (
    IMPORTED_STRATEGY,
    ImportedFoldsPreparation,
    build_imported_args,
)
from ccf_tabformer_dl.data_contract.folds.preparation.preparation import FoldsPreparation

_REGISTRY: dict[str, type[FoldsPreparation]] = {
    IMPORTED_STRATEGY: ImportedFoldsPreparation,
}
_ARGS_BUILDERS = {
    IMPORTED_STRATEGY: build_imported_args,
}


def oracle(strategy: str) -> type[FoldsPreparation]:
    if not isinstance(strategy, str) or not strategy:
        raise TypeError("strategy must be a non-empty str")
    if strategy not in _REGISTRY:
        raise NotImplementedError(
            f"unknown folds preparation strategy: {strategy!r}"
        )
    return _REGISTRY[strategy]


def build_folds_preparation(*, strategy: str, args: dict) -> FoldsPreparation:
    if not isinstance(args, dict):
        raise TypeError("args must be a dict")
    preparation_class = oracle(strategy)
    built_args = _ARGS_BUILDERS[strategy](args)
    return preparation_class(**built_args)
