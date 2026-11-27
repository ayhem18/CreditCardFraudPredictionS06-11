"""Strategy name → inner_table-preparation class."""

from __future__ import annotations

from ccf_tabformer_dl.data_contract.inner_table.preparation.fixed_width import (
    FIXED_WIDTH_PADDED_STRATEGY,
    FixedWidthPaddedPreparation,
    build_fixed_width_padded_args,
)
from ccf_tabformer_dl.data_contract.inner_table.preparation.preparation import InnerTablePreparation

_REGISTRY: dict[str, type[InnerTablePreparation]] = {
    FIXED_WIDTH_PADDED_STRATEGY: FixedWidthPaddedPreparation,
}
_ARGS_BUILDERS = {
    FIXED_WIDTH_PADDED_STRATEGY: build_fixed_width_padded_args,
}


def oracle(strategy: str) -> type[InnerTablePreparation]:
    if not isinstance(strategy, str) or not strategy:
        raise TypeError("strategy must be a non-empty str")
    if strategy not in _REGISTRY:
        raise NotImplementedError(
            f"unknown inner_table preparation strategy: {strategy!r}"
        )
    return _REGISTRY[strategy]


def build_inner_table_preparation(
    *, strategy: str, args: dict
) -> InnerTablePreparation:
    if not isinstance(args, dict):
        raise TypeError("args must be a dict")
    preparation_class = oracle(strategy)
    built_args = _ARGS_BUILDERS[strategy](args)
    return preparation_class(**built_args)
