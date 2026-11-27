"""Strategy name → row_subset-preparation class."""

from __future__ import annotations

from ccf_tabformer_dl.data_contract.row_subset.preparation.alarm_label import (
    BALANCED_ALARM_LABEL_STRATEGY,
    BalancedAlarmLabelPreparation,
    build_balanced_alarm_label_args,
)
from ccf_tabformer_dl.data_contract.row_subset.preparation.preparation import RowSubsetPreparation
from ccf_tabformer_dl.data_contract.row_subset.preparation.stratified import (
    STRATIFIED_BY_LABEL_STRATEGY,
    StratifiedByLabelPreparation,
    build_stratified_by_label_args,
)

_REGISTRY: dict[str, type[RowSubsetPreparation]] = {
    STRATIFIED_BY_LABEL_STRATEGY: StratifiedByLabelPreparation,
    BALANCED_ALARM_LABEL_STRATEGY: BalancedAlarmLabelPreparation,
}
_ARGS_BUILDERS = {
    STRATIFIED_BY_LABEL_STRATEGY: build_stratified_by_label_args,
    BALANCED_ALARM_LABEL_STRATEGY: build_balanced_alarm_label_args,
}


def oracle(strategy: str) -> type[RowSubsetPreparation]:
    if not isinstance(strategy, str) or not strategy:
        raise TypeError("strategy must be a non-empty str")
    if strategy not in _REGISTRY:
        raise NotImplementedError(
            f"unknown row_subset preparation strategy: {strategy!r}"
        )
    return _REGISTRY[strategy]


def build_row_subset_preparation(
    *, strategy: str, args: dict
) -> RowSubsetPreparation:
    if not isinstance(args, dict):
        raise TypeError("args must be a dict")
    preparation_class = oracle(strategy)
    built_args = _ARGS_BUILDERS[strategy](args)
    return preparation_class(**built_args)
