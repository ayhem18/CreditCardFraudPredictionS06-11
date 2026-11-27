"""Strategy name → field_spec-preparation class."""

from __future__ import annotations

from ccf_tabformer_dl.data_contract.field_spec.preparation.declared import (
    DECLARED_STRATEGY,
    DeclaredFieldSpecPreparation,
    build_declared_args,
)
from ccf_tabformer_dl.data_contract.field_spec.preparation.preparation import FieldSpecPreparation

_REGISTRY: dict[str, type[FieldSpecPreparation]] = {
    DECLARED_STRATEGY: DeclaredFieldSpecPreparation,
}
_ARGS_BUILDERS = {
    DECLARED_STRATEGY: build_declared_args,
}


def oracle(strategy: str) -> type[FieldSpecPreparation]:
    if not isinstance(strategy, str) or not strategy:
        raise TypeError("strategy must be a non-empty str")
    if strategy not in _REGISTRY:
        raise NotImplementedError(
            f"unknown field_spec preparation strategy: {strategy!r}"
        )
    return _REGISTRY[strategy]


def build_field_spec_preparation(
    *, strategy: str, args: dict
) -> FieldSpecPreparation:
    if not isinstance(args, dict):
        raise TypeError("args must be a dict")
    preparation_class = oracle(strategy)
    built_args = _ARGS_BUILDERS[strategy](args)
    return preparation_class(**built_args)
