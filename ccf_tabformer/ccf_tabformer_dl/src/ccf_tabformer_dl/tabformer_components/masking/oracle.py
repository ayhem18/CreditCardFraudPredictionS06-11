"""Strategy name → Masker class, and the `build_masker` entry point."""

from __future__ import annotations

from ccf_tabformer_dl.tabformer_components.masking.field_level import (
    FIELD_LEVEL_STRATEGY,
    FieldLevelMasker,
    build_field_level_args,
)
from ccf_tabformer_dl.tabformer_components.masking.masker import Masker
from ccf_tabformer_dl.tabformer_components.table_spec import TableSpec

_REGISTRY: dict[str, type[Masker]] = {
    FIELD_LEVEL_STRATEGY: FieldLevelMasker,
}
_ARGS_BUILDERS = {
    FIELD_LEVEL_STRATEGY: build_field_level_args,
}


def oracle(strategy: str) -> type[Masker]:
    if not isinstance(strategy, str) or not strategy:
        raise TypeError("strategy must be a non-empty str")
    if strategy not in _REGISTRY:
        raise NotImplementedError(f"unknown masker strategy: {strategy!r}")
    return _REGISTRY[strategy]


def build_masker(*, strategy: str, args: dict, spec: TableSpec) -> Masker:
    if not isinstance(args, dict):
        raise TypeError("args must be a dict")
    masker_class = oracle(strategy)
    built_args = _ARGS_BUILDERS[strategy](args)
    return masker_class(spec=spec, **built_args)
