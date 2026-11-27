"""Strategy name → Head class, and the `build_head` entry point."""

from __future__ import annotations

from ccf_tabformer_dl.tabformer_components.head.head import Head
from ccf_tabformer_dl.tabformer_components.head.support_masked import (
    SHARED_SUPPORT_MASKED_STRATEGY,
    SupportMaskedHead,
    build_shared_support_masked_args,
)
from ccf_tabformer_dl.tabformer_components.table_spec import TableSpec

_REGISTRY: dict[str, type[Head]] = {
    SHARED_SUPPORT_MASKED_STRATEGY: SupportMaskedHead,
}
_ARGS_BUILDERS = {
    SHARED_SUPPORT_MASKED_STRATEGY: build_shared_support_masked_args,
}


def oracle(strategy: str) -> type[Head]:
    if not isinstance(strategy, str) or not strategy:
        raise TypeError("strategy must be a non-empty str")
    if strategy not in _REGISTRY:
        raise NotImplementedError(f"unknown head strategy: {strategy!r}")
    return _REGISTRY[strategy]


def build_head(*, strategy: str, args: dict, spec: TableSpec) -> Head:
    if not isinstance(args, dict):
        raise TypeError("args must be a dict")
    head_class = oracle(strategy)
    built_args = _ARGS_BUILDERS[strategy](args)
    return head_class(spec=spec, **built_args)
