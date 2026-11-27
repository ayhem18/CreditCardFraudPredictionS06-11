"""Strategy name → Embedder class, and the `build_embedder` entry point."""

from __future__ import annotations

from ccf_tabformer_dl.tabformer_components.embedding.additive import (
    ADDITIVE_STRATEGY,
    AdditiveEmbedder,
    build_additive_args,
)
from ccf_tabformer_dl.tabformer_components.embedding.embedder import Embedder
from ccf_tabformer_dl.tabformer_components.table_spec import TableSpec

_REGISTRY: dict[str, type[Embedder]] = {
    ADDITIVE_STRATEGY: AdditiveEmbedder,
}
_ARGS_BUILDERS = {
    ADDITIVE_STRATEGY: build_additive_args,
}


def oracle(strategy: str) -> type[Embedder]:
    if not isinstance(strategy, str) or not strategy:
        raise TypeError("strategy must be a non-empty str")
    if strategy not in _REGISTRY:
        raise NotImplementedError(f"unknown embedder strategy: {strategy!r}")
    return _REGISTRY[strategy]


def build_embedder(*, strategy: str, args: dict, spec: TableSpec) -> Embedder:
    if not isinstance(args, dict):
        raise TypeError("args must be a dict")
    embedder_class = oracle(strategy)
    built_args = _ARGS_BUILDERS[strategy](args)
    return embedder_class(spec=spec, **built_args)
