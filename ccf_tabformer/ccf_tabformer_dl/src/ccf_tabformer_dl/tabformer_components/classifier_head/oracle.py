"""Strategy name → ClassifierHead class, and the `build_classifier_head` entry point."""

from __future__ import annotations

from ccf_tabformer_dl.tabformer_components.classifier_head.classifier_head import (
    ClassifierHead,
)
from ccf_tabformer_dl.tabformer_components.classifier_head.mean_pool import (
    MEAN_POOL_STRATEGY,
    MeanPoolClassifierHead,
    build_mean_pool_args,
)

_REGISTRY: dict[str, type[ClassifierHead]] = {
    MEAN_POOL_STRATEGY: MeanPoolClassifierHead,
}
_ARGS_BUILDERS = {
    MEAN_POOL_STRATEGY: build_mean_pool_args,
}


def oracle(strategy: str) -> type[ClassifierHead]:
    if not isinstance(strategy, str) or not strategy:
        raise TypeError("strategy must be a non-empty str")
    if strategy not in _REGISTRY:
        raise NotImplementedError(f"unknown classifier head strategy: {strategy!r}")
    return _REGISTRY[strategy]


def build_classifier_head(*, strategy: str, args: dict) -> ClassifierHead:
    if not isinstance(args, dict):
        raise TypeError("args must be a dict")
    head_class = oracle(strategy)
    built_args = _ARGS_BUILDERS[strategy](args)
    return head_class(**built_args)
