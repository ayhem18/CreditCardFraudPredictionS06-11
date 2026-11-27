"""Strategy name → Encoder class, and the `build_encoder` entry point."""

from __future__ import annotations

from ccf_tabformer_dl.tabformer_components.encoder.encoder import Encoder
from ccf_tabformer_dl.tabformer_components.encoder.prenorm import (
    PRENORM_TRANSFORMER_STRATEGY,
    PreNormTransformerEncoder,
    build_prenorm_transformer_args,
)

_REGISTRY: dict[str, type[Encoder]] = {
    PRENORM_TRANSFORMER_STRATEGY: PreNormTransformerEncoder,
}
_ARGS_BUILDERS = {
    PRENORM_TRANSFORMER_STRATEGY: build_prenorm_transformer_args,
}


def oracle(strategy: str) -> type[Encoder]:
    if not isinstance(strategy, str) or not strategy:
        raise TypeError("strategy must be a non-empty str")
    if strategy not in _REGISTRY:
        raise NotImplementedError(f"unknown encoder strategy: {strategy!r}")
    return _REGISTRY[strategy]


def build_encoder(*, strategy: str, args: dict) -> Encoder:
    if not isinstance(args, dict):
        raise TypeError("args must be a dict")
    encoder_class = oracle(strategy)
    built_args = _ARGS_BUILDERS[strategy](args)
    return encoder_class(**built_args)
