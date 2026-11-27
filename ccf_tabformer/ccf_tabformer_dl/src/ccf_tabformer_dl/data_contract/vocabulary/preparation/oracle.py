"""Strategy name → vocabulary-preparation class."""

from __future__ import annotations

from ccf_tabformer_dl.data_contract.vocabulary.preparation.preparation import VocabularyPreparation
from ccf_tabformer_dl.data_contract.vocabulary.preparation.universal_digits import (
    UNIVERSAL_DIGITS_STRATEGY,
    UniversalDigitsPreparation,
    build_universal_digits_args,
)

_REGISTRY: dict[str, type[VocabularyPreparation]] = {
    UNIVERSAL_DIGITS_STRATEGY: UniversalDigitsPreparation,
}
_ARGS_BUILDERS = {
    UNIVERSAL_DIGITS_STRATEGY: build_universal_digits_args,
}


def oracle(strategy: str) -> type[VocabularyPreparation]:
    if not isinstance(strategy, str) or not strategy:
        raise TypeError("strategy must be a non-empty str")
    if strategy not in _REGISTRY:
        raise NotImplementedError(
            f"unknown vocabulary preparation strategy: {strategy!r}"
        )
    return _REGISTRY[strategy]


def build_vocabulary_preparation(
    *, strategy: str, args: dict
) -> VocabularyPreparation:
    if not isinstance(args, dict):
        raise TypeError("args must be a dict")
    preparation_class = oracle(strategy)
    built_args = _ARGS_BUILDERS[strategy](args)
    return preparation_class(**built_args)
