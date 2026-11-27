"""Named generating-process executions for universal_digits."""

from __future__ import annotations

from dl_toolbox.base.hashing.identity import is_full_sha256
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str

from ccf_tabformer_dl.data_contract.vocabulary.entity import MASK_TOKEN_DEFAULT
from ccf_tabformer_dl.data_contract.vocabulary.preparation.oracle import (
    build_vocabulary_preparation,
)
from ccf_tabformer_dl.data_contract.vocabulary.preparation.preparation import VocabularyPreparation
from ccf_tabformer_dl.data_contract.vocabulary.preparation.universal_digits import (
    BOOLEAN_POLICY_PER_FLAG,
    UNIVERSAL_DIGITS_STRATEGY,
)

UNIVERSAL_DIGITS_V1 = "universal_digits_v1"


def load_universal_digits_v1(*, field_spec_hash: str) -> VocabularyPreparation:
    digest = parse_non_empty_str(field_spec_hash, label="field_spec_hash")
    if not is_full_sha256(digest):
        raise ValueError(
            f"field_spec_hash must be a collection hash, got {field_spec_hash!r}"
        )
    return build_vocabulary_preparation(
        strategy=UNIVERSAL_DIGITS_STRATEGY,
        args={
            "field_spec_hash": digest,
            "boolean_policy": BOOLEAN_POLICY_PER_FLAG,
            "mask_token": MASK_TOKEN_DEFAULT,
        },
    )
