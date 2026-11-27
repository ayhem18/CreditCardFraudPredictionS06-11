"""``universal_digits``: shared 0–9, per-column cat/flag blocks, one ``[MASK]``."""

from __future__ import annotations

from dl_toolbox.base.hashing.identity import is_full_sha256
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import (
    one_of_message,
    parse_one_of,
    require_exact_keys,
)

from ccf_tabformer_dl.data_contract.field_spec.collection import FieldSpecCollection
from ccf_tabformer_dl.data_contract.vocabulary.entity import (
    MASK_TOKEN_DEFAULT,
    build_universal_digits_inventory,
)
from ccf_tabformer_dl.data_contract.vocabulary.preparation.preparation import VocabularyPreparation

UNIVERSAL_DIGITS_STRATEGY = "universal_digits"
BOOLEAN_POLICY_PER_FLAG = "per_flag"
_ARG_KEYS = ("field_spec_hash", "boolean_policy", "mask_token")


class UniversalDigitsPreparation(VocabularyPreparation):
    strategy = UNIVERSAL_DIGITS_STRATEGY

    def __init__(
        self,
        *,
        field_spec_hash: str,
        boolean_policy: str,
        mask_token: str,
    ) -> None:
        digest = parse_non_empty_str(field_spec_hash, label="field_spec_hash")
        if not is_full_sha256(digest):
            raise ValueError(
                f"field_spec_hash must be a collection hash, got {field_spec_hash!r}"
            )
        policy = parse_one_of(
            boolean_policy,
            allowed=(BOOLEAN_POLICY_PER_FLAG,),
            label="boolean_policy",
            error_type=ValueError,
            error_message=one_of_message,
        )
        self._field_spec_hash = digest
        self._boolean_policy = policy
        self._mask_token = parse_non_empty_str(mask_token, label="mask_token")

    def generating_process_signature(self) -> dict:
        return {
            "strategy": self.strategy,
            "args": {
                "field_spec_hash": self._field_spec_hash,
                "boolean_policy": self._boolean_policy,
                "mask_token": self._mask_token,
            },
        }

    def collect_inventory(self) -> dict:
        field_spec = FieldSpecCollection.load(self._field_spec_hash)
        inventory = build_universal_digits_inventory(
            specs=field_spec.columns(),
            mask_token=self._mask_token,
        )
        return inventory.to_dict()


def build_universal_digits_args(args: dict) -> dict:
    require_exact_keys(args, _ARG_KEYS, label="args")
    mask_token = args["mask_token"]
    if mask_token is None:
        mask_token = MASK_TOKEN_DEFAULT
    return {
        "field_spec_hash": args["field_spec_hash"],
        "boolean_policy": args["boolean_policy"],
        "mask_token": mask_token,
    }
