"""Global token inventory: digits, per-column value blocks, one ``[MASK]``."""

from __future__ import annotations

from dl_toolbox.base.parsing.parse_iterable import parse_mapping, parse_sequence
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str

from ccf_tabformer_dl.data_contract.field_spec.entities import (
    BooleanColumnSpec,
    CategoricalColumnSpec,
    NumericColumnSpec,
)
from ccf_tabformer_dl.data_contract.written_numbers import DIGIT_VOCAB_SIZE

KIND_DIGIT = "digit"
KIND_CATEGORICAL = "categorical"
KIND_BOOLEAN = "boolean"
KIND_SPECIAL = "special"
MASK_TOKEN_DEFAULT = "[MASK]"


class VocabularyInventory:
    def __init__(
        self,
        *,
        mask_token: str,
        tokens: list[dict],
        column_blocks: dict,
    ) -> None:
        self.mask_token = parse_non_empty_str(mask_token, label="mask_token")
        parsed_tokens, mask_id, vocab_size = _parse_tokens(tokens)
        blocks = _parse_column_blocks(column_blocks)
        if parsed_tokens[mask_id]["value"] != self.mask_token:
            raise ValueError(
                f"mask token at id {mask_id} is {parsed_tokens[mask_id]['value']!r}, "
                f"expected {self.mask_token!r}"
            )
        self.tokens = parsed_tokens
        self.column_blocks = blocks
        self.mask_id = mask_id
        self.vocab_size = vocab_size

    def global_id(self, *, column: str, written: str) -> int:
        if column not in self.column_blocks:
            raise KeyError(f"no vocabulary block for column {column!r}")
        mapping = self.column_blocks[column]["global_value_to_id"]
        if written not in mapping:
            raise KeyError(f"column {column!r} has no value {written!r}")
        return int(mapping[written])

    def allowed_ids_for_column(self, *, column: str) -> tuple[int, ...]:
        if column not in self.column_blocks:
            raise KeyError(f"no vocabulary block for column {column!r}")
        block = self.column_blocks[column]
        start = int(block["start"])
        n = int(block["n"])
        ids = [0] * n
        for i in range(n):
            ids[i] = start + i
        return tuple(ids)

    def to_dict(self) -> dict:
        n_tokens = len(self.tokens)
        tokens = [None] * n_tokens
        for i in range(n_tokens):
            tokens[i] = dict(self.tokens[i])
        columns = sorted(self.column_blocks)
        n_columns = len(columns)
        blocks: dict = {}
        for i in range(n_columns):
            name = columns[i]
            block = self.column_blocks[name]
            blocks[name] = {
                "start": int(block["start"]),
                "n": int(block["n"]),
                "global_value_to_id": dict(block["global_value_to_id"]),
            }
        return {
            "mask_token": self.mask_token,
            "mask_id": self.mask_id,
            "vocab_size": self.vocab_size,
            "tokens": tokens,
            "column_blocks": blocks,
        }

    @classmethod
    def from_payload(cls, document: dict) -> VocabularyInventory:
        return cls(
            mask_token=document["mask_token"],
            tokens=document["tokens"],
            column_blocks=document["column_blocks"],
        )


def build_universal_digits_inventory(
    *,
    specs: tuple,
    mask_token: str,
) -> VocabularyInventory:
    """Digits 0–9, then one contiguous block per non-numeric column, then MASK."""
    token = parse_non_empty_str(mask_token, label="mask_token")
    n_digits = DIGIT_VOCAB_SIZE
    tokens: list[dict] = [None] * n_digits
    for digit in range(n_digits):
        tokens[digit] = {
            "id": digit,
            "kind": KIND_DIGIT,
            "value": str(digit),
            "column": None,
        }
    next_id = n_digits
    column_blocks: dict = {}
    n_specs = len(specs)
    for i in range(n_specs):
        spec = specs[i]
        if isinstance(spec, NumericColumnSpec):
            continue
        if not isinstance(spec, (CategoricalColumnSpec, BooleanColumnSpec)):
            raise TypeError(f"unknown column spec type {type(spec)!r}")
        kind = KIND_BOOLEAN if isinstance(spec, BooleanColumnSpec) else KIND_CATEGORICAL
        local = spec.value_to_id
        n_values = len(local)
        start = next_id
        ordered = sorted(local, key=lambda value: local[value])
        global_map: dict[str, int] = {}
        extra = [None] * n_values
        for j in range(n_values):
            value = ordered[j]
            gid = start + int(local[value])
            extra[j] = {
                "id": gid,
                "kind": kind,
                "value": value,
                "column": spec.name,
            }
            global_map[value] = gid
        tokens.extend(extra)
        column_blocks[spec.name] = {
            "start": start,
            "n": n_values,
            "global_value_to_id": global_map,
        }
        next_id = start + n_values
    mask_id = next_id
    tokens.append(
        {
            "id": mask_id,
            "kind": KIND_SPECIAL,
            "value": token,
            "column": None,
        }
    )
    return VocabularyInventory(
        mask_token=token,
        tokens=tokens,
        column_blocks=column_blocks,
    )


def _parse_tokens(raw: object) -> tuple[list[dict], int, int]:
    sequence = parse_sequence(raw, label="tokens")
    n = len(sequence)
    if n < DIGIT_VOCAB_SIZE + 1:
        raise ValueError(
            f"tokens must include digits 0-9 and a mask token, got length {n}"
        )
    parsed = [None] * n
    mask_id = None
    for i in range(n):
        item = parse_mapping(sequence[i], label=f"tokens[{i}]")
        token_id = int(item["id"])
        if token_id != i:
            raise ValueError(f"tokens[{i}].id must be {i}, got {token_id!r}")
        kind = parse_non_empty_str(item["kind"], label=f"tokens[{i}].kind")
        value = parse_non_empty_str(item["value"], label=f"tokens[{i}].value")
        column = item.get("column")
        if column is not None:
            column = parse_non_empty_str(column, label=f"tokens[{i}].column")
        parsed[i] = {"id": token_id, "kind": kind, "value": value, "column": column}
        if kind == KIND_SPECIAL:
            mask_id = token_id
    if mask_id is None:
        raise ValueError("tokens must include a special mask token")
    return parsed, mask_id, n


def _parse_column_blocks(raw: object) -> dict:
    mapping = parse_mapping(raw, label="column_blocks")
    keys = list(mapping)
    n_keys = len(keys)
    blocks: dict = {}
    for i in range(n_keys):
        name = parse_non_empty_str(keys[i], label="column_blocks key")
        block = parse_mapping(mapping[keys[i]], label=f"column_blocks[{name!r}]")
        start = int(block["start"])
        n = int(block["n"])
        value_map = parse_mapping(
            block["global_value_to_id"],
            label=f"column_blocks[{name!r}].global_value_to_id",
        )
        canonical: dict[str, int] = {}
        values = list(value_map)
        n_values = len(values)
        if n_values != n:
            raise ValueError(
                f"column_blocks[{name!r}].n {n} != global_value_to_id size {n_values}"
            )
        for j in range(n_values):
            written = parse_non_empty_str(
                values[j], label=f"column_blocks[{name!r}] value"
            )
            canonical[written] = int(value_map[values[j]])
        blocks[name] = {
            "start": start,
            "n": n,
            "global_value_to_id": canonical,
        }
    return blocks
