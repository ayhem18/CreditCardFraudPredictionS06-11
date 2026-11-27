"""Role-specific column specs. In-between routed ``as_cat`` becomes categorical."""

from __future__ import annotations

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import one_of_message, parse_one_of, require_exact_keys

from ccf_tabformer_dl.data_contract.columns import (
    BOOLEAN_FALSE,
    BOOLEAN_TRUE,
    ROLE_BOOLEAN,
    ROLE_CATEGORICAL,
    ROLE_NUMERIC,
    ROLES,
)

ColumnSpec = "NumericColumnSpec | CategoricalColumnSpec | BooleanColumnSpec"


class NumericColumnSpec:
    def __init__(self, *, name: str, int_slots: int, decimal_slots: int) -> None:
        self.name = parse_non_empty_str(name, label="name")
        if not isinstance(int_slots, int) or int_slots < 1:
            raise ValueError(f"{self.name}: int_slots must be >= 1, got {int_slots!r}")
        if not isinstance(decimal_slots, int) or decimal_slots < 0:
            raise ValueError(
                f"{self.name}: decimal_slots must be >= 0, got {decimal_slots!r}"
            )
        self.int_slots = int_slots
        self.decimal_slots = decimal_slots

    @property
    def digit_slots(self) -> int:
        return self.int_slots + self.decimal_slots

    @property
    def role(self) -> str:
        return ROLE_NUMERIC

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": ROLE_NUMERIC,
            "int_slots": self.int_slots,
            "decimal_slots": self.decimal_slots,
            "digit_slots": self.digit_slots,
        }

    @classmethod
    def from_payload(cls, document: dict) -> NumericColumnSpec:
        require_exact_keys(
            document,
            ("name", "role", "int_slots", "decimal_slots", "digit_slots"),
            label="numeric column",
        )
        spec = cls(
            name=document["name"],
            int_slots=int(document["int_slots"]),
            decimal_slots=int(document["decimal_slots"]),
        )
        if int(document["digit_slots"]) != spec.digit_slots:
            raise ValueError(
                f"{spec.name}: digit_slots {document['digit_slots']!r} != "
                f"{spec.digit_slots}"
            )
        return spec


class CategoricalColumnSpec:
    def __init__(self, *, name: str, value_to_id: dict[str, int]) -> None:
        self.name = parse_non_empty_str(name, label="name")
        self.value_to_id = _lexicographic_value_to_id(value_to_id, column=self.name)

    @property
    def role(self) -> str:
        return ROLE_CATEGORICAL

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": ROLE_CATEGORICAL,
            "value_to_id": dict(self.value_to_id),
        }

    @classmethod
    def from_payload(cls, document: dict) -> CategoricalColumnSpec:
        require_exact_keys(
            document,
            ("name", "role", "value_to_id"),
            label="categorical column",
        )
        return cls(name=document["name"], value_to_id=document["value_to_id"])


class BooleanColumnSpec:
    def __init__(self, *, name: str, value_to_id: dict[str, int]) -> None:
        self.name = parse_non_empty_str(name, label="name")
        mapped = _lexicographic_value_to_id(value_to_id, column=self.name)
        expected = {BOOLEAN_FALSE: 0, BOOLEAN_TRUE: 1}
        if mapped != expected:
            raise ValueError(
                f"{self.name}: boolean value_to_id must be {expected}, got {mapped}"
            )
        self.value_to_id = mapped

    @property
    def role(self) -> str:
        return ROLE_BOOLEAN

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": ROLE_BOOLEAN,
            "value_to_id": dict(self.value_to_id),
        }

    @classmethod
    def from_payload(cls, document: dict) -> BooleanColumnSpec:
        require_exact_keys(
            document,
            ("name", "role", "value_to_id"),
            label="boolean column",
        )
        return cls(name=document["name"], value_to_id=document["value_to_id"])


def column_spec_from_payload(document: dict) -> NumericColumnSpec | CategoricalColumnSpec | BooleanColumnSpec:
    payload = parse_mapping(document, label="column spec")
    role = parse_one_of(
        payload.get("role"),
        allowed=ROLES,
        label="role",
        error_type=ValueError,
        error_message=one_of_message,
    )
    if role == ROLE_NUMERIC:
        return NumericColumnSpec.from_payload(payload)
    if role == ROLE_CATEGORICAL:
        return CategoricalColumnSpec.from_payload(payload)
    if role == ROLE_BOOLEAN:
        return BooleanColumnSpec.from_payload(payload)
    raise ValueError(
        f"in_between is a declaration role, not a stored entity role, got {role!r}"
    )


def _lexicographic_value_to_id(raw: object, *, column: str) -> dict[str, int]:
    mapping = parse_mapping(raw, label=f"{column} value_to_id")
    keys = list(mapping)
    n_keys = len(keys)
    if n_keys < 1:
        raise ValueError(f"{column}: value_to_id must be non-empty")
    values = [""] * n_keys
    for i in range(n_keys):
        values[i] = parse_non_empty_str(keys[i], label=f"{column} value")
    ordered = sorted(values)
    canonical: dict[str, int] = {}
    for i in range(n_keys):
        value = ordered[i]
        token_id = mapping[value]
        if not isinstance(token_id, int) or token_id != i:
            raise ValueError(
                f"{column}: value {value!r} must have lexicographic id {i}, got {token_id!r}"
            )
        canonical[value] = i
    if len(canonical) != n_keys:
        raise ValueError(f"{column}: duplicate values in value_to_id")
    return canonical
