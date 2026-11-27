"""``declared``: roles and in-between routing are given; facts are measured."""

from __future__ import annotations

import pandas as pd
from dl_toolbox.base.parsing.parse_iterable import parse_mapping, parse_sequence
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import (
    one_of_message,
    parse_one_of,
    require_exact_keys,
)

from ccf_tabformer_dl.data_contract.columns import (
    BOOLEAN_FALSE,
    BOOLEAN_TRUE,
    ROLE_BOOLEAN,
    ROLE_CATEGORICAL,
    ROLE_IN_BETWEEN,
    ROLE_NUMERIC,
    ROLES,
    ROUTING_AS_CAT,
    ROUTING_AS_NUM,
)
from ccf_tabformer_dl.data_contract.corrected_table import (
    corrected_table_identity,
    load_corrected_frames,
    pooled_feature_series,
)
from ccf_tabformer_dl.data_contract.field_spec.entities import (
    BooleanColumnSpec,
    CategoricalColumnSpec,
    NumericColumnSpec,
)
from ccf_tabformer_dl.data_contract.field_spec.preparation.preparation import FieldSpecPreparation
from ccf_tabformer_dl.data_contract.written_numbers import measure_numeric_slots

DECLARED_STRATEGY = "declared"
_ARG_KEYS = ("columns",)


class DeclaredFieldSpecPreparation(FieldSpecPreparation):
    strategy = DECLARED_STRATEGY

    def __init__(self, *, columns: object) -> None:
        self._columns = _parse_declarations(columns)

    def generating_process_signature(self) -> dict:
        n = len(self._columns)
        columns = [None] * n
        for i in range(n):
            columns[i] = dict(self._columns[i])
        return {
            "strategy": self.strategy,
            "args": {
                "columns": columns,
                "corrected_table": corrected_table_identity(),
            },
        }

    def collect_members(self) -> dict:
        train, test = load_corrected_frames()
        n = len(self._columns)
        members: dict = {}
        for i in range(n):
            declaration = self._columns[i]
            name = declaration["name"]
            spec = measure_column(
                train=train,
                test=test,
                declaration=declaration,
            )
            members[name] = spec.to_dict()
        return members


def measure_column(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    declaration: dict,
) -> NumericColumnSpec | CategoricalColumnSpec | BooleanColumnSpec:
    name = declaration["name"]
    role = declaration["role"]
    series = pooled_feature_series(train=train, test=test, column=name)
    if role == ROLE_NUMERIC:
        int_slots, decimal_slots = measure_numeric_slots(series=series, column=name)
        return NumericColumnSpec(
            name=name, int_slots=int_slots, decimal_slots=decimal_slots
        )
    if role == ROLE_IN_BETWEEN:
        routing = declaration["routing"]
        if routing != ROUTING_AS_CAT:
            raise NotImplementedError(
                f"{name}: in_between routing {routing!r} is not implemented; "
                f"use {ROUTING_AS_CAT!r}"
            )
        return CategoricalColumnSpec(
            name=name,
            value_to_id=_unique_value_to_id(series=series, column=name),
        )
    if role == ROLE_CATEGORICAL:
        return CategoricalColumnSpec(
            name=name,
            value_to_id=_unique_value_to_id(series=series, column=name),
        )
    if role == ROLE_BOOLEAN:
        _require_boolean_written(series=series, column=name)
        return BooleanColumnSpec(
            name=name,
            value_to_id={BOOLEAN_FALSE: 0, BOOLEAN_TRUE: 1},
        )
    raise ValueError(f"{name}: unknown role {role!r}")


def _unique_value_to_id(*, series: pd.Series, column: str) -> dict[str, int]:
    stripped = series.astype(str).str.strip()
    if bool(stripped.eq("").any()) or bool(stripped.isna().any()):
        raise ValueError(f"{column}: empty or missing written values")
    unique_values = sorted(stripped.unique().tolist())
    n_values = len(unique_values)
    if n_values < 1:
        raise ValueError(f"{column}: no unique values")
    mapping: dict[str, int] = {}
    for i in range(n_values):
        mapping[str(unique_values[i])] = i
    return mapping


def _require_boolean_written(*, series: pd.Series, column: str) -> None:
    stripped = series.astype(str).str.strip()
    unique_values = set(stripped.unique().tolist())
    allowed = {BOOLEAN_FALSE, BOOLEAN_TRUE}
    extra = unique_values - allowed
    if extra:
        raise ValueError(
            f"{column}: boolean values must be {sorted(allowed)}, extra={sorted(extra)}"
        )
    missing = allowed - unique_values
    if missing:
        raise ValueError(
            f"{column}: boolean values missing {sorted(missing)}"
        )


def _parse_declarations(raw: object) -> tuple[dict, ...]:
    sequence = parse_sequence(raw, label="columns")
    n = len(sequence)
    if n < 1:
        raise ValueError("columns must be non-empty")
    parsed = [None] * n
    seen: dict[str, None] = {}
    for i in range(n):
        item = parse_mapping(sequence[i], label=f"columns[{i}]")
        name = parse_non_empty_str(item.get("name"), label=f"columns[{i}].name")
        if name in seen:
            raise ValueError(f"duplicate column {name!r}")
        seen[name] = None
        role = parse_one_of(
            item.get("role"),
            allowed=ROLES,
            label=f"columns[{i}].role",
            error_type=ValueError,
            error_message=one_of_message,
        )
        if role == ROLE_IN_BETWEEN:
            require_exact_keys(
                item, ("name", "role", "routing"), label=f"columns[{i}]"
            )
            routing = parse_one_of(
                item["routing"],
                allowed=(ROUTING_AS_CAT, ROUTING_AS_NUM),
                label=f"columns[{i}].routing",
                error_type=ValueError,
                error_message=one_of_message,
            )
            parsed[i] = {"name": name, "role": role, "routing": routing}
        else:
            require_exact_keys(item, ("name", "role"), label=f"columns[{i}]")
            parsed[i] = {"name": name, "role": role}
    return tuple(parsed)


def build_declared_args(args: dict) -> dict:
    require_exact_keys(args, _ARG_KEYS, label="args")
    return {"columns": args["columns"]}
