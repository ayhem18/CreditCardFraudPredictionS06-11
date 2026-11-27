"""Row-independent inner-table layout: feature_index, slot_index, allowed ids."""

from __future__ import annotations

from ccf_tabformer_dl.data_contract.field_spec.entities import (
    BooleanColumnSpec,
    CategoricalColumnSpec,
    NumericColumnSpec,
)
from ccf_tabformer_dl.data_contract.vocabulary.entity import VocabularyInventory
from ccf_tabformer_dl.data_contract.written_numbers import (
    DIGIT_VOCAB_SIZE,
    NON_DIGIT_SLOT,
    max_place_slots,
    place_slot_ids_for_numeric,
)
from ccf_tabformer_dl.tabformer_components.table_spec import FrozenTableSpec, support_additive_from_allowed


class InnerTableLayout:
    def __init__(
        self,
        *,
        feature_index: tuple[int, ...],
        slot_index: tuple[int, ...],
        allowed_ids: tuple[tuple[int, ...], ...],
        n_features: int,
        max_slots: int,
        vocab_size: int,
        mask_id: int,
        column_spans: tuple[tuple[str, int, int], ...],
    ) -> None:
        seq_len = len(feature_index)
        if seq_len < 1:
            raise ValueError("layout seq_len must be positive")
        if len(slot_index) != seq_len or len(allowed_ids) != seq_len:
            raise ValueError("feature_index, slot_index, allowed_ids must share seq_len")
        self.feature_index = feature_index
        self.slot_index = slot_index
        self.allowed_ids = allowed_ids
        self.seq_len = seq_len
        self.n_features = n_features
        self.max_slots = max_slots
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.column_spans = column_spans

    def to_dict(self) -> dict:
        n = self.seq_len
        allowed = [None] * n
        for i in range(n):
            allowed[i] = list(self.allowed_ids[i])
        n_spans = len(self.column_spans)
        spans = [None] * n_spans
        for i in range(n_spans):
            name, start, width = self.column_spans[i]
            spans[i] = {"name": name, "start": start, "width": width}
        return {
            "seq_len": self.seq_len,
            "n_features": self.n_features,
            "max_slots": self.max_slots,
            "vocab_size": self.vocab_size,
            "mask_id": self.mask_id,
            "feature_index": list(self.feature_index),
            "slot_index": list(self.slot_index),
            "allowed_ids": allowed,
            "column_spans": spans,
        }

    def to_table_spec(self) -> FrozenTableSpec:
        return FrozenTableSpec(
            seq_len=self.seq_len,
            n_features=self.n_features,
            max_slots=self.max_slots,
            vocab_size=self.vocab_size,
            mask_id=self.mask_id,
            feature_index=self.feature_index,
            slot_index=self.slot_index,
            support_additive=support_additive_from_allowed(
                seq_len=self.seq_len,
                vocab_size=self.vocab_size,
                allowed_ids=self.allowed_ids,
            ),
        )

    @classmethod
    def from_payload(cls, document: dict) -> InnerTableLayout:
        allowed_raw = document["allowed_ids"]
        n = len(allowed_raw)
        allowed = [None] * n
        for i in range(n):
            allowed[i] = tuple(int(v) for v in allowed_raw[i])
        spans_raw = document["column_spans"]
        n_spans = len(spans_raw)
        spans = [None] * n_spans
        for i in range(n_spans):
            item = spans_raw[i]
            spans[i] = (item["name"], int(item["start"]), int(item["width"]))
        return cls(
            feature_index=tuple(int(v) for v in document["feature_index"]),
            slot_index=tuple(int(v) for v in document["slot_index"]),
            allowed_ids=tuple(allowed),
            n_features=int(document["n_features"]),
            max_slots=int(document["max_slots"]),
            vocab_size=int(document["vocab_size"]),
            mask_id=int(document["mask_id"]),
            column_spans=tuple(spans),
        )


def build_inner_table_layout(
    *,
    specs: tuple,
    inventory: VocabularyInventory,
) -> InnerTableLayout:
    n_features = len(specs)
    if n_features < 1:
        raise ValueError("field_spec has no columns")
    max_int, max_dec = _numeric_place_bounds(specs)
    if max_int == 0:
        max_slots = 1
    else:
        max_slots = max_place_slots(
            max_int_slots=max_int, max_decimal_slots=max_dec
        )
    digit_ids = tuple(range(DIGIT_VOCAB_SIZE))
    feature_ids: list[int] = []
    slot_ids: list[int] = []
    allowed: list[tuple[int, ...]] = []
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for feature in range(n_features):
        spec = specs[feature]
        start = cursor
        if isinstance(spec, NumericColumnSpec):
            slots = place_slot_ids_for_numeric(
                int_slots=spec.int_slots,
                decimal_slots=spec.decimal_slots,
                max_int_slots=max_int,
            )
            width = len(slots)
            for j in range(width):
                feature_ids.append(feature)
                slot_ids.append(slots[j])
                allowed.append(digit_ids)
        elif isinstance(spec, (CategoricalColumnSpec, BooleanColumnSpec)):
            width = 1
            feature_ids.append(feature)
            slot_ids.append(NON_DIGIT_SLOT)
            allowed.append(inventory.allowed_ids_for_column(column=spec.name))
        else:
            raise TypeError(f"unknown column spec type {type(spec)!r}")
        spans.append((spec.name, start, width))
        cursor += width
    return InnerTableLayout(
        feature_index=tuple(feature_ids),
        slot_index=tuple(slot_ids),
        allowed_ids=tuple(allowed),
        n_features=n_features,
        max_slots=max_slots,
        vocab_size=inventory.vocab_size,
        mask_id=inventory.mask_id,
        column_spans=tuple(spans),
    )


def _numeric_place_bounds(specs: tuple) -> tuple[int, int]:
    max_int = 0
    max_dec = 0
    n = len(specs)
    for i in range(n):
        spec = specs[i]
        if isinstance(spec, NumericColumnSpec):
            if spec.int_slots > max_int:
                max_int = spec.int_slots
            if spec.decimal_slots > max_dec:
                max_dec = spec.decimal_slots
    return max_int, max_dec
