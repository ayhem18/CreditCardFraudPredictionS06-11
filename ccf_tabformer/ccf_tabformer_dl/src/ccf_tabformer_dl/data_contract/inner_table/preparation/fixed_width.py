"""``fixed_width_padded``: place-value digit pad + one token per cat/flag."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from dl_toolbox.base.hashing.identity import is_full_sha256
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys
from tqdm import tqdm

from ccf_tabformer_dl.data_contract.corrected_table import (
    corrected_table_identity,
    load_corrected_frames,
    pooled_feature_series,
    pooled_labels,
    pooled_row_ids,
    pooled_split,
)
from ccf_tabformer_dl.data_contract.field_spec.entities import (
    BooleanColumnSpec,
    CategoricalColumnSpec,
    NumericColumnSpec,
)
from ccf_tabformer_dl.data_contract.inner_table.collection import (
    LABELS_FILE,
    ROW_IDS_FILE,
    SPLIT_FILE,
    TOKEN_IDS_FILE,
    payload_meta_from_dir,
)
from ccf_tabformer_dl.data_contract.inner_table.layout import build_inner_table_layout
from ccf_tabformer_dl.data_contract.inner_table.preparation.preparation import InnerTablePreparation
from ccf_tabformer_dl.data_contract.vocabulary.collection import VocabularyCollection
from ccf_tabformer_dl.data_contract.vocabulary.entity import VocabularyInventory
from ccf_tabformer_dl.data_contract.written_numbers import digit_codes_for_numeric

FIXED_WIDTH_PADDED_STRATEGY = "fixed_width_padded"
_ARG_KEYS = ("vocabulary_hash",)


class FixedWidthPaddedPreparation(InnerTablePreparation):
    strategy = FIXED_WIDTH_PADDED_STRATEGY

    def __init__(self, *, vocabulary_hash: str) -> None:
        digest = parse_non_empty_str(vocabulary_hash, label="vocabulary_hash")
        if not is_full_sha256(digest):
            raise ValueError(
                f"vocabulary_hash must be a collection hash, got {vocabulary_hash!r}"
            )
        self._vocabulary_hash = digest

    def generating_process_signature(self) -> dict:
        return {
            "strategy": self.strategy,
            "args": {
                "vocabulary_hash": self._vocabulary_hash,
                "corrected_table": corrected_table_identity(),
            },
        }

    def write_payload(self, *, artifacts_dir: Path) -> tuple[dict, dict]:
        vocabulary = VocabularyCollection.load(self._vocabulary_hash)
        field_spec = vocabulary.parent_collection()
        expected_table = field_spec.generating_process_signature["args"][
            "corrected_table"
        ]
        current_table = corrected_table_identity()
        if expected_table != current_table:
            raise ValueError(
                "corrected_table identity drifted between field_spec and "
                f"inner_table: field_spec={expected_table} current={current_table}"
            )
        specs = field_spec.columns()
        inventory = vocabulary.inventory()
        layout = build_inner_table_layout(specs=specs, inventory=inventory)
        train, test = load_corrected_frames()
        n_train = int(train.shape[0])
        n_test = int(test.shape[0])
        token_ids = encode_inner_table(
            train=train,
            test=test,
            specs=specs,
            inventory=inventory,
            layout=layout,
        )
        np.save(artifacts_dir / TOKEN_IDS_FILE, token_ids)
        np.save(
            artifacts_dir / ROW_IDS_FILE,
            pooled_row_ids(train=train, test=test),
        )
        np.save(
            artifacts_dir / SPLIT_FILE,
            pooled_split(n_train=n_train, n_test=n_test),
        )
        np.save(
            artifacts_dir / LABELS_FILE,
            pooled_labels(train=train, n_test=n_test),
        )
        layout_document = layout.to_dict()
        payload_meta = payload_meta_from_dir(
            artifacts_dir, seq_len=layout.seq_len
        )
        return layout_document, payload_meta


def encode_inner_table(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    specs: tuple,
    inventory: VocabularyInventory,
    layout,
) -> np.ndarray:
    n_rows = int(train.shape[0]) + int(test.shape[0])
    token_ids = np.empty((n_rows, layout.seq_len), dtype=np.int32)
    n_spans = len(layout.column_spans)
    for i in tqdm(range(n_spans), desc="inner_table columns"):
        name, start, width = layout.column_spans[i]
        spec = specs[i]
        if spec.name != name:
            raise ValueError(
                f"layout span {name!r} does not match spec {spec.name!r}"
            )
        series = pooled_feature_series(train=train, test=test, column=name)
        block = encode_column(series=series, spec=spec, inventory=inventory)
        if block.shape != (n_rows, width):
            raise ValueError(
                f"{name}: encoded shape {block.shape} != ({n_rows}, {width})"
            )
        token_ids[:, start : start + width] = block
    return token_ids


def encode_column(
    *,
    series: pd.Series,
    spec,
    inventory: VocabularyInventory,
) -> np.ndarray:
    if isinstance(spec, NumericColumnSpec):
        return digit_codes_for_numeric(
            series=series,
            column=spec.name,
            int_slots=spec.int_slots,
            decimal_slots=spec.decimal_slots,
        )
    if isinstance(spec, (CategoricalColumnSpec, BooleanColumnSpec)):
        return _map_written_values(
            series=series,
            column=spec.name,
            inventory=inventory,
        )
    raise TypeError(f"unknown column spec type {type(spec)!r}")


def _map_written_values(
    *,
    series: pd.Series,
    column: str,
    inventory: VocabularyInventory,
) -> np.ndarray:
    stripped = series.astype(str).str.strip()
    mapping = inventory.column_blocks[column]["global_value_to_id"]
    mapped = stripped.map(mapping)
    if bool(mapped.isna().any()):
        bad_index = int(mapped.isna().to_numpy().nonzero()[0][0])
        raise ValueError(
            f"{column}: value {stripped.iloc[bad_index]!r} is not in the vocabulary"
        )
    return mapped.to_numpy(dtype=np.int32).reshape(-1, 1)


def build_fixed_width_padded_args(args: dict) -> dict:
    require_exact_keys(args, _ARG_KEYS, label="args")
    return {"vocabulary_hash": args["vocabulary_hash"]}
