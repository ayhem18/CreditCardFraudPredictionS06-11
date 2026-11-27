"""Vocabulary layout, inner-table encode, and row-index dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ccf_tabformer_dl.data_contract.field_spec.entities import (
    BooleanColumnSpec,
    CategoricalColumnSpec,
    NumericColumnSpec,
)
from ccf_tabformer_dl.data_contract.inner_table.layout import build_inner_table_layout
from ccf_tabformer_dl.data_contract.inner_table.preparation.fixed_width import encode_inner_table
from ccf_tabformer_dl.data_contract.inner_table_dataset import InnerTableIndexDataset
from ccf_tabformer_dl.data_contract.vocabulary.entity import (
    MASK_TOKEN_DEFAULT,
    build_universal_digits_inventory,
)
from ccf_tabformer_dl.tabformer_components.reconstruction.steps import IDS_KEY


def _tiny_specs():
    return (
        NumericColumnSpec(name="amount_usd", int_slots=2, decimal_slots=1),
        CategoricalColumnSpec(
            name="card_type", value_to_id={"Amex": 0, "Visa": 1}
        ),
        BooleanColumnSpec(
            name="used_vpn", value_to_id={"False": 0, "True": 1}
        ),
    )


def test_universal_digits_inventory_ranges() -> None:
    inventory = build_universal_digits_inventory(
        specs=_tiny_specs(), mask_token=MASK_TOKEN_DEFAULT
    )
    assert inventory.vocab_size == 15
    assert inventory.mask_id == 14
    assert inventory.global_id(column="card_type", written="Visa") == 11
    assert inventory.global_id(column="used_vpn", written="True") == 13
    assert inventory.allowed_ids_for_column(column="card_type") == (10, 11)


def test_encode_inner_table_and_index_dataset() -> None:
    specs = _tiny_specs()
    inventory = build_universal_digits_inventory(
        specs=specs, mask_token=MASK_TOKEN_DEFAULT
    )
    layout = build_inner_table_layout(specs=specs, inventory=inventory)
    assert layout.seq_len == 5
    assert layout.n_features == 3
    assert layout.slot_index[0:3] == (1, 2, 3)
    assert layout.slot_index[3:] == (0, 0)
    train = pd.DataFrame(
        {
            "amount_usd": ["12.5", "3.0"],
            "card_type": ["Visa", "Amex"],
            "used_vpn": ["True", "False"],
        }
    )
    test = pd.DataFrame(
        {
            "amount_usd": ["1.2"],
            "card_type": ["Amex"],
            "used_vpn": ["False"],
        }
    )
    token_ids = encode_inner_table(
        train=train,
        test=test,
        specs=specs,
        inventory=inventory,
        layout=layout,
    )
    expected = np.array(
        [
            [1, 2, 5, 11, 13],
            [0, 3, 0, 10, 12],
            [0, 1, 2, 10, 12],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(token_ids, expected)
    dataset = InnerTableIndexDataset(
        token_ids=token_ids, row_index=np.array([2, 0], dtype=np.int64)
    )
    assert len(dataset) == 2
    first = dataset[0][IDS_KEY]
    assert torch.equal(first, torch.tensor(expected[2], dtype=torch.long))
    spec = layout.to_table_spec()
    assert spec.vocab_size == 15
    assert spec.mask_id == 14
    assert int(spec.feature_index[0]) == 0
    assert int(spec.feature_index[3]) == 1
