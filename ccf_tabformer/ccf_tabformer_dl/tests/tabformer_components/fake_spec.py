"""A tiny `FrozenTableSpec` for reconstruction-component tests."""

from __future__ import annotations

import torch

from ccf_tabformer_dl.tabformer_components.table_spec import FrozenTableSpec, support_additive_from_allowed

DIGIT_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
CAT_IDS = (10, 11)
MASK_ID = 12
VOCAB_SIZE = 13
SEQ_LEN = 3
N_FEATURES = 2
MAX_SLOTS = 3


def tiny_table_spec() -> FrozenTableSpec:
    """Two features: a 2-digit number and a one-token categorical."""
    allowed_ids = (DIGIT_IDS, DIGIT_IDS, CAT_IDS)
    return FrozenTableSpec(
        seq_len=SEQ_LEN,
        n_features=N_FEATURES,
        max_slots=MAX_SLOTS,
        vocab_size=VOCAB_SIZE,
        mask_id=MASK_ID,
        feature_index=torch.tensor([0, 0, 1], dtype=torch.long),
        slot_index=torch.tensor([1, 2, 0], dtype=torch.long),
        support_additive=support_additive_from_allowed(
            seq_len=SEQ_LEN,
            vocab_size=VOCAB_SIZE,
            allowed_ids=allowed_ids,
        ),
    )


def tiny_ids() -> torch.Tensor:
    return torch.tensor(
        [
            [1, 2, 10],
            [0, 9, 11],
            [3, 4, 10],
            [8, 0, 11],
        ],
        dtype=torch.long,
    )
