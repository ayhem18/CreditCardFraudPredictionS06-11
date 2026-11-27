"""Additive embedder: value + feature + slot, shape and mask-id embedding."""

from __future__ import annotations

import torch

from ccf_tabformer_dl.tabformer_components.embedding.oracle import build_embedder
from tests.tabformer_components.fake_spec import MASK_ID, tiny_ids, tiny_table_spec

D_MODEL = 8


def test_additive_forward_shape_and_structural_add() -> None:
    spec = tiny_table_spec()
    embedder = build_embedder(
        strategy="additive", args={"d_model": D_MODEL}, spec=spec
    )
    ids = tiny_ids()
    hidden = embedder(ids)
    assert tuple(hidden.shape) == (ids.shape[0], spec.seq_len, D_MODEL)

    value = embedder.value_emb(ids)
    feature = embedder.feature_emb(embedder.feature_index).unsqueeze(0)
    slot = embedder.slot_emb(embedder.slot_index).unsqueeze(0)
    torch.testing.assert_close(hidden, value + feature + slot)


def test_mask_id_keeps_feature_and_slot() -> None:
    spec = tiny_table_spec()
    embedder = build_embedder(
        strategy="additive", args={"d_model": D_MODEL}, spec=spec
    )
    ids = tiny_ids()
    masked = ids.clone()
    masked[0, 0] = MASK_ID
    hidden = embedder(masked)
    expected = (
        embedder.value_emb(masked)
        + embedder.feature_emb(embedder.feature_index).unsqueeze(0)
        + embedder.slot_emb(embedder.slot_index).unsqueeze(0)
    )
    torch.testing.assert_close(hidden, expected)
    assert not torch.equal(hidden[0, 0], embedder(ids)[0, 0])
