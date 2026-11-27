"""Field-level masker: whole features, at least one position, seeded draw."""

from __future__ import annotations

import torch

from ccf_tabformer_dl.tabformer_components.masking.oracle import build_masker
from tests.tabformer_components.fake_spec import MASK_ID, tiny_ids, tiny_table_spec


def test_field_level_masks_whole_features_and_writes_mask_id() -> None:
    spec = tiny_table_spec()
    masker = build_masker(
        strategy="field_level", args={"feature_mask_prob": 0.5}, spec=spec
    )
    ids = tiny_ids()
    generator = torch.Generator().manual_seed(0)
    result = masker.mask(ids, generator=generator)
    assert tuple(result.masked_ids.shape) == tuple(ids.shape)
    assert tuple(result.target_mask.shape) == tuple(ids.shape)
    assert result.target_mask.dtype == torch.bool
    assert bool(result.target_mask.any(dim=1).all())
    assert torch.equal(result.masked_ids[result.target_mask], torch.full(
        (int(result.target_mask.sum()),), MASK_ID, dtype=torch.long
    ))
    assert torch.equal(result.masked_ids[~result.target_mask], ids[~result.target_mask])
    feature_index = spec.feature_index
    n_features = spec.n_features
    for feature in range(n_features):
        positions = feature_index == feature
        n_pos = int(positions.sum())
        per_row = result.target_mask[:, positions].sum(dim=1)
        ok = (per_row == 0) | (per_row == n_pos)
        assert bool(ok.all())


def test_field_level_seeded_generator_is_deterministic() -> None:
    spec = tiny_table_spec()
    masker = build_masker(
        strategy="field_level", args={"feature_mask_prob": 0.4}, spec=spec
    )
    ids = tiny_ids()
    first = masker.mask(ids, generator=torch.Generator().manual_seed(7))
    second = masker.mask(ids, generator=torch.Generator().manual_seed(7))
    third = masker.mask(ids, generator=torch.Generator().manual_seed(8))
    assert torch.equal(first.target_mask, second.target_mask)
    assert torch.equal(first.masked_ids, second.masked_ids)
    assert not torch.equal(first.target_mask, third.target_mask)


def test_field_level_forces_at_least_one_position() -> None:
    spec = tiny_table_spec()
    masker = build_masker(
        strategy="field_level", args={"feature_mask_prob": 0.001}, spec=spec
    )
    ids = tiny_ids().repeat(16, 1)
    result = masker.mask(ids, generator=torch.Generator().manual_seed(1))
    assert bool(result.target_mask.any(dim=1).all())
