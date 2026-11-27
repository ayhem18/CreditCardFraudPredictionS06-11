"""`field_level` masker: mask whole features (the decided default, B.2).

Picks features per row and masks *every* position of each chosen feature, so the
model cannot rebuild a masked digit from the same number's other digits — it must
use the other fields (`P(field | other fields)`). The mixed digit+feature draw of
the old `mfm/masking.py` is a *separate* ablation strategy, not this one.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ccf_tabformer_dl.tabformer_components.masking.masker import Masker, MaskResult
from ccf_tabformer_dl.tabformer_components.table_spec import TableSpec

FIELD_LEVEL_STRATEGY = "field_level"
_ARG_KEYS = ("feature_mask_prob",)


class FieldLevelMasker(Masker):
    strategy = FIELD_LEVEL_STRATEGY

    def __init__(self, *, spec: TableSpec, feature_mask_prob: float) -> None:
        super().__init__(spec=spec)
        if not isinstance(feature_mask_prob, (int, float)):
            raise TypeError("feature_mask_prob must be a number")
        if not 0.0 < float(feature_mask_prob) < 1.0:
            raise ValueError(
                f"feature_mask_prob must be in (0, 1), got {feature_mask_prob!r}"
            )
        self._feature_mask_prob = float(feature_mask_prob)

    def mask(self, ids: Tensor, *, generator: torch.Generator | None) -> MaskResult:
        if ids.ndim != 2:
            raise ValueError(f"ids must be long (B, S), got shape {tuple(ids.shape)}")
        n_rows = int(ids.shape[0])
        seq_len = int(ids.shape[1])
        feature_index = self._spec.feature_index.to(device=ids.device, dtype=torch.long)
        if int(feature_index.shape[0]) != seq_len:
            raise ValueError(
                f"ids seq_len {seq_len} != spec.feature_index length "
                f"{int(feature_index.shape[0])}"
            )
        n_features = int(self._spec.n_features)
        feature_draw = _bernoulli_features(
            n_rows=n_rows,
            n_features=n_features,
            prob=self._feature_mask_prob,
            device=ids.device,
            generator=generator,
        )
        empty = ~feature_draw.any(dim=1)
        if bool(empty.any()):
            empty_rows = torch.nonzero(empty, as_tuple=False).squeeze(1)
            n_empty = int(empty_rows.shape[0])
            forced = _randint(
                n_empty,
                n_features,
                device=ids.device,
                generator=generator,
            )
            feature_draw[empty_rows, forced] = True
        target_mask = feature_draw[:, feature_index]
        masked_ids = ids.clone()
        masked_ids[target_mask] = int(self._spec.mask_id)
        return MaskResult(masked_ids=masked_ids, target_mask=target_mask)


def _bernoulli_features(
    *,
    n_rows: int,
    n_features: int,
    prob: float,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    if generator is None:
        draw = torch.rand(n_rows, n_features, device=device)
    else:
        draw = torch.rand(n_rows, n_features, generator=generator).to(device=device)
    return draw < prob


def _randint(
    n: int,
    n_features: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    if generator is None:
        return torch.randint(0, n_features, (n,), device=device)
    return torch.randint(0, n_features, (n,), generator=generator).to(device=device)


def build_field_level_args(args: dict) -> dict:
    if set(args) != set(_ARG_KEYS):
        raise ValueError(
            f"field_level args must have exactly {_ARG_KEYS}, got {sorted(args)}"
        )
    return {"feature_mask_prob": args["feature_mask_prob"]}
