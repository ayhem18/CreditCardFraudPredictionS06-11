"""Masked-feature mask sampler: a mix of whole-feature and single-token masks.

Two independent draws per row, on the tokens' device:
- a whole-feature draw (mask every slot of a feature) — teaches cross-feature
  structure;
- a single-token draw (mask individual digit slots) — teaches intra-number
  digit structure, the point of the digit-sequence tokenizer.
Their union is the mask; at least one token per row is always masked so the
reconstruction loss is never empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class MaskSamplerConfig:
    token_mask_prob: float  # per-token single-slot mask probability
    feature_mask_prob: float  # per-feature whole-feature mask probability


class TokenMaskSampler:
    def __init__(self, *, feature_index: np.ndarray, n_features: int, config: MaskSamplerConfig) -> None:
        if not 0.0 <= config.token_mask_prob < 1.0:
            raise ValueError(f"token_mask_prob must be in [0,1), got {config.token_mask_prob}")
        if not 0.0 <= config.feature_mask_prob < 1.0:
            raise ValueError(f"feature_mask_prob must be in [0,1), got {config.feature_mask_prob}")
        self._feature_index = torch.as_tensor(feature_index, dtype=torch.long)
        self._n_features = n_features
        self._config = config

    def sample(self, *, tokens: Tensor) -> Tensor:
        """tokens (N, T) -> bool mask (N, T) on the same device."""
        n_rows, n_tokens = tokens.shape
        device = tokens.device
        feature_index = self._feature_index.to(device)
        single = torch.rand(n_rows, n_tokens, device=device) < self._config.token_mask_prob
        feature_draw = (
            torch.rand(n_rows, self._n_features, device=device) < self._config.feature_mask_prob
        )
        whole = feature_draw[:, feature_index]
        mask = single | whole
        empty = ~mask.any(dim=1)
        if bool(empty.any()):
            forced = torch.randint(0, n_tokens, (n_rows,), device=device)
            rows = torch.nonzero(empty, as_tuple=False).squeeze(1)
            mask[rows, forced[rows]] = True
        return mask
