"""Torch dataset over global-id token rows.

Rows are pre-mapped to global ids (see ``model.vocab.to_global_ids``). The
dataset yields one row of token ids and (optionally) its label; masking is
applied later, per step, in the FitSteps ``prepare_tensors`` so each step sees
a fresh mask.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

TOKENS_KEY = "tokens"
LABEL_KEY = "label"


class TokenizedDataset(Dataset):
    """Global-id token rows, with an optional ``is_fraud`` label."""

    def __init__(self, *, global_ids: np.ndarray, labels: np.ndarray | None) -> None:
        if global_ids.ndim != 2:
            raise ValueError(f"global_ids must be 2-D, got {global_ids.shape}")
        if labels is not None and labels.shape[0] != global_ids.shape[0]:
            raise ValueError(
                f"labels length {labels.shape[0]} != rows {global_ids.shape[0]}"
            )
        self._tokens = torch.as_tensor(global_ids, dtype=torch.long)
        self._labels = None if labels is None else torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self._tokens.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = {TOKENS_KEY: self._tokens[index]}
        if self._labels is not None:
            item[LABEL_KEY] = self._labels[index]
        return item
