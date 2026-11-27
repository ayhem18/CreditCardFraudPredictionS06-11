"""The Masker contract. See documentation/decisions/masking.md.

Design decisions this interface encodes:
- The Masker is built by its oracle from the *same* `TableSpec` as the Embedder;
  the two share only the spec and never reference each other.
- Masking writes the reserved `[MASK]` id (`spec.mask_id`) into the chosen
  positions — it does not reach into any embedding table.
- Targets are the original ids at the `True` positions of `target_mask`; the
  caller holds the original ids, so the Masker returns only the mask.
- A seeded `torch.Generator` yields a deterministic (validation) mask; `None`
  yields fresh per-step (training) masking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from ccf_tabformer_dl.tabformer_components.table_spec import TableSpec


@dataclass(frozen=True)
class MaskResult:
    masked_ids: Tensor  # long (B, S): input ids with spec.mask_id at masked positions
    target_mask: Tensor  # bool (B, S): True where the position must be reconstructed


class Masker(ABC):
    strategy: str

    def __init__(self, *, spec: TableSpec) -> None:
        self._spec = spec

    @abstractmethod
    def mask(self, ids: Tensor, *, generator: torch.Generator | None) -> MaskResult:
        """Build the masked-field-modeling inputs and targets.

        ids: long `(B, S)` -> `MaskResult`.
        generator: seeded for a deterministic (val) mask, `None` for fresh
        per-step (train) masking.

        Invariant: at least one position per row is `True` in `target_mask`, so
        the reconstruction loss is never empty.
        """
        raise NotImplementedError
