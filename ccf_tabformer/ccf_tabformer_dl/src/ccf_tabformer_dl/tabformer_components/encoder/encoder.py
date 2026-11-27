"""The Encoder contract. See documentation/decisions/encoder_and_heads.md.

A pure sequence-to-sequence contextualizer: `(B, S, d_model) -> (B, S, d_model)`.
It does **not** own a row-position embedding (B.7) — order comes only from the
feature/slot streams the `Embedder` already added — and it does **not** own the
CLS/pooling used by the (out-of-scope) fraud head. It needs `d_model` but not the
`TableSpec` (the structural streams live in the Embedder).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Encoder(nn.Module, ABC):
    strategy: str

    @property
    @abstractmethod
    def d_model(self) -> int:
        """The model width; must equal the Embedder's `d_model`."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, hidden: Tensor) -> Tensor:
        """hidden: float `(B, S, d_model)` -> float `(B, S, d_model)`.

        Invariants: shape preserved; attention is over the full set of positions
        (no causal or padding mask — every inner-table position is real).
        """
        raise NotImplementedError
