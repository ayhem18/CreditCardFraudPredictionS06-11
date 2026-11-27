"""The Embedder contract. See documentation/decisions/embedding.md.

Design decisions this interface encodes:
- The Embedder is one object passed to the network, not a bag of `nn.Embedding`s
  inside the encoder.
- The structural streams are row-independent, so the Embedder holds them (from
  the spec) and `forward` takes only the `(B, S)` ids.
- The Embedder knows nothing about masking: a masked position is simply the
  reserved `[MASK]` id in the input, embedded like any other id. (The per-feature
  `mask_emb` of the old `model/network.py` is removed by this decoupling.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from ccf_tabformer_dl.tabformer_components.table_spec import TableSpec


class Embedder(nn.Module, ABC):
    strategy: str

    def __init__(self, *, spec: TableSpec) -> None:
        super().__init__()
        self._spec = spec

    @property
    @abstractmethod
    def d_model(self) -> int:
        """The embedding / model width D of the produced vectors."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, ids: Tensor) -> Tensor:
        """Map token ids to vectors.

        ids: long `(B, S)` with every value in `[0, spec.vocab_size)`.
        returns: float `(B, S, d_model)`.

        Invariants: batch and S are preserved; the result is a pure function of
        `ids` (no masking decided here); positions carrying `spec.mask_id` are
        embedded like any id, their feature/slot context preserved.
        """
        raise NotImplementedError
