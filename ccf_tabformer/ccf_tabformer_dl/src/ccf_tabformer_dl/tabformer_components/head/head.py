"""The reconstruction Head contract. See documentation/decisions/encoder_and_heads.md.

Projects each encoded position to logits over the value vocabulary, with
per-position support masking (disallowed ids → -inf, from `spec.support_additive`).
Returning logits over the whole global vocab `(B, S, vocab_size)` lets *both* the
shared-support-masked design and a future per-field-type design satisfy one
interface (the latter scatters its per-type logits into the global layout). The
Head is decoupled from masking: it scores every position; the training step
selects the `target_mask` positions and the original ids as targets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from ccf_tabformer_dl.tabformer_components.table_spec import TableSpec


class Head(nn.Module, ABC):
    strategy: str

    def __init__(self, *, spec: TableSpec) -> None:
        super().__init__()
        self._spec = spec

    @property
    @abstractmethod
    def d_model(self) -> int:
        """The input width; must equal the Encoder's `d_model`."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, hidden: Tensor) -> Tensor:
        """hidden: float `(B, S, d_model)` -> logits float `(B, S, vocab_size)`.

        Invariant: disallowed value ids at each position carry -inf (via
        `spec.support_additive`), so they never receive probability.
        """
        raise NotImplementedError
