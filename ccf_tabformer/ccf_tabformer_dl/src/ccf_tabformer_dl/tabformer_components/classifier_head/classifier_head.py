"""The fraud classifier-head contract (fine-tuning).

Distinct from the reconstruction `head/` (which returns `(B, S, vocab)`): this
head *pools* the encoded sequence to one vector and projects it to a single fraud
logit `(B,)`. The pooling exists here because the `Encoder` is a pure
seq-to-seq contextualizer with no CLS token (see
documentation/decisions/encoder_and_heads.md). Pooling is the strategy axis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class ClassifierHead(nn.Module, ABC):
    strategy: str

    @property
    @abstractmethod
    def d_model(self) -> int:
        """Input width; must equal the Encoder's (and Embedder's) `d_model`."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, hidden: Tensor) -> Tensor:
        """hidden: float `(B, S, d_model)` -> fraud logit float `(B,)`.

        One logit per row, for `binary_cross_entropy_with_logits`; ROC-AUC reads
        the sigmoid-monotone score. (Fixed-width layout has no padding, so every
        position is real; a padded layout would pass a mask in via the spec.)
        """
        raise NotImplementedError
