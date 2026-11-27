"""The masked-field-modeling model: Embedder -> Encoder -> Head.

The composition is the only thing implemented here (the three components' compute
lives in their own stubs). The model is *given* a built `Embedder`, `Encoder` and
`Head`; it never builds tables or applies masking. Masking is applied by the
training step (via a `Masker`) *before* `forward` — the input `ids` already carry
the `[MASK]` id at masked positions. The step reads the logits at the
`MaskResult.target_mask` positions against the original ids.
"""

from __future__ import annotations

from torch import Tensor, nn

from ccf_tabformer_dl.tabformer_components.embedding.embedder import Embedder
from ccf_tabformer_dl.tabformer_components.encoder.encoder import Encoder
from ccf_tabformer_dl.tabformer_components.head.head import Head


class MfmModel(nn.Module):
    def __init__(self, *, embedder: Embedder, encoder: Encoder, head: Head) -> None:
        super().__init__()
        if not embedder.d_model == encoder.d_model == head.d_model:
            raise ValueError(
                "d_model must agree across components: "
                f"embedder={embedder.d_model}, encoder={encoder.d_model}, "
                f"head={head.d_model}"
            )
        self.embedder = embedder
        self.encoder = encoder
        self.head = head

    def forward(self, ids: Tensor) -> Tensor:
        """ids: long `(B, S)` (already masked) -> logits `(B, S, vocab_size)`."""
        return self.head(self.encoder(self.embedder(ids)))


def build_mfm_model(
    *, embedder: Embedder, encoder: Encoder, head: Head
) -> MfmModel:
    return MfmModel(embedder=embedder, encoder=encoder, head=head)
