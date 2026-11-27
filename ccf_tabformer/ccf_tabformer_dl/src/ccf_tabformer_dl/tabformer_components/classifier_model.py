"""The fine-tuning model: Embedder -> Encoder -> ClassifierHead.

Reuses the *same* `Embedder` and `Encoder` contracts as `MfmModel`, and names its
submodules `embedder` / `encoder` **identically**, so a pretrained MFM checkpoint's
backbone weights transfer by key (see `backbone.load_backbone_from_checkpoint`).
The reconstruction head is dropped; a `ClassifierHead` replaces it. Only the
composition wiring is implemented here; the components' compute lives in stubs.
No masking is involved — at fine-tuning `ids` are the raw (unmasked) row ids.
"""

from __future__ import annotations

from torch import Tensor, nn

from ccf_tabformer_dl.tabformer_components.classifier_head.classifier_head import (
    ClassifierHead,
)
from ccf_tabformer_dl.tabformer_components.embedding.embedder import Embedder
from ccf_tabformer_dl.tabformer_components.encoder.encoder import Encoder


class ClassifierModel(nn.Module):
    def __init__(
        self,
        *,
        embedder: Embedder,
        encoder: Encoder,
        classifier_head: ClassifierHead,
    ) -> None:
        super().__init__()
        if not embedder.d_model == encoder.d_model == classifier_head.d_model:
            raise ValueError(
                "d_model must agree across components: "
                f"embedder={embedder.d_model}, encoder={encoder.d_model}, "
                f"classifier_head={classifier_head.d_model}"
            )
        self.embedder = embedder
        self.encoder = encoder
        self.classifier_head = classifier_head

    def forward(self, ids: Tensor) -> Tensor:
        """ids: long `(B, S)` (unmasked) -> fraud logit `(B,)`."""
        return self.classifier_head(self.encoder(self.embedder(ids)))

    def set_encoder_frozen(self, *, frozen: bool) -> None:
        """Freeze/unfreeze the pretrained backbone (embedder + encoder)."""
        for param in self.embedder.parameters():
            param.requires_grad = not frozen
        for param in self.encoder.parameters():
            param.requires_grad = not frozen


def build_classifier_model(
    *, embedder: Embedder, encoder: Encoder, classifier_head: ClassifierHead
) -> ClassifierModel:
    return ClassifierModel(
        embedder=embedder, encoder=encoder, classifier_head=classifier_head
    )
