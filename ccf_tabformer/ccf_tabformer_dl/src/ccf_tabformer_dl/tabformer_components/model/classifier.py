"""Fine-tune classifier: pretrained encoder + CLS -> MLP head -> fraud logit.

The encoder is the same :class:`TabFormerEncoder` used for MFM pretraining, so
its weights load directly from an MFM checkpoint. Three regimes are expressed
through ``init`` (load pretrained weights or not) and ``freeze_encoder``
(train the encoder or only the head).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from ccf_tabformer_dl.tabformer_components.model.network import TabFormerConfig, TabFormerEncoder
from ccf_tabformer_dl.tabformer_components.model.vocab import GlobalVocab

ENCODER_PREFIX = "net.encoder."


@dataclass(frozen=True)
class HeadConfig:
    hidden: int
    dropout: float


class FineTuneModel(nn.Module):
    """Encoder + MLP classification head on the CLS representation."""

    def __init__(
        self, *, vocab: GlobalVocab, encoder_config: TabFormerConfig, head_config: HeadConfig
    ) -> None:
        super().__init__()
        self.encoder = TabFormerEncoder(vocab=vocab, config=encoder_config)
        d = encoder_config.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, head_config.hidden),
            nn.GELU(),
            nn.Dropout(head_config.dropout),
            nn.Linear(head_config.hidden, 1),
        )

    def forward(self, *, tokens: Tensor) -> Tensor:
        cls_hidden, _ = self.encoder(tokens=tokens, masked=None)
        return self.head(cls_hidden).squeeze(-1)

    def set_encoder_frozen(self, *, frozen: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = not frozen


def load_pretrained_encoder(
    model: FineTuneModel, *, checkpoint: Path, map_location: str
) -> None:
    """Load the MFM encoder weights (``net.encoder.*``) into ``model.encoder``."""
    ckpt = torch.load(checkpoint, map_location=map_location, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    encoder_state = {
        key[len(ENCODER_PREFIX) :]: value
        for key, value in state.items()
        if key.startswith(ENCODER_PREFIX)
    }
    if not encoder_state:
        raise ValueError(f"no '{ENCODER_PREFIX}*' weights found in {checkpoint}")
    model.encoder.load_state_dict(encoder_state, strict=True)


NET_PREFIX = "net."


def load_finetune_checkpoint(
    model: FineTuneModel, *, checkpoint: Path, map_location: str
) -> None:
    """Load a full fine-tune checkpoint (``net.*``) into ``model`` (encoder+head)."""
    ckpt = torch.load(checkpoint, map_location=map_location, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    stripped = {
        key[len(NET_PREFIX) :]: value
        for key, value in state.items()
        if key.startswith(NET_PREFIX)
    }
    if not stripped:
        raise ValueError(f"no '{NET_PREFIX}*' weights found in {checkpoint}")
    model.load_state_dict(stripped, strict=True)
