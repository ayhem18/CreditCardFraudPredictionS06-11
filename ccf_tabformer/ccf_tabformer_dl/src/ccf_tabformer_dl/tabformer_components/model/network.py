"""Digit-token transformer: tokenizer embeddings + encoder + MFM head.

Each token embedding is ``value + column + slot`` (a masked token swaps its
value embedding for a per-feature learned ``[MASK]`` embedding, keeping the
column/slot parts). A learned ``[CLS]`` token is prepended; the encoder is
bidirectional (no attention mask). The reconstruction head projects every
feature token to the global value vocabulary and adds the per-position support
mask so disallowed ids never carry probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ccf_tabformer_dl.tabformer_components.model.vocab import GlobalVocab


@dataclass(frozen=True)
class TabFormerConfig:
    d_model: int
    n_layers: int
    n_heads: int
    ffn_mult: int
    dropout: float


class TabFormerEncoder(nn.Module):
    """Tokenizer embeddings + bidirectional Transformer encoder with a CLS token."""

    def __init__(self, *, vocab: GlobalVocab, config: TabFormerConfig) -> None:
        super().__init__()
        self._config = config
        d = config.d_model
        self.value_emb = nn.Embedding(vocab.global_vocab_size, d)
        self.column_emb = nn.Embedding(vocab.n_features, d)
        self.slot_emb = nn.Embedding(vocab.max_slots, d)
        self.mask_emb = nn.Embedding(vocab.n_features, d)
        self.cls = nn.Parameter(torch.zeros(d))

        nn.init.normal_(self.cls, std=0.02)

        self.register_buffer(
            "feature_index", torch.as_tensor(vocab.feature_index, dtype=torch.long)
        )

        self.register_buffer(
            "slot_index", torch.as_tensor(vocab.slot_index, dtype=torch.long)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=d * config.ffn_mult,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, norm=nn.LayerNorm(d)
        )

    def forward(self, *, 
                tokens: Tensor, 
                masked: Tensor | None) -> tuple[Tensor, Tensor]:
        """tokens (N,T) global ids; masked (N,T) bool or None.

        Returns (cls_hidden (N,d), token_hidden (N,T,d)).
        """
        # Each row in the original table is now a sequence of fixed length. 
        # Each value is mapped to an id (the documentation says global ids, which needs to be verified further)

        if tokens.ndim != 2:
            raise ValueError(f"tokens must be (N, T), got {tuple(tokens.shape)}")

        n_rows = tokens.shape[0]

        # Pytorch repositories have this bad habit of copying a self.{var_name} to a local variable instead of using self.{var_name} directly.
        # am I missing something here ? Just use the damn attribute directly

        # feature_index = self.feature_index
        # value = self.value_emb(tokens)

        if masked is not None:
            replacement = self.mask_emb(self.feature_index).unsqueeze(0).expand_as(value)
            value = torch.where(masked.unsqueeze(-1), replacement, value)
        else:
            # the if-else block is just easier to reason about
            value = self.value_emb(tokens)

        emb = (
            value
            + self.column_emb(self.feature_index).unsqueeze(0)
            + self.slot_emb(self.slot_index).unsqueeze(0)
        )

        cls = self.cls.view(1, 1, -1).expand(n_rows, 1, -1)

        sequence = torch.cat([cls, emb], dim=1)

        hidden = self.encoder(sequence)

        return hidden[:, 0], hidden[:, 1:]


class MaskedFeatureModel(nn.Module):
    """Encoder + per-position-masked reconstruction head for MFM pretraining."""

    def __init__(self, *, vocab: GlobalVocab, config: TabFormerConfig) -> None:
        super().__init__()
        self.encoder = TabFormerEncoder(vocab=vocab, config=config)
        self.head = nn.Linear(config.d_model, vocab.global_vocab_size)
        self.register_buffer(
            "support_additive",
            torch.as_tensor(vocab.support_additive(), dtype=torch.float32),
        )

    def forward(self, *, tokens: Tensor, masked: Tensor) -> Tensor:
        """tokens (N,T) global ids, masked (N,T) bool -> logits (N,T,V_global)."""
        _, token_hidden = self.encoder(tokens=tokens, masked=masked)
        logits = self.head(token_hidden)
        return logits + self.support_additive.unsqueeze(0)
