"""`additive` embedder: value + feature + slot (the B.5 baseline).

Equivalent to a Linear over the concatenation of the three learnable tables, so
it is the honest starting point, not a limitation to engineer around. `film` /
`mlp` compositions are separate strategies registered in the oracle.
"""

from __future__ import annotations

from torch import Tensor, nn

from ccf_tabformer_dl.tabformer_components.embedding.embedder import Embedder
from ccf_tabformer_dl.tabformer_components.table_spec import TableSpec

ADDITIVE_STRATEGY = "additive"
_ARG_KEYS = ("d_model",)


class AdditiveEmbedder(Embedder):
    strategy = ADDITIVE_STRATEGY

    def __init__(self, *, spec: TableSpec, d_model: int) -> None:
        super().__init__(spec=spec)
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model must be a positive int, got {d_model!r}")
        self._d_model = d_model
        self.value_emb = nn.Embedding(spec.vocab_size, d_model)
        self.feature_emb = nn.Embedding(spec.n_features, d_model)
        self.slot_emb = nn.Embedding(spec.max_slots, d_model)
        # row-independent structural streams, held as buffers (not inputs to forward)
        self.register_buffer("feature_index", spec.feature_index.long())
        self.register_buffer("slot_index", spec.slot_index.long())

    @property
    def d_model(self) -> int:
        return self._d_model

    def forward(self, ids: Tensor) -> Tensor:
        seq_len = int(self.feature_index.shape[0])
        if ids.ndim != 2 or ids.shape[-1] != seq_len:
            raise ValueError(
                f"ids must be long (B, {seq_len}), got shape {tuple(ids.shape)}"
            )
        value = self.value_emb(ids)
        feature = self.feature_emb(self.feature_index)
        slot = self.slot_emb(self.slot_index)
        return value + feature.unsqueeze(0) + slot.unsqueeze(0)


def build_additive_args(args: dict) -> dict:
    if set(args) != set(_ARG_KEYS):
        raise ValueError(
            f"additive args must have exactly {_ARG_KEYS}, got {sorted(args)}"
        )
    return {"d_model": args["d_model"]}
