"""`shared_support_masked` head: one linear projection + per-position support mask.

The baseline reconstruction head: a single `Linear(d_model, vocab_size)` whose
logits get `spec.support_additive` added, so each position can only predict a
value the generator writes there. A per-field-type head is a separate strategy.
"""

from __future__ import annotations

from torch import Tensor, nn

from ccf_tabformer_dl.tabformer_components.head.head import Head
from ccf_tabformer_dl.tabformer_components.table_spec import TableSpec

SHARED_SUPPORT_MASKED_STRATEGY = "shared_support_masked"
_ARG_KEYS = ("d_model",)


class SupportMaskedHead(Head):
    strategy = SHARED_SUPPORT_MASKED_STRATEGY

    def __init__(self, *, spec: TableSpec, d_model: int) -> None:
        super().__init__(spec=spec)
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model must be a positive int, got {d_model!r}")
        self._d_model = d_model
        self.proj = nn.Linear(d_model, spec.vocab_size)
        # 0 where allowed, -inf elsewhere; added to logits at forward time
        self.register_buffer("support_additive", spec.support_additive().float())

    @property
    def d_model(self) -> int:
        return self._d_model

    def forward(self, hidden: Tensor) -> Tensor:
        return self.proj(hidden) + self.support_additive.unsqueeze(0)


def build_shared_support_masked_args(args: dict) -> dict:
    if set(args) != set(_ARG_KEYS):
        raise ValueError(
            f"shared_support_masked args must have exactly {_ARG_KEYS}, "
            f"got {sorted(args)}"
        )
    return {"d_model": args["d_model"]}
