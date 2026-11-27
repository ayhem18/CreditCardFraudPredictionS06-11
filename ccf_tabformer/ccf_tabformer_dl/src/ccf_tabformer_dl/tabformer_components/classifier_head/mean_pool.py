"""`mean_pool` classifier head: mean over positions -> MLP -> one logit.

The baseline pooling (the lean in encoder_and_heads.md): average the encoder's
per-position hidden states over `S`, then a small MLP to a scalar. `cls_token`
and `attention_pool` are separate strategies.
"""

from __future__ import annotations

from torch import Tensor, nn

from ccf_tabformer_dl.tabformer_components.classifier_head.classifier_head import (
    ClassifierHead,
)

MEAN_POOL_STRATEGY = "mean_pool"
_ARG_KEYS = ("d_model", "hidden", "dropout")


class MeanPoolClassifierHead(ClassifierHead):
    strategy = MEAN_POOL_STRATEGY

    def __init__(self, *, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model must be a positive int, got {d_model!r}")
        if not isinstance(hidden, int) or hidden <= 0:
            raise ValueError(f"hidden must be a positive int, got {hidden!r}")
        if not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        self._d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 1),
        )

    @property
    def d_model(self) -> int:
        return self._d_model

    def forward(self, hidden: Tensor) -> Tensor:
        raise NotImplementedError(
            "contract stub: hidden.mean(dim=1) -> self.mlp(...) -> squeeze(-1)"
        )


def build_mean_pool_args(args: dict) -> dict:
    if set(args) != set(_ARG_KEYS):
        raise ValueError(
            f"mean_pool args must have exactly {_ARG_KEYS}, got {sorted(args)}"
        )
    return {"d_model": args["d_model"], "hidden": args["hidden"], "dropout": args["dropout"]}
