"""`prenorm_transformer` encoder: a pre-norm bidirectional Transformer stack."""

from __future__ import annotations

from torch import Tensor, nn

from ccf_tabformer_dl.tabformer_components.encoder.encoder import Encoder

PRENORM_TRANSFORMER_STRATEGY = "prenorm_transformer"
_ARG_KEYS = ("d_model", "n_layers", "n_heads", "ffn_mult", "dropout")


class PreNormTransformerEncoder(Encoder):
    strategy = PRENORM_TRANSFORMER_STRATEGY

    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()

        for name, value in (
            ("d_model", d_model),
            ("n_layers", n_layers),
            ("n_heads", n_heads),
            ("ffn_mult", ffn_mult),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")

        if not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        self._d_model = d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

    @property
    def d_model(self) -> int:
        return self._d_model

    def forward(self, hidden: Tensor) -> Tensor:
        return self.encoder(hidden)


def build_prenorm_transformer_args(args: dict) -> dict:
    if set(args) != set(_ARG_KEYS):
        raise ValueError(
            f"prenorm_transformer args must have exactly {_ARG_KEYS}, "
            f"got {sorted(args)}"
        )
    return {
        "d_model": args["d_model"],
        "n_layers": args["n_layers"],
        "n_heads": args["n_heads"],
        "ffn_mult": args["ffn_mult"],
        "dropout": args["dropout"],
    }
