"""Encoder component: contextualize (B, S, d) sequences, oracle-built."""

from ccf_tabformer_dl.tabformer_components.encoder.encoder import Encoder
from ccf_tabformer_dl.tabformer_components.encoder.oracle import build_encoder, oracle

__all__ = ["Encoder", "build_encoder", "oracle"]
