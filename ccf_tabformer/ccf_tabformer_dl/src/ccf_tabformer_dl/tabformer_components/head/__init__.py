"""Reconstruction-head component: (B, S, d) -> (B, S, vocab) logits, oracle-built."""

from ccf_tabformer_dl.tabformer_components.head.head import Head
from ccf_tabformer_dl.tabformer_components.head.oracle import build_head, oracle

__all__ = ["Head", "build_head", "oracle"]
