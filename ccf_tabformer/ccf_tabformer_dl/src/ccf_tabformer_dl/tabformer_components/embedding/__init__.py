"""Embedding component: ids → vectors, built by an oracle from a TableSpec."""

from ccf_tabformer_dl.tabformer_components.embedding.embedder import Embedder
from ccf_tabformer_dl.tabformer_components.embedding.oracle import build_embedder, oracle

__all__ = ["Embedder", "build_embedder", "oracle"]
