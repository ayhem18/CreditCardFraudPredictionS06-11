"""Masking component: build MFM inputs/targets, oracle-built from a TableSpec."""

from ccf_tabformer_dl.tabformer_components.masking.masker import Masker, MaskResult
from ccf_tabformer_dl.tabformer_components.masking.oracle import build_masker, oracle

__all__ = ["Masker", "MaskResult", "build_masker", "oracle"]
