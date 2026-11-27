"""Classifier-head component: pool (B, S, d) -> one fraud logit (B,), oracle-built."""

from ccf_tabformer_dl.tabformer_components.classifier_head.classifier_head import (
    ClassifierHead,
)
from ccf_tabformer_dl.tabformer_components.classifier_head.oracle import (
    build_classifier_head,
    oracle,
)

__all__ = ["ClassifierHead", "build_classifier_head", "oracle"]
