"""FitSteps + masked-CE loss + reconstruction-accuracy metric for MFM.

The mask is sampled per step in ``prepare_tensors`` (on the batch's device), so
every step reconstructs a fresh set of positions. Loss and metric read the same
``tokens``/``mask`` pair, so the head's per-position support mask guarantees the
target id is always inside the allowed set.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dl_toolbox.dl_engine.metrics.contracts import KEY_OUTPUT, KEY_TARGET
from dl_toolbox.dl_engine.metrics.metric_object import MetricObject, concat_dim0
from dl_toolbox.dl_engine.metrics.reducers import max_items, mean_items, min_items, std_items
from dl_toolbox.dl_engine.runs.common.run_config import FitSteps

from ccf_tabformer_dl.data_contract.dataset import TOKENS_KEY
from ccf_tabformer_dl.tabformer_components.mfm.masking import TokenMaskSampler

MASK_KEY = "mask"
KEY_MASK = "mask"
RECON_ACC = "recon_acc"
EPOCH_REDUCERS = ("mean", "std", "min", "max")
_REDUCERS = {"mean": mean_items, "std": std_items, "min": min_items, "max": max_items}


def masked_feature_fit_steps(*, sampler: TokenMaskSampler) -> FitSteps:
    def prepare_tensors(batch: Mapping[str, Any]) -> dict[str, Tensor]:
        tokens = batch[TOKENS_KEY]
        if not isinstance(tokens, Tensor):
            raise TypeError(f"batch[{TOKENS_KEY!r}] must be a Tensor, got {type(tokens)!r}")
        mask = sampler.sample(tokens=tokens)
        return {TOKENS_KEY: tokens, MASK_KEY: mask}

    def forward_fn(net: nn.Module, batch: Mapping[str, Any]) -> Tensor:
        return net(tokens=batch[TOKENS_KEY], masked=batch[MASK_KEY])

    def loss_fn(output: Tensor, batch: Mapping[str, Any]) -> Tensor:
        mask = batch[MASK_KEY]
        selected = output[mask]
        target = batch[TOKENS_KEY][mask]
        return F.cross_entropy(selected, target)

    def build_metric_inputs(output: Tensor, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        return {KEY_OUTPUT: output, KEY_TARGET: batch[TOKENS_KEY], KEY_MASK: batch[MASK_KEY]}

    return FitSteps(
        prepare_tensors_fn=prepare_tensors,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        build_metric_inputs_fn=build_metric_inputs,
    )


def recon_accuracy_metric() -> MetricObject:
    def measure(tensors: Mapping[str, Tensor]) -> Tensor:
        logits = tensors[KEY_OUTPUT]
        target = tensors[KEY_TARGET]
        mask = tensors[KEY_MASK]
        pred = logits.argmax(dim=-1)
        correct = (pred == target)[mask]
        return correct.to(dtype=torch.float32)

    return MetricObject(
        name=RECON_ACC,
        required_keys=frozenset({KEY_OUTPUT, KEY_TARGET, KEY_MASK}),
        metric_function=measure,
        combiner=concat_dim0,
        reducers=_REDUCERS,
    )
