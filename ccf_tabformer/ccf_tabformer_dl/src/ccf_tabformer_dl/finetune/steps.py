"""Fine-tune FitSteps (BCE) + a ROC-AUC metric.

ROC-AUC is a global ranking statistic, so the metric accumulates per-item
``[score, label]`` pairs across the epoch (via ``concat_dim0``) and the ``auc``
reducer computes the AUC over all of them. Fine-tuning runs single-GPU, so the
accumulated pairs are the whole val fold.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F

from dl_toolbox.dl_engine.metrics.contracts import KEY_OUTPUT, KEY_TARGET
from dl_toolbox.dl_engine.metrics.metric_object import MetricObject, concat_dim0
from dl_toolbox.dl_engine.runs.common.run_config import FitSteps

from ccf_tabformer_dl.data.dataset import LABEL_KEY, TOKENS_KEY

AUC = "auc"
AUC_REDUCER = "auc"


def classification_fit_steps() -> FitSteps:
    def prepare_tensors(batch: Mapping[str, Any]) -> dict[str, Tensor]:
        tokens = batch[TOKENS_KEY]
        label = batch[LABEL_KEY]
        if not isinstance(tokens, Tensor) or not isinstance(label, Tensor):
            raise TypeError("batch tokens/label must be Tensors")
        return {TOKENS_KEY: tokens, LABEL_KEY: label}

    def forward_fn(net: nn.Module, batch: Mapping[str, Any]) -> Tensor:
        return net(tokens=batch[TOKENS_KEY])

    def loss_fn(output: Tensor, batch: Mapping[str, Any]) -> Tensor:
        return F.binary_cross_entropy_with_logits(output, batch[LABEL_KEY].to(output.dtype))

    def build_metric_inputs(output: Tensor, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        return {KEY_OUTPUT: output, KEY_TARGET: batch[LABEL_KEY]}

    return FitSteps(
        prepare_tensors_fn=prepare_tensors,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        build_metric_inputs_fn=build_metric_inputs,
    )


def _auc_reduce(pairs: Tensor) -> Tensor:
    data = pairs.detach().float().cpu().numpy()
    scores = data[:, 0]
    labels = data[:, 1]
    if labels.min() == labels.max():
        return pairs.new_tensor(float("nan"))
    return pairs.new_tensor(float(roc_auc_score(labels, scores)))


def auc_metric() -> MetricObject:
    def measure(tensors: Mapping[str, Tensor]) -> Tensor:
        score = tensors[KEY_OUTPUT].reshape(-1).to(torch.float32)
        label = tensors[KEY_TARGET].reshape(-1).to(torch.float32)
        return torch.stack([score, label], dim=-1)

    return MetricObject(
        name=AUC,
        required_keys=frozenset({KEY_OUTPUT, KEY_TARGET}),
        metric_function=measure,
        combiner=concat_dim0,
        reducers={AUC_REDUCER: _auc_reduce},
    )
