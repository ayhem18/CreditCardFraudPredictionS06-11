"""Classification FitSteps and ROC-AUC reducer."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from dl_toolbox.dl_engine.metrics.contracts import KEY_OUTPUT, KEY_TARGET

from ccf_tabformer_dl.data_contract.inner_table_dataset import LABEL_KEY
from ccf_tabformer_dl.tabformer_components.reconstruction.steps import IDS_KEY
from ccf_tabformer_dl.task_engines.finetune.steps import (
    AUC_REDUCER,
    auc_metric,
    classification_bce_metric,
    classification_fit_steps,
)


def test_classification_loss_is_bce_with_logits() -> None:
    steps = classification_fit_steps()
    ids = torch.ones(4, 3, dtype=torch.long)
    label = torch.tensor([0.0, 1.0, 1.0, 0.0])
    batch = steps.prepare_tensors_fn({IDS_KEY: ids, LABEL_KEY: label})
    logits = torch.tensor([0.2, -0.4, 1.1, -1.3])

    class _Net:
        def __call__(self, tokens):
            del tokens
            return logits

    output = steps.forward_fn(_Net(), batch)
    loss = steps.loss_fn(output, batch)
    expected = F.binary_cross_entropy_with_logits(logits, label)
    torch.testing.assert_close(loss, expected)


def test_auc_reducer_ranks_scores() -> None:
    metric = auc_metric()
    pairs = metric.metric_function(
        {
            KEY_OUTPUT: torch.tensor([0.1, 0.9, 0.2, 0.8]),
            KEY_TARGET: torch.tensor([0.0, 1.0, 0.0, 1.0]),
        }
    )
    value = metric.reducers[AUC_REDUCER](pairs)
    assert float(value.item()) == 1.0


def test_classification_bce_is_per_item() -> None:
    metric = classification_bce_metric()
    logits = torch.tensor([0.2, -0.4, 1.1, -1.3])
    label = torch.tensor([0.0, 1.0, 1.0, 0.0])
    items = metric.metric_function({KEY_OUTPUT: logits, KEY_TARGET: label})
    expected = F.binary_cross_entropy_with_logits(
        logits, label, reduction="none"
    )
    torch.testing.assert_close(items, expected)
    torch.testing.assert_close(metric.reducers["mean"](items), expected.mean())


def test_auc_reducer_nan_when_one_class() -> None:
    metric = auc_metric()
    pairs = metric.metric_function(
        {
            KEY_OUTPUT: torch.tensor([0.1, 0.9]),
            KEY_TARGET: torch.tensor([1.0, 1.0]),
        }
    )
    value = metric.reducers[AUC_REDUCER](pairs)
    assert math.isnan(float(value.item()))
