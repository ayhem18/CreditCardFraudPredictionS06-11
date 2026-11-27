"""FitSteps + masked-CE loss + recon_acc and bce metrics for field-level MFM.

Masking happens in ``prepare_tensors`` via a ``Masker``. The model sees already
masked ids; loss and metric read only the ``target_mask`` positions against the
original ids.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dl_toolbox.dl_engine.metrics.contracts import KEY_OUTPUT, KEY_TARGET
from dl_toolbox.dl_engine.metrics.metric_object import MetricObject, concat_dim0
from dl_toolbox.dl_engine.metrics.reducers import max_items, mean_items, min_items, std_items
from dl_toolbox.dl_engine.runs.common.run_config import FitSteps

from ccf_tabformer_dl.tabformer_components.masking.masker import Masker

IDS_KEY = "ids"
MASKED_IDS_KEY = "masked_ids"
TARGET_MASK_KEY = "target_mask"
KEY_MASK = "mask"
RECON_ACC = "recon_acc"
BCE = "bce"
_REDUCERS = {"mean": mean_items, "std": std_items, "min": min_items, "max": max_items}


def reconstruction_fit_steps(
    *, masker: Masker, generator: torch.Generator | None
) -> FitSteps:
    """Train (``generator=None``) or seeded-val (``generator`` set) reconstruction."""

    def prepare_tensors(batch: Mapping[str, Any]) -> dict[str, Tensor]:
        ids = batch[IDS_KEY]
        if not isinstance(ids, Tensor):
            raise TypeError(f"batch[{IDS_KEY!r}] must be a Tensor, got {type(ids)!r}")
        result = masker.mask(ids, generator=generator)
        return {
            IDS_KEY: ids,
            MASKED_IDS_KEY: result.masked_ids,
            TARGET_MASK_KEY: result.target_mask,
        }

    def forward_fn(net: nn.Module, batch: Mapping[str, Any]) -> Tensor:
        return net(batch[MASKED_IDS_KEY])

    def loss_fn(output: Tensor, batch: Mapping[str, Any]) -> Tensor:
        mask = batch[TARGET_MASK_KEY]
        return F.cross_entropy(output[mask], batch[IDS_KEY][mask])

    def build_metric_inputs(
        output: Tensor, batch: Mapping[str, Any]
    ) -> dict[str, Tensor]:
        return {
            KEY_OUTPUT: output,
            KEY_TARGET: batch[IDS_KEY],
            KEY_MASK: batch[TARGET_MASK_KEY],
        }

    return FitSteps(
        prepare_tensors_fn=prepare_tensors,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        build_metric_inputs_fn=build_metric_inputs,
    )


def reconstruction_train_val_fit_steps(
    *, masker: Masker, val_mask_seed: int
) -> FitSteps:
    """Train draws a fresh mask. Val (Lightning inference mode) is seeded.

    The val generator is re-seeded from ``val_mask_seed`` and the batch token
    ids so the same val rows keep the same mask across epochs.
    """

    if isinstance(val_mask_seed, bool) or not isinstance(val_mask_seed, int):
        raise TypeError(
            f"val_mask_seed must be int, got {type(val_mask_seed)!r}"
        )
    if val_mask_seed < 0:
        raise ValueError(f"val_mask_seed must be >= 0, got {val_mask_seed}")

    def prepare_tensors(batch: Mapping[str, Any]) -> dict[str, Tensor]:
        ids = batch[IDS_KEY]
        if not isinstance(ids, Tensor):
            raise TypeError(f"batch[{IDS_KEY!r}] must be a Tensor, got {type(ids)!r}")
        if torch.is_inference_mode_enabled():
            generator = torch.Generator()
            generator.manual_seed(
                _mask_seed_for_ids(ids, base_seed=val_mask_seed)
            )
        else:
            generator = None
        result = masker.mask(ids, generator=generator)
        return {
            IDS_KEY: ids,
            MASKED_IDS_KEY: result.masked_ids,
            TARGET_MASK_KEY: result.target_mask,
        }

    def forward_fn(net: nn.Module, batch: Mapping[str, Any]) -> Tensor:
        return net(batch[MASKED_IDS_KEY])

    def loss_fn(output: Tensor, batch: Mapping[str, Any]) -> Tensor:
        mask = batch[TARGET_MASK_KEY]
        return F.cross_entropy(output[mask], batch[IDS_KEY][mask])

    def build_metric_inputs(
        output: Tensor, batch: Mapping[str, Any]
    ) -> dict[str, Tensor]:
        return {
            KEY_OUTPUT: output,
            KEY_TARGET: batch[IDS_KEY],
            KEY_MASK: batch[TARGET_MASK_KEY],
        }

    return FitSteps(
        prepare_tensors_fn=prepare_tensors,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        build_metric_inputs_fn=build_metric_inputs,
    )


def _mask_seed_for_ids(ids: Tensor, *, base_seed: int) -> int:
    payload = ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    digest = hashlib.blake2b(payload.numpy().tobytes(), digest_size=8).digest()
    mixed = int.from_bytes(digest, "little") ^ int(base_seed)
    return mixed & 0xFFFFFFFFFFFFFFFF


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


def recon_bce_metric() -> MetricObject:
    def measure(tensors: Mapping[str, Tensor]) -> Tensor:
        logits = tensors[KEY_OUTPUT]
        target = tensors[KEY_TARGET]
        mask = tensors[KEY_MASK]
        masked_logits = logits[mask]
        masked_target = target[mask]
        n_classes = masked_logits.shape[-1]
        one_hot = F.one_hot(masked_target, num_classes=n_classes).to(
            dtype=masked_logits.dtype
        )
        finite = torch.isfinite(masked_logits)
        safe_logits = masked_logits.masked_fill(~finite, 0.0)
        per_class = F.binary_cross_entropy_with_logits(
            safe_logits, one_hot, reduction="none"
        )
        per_class = per_class.masked_fill(~finite, 0.0)
        n_finite = finite.to(dtype=per_class.dtype).sum(dim=-1).clamp_min(1.0)
        return per_class.sum(dim=-1) / n_finite

    return MetricObject(
        name=BCE,
        required_keys=frozenset({KEY_OUTPUT, KEY_TARGET, KEY_MASK}),
        metric_function=measure,
        combiner=concat_dim0,
        reducers=_REDUCERS,
    )
