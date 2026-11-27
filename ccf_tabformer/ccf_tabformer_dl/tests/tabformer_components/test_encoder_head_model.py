"""Encoder, support-masked head, composed MFM model, reconstruction steps."""

from __future__ import annotations

import torch

from ccf_tabformer_dl.tabformer_components.embedding.oracle import build_embedder
from ccf_tabformer_dl.tabformer_components.encoder.oracle import build_encoder
from ccf_tabformer_dl.tabformer_components.head.oracle import build_head
from ccf_tabformer_dl.tabformer_components.masking.oracle import build_masker
from ccf_tabformer_dl.tabformer_components.mfm_model import build_mfm_model
from ccf_tabformer_dl.tabformer_components.reconstruction.steps import (
    IDS_KEY,
    TARGET_MASK_KEY,
    recon_accuracy_metric,
    recon_bce_metric,
    reconstruction_fit_steps,
    reconstruction_train_val_fit_steps,
)
from tests.tabformer_components.fake_spec import tiny_ids, tiny_table_spec

D_MODEL = 8
ENCODER_ARGS = {
    "d_model": D_MODEL,
    "n_layers": 1,
    "n_heads": 2,
    "ffn_mult": 2,
    "dropout": 0.0,
}


def _model(spec):
    return build_mfm_model(
        embedder=build_embedder(
            strategy="additive", args={"d_model": D_MODEL}, spec=spec
        ),
        encoder=build_encoder(strategy="prenorm_transformer", args=ENCODER_ARGS),
        head=build_head(
            strategy="shared_support_masked", args={"d_model": D_MODEL}, spec=spec
        ),
    )


def test_encoder_preserves_shape() -> None:
    spec = tiny_table_spec()
    embedder = build_embedder(
        strategy="additive", args={"d_model": D_MODEL}, spec=spec
    )
    encoder = build_encoder(strategy="prenorm_transformer", args=ENCODER_ARGS)
    hidden = encoder(embedder(tiny_ids()))
    assert tuple(hidden.shape) == (tiny_ids().shape[0], spec.seq_len, D_MODEL)


def test_head_masks_disallowed_logits() -> None:
    spec = tiny_table_spec()
    head = build_head(
        strategy="shared_support_masked", args={"d_model": D_MODEL}, spec=spec
    )
    hidden = torch.zeros(2, spec.seq_len, D_MODEL)
    logits = head(hidden)
    assert tuple(logits.shape) == (2, spec.seq_len, spec.vocab_size)
    support = spec.support_additive()
    disallowed = support == float("-inf")
    assert bool(torch.isneginf(logits[0][disallowed]).all())
    assert bool(torch.isfinite(logits[0][~disallowed]).all())


def test_model_and_steps_produce_finite_loss() -> None:
    spec = tiny_table_spec()
    model = _model(spec)
    masker = build_masker(
        strategy="field_level", args={"feature_mask_prob": 0.4}, spec=spec
    )
    steps = reconstruction_fit_steps(
        masker=masker, generator=torch.Generator().manual_seed(0)
    )
    prepared = steps.prepare_tensors_fn({IDS_KEY: tiny_ids()})
    logits = steps.forward_fn(model, prepared)
    loss = steps.loss_fn(logits, prepared)
    assert torch.isfinite(loss)
    metric_inputs = steps.build_metric_inputs_fn(logits, prepared)
    values = recon_accuracy_metric().metric_function(metric_inputs)
    assert values.ndim == 1
    assert bool(((values == 0) | (values == 1)).all())
    bce = recon_bce_metric().metric_function(metric_inputs)
    assert bce.ndim == 1
    assert bce.shape[0] == values.shape[0]
    assert bool(torch.isfinite(bce).all())
    assert bool((bce >= 0).all())


def test_train_val_steps_seed_val_mask_under_inference_mode() -> None:
    spec = tiny_table_spec()
    masker = build_masker(
        strategy="field_level", args={"feature_mask_prob": 0.4}, spec=spec
    )
    steps = reconstruction_train_val_fit_steps(masker=masker, val_mask_seed=0)
    batch = {IDS_KEY: tiny_ids()}
    with torch.inference_mode():
        first = steps.prepare_tensors_fn(batch)
        second = steps.prepare_tensors_fn(batch)
    assert torch.equal(first[TARGET_MASK_KEY], second[TARGET_MASK_KEY])
    train_masks = [
        steps.prepare_tensors_fn(batch)[TARGET_MASK_KEY] for _ in range(8)
    ]
    differ = any(
        not torch.equal(train_masks[0], later) for later in train_masks[1:]
    )
    assert differ
