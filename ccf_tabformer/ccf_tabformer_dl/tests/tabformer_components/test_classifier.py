"""Mean-pool classifier head, ClassifierModel, backbone transfer from MFM ckpt."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ccf_tabformer_dl.tabformer_components.backbone import load_backbone_from_checkpoint
from ccf_tabformer_dl.tabformer_components.classifier_head.oracle import (
    build_classifier_head,
)
from ccf_tabformer_dl.tabformer_components.classifier_model import (
    build_classifier_model,
)
from ccf_tabformer_dl.tabformer_components.embedding.oracle import build_embedder
from ccf_tabformer_dl.tabformer_components.encoder.oracle import build_encoder
from ccf_tabformer_dl.tabformer_components.head.oracle import build_head
from ccf_tabformer_dl.tabformer_components.mfm_model import build_mfm_model
from tests.tabformer_components.fake_spec import tiny_ids, tiny_table_spec

D_MODEL = 8
HEAD_HIDDEN = 16
ENCODER_ARGS = {
    "d_model": D_MODEL,
    "n_layers": 1,
    "n_heads": 2,
    "ffn_mult": 2,
    "dropout": 0.0,
}
CLASSIFIER_HEAD_ARGS = {
    "d_model": D_MODEL,
    "hidden": HEAD_HIDDEN,
    "dropout": 0.0,
}


def _classifier():
    spec = tiny_table_spec()
    return build_classifier_model(
        embedder=build_embedder(
            strategy="additive", args={"d_model": D_MODEL}, spec=spec
        ),
        encoder=build_encoder(strategy="prenorm_transformer", args=ENCODER_ARGS),
        classifier_head=build_classifier_head(
            strategy="mean_pool", args=CLASSIFIER_HEAD_ARGS
        ),
    )


def _mfm():
    spec = tiny_table_spec()
    return build_mfm_model(
        embedder=build_embedder(
            strategy="additive", args={"d_model": D_MODEL}, spec=spec
        ),
        encoder=build_encoder(strategy="prenorm_transformer", args=ENCODER_ARGS),
        head=build_head(
            strategy="shared_support_masked", args={"d_model": D_MODEL}, spec=spec
        ),
    )


def test_mean_pool_forward_shape() -> None:
    head = build_classifier_head(strategy="mean_pool", args=CLASSIFIER_HEAD_ARGS)
    hidden = torch.randn(4, tiny_table_spec().seq_len, D_MODEL)
    logits = head(hidden)
    assert tuple(logits.shape) == (4,)
    assert bool(torch.isfinite(logits).all())


def test_classifier_model_forward_shape() -> None:
    net = _classifier()
    ids = tiny_ids()
    logits = net(ids)
    assert tuple(logits.shape) == (int(ids.shape[0]),)
    assert bool(torch.isfinite(logits).all())


def test_set_encoder_frozen_toggles_backbone_only() -> None:
    net = _classifier()
    net.set_encoder_frozen(frozen=True)
    for param in net.embedder.parameters():
        assert param.requires_grad is False
    for param in net.encoder.parameters():
        assert param.requires_grad is False
    for param in net.classifier_head.parameters():
        assert param.requires_grad is True
    net.set_encoder_frozen(frozen=False)
    for param in net.embedder.parameters():
        assert param.requires_grad is True


def _assert_backbone_close(dest, source) -> None:
    for name, tensor in source.embedder.state_dict().items():
        torch.testing.assert_close(dest.embedder.state_dict()[name], tensor)
    for name, tensor in source.encoder.state_dict().items():
        torch.testing.assert_close(dest.encoder.state_dict()[name], tensor)


def test_load_backbone_from_lightning_checkpoint(tmp_path: Path) -> None:
    source = _mfm()
    dest = _classifier()
    with torch.no_grad():
        dest.embedder.value_emb.weight.add_(1.0)
    shell = nn.Module()
    shell.add_module("net", source)
    path = tmp_path / "mfm.ckpt"
    torch.save({"state_dict": shell.state_dict()}, path)
    head_before = {
        name: tensor.detach().clone()
        for name, tensor in dest.classifier_head.state_dict().items()
    }
    load_backbone_from_checkpoint(
        dest, checkpoint_path=path, map_location=torch.device("cpu")
    )
    _assert_backbone_close(dest, source)
    for name, tensor in dest.classifier_head.state_dict().items():
        torch.testing.assert_close(tensor, head_before[name])


def test_load_backbone_from_unprefixed_state_dict(tmp_path: Path) -> None:
    source = _mfm()
    dest = _classifier()
    path = tmp_path / "mfm_raw.ckpt"
    torch.save({"state_dict": source.state_dict()}, path)
    load_backbone_from_checkpoint(
        dest, checkpoint_path=path, map_location=torch.device("cpu")
    )
    _assert_backbone_close(dest, source)


def test_load_backbone_rejects_classifier_checkpoint(tmp_path: Path) -> None:
    dest = _classifier()
    path = tmp_path / "classifier.ckpt"
    torch.save({"state_dict": dest.state_dict()}, path)
    with pytest.raises(ValueError, match="missing_keys"):
        load_backbone_from_checkpoint(
            dest, checkpoint_path=path, map_location=torch.device("cpu")
        )
