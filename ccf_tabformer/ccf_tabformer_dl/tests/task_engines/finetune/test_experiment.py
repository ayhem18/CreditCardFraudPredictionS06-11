"""Fine-tune experiment assembles TrainValFitParts from oracles."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.exp_engine.train_val_engine.experiment import TrainValFitParts
from dl_toolbox.dl_engine.runs.common.run_config import OptimizationConfig

from ccf_tabformer_dl.project import resolve_finetune_recipe
from ccf_tabformer_dl.tabformer_components.classifier_model import ClassifierModel
from ccf_tabformer_dl.task_engines.finetune.experiment import FineTuneExperiment
from tests.tabformer_components.fake_spec import tiny_ids, tiny_table_spec

_EXPERIMENT = "ccf_tabformer_dl.task_engines.finetune.experiment"

_BASE = {
    "data": {
        "inner_table": "fixed_width_v1",
        "folds": "alarm_profile_5fold_v1",
        "fold": 0,
        "batch_size": 2,
        "num_workers": 0,
    },
    "embedding": {"strategy": "additive", "args": {"d_model": 8}},
    "encoder": {
        "strategy": "prenorm_transformer",
        "args": {
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 2,
            "ffn_mult": 2,
            "dropout": 0.0,
        },
    },
    "classifier_head": {
        "strategy": "mean_pool",
        "args": {"d_model": 8, "hidden": 16, "dropout": 0.0},
    },
    "lr_config": {"learning_rate": 0.0001},
    "optimizer_config": {
        "name": "adamw",
        "weight_decay": 0.0,
        "betas": [0.9, 0.95],
    },
    "lr_scheduler_config": {"name": "constant"},
}


def _config(*, regime: str, checkpoint: str) -> dict:
    payload = dict(_BASE)
    payload["regime"] = regime
    payload["pretrained_checkpoint"] = checkpoint
    return payload


def _patch_loaders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    ids = tiny_ids()
    train_loader = DataLoader(TensorDataset(ids), batch_size=2)
    val_loader = DataLoader(TensorDataset(ids), batch_size=2)

    def _fake_loaders(config):
        del config
        return train_loader, val_loader

    monkeypatch.setattr(
        f"{_EXPERIMENT}.table_spec_from_inner_table",
        lambda inner_table: tiny_table_spec(),
    )
    monkeypatch.setattr(
        f"{_EXPERIMENT}.build_train_val_loaders", _fake_loaders
    )
    return train_loader, val_loader, tmp_path


def test_build_fit_random_monitors_val_auc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_loader, val_loader, root = _patch_loaders(monkeypatch, tmp_path)
    called = {"n": 0}

    def _fake_load(model, *, checkpoint_path, map_location):
        del model, checkpoint_path, map_location
        called["n"] += 1

    monkeypatch.setattr(
        f"{_EXPERIMENT}.load_backbone_from_checkpoint", _fake_load
    )
    built = FineTuneExperiment().build_fit(
        _config(regime="random", checkpoint=""), project_root=root
    )
    assert isinstance(built, TrainValFitParts)
    assert isinstance(built.net, ClassifierModel)
    assert built.loaders.train is train_loader
    assert built.loaders.val is val_loader
    assert built.optimization.scheduler is OptimizationConfig.ABSENT
    assert isinstance(built.optimization.optimizer, AdamW)
    assert built.metrics.monitor.key == "auc/val/epoch/auc"
    assert built.metrics.monitor.mode == "max"
    assert called["n"] == 0
    for param in built.net.embedder.parameters():
        assert param.requires_grad is True


def test_build_fit_frozen_loads_and_freezes_backbone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_loaders(monkeypatch, tmp_path)
    checkpoint = tmp_path / "backbone.ckpt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    called = {"path": None}

    def _fake_load(model, *, checkpoint_path, map_location):
        del model, map_location
        called["path"] = checkpoint_path

    monkeypatch.setattr(
        f"{_EXPERIMENT}.load_backbone_from_checkpoint", _fake_load
    )
    built = FineTuneExperiment().build_fit(
        _config(regime="frozen", checkpoint=str(checkpoint)),
        project_root=tmp_path,
    )
    assert called["path"] == checkpoint.resolve()
    for param in built.net.embedder.parameters():
        assert param.requires_grad is False
    for param in built.net.encoder.parameters():
        assert param.requires_grad is False
    for param in built.net.classifier_head.parameters():
        assert param.requires_grad is True
    trainable = [
        param for param in built.net.parameters() if param.requires_grad
    ]
    assert len(trainable) > 0


def test_build_fit_pretrained_missing_checkpoint_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_loaders(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="pretrained checkpoint"):
        FineTuneExperiment().build_fit(
            _config(regime="pretrained", checkpoint="missing.ckpt"),
            project_root=tmp_path,
        )


def test_resolve_finetune_recipe_finds_named_file() -> None:
    recipe = resolve_finetune_recipe(
        Path(__file__).resolve(), "fivefold.json"
    )
    assert recipe.is_file()
    assert recipe.name == "fivefold.json"
