"""Overfit experiment assembles TrainOnlyFitParts from oracles."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.exp_engine.train_only_engine.experiment import TrainOnlyFitParts
from dl_toolbox.dl_engine.runs.common.run_config import OptimizationConfig

from ccf_tabformer_dl.project import resolve_overfit_recipe
from ccf_tabformer_dl.tabformer_components.mfm_model import MfmModel
from ccf_tabformer_dl.task_engines.overfit.experiment import OverfitExperiment
from tests.tabformer_components.fake_spec import tiny_ids, tiny_table_spec

_EXPERIMENT = "ccf_tabformer_dl.task_engines.overfit.experiment"

_EXPERIMENT_CONFIG = {
    "data": {
        "row_subset": "alarm_label_1k_seed0",
        "batch_size": 2,
        "num_workers": 0,
    },
    "embedding": {"strategy": "additive", "args": {"d_model": 8}},
    "masking": {"strategy": "field_level", "args": {"feature_mask_prob": 0.4}},
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
    "head": {"strategy": "shared_support_masked", "args": {"d_model": 8}},
    "lr_config": {"learning_rate": 0.001},
    "optimizer_config": {
        "name": "adamw",
        "weight_decay": 0.0,
        "betas": [0.9, 0.95],
    },
    "lr_scheduler_config": {"name": "constant"},
}


def test_build_train_only_uses_absolute_lr_and_one_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = tiny_ids()
    loader = DataLoader(TensorDataset(ids), batch_size=2)

    def _fake_loader(config):
        del config
        return loader

    monkeypatch.setattr(
        f"{_EXPERIMENT}.table_spec_from_subset",
        lambda row_subset: tiny_table_spec(),
    )
    monkeypatch.setattr(f"{_EXPERIMENT}.build_overfit_loader", _fake_loader)
    built = OverfitExperiment().build_train_only(
        _EXPERIMENT_CONFIG, project_root=tmp_path
    )
    assert isinstance(built, TrainOnlyFitParts)
    assert isinstance(built.net, MfmModel)
    assert built.loaders.train is loader
    assert built.optimization.scheduler is OptimizationConfig.ABSENT
    assert isinstance(built.optimization.optimizer, AdamW)
    assert built.optimization.optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert "/train/" in built.metrics.monitor.key
    assert built.metrics.monitor.mode == "max"
    names = [spec[0].name for spec in built.metrics.train_metrics]
    assert names == ["recon_acc", "bce"]


def test_resolve_overfit_recipe_finds_named_file() -> None:
    recipe = resolve_overfit_recipe(
        Path(__file__).resolve(), "mfm_overfit_alarm_label_1k.json"
    )
    assert recipe.is_file()
    assert recipe.name == "mfm_overfit_alarm_label_1k.json"
