"""Full-run experiment assembles TrainValFitParts from oracles."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.exp_engine.train_val_engine.experiment import TrainValFitParts
from dl_toolbox.dl_engine.runs.common.run_config import OptimizationConfig

from ccf_tabformer_dl.project import resolve_full_run_recipe
from ccf_tabformer_dl.tabformer_components.mfm_model import MfmModel
from ccf_tabformer_dl.task_engines.full_run.experiment import FullRunExperiment
from tests.tabformer_components.fake_spec import tiny_ids, tiny_table_spec

_EXPERIMENT = "ccf_tabformer_dl.task_engines.full_run.experiment"

_EXPERIMENT_CONFIG = {
    "data": {
        "inner_table": "fixed_width_v1",
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
    "lr_config": {"learning_rate": 0.000316227766016838},
    "optimizer_config": {
        "name": "adamw",
        "weight_decay": 0.0,
        "betas": [0.9, 0.95],
    },
    "lr_scheduler_config": {"name": "constant"},
}


def test_build_fit_uses_two_loaders_and_val_bce_monitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    monkeypatch.setattr(f"{_EXPERIMENT}.build_train_val_loaders", _fake_loaders)
    built = FullRunExperiment().build_fit(
        _EXPERIMENT_CONFIG, project_root=tmp_path
    )
    assert isinstance(built, TrainValFitParts)
    assert isinstance(built.net, MfmModel)
    assert built.loaders.train is train_loader
    assert built.loaders.val is val_loader
    assert built.optimization.scheduler is OptimizationConfig.ABSENT
    assert isinstance(built.optimization.optimizer, AdamW)
    assert built.optimization.optimizer.param_groups[0]["lr"] == pytest.approx(
        0.000316227766016838
    )
    assert built.metrics.monitor.key == "bce/val/epoch/mean"
    assert built.metrics.monitor.mode == "min"
    train_names = [spec[0].name for spec in built.metrics.train_metrics]
    val_names = [spec[0].name for spec in built.metrics.val_metrics]
    assert train_names == ["recon_acc", "bce"]
    assert val_names == ["recon_acc", "bce"]


def test_resolve_full_run_recipe_finds_named_file() -> None:
    recipe = resolve_full_run_recipe(
        Path(__file__).resolve(), "mfm_full_run_fixed_width_v1.json"
    )
    assert recipe.is_file()
    assert recipe.name == "mfm_full_run_fixed_width_v1.json"
