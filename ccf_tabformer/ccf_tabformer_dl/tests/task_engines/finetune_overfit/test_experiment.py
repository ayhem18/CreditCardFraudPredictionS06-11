"""Classification overfit experiment assembles TrainOnlyFitParts."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from dl_toolbox.dl_engine.exp_engine.train_only_engine.experiment import TrainOnlyFitParts
from dl_toolbox.dl_engine.runs.common.run_config import OptimizationConfig

from ccf_tabformer_dl.tabformer_components.classifier_model import ClassifierModel
from ccf_tabformer_dl.task_engines.finetune_overfit.experiment import (
    FineTuneOverfitExperiment,
)
from tests.tabformer_components.fake_spec import tiny_ids, tiny_table_spec

_EXPERIMENT = "ccf_tabformer_dl.task_engines.finetune_overfit.experiment"

_EXPERIMENT_CONFIG = {
    "data": {
        "row_subset": "alarm_label_1k_seed0",
        "batch_size": 2,
        "num_workers": 0,
    },
    "regime": "random",
    "pretrained_checkpoint": "",
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


def test_build_train_only_monitors_train_auc(
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
    monkeypatch.setattr(
        f"{_EXPERIMENT}.build_finetune_overfit_loader", _fake_loader
    )
    built = FineTuneOverfitExperiment().build_train_only(
        _EXPERIMENT_CONFIG, project_root=tmp_path
    )
    assert isinstance(built, TrainOnlyFitParts)
    assert isinstance(built.net, ClassifierModel)
    assert built.loaders.train is loader
    assert built.optimization.scheduler is OptimizationConfig.ABSENT
    assert isinstance(built.optimization.optimizer, AdamW)
    assert built.metrics.monitor.key == "auc/train/epoch/auc"
    assert built.metrics.monitor.mode == "max"
    names = [spec[0].name for spec in built.metrics.train_metrics]
    assert names == ["auc", "bce"]
    for param in built.net.embedder.parameters():
        assert param.requires_grad is True
