"""Classification-overfit experiment_config: row_subset data, shared classifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccf_tabformer_dl.project import resolve_finetune_overfit_recipe
from ccf_tabformer_dl.task_engines.finetune.config import REGIME_RANDOM
from ccf_tabformer_dl.task_engines.finetune_overfit.config import (
    FineTuneOverfitConfigParser,
)

_VALID = {
    "data": {
        "row_subset": "alarm_label_1k_seed0",
        "batch_size": 32,
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
        "args": {"d_model": 8, "hidden": 16, "dropout": 0.1},
    },
    "lr_config": {"learning_rate": 0.0001},
    "optimizer_config": {
        "name": "adamw",
        "weight_decay": 0.0,
        "betas": [0.9, 0.95],
    },
    "lr_scheduler_config": {"name": "constant"},
}


def test_parser_reads_row_subset_and_classifier() -> None:
    parsed = FineTuneOverfitConfigParser(_VALID)
    assert parsed.data.row_subset == "alarm_label_1k_seed0"
    assert parsed.data.batch_size == 32
    assert parsed.regime == REGIME_RANDOM
    assert parsed.load_backbone() is False
    assert parsed.freeze_encoder() is False
    assert parsed.classifier_head.strategy == "mean_pool"
    assert parsed.lr.learning_rate == pytest.approx(0.0001)


def test_parser_rejects_folds_data_keys() -> None:
    payload = dict(_VALID)
    payload["data"] = {
        "inner_table": "fixed_width_v1",
        "folds": "alarm_profile_5fold_v1",
        "fold": 0,
        "batch_size": 32,
        "num_workers": 0,
    }
    with pytest.raises(ValueError, match="data"):
        FineTuneOverfitConfigParser(payload)


def test_parser_rejects_masking_key() -> None:
    payload = dict(_VALID)
    payload["masking"] = {"strategy": "field_level", "args": {}}
    with pytest.raises(ValueError, match="experiment_config"):
        FineTuneOverfitConfigParser(payload)


def test_starting_recipe_parses() -> None:
    recipe = resolve_finetune_overfit_recipe(
        Path(__file__).resolve(), "cls_overfit_alarm_label_1k.json"
    )
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    parsed = FineTuneOverfitConfigParser(payload["experiment_config"])
    assert parsed.data.row_subset == "alarm_label_1k_seed0"
    assert payload["trainer"]["devices"] == 1
    assert payload["early_stopping"]["name"] == "none"
    assert payload["run_group"] == [
        "finetune",
        "overfit_sanity_check",
        "cls",
    ]
