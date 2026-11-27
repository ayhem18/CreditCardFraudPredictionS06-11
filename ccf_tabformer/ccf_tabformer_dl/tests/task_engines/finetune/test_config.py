"""Fine-tune experiment_config parse: regimes, folds data, recipes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dl_toolbox.dl_engine.exp_engine.train_val_engine.recipe import parse_inspect_recipe

from ccf_tabformer_dl.project import resolve_finetune_recipe
from ccf_tabformer_dl.task_engines.finetune.config import (
    REGIME_FROZEN,
    REGIME_PRETRAINED,
    REGIME_RANDOM,
    FineTuneConfigParser,
)

_VALID = {
    "data": {
        "inner_table": "fixed_width_v1",
        "folds": "alarm_profile_5fold_v1",
        "fold": 0,
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


def test_parser_reads_random_regime() -> None:
    parsed = FineTuneConfigParser(_VALID)
    assert parsed.data.inner_table == "fixed_width_v1"
    assert parsed.data.folds == "alarm_profile_5fold_v1"
    assert parsed.data.fold == 0
    assert parsed.regime == REGIME_RANDOM
    assert parsed.pretrained_checkpoint == ""
    assert parsed.load_backbone() is False
    assert parsed.freeze_encoder() is False
    assert parsed.classifier_head.strategy == "mean_pool"
    assert parsed.lr.learning_rate == pytest.approx(0.0001)


def test_parser_pretrained_requires_checkpoint() -> None:
    payload = dict(_VALID)
    payload["regime"] = REGIME_PRETRAINED
    payload["pretrained_checkpoint"] = "SET_AFTER_FULL_RUN.ckpt"
    parsed = FineTuneConfigParser(payload)
    assert parsed.load_backbone() is True
    assert parsed.freeze_encoder() is False


def test_parser_frozen_loads_and_freezes() -> None:
    payload = dict(_VALID)
    payload["regime"] = REGIME_FROZEN
    payload["pretrained_checkpoint"] = "SET_AFTER_FULL_RUN.ckpt"
    parsed = FineTuneConfigParser(payload)
    assert parsed.load_backbone() is True
    assert parsed.freeze_encoder() is True


def test_parser_rejects_random_with_checkpoint() -> None:
    payload = dict(_VALID)
    payload["pretrained_checkpoint"] = "some.ckpt"
    with pytest.raises(ValueError, match="empty"):
        FineTuneConfigParser(payload)


def test_parser_rejects_pretrained_with_empty_checkpoint() -> None:
    payload = dict(_VALID)
    payload["regime"] = REGIME_PRETRAINED
    payload["pretrained_checkpoint"] = ""
    with pytest.raises(ValueError, match="pretrained_checkpoint"):
        FineTuneConfigParser(payload)


def test_parser_rejects_masking_key() -> None:
    payload = dict(_VALID)
    payload["masking"] = {"strategy": "field_level", "args": {}}
    with pytest.raises(ValueError, match="experiment_config"):
        FineTuneConfigParser(payload)


def test_recipe_experiment_configs_parse() -> None:
    names = (
        "pilot_fold0_random.json",
        "pilot_fold0_frozen.json",
        "pilot_fold0_pretrained.json",
        "fivefold.json",
        "smoke_local.json",
    )
    script = Path(__file__).resolve()
    for name in names:
        recipe = resolve_finetune_recipe(script, name)
        payload = json.loads(recipe.read_text(encoding="utf-8"))
        parsed = FineTuneConfigParser(payload["experiment_config"])
        assert payload["trainer"]["devices"] == 1
        assert payload["early_stopping"]["name"] == "none"
        assert payload["checkpoint_policy"]["save_last"] is True
        assert parsed.data.folds == "alarm_profile_5fold_v1"
        assert parsed.embedding.args["d_model"] == 1024
        assert parsed.encoder.args["n_heads"] == 8
        assert parsed.encoder.args["n_layers"] == 6


def test_inspect_last_recipe_parses() -> None:
    recipe = resolve_finetune_recipe(
        Path(__file__).resolve(), "inspect_last.json"
    )
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    parsed = parse_inspect_recipe(payload)
    assert parsed.selector == "last"
