"""Full-run experiment_config parse: inner_table data, required strategies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccf_tabformer_dl.project import resolve_full_run_recipe
from ccf_tabformer_dl.task_engines.full_run.config import FullRunConfigParser

_VALID = {
    "data": {
        "inner_table": "fixed_width_v1",
        "batch_size": 32,
        "num_workers": 0,
    },
    "embedding": {"strategy": "additive", "args": {"d_model": 8}},
    "masking": {"strategy": "field_level", "args": {"feature_mask_prob": 0.15}},
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


def test_parser_reads_required_blocks() -> None:
    parsed = FullRunConfigParser(_VALID)
    assert parsed.data.inner_table == "fixed_width_v1"
    assert parsed.data.batch_size == 32
    assert parsed.embedding.strategy == "additive"
    assert parsed.masking.strategy == "field_level"
    assert parsed.encoder.strategy == "prenorm_transformer"
    assert parsed.head.strategy == "shared_support_masked"
    assert parsed.lr.learning_rate == pytest.approx(0.001)
    assert parsed.optimizer.betas == (0.9, 0.95)
    assert parsed.scheduler.name == "constant"


def test_parser_rejects_row_subset_data_key() -> None:
    payload = dict(_VALID)
    payload["data"] = {
        "row_subset": "alarm_label_1k_seed0",
        "batch_size": 32,
        "num_workers": 0,
    }
    with pytest.raises(ValueError, match="data"):
        FullRunConfigParser(payload)


def test_parser_rejects_extra_experiment_key() -> None:
    payload = dict(_VALID)
    payload["loss_config"] = {"name": "ce"}
    with pytest.raises(ValueError, match="experiment_config"):
        FullRunConfigParser(payload)


def test_recipe_experiment_config_parses() -> None:
    recipe = resolve_full_run_recipe(
        Path(__file__).resolve(), "mfm_full_run_fixed_width_v1.json"
    )
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    parsed = FullRunConfigParser(payload["experiment_config"])
    assert parsed.data.inner_table == "fixed_width_v1"
    early = payload["early_stopping"]
    assert early["name"] == "patience"
    assert early["patience"] == 50
    assert early["min_delta"] == 0.0
    assert type(early["min_delta"]) is float
    assert payload["trainer"]["max_epochs"] == 500
