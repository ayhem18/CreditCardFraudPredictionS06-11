"""Overfit experiment_config parse: exact keys, required strategies."""

from __future__ import annotations

import pytest

from ccf_tabformer_dl.task_engines.overfit.config import OverfitConfigParser

_VALID = {
    "data": {
        "row_subset": "alarm_label_1k_seed0",
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
    parsed = OverfitConfigParser(_VALID)
    assert parsed.data.row_subset == "alarm_label_1k_seed0"
    assert parsed.data.batch_size == 32
    assert parsed.embedding.strategy == "additive"
    assert parsed.masking.strategy == "field_level"
    assert parsed.encoder.strategy == "prenorm_transformer"
    assert parsed.head.strategy == "shared_support_masked"
    assert parsed.lr.learning_rate == pytest.approx(0.001)
    assert parsed.optimizer.betas == (0.9, 0.95)
    assert parsed.scheduler.name == "constant"


def test_parser_rejects_extra_experiment_key() -> None:
    payload = dict(_VALID)
    payload["loss_config"] = {"name": "ce"}
    with pytest.raises(ValueError, match="experiment_config"):
        OverfitConfigParser(payload)


def test_parser_rejects_unknown_optimizer() -> None:
    payload = dict(_VALID)
    payload["optimizer_config"] = {
        "name": "sgd",
        "weight_decay": 0.0,
        "betas": [0.9, 0.95],
    }
    with pytest.raises(ValueError, match="adamw"):
        OverfitConfigParser(payload)
