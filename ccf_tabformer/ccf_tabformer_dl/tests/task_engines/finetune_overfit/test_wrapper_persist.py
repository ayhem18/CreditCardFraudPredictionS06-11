"""Classification-overfit grid writer stamps exp_i recipes and run_group."""

from __future__ import annotations

from pathlib import Path

import pytest

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import JsonTextForm, read_object, write_object
from dl_toolbox.dl_engine.exp_engine.grid_search import (
    GridSearchPersist,
    parse_grid_search_document,
)

from ccf_tabformer_dl.task_engines.finetune_overfit.config import (
    FineTuneOverfitConfigParser,
)
from ccf_tabformer_dl.task_engines.finetune_overfit.wrapper.persist import (
    write_finetune_overfit_grid_search,
)

_STARTING = {
    "human_readable_name": "cls_overfit_alarm_label_1k",
    "run_group": ["finetune", "overfit_sanity_check", "cls"],
    "seed": 0,
    "trainer": {
        "max_epochs": 2,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "bf16-mixed",
        "enable_progress_bar": True,
        "accumulate_grad_batches": 1,
    },
    "checkpoint_policy": {"save_top_k": 1, "save_last": False},
    "early_stopping": {"name": "none"},
    "logger": {
        "experiment": "ccf_tabformer_finetune_overfit",
        "tags": {"task": "finetune"},
    },
    "experiment_config": {
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
    },
}


def _install_tmp_project(tmp_path: Path, monkeypatch, payload: dict) -> None:
    import ccf_tabformer_dl.task_engines.finetune_overfit.wrapper.persist as persist

    recipes_dir = (
        tmp_path
        / "run_recipes"
        / "classification_fine_tune"
        / "overfit_grid_search"
    )
    recipes_dir.mkdir(parents=True)
    starting = write_object(
        directory=Directory.from_absolute(
            path=recipes_dir, policy=DirectoryParsePolicy.MUST_EXIST
        ),
        name="base.json",
        payload=payload,
        form=JsonTextForm.DOCUMENT,
    )
    monkeypatch.setattr(persist, "project_root_from_start", lambda start: tmp_path)
    monkeypatch.setattr(
        persist,
        "resolve_finetune_overfit_recipe",
        lambda start, recipe: starting.path,
    )


def test_write_finetune_overfit_grid_search_uses_exp_i(
    tmp_path: Path, monkeypatch
) -> None:
    _install_tmp_project(tmp_path, monkeypatch, _STARTING)
    document = parse_grid_search_document(
        {
            "starting_recipe": "base.json",
            "grid_search_exp_name": "cls_alarm_label_1k_lr",
            "grid_search_config": [
                {
                    "param_name": "lr_config.learning_rate",
                    "param_values": [0.0001, 0.001],
                }
            ],
        }
    )
    paths = write_finetune_overfit_grid_search(
        start=tmp_path / "script.py", document=document, keep=None
    )
    assert [path.name for path in paths] == ["exp_0.json", "exp_1.json"]
    first = read_object(file=File.from_absolute(path=paths[0]))
    parsed = FineTuneOverfitConfigParser(first["experiment_config"])
    assert parsed.lr.learning_rate == pytest.approx(0.0001)
    assert first["human_readable_name"] == "cls_alarm_label_1k_lr_exp_0"
    assert first["run_group"] == [
        "finetune",
        "overfit_sanity_check",
        "grid_search_overfit",
        "cls_alarm_label_1k_lr",
    ]
    index = read_object(
        file=File.from_absolute(path=paths[0].parent / GridSearchPersist.INDEX_FILENAME)
    )
    assert index["experiments"][1]["params"]["lr_config.learning_rate"] == 0.001
    assert index["skips"] == []
