"""Fold-recipe writer stamps data.fold and human_readable_name."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import JsonTextForm, read_object, write_object

from ccf_tabformer_dl.task_engines.finetune.config import FineTuneConfigParser
from ccf_tabformer_dl.task_engines.finetune.wrapper.persist import (
    INDEX_FILENAME,
    write_fold_recipes,
)

_STARTING = {
    "human_readable_name": "mfm_finetune_pretrained",
    "run_group": ["finetune", "pretrained"],
    "seed": 0,
    "trainer": {
        "max_epochs": 2,
        "accelerator": "gpu",
        "devices": 1,
        "precision": "bf16-mixed",
        "enable_progress_bar": True,
        "accumulate_grad_batches": 1,
    },
    "checkpoint_policy": {"save_top_k": 0, "save_last": True},
    "early_stopping": {"name": "none"},
    "logger": {
        "experiment": "ccf_tabformer_finetune",
        "tags": {"regime": "pretrained"},
    },
    "experiment_config": {
        "data": {
            "inner_table": "fixed_width_v1",
            "folds": "alarm_profile_5fold_v1",
            "fold": 0,
            "batch_size": 32,
            "num_workers": 0,
        },
        "regime": "pretrained",
        "pretrained_checkpoint": "SET_AFTER_FULL_RUN.ckpt",
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


def _install_tmp_project(tmp_path: Path, monkeypatch, payload: dict) -> Path:
    import ccf_tabformer_dl.task_engines.finetune.wrapper.persist as persist

    recipes_dir = tmp_path / "run_recipes" / "finetune"
    recipes_dir.mkdir(parents=True)
    starting = write_object(
        directory=Directory.from_absolute(
            path=recipes_dir, policy=DirectoryParsePolicy.MUST_EXIST
        ),
        name="fivefold.json",
        payload=payload,
        form=JsonTextForm.DOCUMENT,
    )
    monkeypatch.setattr(persist, "project_root_from_start", lambda start: tmp_path)
    monkeypatch.setattr(
        persist,
        "resolve_finetune_recipe",
        lambda start, recipe: starting.path,
    )
    return starting.path


def test_write_fold_recipes_stamps_fold_and_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    _install_tmp_project(tmp_path, monkeypatch, _STARTING)
    checkpoint = str(tmp_path / "backbone.ckpt")
    paths = write_fold_recipes(
        start=tmp_path / "script.py",
        starting_recipe=tmp_path / "run_recipes" / "finetune" / "fivefold.json",
        n_splits=2,
        pretrained_checkpoint=checkpoint,
    )
    assert [path.name for path in paths] == ["fold_0.json", "fold_1.json"]
    first = read_object(file=File.from_absolute(path=paths[0]))
    parsed = FineTuneConfigParser(first["experiment_config"])
    assert parsed.data.fold == 0
    assert parsed.pretrained_checkpoint == checkpoint
    assert first["human_readable_name"] == "mfm_finetune_pretrained_fold0"
    assert first["logger"]["tags"]["fold"] == "0"
    second = read_object(file=File.from_absolute(path=paths[1]))
    assert FineTuneConfigParser(second["experiment_config"]).data.fold == 1
    assert second["human_readable_name"] == "mfm_finetune_pretrained_fold1"
    index = read_object(
        file=File.from_absolute(path=paths[0].parent / INDEX_FILENAME)
    )
    assert index["n_splits"] == 2
    assert index["folds"][1]["file"] == "fold_1.json"
