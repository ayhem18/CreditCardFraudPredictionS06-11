"""Write one fine-tune recipe per imported fold from a starting recipe."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import JsonTextForm, read_object, write_object
from dl_toolbox.base.parsing.parse_numerical import parse_positive_int
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.project_layout.names import RUN_RECIPES_DIRNAME

from ccf_tabformer_dl.project import (
    FINETUNE_DIRNAME,
    FINETUNE_FOLDS_DIRNAME,
    project_root_from_start,
    resolve_finetune_recipe,
)
from ccf_tabformer_dl.task_engines.finetune.config import (
    REGIME_RANDOM,
    FineTuneConfigParser,
)

INDEX_FILENAME = "index.json"


def fold_recipes_output_dir(script: Path, family_name: str) -> Path:
    root = project_root_from_start(script)
    name = parse_non_empty_str(family_name, label="family_name")
    return (
        root
        / RUN_RECIPES_DIRNAME
        / FINETUNE_DIRNAME
        / FINETUNE_FOLDS_DIRNAME
        / name
    )


def write_fold_recipes(
    *,
    start: Path,
    starting_recipe: Path,
    n_splits: int,
    pretrained_checkpoint: str | None,
) -> tuple[Path, ...]:
    """Stamp ``data.fold`` 0..n_splits-1 under ``run_recipes/finetune/folds/``."""
    splits = parse_positive_int(n_splits, label="n_splits")
    starting_path = resolve_finetune_recipe(start, str(starting_recipe))
    starting = read_object(file=File.from_absolute(path=starting_path))
    if not isinstance(starting, dict):
        raise TypeError(f"starting recipe must be a dict, got {type(starting)!r}")
    experiment_config = starting.get("experiment_config")
    if not isinstance(experiment_config, dict):
        raise TypeError(
            f"experiment_config must be a dict, got {type(experiment_config)!r}"
        )
    parsed = FineTuneConfigParser(dict(experiment_config))
    if pretrained_checkpoint is not None:
        if parsed.regime == REGIME_RANDOM:
            raise ValueError(
                "pretrained_checkpoint override is illegal when regime is random"
            )
        experiment_config = dict(experiment_config)
        experiment_config["pretrained_checkpoint"] = parse_non_empty_str(
            pretrained_checkpoint, label="pretrained_checkpoint"
        )
        starting = dict(starting)
        starting["experiment_config"] = experiment_config
        FineTuneConfigParser(dict(experiment_config))
    base_name = parse_non_empty_str(
        starting["human_readable_name"], label="human_readable_name"
    )
    family = Directory.from_absolute(
        path=fold_recipes_output_dir(start, base_name),
        policy=DirectoryParsePolicy.CREATE,
    )
    paths: list[Path | None] = [None] * splits
    index_rows: list[dict[str, Any] | None] = [None] * splits
    for fold in range(splits):
        recipe = _stamp_fold_recipe(starting, fold=fold, base_name=base_name)
        FineTuneConfigParser(dict(recipe["experiment_config"]))
        written = write_object(
            directory=family,
            name=f"fold_{fold}.json",
            payload=recipe,
            form=JsonTextForm.DOCUMENT,
        )
        paths[fold] = written.path
        index_rows[fold] = {
            "fold": fold,
            "file": f"fold_{fold}.json",
            "human_readable_name": recipe["human_readable_name"],
        }
    write_object(
        directory=family,
        name=INDEX_FILENAME,
        payload={
            "starting_recipe": starting_path.name,
            "n_splits": splits,
            "folds": [row for row in index_rows if row is not None],
        },
        form=JsonTextForm.DOCUMENT,
    )
    return tuple(path for path in paths if path is not None)


def _stamp_fold_recipe(
    starting: dict[str, Any], *, fold: int, base_name: str
) -> dict[str, Any]:
    recipe = copy.deepcopy(starting)
    experiment_config = recipe["experiment_config"]
    if not isinstance(experiment_config, dict):
        raise TypeError(
            f"experiment_config must be a dict, got {type(experiment_config)!r}"
        )
    data = experiment_config.get("data")
    if not isinstance(data, dict):
        raise TypeError(f"data must be a dict, got {type(data)!r}")
    data = dict(data)
    data["fold"] = fold
    experiment_config = dict(experiment_config)
    experiment_config["data"] = data
    recipe["experiment_config"] = experiment_config
    recipe["human_readable_name"] = f"{base_name}_fold{fold}"
    logger = recipe.get("logger")
    if isinstance(logger, dict):
        recipe["logger"] = dict(logger)
        tags = logger.get("tags")
        updated_tags = dict(tags) if isinstance(tags, dict) else {}
        updated_tags["fold"] = str(fold)
        recipe["logger"]["tags"] = updated_tags
    return recipe
