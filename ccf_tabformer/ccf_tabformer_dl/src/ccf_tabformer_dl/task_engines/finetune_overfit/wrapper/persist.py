"""Write kept classification-overfit grid members under run_recipes/finetune."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dl_toolbox.base.file_management.directory import (
    Directory,
    DirectoryParsePolicy,
)
from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import read_object
from dl_toolbox.dl_engine.exp_engine.grid_search import (
    GridSearchDocument,
    KeepFn,
    stamp_grid_recipe,
    write_grid_search,
)
from dl_toolbox.project_layout.names import RUN_RECIPES_DIRNAME

from ccf_tabformer_dl.project import (
    FINETUNE_DIRNAME,
    FINETUNE_OVERFIT_RUN_GROUP,
    FINETUNE_OVERFIT_SANITY_CHECKS_DIRNAME,
    GRID_SEARCH_OVERFIT_DIRNAME,
    project_root_from_start,
    resolve_finetune_overfit_recipe,
)
from ccf_tabformer_dl.task_engines.finetune_overfit.config import (
    FineTuneOverfitConfigParser,
)


def validate_finetune_overfit_experiment_config(payload: object) -> None:
    if not isinstance(payload, dict):
        raise TypeError(
            f"experiment_config must be a dict, got {type(payload)!r}"
        )
    FineTuneOverfitConfigParser(payload)


def grid_search_output_dir(script: Path, grid_search_exp_name: str) -> Path:
    root = project_root_from_start(script)
    return (
        root
        / RUN_RECIPES_DIRNAME
        / FINETUNE_DIRNAME
        / FINETUNE_OVERFIT_SANITY_CHECKS_DIRNAME
        / GRID_SEARCH_OVERFIT_DIRNAME
        / grid_search_exp_name
    )


def write_finetune_overfit_grid_search(
    *,
    start: Path,
    document: GridSearchDocument,
    keep: KeepFn | None,
) -> tuple[Path, ...]:
    starting_path = resolve_finetune_overfit_recipe(start, document.starting_recipe)
    starting = read_object(file=File.from_absolute(path=starting_path))
    family = Directory.from_absolute(
        path=grid_search_output_dir(start, document.grid_search_exp_name),
        policy=DirectoryParsePolicy.CREATE,
    )
    return write_grid_search(
        starting_recipe=starting,
        document=document,
        family=family,
        validate=validate_finetune_overfit_experiment_config,
        keep=keep,
        stamp_recipe=_stamp_finetune_overfit_recipe,
    )


def _stamp_finetune_overfit_recipe(
    starting: dict[str, Any],
    *,
    experiment_config: dict[str, Any],
    grid_search_exp_name: str,
    index: int,
    cartesian_index: int,
) -> dict[str, Any]:
    return stamp_grid_recipe(
        starting,
        experiment_config=experiment_config,
        grid_search_exp_name=grid_search_exp_name,
        index=index,
        cartesian_index=cartesian_index,
        run_group=FINETUNE_OVERFIT_RUN_GROUP
        + (GRID_SEARCH_OVERFIT_DIRNAME, grid_search_exp_name),
    )
