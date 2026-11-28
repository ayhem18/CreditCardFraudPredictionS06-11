"""Write kept overfit grid members under run_recipes/supervised_reconstruction."""

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
    GRID_SEARCH_OVERFIT_RUN_GROUP,
    OVERFIT_GRID_SEARCH_DIRNAME,
    OVERFIT_RUN_GROUP,
    SUPERVISED_RECONSTRUCTION_DIRNAME,
    project_root_from_start,
    resolve_overfit_recipe,
)
from ccf_tabformer_dl.task_engines.overfit.config import OverfitConfigParser


def validate_overfit_experiment_config(payload: object) -> None:
    if not isinstance(payload, dict):
        raise TypeError(
            f"experiment_config must be a dict, got {type(payload)!r}"
        )
    OverfitConfigParser(payload)


def grid_search_output_dir(script: Path, grid_search_exp_name: str) -> Path:
    root = project_root_from_start(script)
    return (
        root
        / RUN_RECIPES_DIRNAME
        / SUPERVISED_RECONSTRUCTION_DIRNAME
        / OVERFIT_GRID_SEARCH_DIRNAME
        / grid_search_exp_name
    )


def write_overfit_grid_search(
    *,
    start: Path,
    document: GridSearchDocument,
    keep: KeepFn | None,
) -> tuple[Path, ...]:
    starting_path = resolve_overfit_recipe(start, document.starting_recipe)
    starting = read_object(file=File.from_absolute(path=starting_path))
    family = Directory.from_absolute(
        path=grid_search_output_dir(start, document.grid_search_exp_name),
        policy=DirectoryParsePolicy.CREATE,
    )
    return write_grid_search(
        starting_recipe=starting,
        document=document,
        family=family,
        validate=validate_overfit_experiment_config,
        keep=keep,
        stamp_recipe=_stamp_overfit_recipe,
    )


def _stamp_overfit_recipe(
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
        run_group=(
            OVERFIT_RUN_GROUP,
            GRID_SEARCH_OVERFIT_RUN_GROUP,
            grid_search_exp_name,
        ),
    )
