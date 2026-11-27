"""Resolve the ccf_tabformer ProjectLayout from any path inside the tree."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import read_object
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import one_of_message, parse_one_of
from dl_toolbox.project_layout.names import (
    MARKER_FILENAME,
    RUN_RECIPES_DIRNAME,
    TRACKING_BACKEND_FILENAME,
)
from dl_toolbox.project_layout.project_layout import ProjectLayout, TrackingBackend

OVERFIT_SANITY_CHECKS_DIRNAME = "overfit_sanity_checks"
OVERFIT_RUN_GROUP = "overfit_sanity_check"
GRID_SEARCH_OVERFIT_DIRNAME = "grid_search_overfit"
FULL_RUN_DIRNAME = "full_run"
FULL_RUN_GROUP = "full_run"
FINETUNE_DIRNAME = "finetune"
FINETUNE_RUN_GROUP = "finetune"
FINETUNE_FOLDS_DIRNAME = "folds"
FINETUNE_OVERFIT_SANITY_CHECKS_DIRNAME = "overfit_sanity_checks"
FINETUNE_OVERFIT_RUN_GROUP = ("finetune", "overfit_sanity_check")


def project_root_from_start(start: Path) -> Path:
    """Directory that holds ``dl_toolbox_project.json``, walking up from ``start``."""
    start_file = File.from_absolute(path=start)
    marker = File.from_parent_walk(start=start_file, basename=MARKER_FILENAME)
    return marker.parent().path


def resolve_overfit_recipe(script: Path, recipe: str) -> Path:
    """Resolve a recipe path or a name under ``run_recipes/overfit_sanity_checks``."""
    return _resolve_named_recipe(
        script,
        recipe,
        folder=OVERFIT_SANITY_CHECKS_DIRNAME,
        label="recipe",
    )


def resolve_finetune_recipe(script: Path, recipe: str) -> Path:
    """Resolve a recipe path or a name under ``run_recipes/finetune``."""
    return _resolve_named_recipe(
        script,
        recipe,
        folder=FINETUNE_DIRNAME,
        label="recipe",
    )


def resolve_finetune_overfit_recipe(script: Path, recipe: str) -> Path:
    """Resolve a recipe under ``run_recipes/finetune/overfit_sanity_checks``."""
    return _resolve_named_recipe(
        script,
        recipe,
        folder=f"{FINETUNE_DIRNAME}/{FINETUNE_OVERFIT_SANITY_CHECKS_DIRNAME}",
        label="recipe",
    )


def resolve_full_run_recipe(script: Path, recipe: str) -> Path:
    """Resolve a recipe path or a name under ``run_recipes/full_run``."""
    return _resolve_named_recipe(
        script,
        recipe,
        folder=FULL_RUN_DIRNAME,
        label="recipe",
    )


def _resolve_named_recipe(
    script: Path, recipe: str, *, folder: str, label: str
) -> Path:
    text = parse_non_empty_str(recipe, label=label)
    root = project_root_from_start(script)
    recipes_dir = root / RUN_RECIPES_DIRNAME / folder
    path = Path(text)
    if path.is_absolute():
        if path.is_file():
            return path
        raise FileNotFoundError(f"{label} not found: {path}")
    relative_to_script = (script.parent / path).resolve()
    if relative_to_script.is_file():
        return relative_to_script
    under_folder = recipes_dir / path
    if under_folder.is_file():
        return under_folder
    direct = recipes_dir / path.name
    if direct.is_file():
        return direct
    raise FileNotFoundError(
        f"{label} not found as {text!r} (cwd/script) nor under {recipes_dir}"
    )


def resolve_finetune_overfit_grid_search_document(script: Path, document: str) -> Path:
    """Resolve a grid-search JSON under ``run_recipes/finetune/overfit_sanity_checks``."""
    return _resolve_named_recipe(
        script,
        document,
        folder=(
            f"{FINETUNE_DIRNAME}/{FINETUNE_OVERFIT_SANITY_CHECKS_DIRNAME}/"
            f"{GRID_SEARCH_OVERFIT_DIRNAME}"
        ),
        label="grid_search",
    )


def resolve_grid_search_document(script: Path, document: str) -> Path:
    """Resolve a grid-search JSON under ``run_recipes/overfit_sanity_checks``."""
    text = parse_non_empty_str(document, label="grid_search")
    root = project_root_from_start(script)
    grids_dir = (
        root
        / RUN_RECIPES_DIRNAME
        / OVERFIT_SANITY_CHECKS_DIRNAME
        / GRID_SEARCH_OVERFIT_DIRNAME
    )
    path = Path(text)
    if path.is_absolute():
        if path.is_file():
            return path
        raise FileNotFoundError(f"grid search document not found: {path}")
    relative_to_script = (script.parent / path).resolve()
    if relative_to_script.is_file():
        return relative_to_script
    under_folder = grids_dir / path
    if under_folder.is_file():
        return under_folder
    direct = grids_dir / path.name
    if direct.is_file():
        return direct
    raise FileNotFoundError(
        f"grid search document not found as {text!r} (cwd/script) nor under {grids_dir}"
    )


def load_project_layout(*, start: Path, tracking_backend: str | None) -> ProjectLayout:
    start_file = File.from_absolute(path=start)
    backend = (
        tracking_backend
        if tracking_backend is not None
        else read_project_tracking_backend(start=start_file)
    )
    marker = read_object(
        file=File.from_parent_walk(start=start_file, basename=MARKER_FILENAME)
    )
    name = marker["name"]
    if not isinstance(name, str) or not name:
        raise TypeError(f"marker name must be a non-empty str, got {name!r}")
    return ProjectLayout.from_start(
        start=start_file, project_name=name, tracking_backend=backend
    )


def read_project_tracking_backend(*, start: File) -> str:
    marker = File.from_parent_walk(start=start, basename=MARKER_FILENAME)
    backend_file = File.from_under(
        root=marker.parent(), relative_path=TRACKING_BACKEND_FILENAME
    )
    text = parse_non_empty_str(
        backend_file.path.read_text(encoding="utf-8").strip(), label="tracking_backend"
    )
    parsed = parse_one_of(
        text,
        allowed=TrackingBackend.BACKENDS,
        label="tracking_backend",
        error_type=ValueError,
        error_message=one_of_message,
    )
    if not isinstance(parsed, str):
        raise TypeError(f"tracking_backend must be str, got {type(parsed)!r}")
    return parsed
