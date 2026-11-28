"""Run MFM pretraining from a recipe.

  python scripts/unsupervised_reconstruction/pretrain.py <recipe.json>
  python scripts/unsupervised_reconstruction/pretrain.py <name>            # under run_recipes/supervised_reconstruction/pretrain/
  python scripts/unsupervised_reconstruction/pretrain.py <recipe> --force  # overwrite this run name

Uses the project's dedicated venv. The tracking backend defaults to the one in
.tracking-backend (mlflow).
"""

from __future__ import annotations

import argparse                     
from pathlib import Path

from dl_toolbox.project_layout.names import RUN_RECIPES_DIRNAME

from ccf_tabformer_dl.project import (
    RECONSTRUCTION_PRETRAIN_RECIPES,
    project_root_from_start,
)
from ccf_tabformer_dl.task_engines.pretraining.engine import make_pretrain_engine


def _resolve_recipe(script: Path, recipe: str) -> Path:
    direct = Path(recipe)
    if direct.is_file():
        return direct.resolve()
    tree_root = project_root_from_start(script)
    recipes_dir = tree_root / RUN_RECIPES_DIRNAME / RECONSTRUCTION_PRETRAIN_RECIPES
    under = recipes_dir / recipe
    if under.is_file():
        return under.resolve()
    raise FileNotFoundError(
        f"recipe not found as {recipe!r} (cwd) nor under {recipes_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ccf_tabformer MFM pretraining.")
    parser.add_argument(
        "recipe",
        help=(
            "Recipe JSON path, or a name under "
            "run_recipes/supervised_reconstruction/pretrain/."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite this run name, then fit.")
    parser.add_argument("--tracking-backend", default=None, choices=["mlflow", "tensorboard"])
    args = parser.parse_args()
    script = Path(__file__).resolve()
    recipe = _resolve_recipe(script, args.recipe)
    make_pretrain_engine(
        script, recipe, tracking_backend=args.tracking_backend, force=args.force
    ).train()


if __name__ == "__main__":
    main()
