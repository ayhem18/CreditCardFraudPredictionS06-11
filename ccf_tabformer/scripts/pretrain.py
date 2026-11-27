"""Run MFM pretraining from a recipe.

  python scripts/pretrain.py <recipe.json>
  python scripts/pretrain.py <name>            # under run_recipes/pretrain/
  python scripts/pretrain.py <recipe> --force  # overwrite this run name

Uses the project's dedicated venv. The tracking backend defaults to the one in
.tracking-backend (mlflow).
"""

from __future__ import annotations

import argparse                     
from pathlib import Path

from ccf_tabformer_dl.task_engines.pretraining.engine import make_pretrain_engine

RECIPES_SUBDIR = "run_recipes/pretrain"


def _resolve_recipe(script: Path, recipe: str) -> Path:
    direct = Path(recipe)
    if direct.is_file():
        return direct.resolve()
    tree_root = script.parent.parent  # scripts/ -> project tree root
    under = tree_root / RECIPES_SUBDIR / recipe
    if under.is_file():
        return under.resolve()
    raise FileNotFoundError(
        f"recipe not found as {recipe!r} (cwd) nor under {tree_root / RECIPES_SUBDIR}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ccf_tabformer MFM pretraining.")
    parser.add_argument("recipe", help="Recipe JSON path, or a name under run_recipes/pretrain/.")
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
