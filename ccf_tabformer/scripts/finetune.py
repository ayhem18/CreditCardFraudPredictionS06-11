"""Run one fold's fine-tune from a recipe.

  python scripts/finetune.py <recipe.json>
  python scripts/finetune.py <name>                       # under run_recipes/finetune/
  python scripts/finetune.py <name> --checkpoint <ckpt>   # override pretrained_checkpoint
  python scripts/finetune.py <name> --fold 2 --force      # override the fold

``--checkpoint`` lets the three pilot regimes share one pretrained .ckpt whose
hashed filename is only known after pretraining, without editing the JSON.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from ccf_tabformer_dl.task_engines.finetune.engine import make_finetune_engine

RECIPES_SUBDIR = "run_recipes/finetune"


def _resolve_recipe(script: Path, recipe: str) -> Path:
    direct = Path(recipe)
    if direct.is_file():
        return direct.resolve()
    under = script.parent.parent / RECIPES_SUBDIR / recipe
    if under.is_file():
        return under.resolve()
    raise FileNotFoundError(
        f"recipe not found as {recipe!r} (cwd) nor under {script.parent.parent / RECIPES_SUBDIR}"
    )


def _apply_overrides(recipe_path: Path, *, checkpoint: str | None, fold: int | None) -> Path:
    if checkpoint is None and fold is None:
        return recipe_path
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    if checkpoint is not None:
        payload["experiment_config"]["pretrained_checkpoint"] = checkpoint
    if fold is not None:
        payload["experiment_config"]["fold"] = fold
        payload["human_readable_name"] = f"{payload['human_readable_name']}_fold{fold}"
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(payload, handle, indent=2)
    handle.close()
    return Path(handle.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ccf_tabformer fine-tune for one fold.")
    parser.add_argument("recipe", help="Recipe JSON path, or a name under run_recipes/finetune/.")
    parser.add_argument("--checkpoint", default=None, help="Override pretrained_checkpoint.")
    parser.add_argument("--fold", type=int, default=None, help="Override the fold id.")
    parser.add_argument("--force", action="store_true", help="Overwrite this run name, then fit.")
    parser.add_argument("--tracking-backend", default=None, choices=["mlflow", "tensorboard"])
    args = parser.parse_args()
    script = Path(__file__).resolve()
    recipe = _resolve_recipe(script, args.recipe)
    recipe = _apply_overrides(recipe, checkpoint=args.checkpoint, fold=args.fold)
    make_finetune_engine(
        script, recipe, tracking_backend=args.tracking_backend, force=args.force
    ).train()


if __name__ == "__main__":
    main()
