"""Train one fine-tune recipe (one imported fold).

  python scripts/fine_tune_classification/finetune.py fivefold.json --checkpoint path/to/full_run.ckpt
  python scripts/fine_tune_classification/finetune.py pilot_fold0_random.json --force
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from dl_toolbox.project_layout.project_layout import TrackingBackend

from ccf_tabformer_dl.project import resolve_finetune_recipe
from ccf_tabformer_dl.task_engines.finetune.engine import make_finetune_engine


def _apply_overrides(
    recipe_path: Path, *, checkpoint: str | None, fold: int | None
) -> Path:
    if checkpoint is None and fold is None:
        return recipe_path
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    experiment_config = payload["experiment_config"]
    if checkpoint is not None:
        experiment_config["pretrained_checkpoint"] = checkpoint
    if fold is not None:
        experiment_config["data"]["fold"] = fold
        payload["human_readable_name"] = (
            f"{payload['human_readable_name']}_fold{fold}"
        )
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(payload, handle, indent=2)
    handle.close()
    return Path(handle.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ccf_tabformer fine-tune recipe on one fold."
    )
    parser.add_argument(
        "recipe",
        help="Recipe JSON path or filename under run_recipes/finetune.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override experiment_config.pretrained_checkpoint.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Override experiment_config.data.fold.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite this human_readable_name, then fit.",
    )
    parser.add_argument(
        "--tracking-backend",
        default=None,
        choices=sorted(TrackingBackend.BACKENDS),
        help=(
            "Override .tracking-backend for this process. Does not rewrite "
            "the file. Omit to use the project default (mlflow)."
        ),
    )
    args = parser.parse_args()
    script = Path(__file__).resolve()
    recipe = resolve_finetune_recipe(script, args.recipe)
    recipe = _apply_overrides(
        recipe, checkpoint=args.checkpoint, fold=args.fold
    )
    make_finetune_engine(
        script,
        recipe,
        tracking_backend=args.tracking_backend,
        force=args.force,
    ).train()


if __name__ == "__main__":
    main()
