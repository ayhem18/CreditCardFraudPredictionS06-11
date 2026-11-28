"""Inspect a finished fine-tune run: write val/test logits under inspection_runs.

  python scripts/fine_tune_classification/finetune_inspect.py mfm_finetune_pretrained_fold0.json
  python scripts/fine_tune_classification/finetune_inspect.py fold_0.json --inspect-recipe inspect_last.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dl_toolbox.project_layout.project_layout import TrackingBackend

from ccf_tabformer_dl.project import resolve_finetune_recipe
from ccf_tabformer_dl.task_engines.finetune.engine import make_finetune_inspect_engine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a finished ccf_tabformer fine-tune recipe."
    )
    parser.add_argument(
        "recipe",
        help=(
            "Fit recipe JSON path or filename under "
            "run_recipes/classification_fine_tune/full_run."
        ),
    )
    parser.add_argument(
        "--inspect-recipe",
        default="inspect_last.json",
        help=(
            "Inspect recipe JSON under "
            "run_recipes/classification_fine_tune/full_run (selector last)."
        ),
    )
    parser.add_argument(
        "--eval-device",
        default=None,
        help="torch device string, e.g. cuda or cpu. Default: cuda if available.",
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
    inspect_recipe = resolve_finetune_recipe(script, args.inspect_recipe)
    if args.eval_device is None:
        eval_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        eval_device = torch.device(args.eval_device)
    make_finetune_inspect_engine(
        script,
        recipe,
        inspect_recipe,
        tracking_backend=args.tracking_backend,
        eval_device=eval_device,
    ).inspect()


if __name__ == "__main__":
    main()
