"""Train an overfit sanity-check recipe on a frozen row_subset.

  python scripts/unsupervised_reconstruction/overfit_train.py mfm_overfit_alarm_label_1k.json
  python scripts/unsupervised_reconstruction/overfit_train.py mfm_overfit_alarm_label_1k.json --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dl_toolbox.project_layout.project_layout import TrackingBackend

from ccf_tabformer_dl.project import resolve_overfit_recipe
from ccf_tabformer_dl.task_engines.overfit.engine import make_overfit_engine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ccf_tabformer overfit sanity-check recipe."
    )
    parser.add_argument(
        "recipe",
        help=(
            "Recipe JSON path or filename under "
            "run_recipes/supervised_reconstruction/overfit_grid_search."
        ),
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
    recipe = resolve_overfit_recipe(script, args.recipe)
    make_overfit_engine(
        script,
        recipe,
        tracking_backend=args.tracking_backend,
        force=args.force,
    ).train()


if __name__ == "__main__":
    main()
