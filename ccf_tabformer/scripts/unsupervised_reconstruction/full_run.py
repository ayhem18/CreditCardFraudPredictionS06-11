"""Train a full-run recipe on the frozen inner_table (val = test split).

  python scripts/unsupervised_reconstruction/full_run.py mfm_full_run_fixed_width_v1.json
  python scripts/unsupervised_reconstruction/full_run.py mfm_full_run_fixed_width_v1.json --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dl_toolbox.project_layout.project_layout import TrackingBackend

from ccf_tabformer_dl.project import resolve_full_run_recipe
from ccf_tabformer_dl.task_engines.full_run.engine import make_full_run_engine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ccf_tabformer TrainVal full-run recipe."
    )
    parser.add_argument(
        "recipe",
        help="Recipe JSON path or filename under run_recipes/full_run.",
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
    recipe = resolve_full_run_recipe(script, args.recipe)
    make_full_run_engine(
        script,
        recipe,
        tracking_backend=args.tracking_backend,
        force=args.force,
    ).train()


if __name__ == "__main__":
    main()
