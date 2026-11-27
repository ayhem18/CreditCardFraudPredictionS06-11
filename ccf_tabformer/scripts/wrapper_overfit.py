"""Write an overfit grid from a grid-search document, then train each member.

  python scripts/wrapper_overfit.py alarm_label_1k_lr.json
  python scripts/wrapper_overfit.py alarm_label_1k_lr.json --force
  python scripts/wrapper_overfit.py alarm_label_1k_lr.json --min-free-gib 20
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import read_object
from dl_toolbox.base.parsing.parse_numerical import parse_positive_int
from dl_toolbox.dl_engine.dl_engine_utils.visible_gpus import (
    gib_to_mib,
    query_free_gpu_indices,
)
from dl_toolbox.dl_engine.exp_engine.grid_search import (
    parse_grid_search_document,
    run_recipes_on_free_gpus,
)
from dl_toolbox.project_layout.project_layout import TrackingBackend

from ccf_tabformer_dl.project import resolve_grid_search_document
from ccf_tabformer_dl.task_engines.overfit.wrapper import write_overfit_grid_search

MIN_FREE_GIB = 4


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read a grid-search document, write kept exp_i recipes under "
            "run_recipes/overfit_sanity_checks/grid_search_overfit/"
            "<grid_search_exp_name>/, then train each one on a free GPU."
        )
    )
    parser.add_argument(
        "grid_search",
        help=(
            "Grid-search JSON path or filename under "
            "run_recipes/overfit_sanity_checks/grid_search_overfit."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to each overfit train process.",
    )
    parser.add_argument(
        "--min-free-gib",
        type=int,
        default=MIN_FREE_GIB,
        help="nvidia-smi free-memory floor in GiB for a GPU to be used.",
    )
    parser.add_argument(
        "--tracking-backend",
        default=None,
        choices=sorted(TrackingBackend.BACKENDS),
        help=(
            "Override .tracking-backend for each child. Does not rewrite "
            "the file. Omit to use the project default (mlflow)."
        ),
    )
    args = parser.parse_args()
    min_free_gib = parse_positive_int(args.min_free_gib, label="min_free_gib")
    min_free_mib = gib_to_mib(min_free_gib=min_free_gib)
    script = Path(__file__).resolve()
    grid_path = resolve_grid_search_document(script, args.grid_search)
    document = parse_grid_search_document(
        read_object(file=File.from_absolute(path=grid_path))
    )
    recipes = write_overfit_grid_search(
        start=script, document=document, keep=None
    )
    n_recipes = len(recipes)
    for i in range(n_recipes):
        print(f"grid member: {recipes[i].name}", flush=True)
    print(f"min_free_gib: {min_free_gib}", flush=True)
    run_recipes_on_free_gpus(
        recipes=recipes,
        train_script=script.parent / "overfit_train.py",
        force=args.force,
        tracking_backend=args.tracking_backend,
        python=sys.executable,
        query_free=lambda: query_free_gpu_indices(min_free_mib=min_free_mib),
        popen=subprocess.Popen,
    )


if __name__ == "__main__":
    main()
