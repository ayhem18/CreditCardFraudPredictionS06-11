"""Write 5 fold recipes from a starting recipe, train them, inspect, assemble OOF.

  python scripts/fine_tune_classification/wrapper_finetune.py fivefold.json --checkpoint path/to/full_run.ckpt
  python scripts/fine_tune_classification/wrapper_finetune.py fivefold.json --checkpoint path/to.ckpt --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from dl_toolbox.base.file_management.file import File
from dl_toolbox.base.input_output.json_io import read_object
from dl_toolbox.base.parsing.parse_numerical import parse_positive_int
from dl_toolbox.dl_engine.dl_engine_utils.visible_gpus import (
    gib_to_mib,
    query_free_gpu_indices,
)
from dl_toolbox.dl_engine.exp_engine.grid_search import run_recipes_on_free_gpus
from dl_toolbox.project_layout.project_layout import TrackingBackend

from ccf_tabformer_dl.data_contract.folds.collection import FoldsCollection
from ccf_tabformer_dl.project import resolve_finetune_recipe
from ccf_tabformer_dl.task_engines.finetune.config import FineTuneConfigParser
from ccf_tabformer_dl.task_engines.finetune.engine import make_finetune_inspect_engine
from ccf_tabformer_dl.task_engines.finetune.oof import (
    assemble_oof,
    average_test_scores,
    inspection_payload_directory,
    write_oof_csv,
    write_submission_csv,
)
from ccf_tabformer_dl.task_engines.finetune.wrapper.persist import write_fold_recipes

MIN_FREE_GIB = 4


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write one recipe per imported fold, train each on a free GPU, "
            "inspect last checkpoints, then write OOF AUC and a mean-of-folds "
            "test submission."
        )
    )
    parser.add_argument(
        "starting_recipe",
        help=(
            "Recipe JSON path or filename under "
            "run_recipes/classification_fine_tune/full_run."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override pretrained_checkpoint on every fold recipe.",
    )
    parser.add_argument(
        "--inspect-recipe",
        default="inspect_last.json",
        help=(
            "Inspect recipe JSON under "
            "run_recipes/classification_fine_tune/full_run."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to each finetune train process.",
    )
    parser.add_argument(
        "--min-free-gib",
        type=int,
        default=MIN_FREE_GIB,
        help="nvidia-smi free-memory floor in GiB for a GPU to be used.",
    )
    parser.add_argument(
        "--eval-device",
        default=None,
        help="Device for inspect after train. Default: cuda if available.",
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
    starting_path = resolve_finetune_recipe(script, args.starting_recipe)
    starting = read_object(file=File.from_absolute(path=starting_path))
    parsed = FineTuneConfigParser(dict(starting["experiment_config"]))
    fold_set = FoldsCollection.load(parsed.data.folds)
    recipes = write_fold_recipes(
        start=script,
        starting_recipe=starting_path,
        n_splits=fold_set.n_splits(),
        pretrained_checkpoint=args.checkpoint,
    )
    n_recipes = len(recipes)
    for i in range(n_recipes):
        print(f"fold recipe: {recipes[i].name}", flush=True)
    print(f"min_free_gib: {min_free_gib}", flush=True)
    run_recipes_on_free_gpus(
        recipes=recipes,
        train_script=script.parent / "finetune.py",
        force=args.force,
        tracking_backend=args.tracking_backend,
        python=sys.executable,
        query_free=lambda: query_free_gpu_indices(min_free_mib=min_free_mib),
        popen=subprocess.Popen,
    )
    inspect_recipe = resolve_finetune_recipe(script, args.inspect_recipe)
    if args.eval_device is None:
        eval_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        eval_device = torch.device(args.eval_device)
    payloads: list[Path] = []
    for i in range(n_recipes):
        engine = make_finetune_inspect_engine(
            script,
            recipes[i],
            inspect_recipe,
            tracking_backend=args.tracking_backend,
            eval_device=eval_device,
        )
        engine.inspect()
        payloads.append(
            inspection_payload_directory(engine.get_artifacts_directory().path)
        )
    assembled = assemble_oof(fold_payloads=payloads)
    test_ids, test_scores = average_test_scores(fold_payloads=payloads)
    out_dir = recipes[0].parent
    write_oof_csv(assembled, dest=out_dir / "oof.csv")
    write_submission_csv(
        row_ids=test_ids, scores=test_scores, dest=out_dir / "submission.csv"
    )
    metrics = {
        "n_folds": assembled["n_folds"],
        "n_oof": assembled["n_oof"],
        "oof_auc": assembled["oof_auc"],
        "per_fold_auc": assembled["per_fold_auc"],
        "stack_oof_auc": assembled["stack_oof_auc"],
        "oof_minus_stack": assembled["oof_minus_stack"],
        "n_test": int(test_ids.shape[0]),
    }
    (out_dir / "oof_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"OOF AUC {assembled['oof_auc']:.5f} "
        f"(stack {assembled['stack_oof_auc']:.5f}, "
        f"delta {assembled['oof_minus_stack']:+.5f})",
        flush=True,
    )
    print(f"wrote {out_dir / 'oof.csv'}", flush=True)
    print(f"wrote {out_dir / 'submission.csv'}", flush=True)


if __name__ == "__main__":
    main()
