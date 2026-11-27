"""Fine-tune engine. TrainVal Lightning fit on one imported fold."""

from __future__ import annotations

from pathlib import Path

import torch

from dl_toolbox.base.file_management.file import File
from dl_toolbox.dl_engine.exp_engine.train_val_engine.default_layout import (
    DefaultLayoutTrainValEngine,
)
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    EnginePurpose,
    IdentityOverwritePolicy,
)

from ccf_tabformer_dl.project import load_project_layout
from ccf_tabformer_dl.task_engines.finetune.experiment import FineTuneExperiment


def make_finetune_engine(
    start: Path,
    recipe: Path,
    *,
    tracking_backend: str | None = None,
    force: bool = False,
) -> DefaultLayoutTrainValEngine:
    layout = load_project_layout(start=start, tracking_backend=tracking_backend)
    return DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        fit_recipe=File.from_absolute(path=recipe),
        experiment=FineTuneExperiment(),
        force=(
            IdentityOverwritePolicy.OVERWRITE_NAME
            if force
            else IdentityOverwritePolicy.OFF
        ),
    )


def make_finetune_inspect_engine(
    start: Path,
    recipe: Path,
    inspect_recipe: Path,
    *,
    tracking_backend: str | None = None,
    eval_device: torch.device,
) -> DefaultLayoutTrainValEngine:
    layout = load_project_layout(start=start, tracking_backend=tracking_backend)
    return DefaultLayoutTrainValEngine(
        purpose=EnginePurpose.INSPECT,
        layout=layout,
        fit_recipe=File.from_absolute(path=recipe),
        experiment=FineTuneExperiment(),
        inspect_recipe=File.from_absolute(path=inspect_recipe),
        eval_device=eval_device,
    )
