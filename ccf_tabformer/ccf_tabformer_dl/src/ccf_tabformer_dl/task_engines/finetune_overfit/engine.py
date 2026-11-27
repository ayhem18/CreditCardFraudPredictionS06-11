"""Classification overfit engine. TrainOnly Lightning fit on a frozen subset."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.file_management.file import File
from dl_toolbox.dl_engine.exp_engine.train_only_engine.default_layout import (
    DefaultLayoutTrainOnlyEngine,
)
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    EnginePurpose,
    IdentityOverwritePolicy,
)

from ccf_tabformer_dl.project import load_project_layout
from ccf_tabformer_dl.task_engines.finetune_overfit.experiment import (
    FineTuneOverfitExperiment,
)


def make_finetune_overfit_engine(
    start: Path,
    recipe: Path,
    *,
    tracking_backend: str | None = None,
    force: bool = False,
) -> DefaultLayoutTrainOnlyEngine:
    layout = load_project_layout(start=start, tracking_backend=tracking_backend)
    return DefaultLayoutTrainOnlyEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        train_recipe=File.from_absolute(path=recipe),
        experiment=FineTuneOverfitExperiment(),
        force=(
            IdentityOverwritePolicy.OVERWRITE_NAME
            if force
            else IdentityOverwritePolicy.OFF
        ),
    )
