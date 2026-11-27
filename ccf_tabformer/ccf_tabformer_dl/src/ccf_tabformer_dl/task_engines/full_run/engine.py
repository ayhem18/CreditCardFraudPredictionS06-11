"""Full-run engine. TrainVal Lightning fit on the frozen inner_table splits."""

from __future__ import annotations

from pathlib import Path

from dl_toolbox.base.file_management.file import File
from dl_toolbox.dl_engine.exp_engine.train_val_engine.default_layout import (
    DefaultLayoutTrainValEngine,
)
from dl_toolbox.dl_engine.exp_engine.train_val_engine.train_val_engine import (
    EnginePurpose,
    IdentityOverwritePolicy,
)

from ccf_tabformer_dl.project import load_project_layout
from ccf_tabformer_dl.task_engines.full_run.experiment import FullRunExperiment


def make_full_run_engine(
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
        experiment=FullRunExperiment(),
        force=(
            IdentityOverwritePolicy.OVERWRITE_NAME
            if force
            else IdentityOverwritePolicy.OFF
        ),
    )
