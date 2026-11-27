"""Factory for the MFM pretraining engine (train-only, default layout)."""

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
from ccf_tabformer_dl.task_engines.pretraining.experiment import PretrainExperiment


def make_pretrain_engine(
    start: Path,
    recipe: Path,
    *,
    tracking_backend: str | None = None,
    force: bool = False,
) -> DefaultLayoutTrainOnlyEngine:
    # Everything in this file is a wrapper around the existing engine code with the exception of PretrainExperiment()
    # that's what needs to be understood

    layout = load_project_layout(start=start, tracking_backend=tracking_backend)

    return DefaultLayoutTrainOnlyEngine(
        purpose=EnginePurpose.FIT,
        layout=layout,
        train_recipe=File.from_absolute(path=recipe),

        experiment=PretrainExperiment(),
        force=(
            IdentityOverwritePolicy.OVERWRITE_NAME
            if force
            else IdentityOverwritePolicy.OFF
        ),
    )
