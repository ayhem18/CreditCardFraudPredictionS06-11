"""Overfit construction. One row_subset loader, train-only reconstruction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.dl_engine.dl_engine_utils.batches import mapping_to_device
from dl_toolbox.dl_engine.exp_engine.train_only_engine.experiment import (
    TrainOnlyEvalParts,
    TrainOnlyFitParts,
)
from dl_toolbox.dl_engine.metrics.metric_object import LEVEL_EPOCH, metric_log_name
from dl_toolbox.dl_engine.runs.common.run_config import MonitorConfig, OptimizationConfig
from dl_toolbox.dl_engine.runs.eval_run.eval_config import (
    EvalContext,
    EvalLoaders,
    LogItem,
)
from dl_toolbox.dl_engine.runs.train_only_run.train_only_config import (
    TrainOnlyLoaders,
    TrainOnlyMetrics,
)

from ccf_tabformer_dl.tabformer_components.embedding.oracle import build_embedder
from ccf_tabformer_dl.tabformer_components.encoder.oracle import build_encoder
from ccf_tabformer_dl.tabformer_components.head.oracle import build_head
from ccf_tabformer_dl.tabformer_components.masking.oracle import build_masker
from ccf_tabformer_dl.tabformer_components.mfm_model import build_mfm_model
from ccf_tabformer_dl.tabformer_components.reconstruction.steps import (
    RECON_ACC,
    recon_accuracy_metric,
    reconstruction_fit_steps,
)
from ccf_tabformer_dl.tabformer_components.table_spec import FrozenTableSpec
from ccf_tabformer_dl.task_engines.overfit.config import OverfitConfigParser
from ccf_tabformer_dl.task_engines.overfit.data import (
    build_overfit_loader,
    table_spec_from_subset,
)

OVERFIT_EVAL_LOADER_NAME = "overfit"
EPOCH_REDUCERS = ("mean", "std", "min", "max")
INSPECT_MASK_SEED = 0


class OverfitExperiment:
    def build_train_only(
        self, experiment_config: Mapping[str, Any], *, project_root: object
    ) -> TrainOnlyFitParts:
        del project_root
        parsed = OverfitConfigParser(dict(experiment_config))
        spec = table_spec_from_subset(parsed.data.row_subset)
        loader = build_overfit_loader(parsed.data)
        return assemble_overfit_train_only(
            parsed=parsed, spec=spec, train_loader=loader
        )

    def build_eval(
        self,
        experiment_config: Mapping[str, Any],
        evaluate: dict[str, Any],
        *,
        project_root: Directory,
        payload_dir: Directory,
    ) -> TrainOnlyEvalParts:
        del evaluate, project_root
        parsed = OverfitConfigParser(dict(experiment_config))
        loader = build_overfit_loader(parsed.data)

        def evaluate_fn(context: EvalContext) -> Sequence[LogItem]:
            return _evaluate_overfit_recon(
                context,
                parsed=parsed,
                payload_dir=payload_dir,
            )

        return TrainOnlyEvalParts(
            loaders=EvalLoaders(named={OVERFIT_EVAL_LOADER_NAME: loader}),
            evaluate_fn=evaluate_fn,
        )


def assemble_overfit_train_only(
    *,
    parsed: OverfitConfigParser,
    spec: FrozenTableSpec,
    train_loader: DataLoader,
) -> TrainOnlyFitParts:
    if not isinstance(train_loader, DataLoader):
        raise TypeError(
            f"train_loader must be DataLoader, got {type(train_loader)!r}"
        )
    masker = build_masker(
        strategy=parsed.masking.strategy, args=parsed.masking.args, spec=spec
    )
    net = build_mfm_model(
        embedder=build_embedder(
            strategy=parsed.embedding.strategy,
            args=parsed.embedding.args,
            spec=spec,
        ),
        encoder=build_encoder(
            strategy=parsed.encoder.strategy, args=parsed.encoder.args
        ),
        head=build_head(
            strategy=parsed.head.strategy, args=parsed.head.args, spec=spec
        ),
    )
    optimizer = AdamW(
        net.parameters(),
        lr=parsed.lr.learning_rate,
        weight_decay=parsed.optimizer.weight_decay,
        betas=parsed.optimizer.betas,
    )
    monitor_key = metric_log_name(RECON_ACC, "train", LEVEL_EPOCH, "mean")
    return TrainOnlyFitParts(
        net=net,
        loaders=TrainOnlyLoaders(train=train_loader),
        steps=reconstruction_fit_steps(masker=masker, generator=None),
        metrics=TrainOnlyMetrics(
            train_metrics=[
                (recon_accuracy_metric(), (), EPOCH_REDUCERS),
            ],
            monitor=MonitorConfig(key=monitor_key, mode="max"),
        ),
        optimization=OptimizationConfig(
            optimizer=optimizer,
            scheduler=OptimizationConfig.ABSENT,
        ),
    )


def _evaluate_overfit_recon(
    context: EvalContext,
    *,
    parsed: OverfitConfigParser,
    payload_dir: Directory,
) -> list[LogItem]:
    spec = table_spec_from_subset(parsed.data.row_subset)
    masker = build_masker(
        strategy=parsed.masking.strategy, args=parsed.masking.args, spec=spec
    )
    generator = torch.Generator()
    generator.manual_seed(INSPECT_MASK_SEED)
    steps = reconstruction_fit_steps(masker=masker, generator=generator)
    loader = context.loaders.named[OVERFIT_EVAL_LOADER_NAME]
    total_loss = 0.0
    total_acc = 0.0
    n_items = 0
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise TypeError(
                    f"overfit batch must be a mapping, got {type(batch)!r}"
                )
            prepared = steps.prepare_tensors_fn(
                mapping_to_device(mapping=batch, device=context.device)
            )
            output = steps.forward_fn(context.net, prepared)
            if not isinstance(output, Tensor):
                raise TypeError(
                    f"overfit forward must return a Tensor, got {type(output)!r}"
                )
            loss = steps.loss_fn(output, prepared)
            metric_inputs = steps.build_metric_inputs_fn(output, prepared)
            acc_items = recon_accuracy_metric().metric_function(metric_inputs)
            n_batch = int(acc_items.shape[0])
            total_loss += float(loss.detach().cpu().item()) * n_batch
            total_acc += float(acc_items.sum().detach().cpu().item())
            n_items += n_batch
    if n_items < 1:
        raise RuntimeError("overfit inspect loader yielded no masked positions")
    mean_loss = total_loss / float(n_items)
    mean_acc = total_acc / float(n_items)
    (payload_dir.path / "recon_loss.txt").write_text(
        f"{mean_loss}\n", encoding="utf-8"
    )
    (payload_dir.path / "recon_acc.txt").write_text(
        f"{mean_acc}\n", encoding="utf-8"
    )
    return [
        LogItem.scalars(
            step=context.step,
            values={
                "inspect/recon_loss": mean_loss,
                "inspect/recon_acc": mean_acc,
            },
        )
    ]
