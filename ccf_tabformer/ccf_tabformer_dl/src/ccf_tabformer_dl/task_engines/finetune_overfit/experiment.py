"""Classification overfit construction. One row_subset loader, train-only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
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

from ccf_tabformer_dl.tabformer_components.backbone import load_backbone_from_checkpoint
from ccf_tabformer_dl.tabformer_components.classifier_head.oracle import (
    build_classifier_head,
)
from ccf_tabformer_dl.tabformer_components.classifier_model import build_classifier_model
from ccf_tabformer_dl.tabformer_components.embedding.oracle import build_embedder
from ccf_tabformer_dl.tabformer_components.encoder.oracle import build_encoder
from ccf_tabformer_dl.tabformer_components.table_spec import FrozenTableSpec
from ccf_tabformer_dl.task_engines.finetune.steps import (
    AUC,
    AUC_REDUCER,
    BCE,
    EPOCH_REDUCERS,
    auc_metric,
    classification_bce_metric,
    classification_fit_steps,
)
from ccf_tabformer_dl.task_engines.finetune_overfit.config import FineTuneOverfitConfigParser
from ccf_tabformer_dl.task_engines.finetune_overfit.data import (
    build_finetune_overfit_loader,
    table_spec_from_subset,
)

OVERFIT_EVAL_LOADER_NAME = "overfit"


class FineTuneOverfitExperiment:
    def build_train_only(
        self, experiment_config: Mapping[str, Any], *, project_root: object
    ) -> TrainOnlyFitParts:
        parsed = FineTuneOverfitConfigParser(dict(experiment_config))
        spec = table_spec_from_subset(parsed.data.row_subset)
        loader = build_finetune_overfit_loader(parsed.data)
        return assemble_finetune_overfit_train_only(
            parsed=parsed,
            spec=spec,
            train_loader=loader,
            project_root=project_root,
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
        parsed = FineTuneOverfitConfigParser(dict(experiment_config))
        loader = build_finetune_overfit_loader(parsed.data)

        def evaluate_fn(context: EvalContext) -> Sequence[LogItem]:
            return _evaluate_finetune_overfit(
                context,
                payload_dir=payload_dir,
            )

        return TrainOnlyEvalParts(
            loaders=EvalLoaders(named={OVERFIT_EVAL_LOADER_NAME: loader}),
            evaluate_fn=evaluate_fn,
        )


def assemble_finetune_overfit_train_only(
    *,
    parsed: FineTuneOverfitConfigParser,
    spec: FrozenTableSpec,
    train_loader: DataLoader,
    project_root: object,
) -> TrainOnlyFitParts:
    if not isinstance(train_loader, DataLoader):
        raise TypeError(
            f"train_loader must be DataLoader, got {type(train_loader)!r}"
        )
    net = build_classifier_model(
        embedder=build_embedder(
            strategy=parsed.embedding.strategy,
            args=parsed.embedding.args,
            spec=spec,
        ),
        encoder=build_encoder(
            strategy=parsed.encoder.strategy, args=parsed.encoder.args
        ),
        classifier_head=build_classifier_head(
            strategy=parsed.classifier_head.strategy,
            args=parsed.classifier_head.args,
        ),
    )
    if parsed.load_backbone():
        checkpoint = _resolve_checkpoint(
            parsed.pretrained_checkpoint, project_root=project_root
        )
        load_backbone_from_checkpoint(
            net, checkpoint_path=checkpoint, map_location=torch.device("cpu")
        )
    net.set_encoder_frozen(frozen=parsed.freeze_encoder())
    trainable = [param for param in net.parameters() if param.requires_grad]
    if len(trainable) == 0:
        raise ValueError("fine-tune overfit net has no trainable parameters")
    optimizer = AdamW(
        trainable,
        lr=parsed.lr.learning_rate,
        weight_decay=parsed.optimizer.weight_decay,
        betas=parsed.optimizer.betas,
    )
    monitor_key = metric_log_name(AUC, "train", LEVEL_EPOCH, AUC_REDUCER)
    return TrainOnlyFitParts(
        net=net,
        loaders=TrainOnlyLoaders(train=train_loader),
        steps=classification_fit_steps(),
        metrics=TrainOnlyMetrics(
            train_metrics=[
                (auc_metric(), (), (AUC_REDUCER,)),
                (classification_bce_metric(), (), EPOCH_REDUCERS),
            ],
            monitor=MonitorConfig(key=monitor_key, mode="max"),
        ),
        optimization=OptimizationConfig(
            optimizer=optimizer,
            scheduler=OptimizationConfig.ABSENT,
        ),
    )


def _resolve_checkpoint(text: str, *, project_root: object) -> Path:
    path = Path(text)
    if path.is_absolute():
        resolved = path
    elif isinstance(project_root, Directory):
        resolved = project_root.path / path
    else:
        resolved = Path(str(project_root)) / path
    if not resolved.is_file():
        raise FileNotFoundError(f"pretrained checkpoint not found: {resolved}")
    return resolved.resolve()


def _evaluate_finetune_overfit(
    context: EvalContext, *, payload_dir: Directory
) -> list[LogItem]:
    steps = classification_fit_steps()
    auc = auc_metric()
    bce = classification_bce_metric()
    loader = context.loaders.named[OVERFIT_EVAL_LOADER_NAME]
    pairs: Tensor | None = None
    bce_items: Tensor | None = None
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise TypeError(
                    f"finetune overfit batch must be a mapping, got {type(batch)!r}"
                )
            prepared = steps.prepare_tensors_fn(
                mapping_to_device(mapping=batch, device=context.device)
            )
            output = steps.forward_fn(context.net, prepared)
            if not isinstance(output, Tensor):
                raise TypeError(
                    f"classifier forward must return a Tensor, got {type(output)!r}"
                )
            metric_inputs = steps.build_metric_inputs_fn(output, prepared)
            batch_pairs = auc.metric_function(metric_inputs)
            batch_bce = bce.metric_function(metric_inputs)
            if pairs is None:
                pairs = batch_pairs
                bce_items = batch_bce
            else:
                pairs = auc.combiner(pairs, batch_pairs)
                bce_items = bce.combiner(bce_items, batch_bce)
    if pairs is None or bce_items is None:
        raise RuntimeError("finetune overfit inspect loader was empty")
    auc_value = float(auc.reducers[AUC_REDUCER](pairs).detach().cpu().item())
    bce_value = float(bce.reducers["mean"](bce_items).detach().cpu().item())
    (payload_dir.path / "auc.txt").write_text(f"{auc_value}\n", encoding="utf-8")
    (payload_dir.path / "bce.txt").write_text(f"{bce_value}\n", encoding="utf-8")
    return [
        LogItem.scalars(
            step=context.step,
            values={"inspect/auc": auc_value, "inspect/bce": bce_value},
        )
    ]
