"""Full-run construction. Train split fit, test split as val, seeded val mask."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.dl_engine.dl_engine_utils.batches import mapping_to_device
from dl_toolbox.dl_engine.exp_engine.train_val_engine.experiment import (
    TrainValEvalParts,
    TrainValFitParts,
)
from dl_toolbox.dl_engine.metrics.metric_object import LEVEL_EPOCH, metric_log_name
from dl_toolbox.dl_engine.runs.common.run_config import MonitorConfig, OptimizationConfig
from dl_toolbox.dl_engine.runs.eval_run.eval_config import (
    EvalContext,
    EvalLoaders,
    LogItem,
)
from dl_toolbox.dl_engine.runs.train_val_run.train_val_config import (
    TrainValLoaders,
    TrainValMetrics,
)

from ccf_tabformer_dl.tabformer_components.embedding.oracle import build_embedder
from ccf_tabformer_dl.tabformer_components.encoder.oracle import build_encoder
from ccf_tabformer_dl.tabformer_components.head.oracle import build_head
from ccf_tabformer_dl.tabformer_components.masking.oracle import build_masker
from ccf_tabformer_dl.tabformer_components.mfm_model import build_mfm_model
from ccf_tabformer_dl.tabformer_components.reconstruction.steps import (
    BCE,
    recon_accuracy_metric,
    recon_bce_metric,
    reconstruction_fit_steps,
    reconstruction_train_val_fit_steps,
)
from ccf_tabformer_dl.tabformer_components.table_spec import FrozenTableSpec
from ccf_tabformer_dl.task_engines.full_run.config import FullRunConfigParser
from ccf_tabformer_dl.task_engines.full_run.data import (
    build_train_val_loaders,
    table_spec_from_inner_table,
)

FULL_RUN_EVAL_LOADER_NAME = "val"
EPOCH_REDUCERS = ("mean", "std", "min", "max")
VAL_MASK_SEED = 0
INSPECT_MASK_SEED = 0


class FullRunExperiment:
    def build_fit(
        self, experiment_config: Mapping[str, Any], *, project_root: object
    ) -> TrainValFitParts:
        del project_root
        parsed = FullRunConfigParser(dict(experiment_config))
        spec = table_spec_from_inner_table(parsed.data.inner_table)
        train_loader, val_loader = build_train_val_loaders(parsed.data)
        return assemble_full_run_train_val(
            parsed=parsed,
            spec=spec,
            train_loader=train_loader,
            val_loader=val_loader,
        )

    def build_eval(
        self,
        experiment_config: Mapping[str, Any],
        evaluate: dict[str, Any],
        *,
        project_root: Directory,
        payload_dir: Directory,
    ) -> TrainValEvalParts:
        del evaluate, project_root
        parsed = FullRunConfigParser(dict(experiment_config))
        _train_loader, val_loader = build_train_val_loaders(parsed.data)
        del _train_loader

        def evaluate_fn(context: EvalContext) -> Sequence[LogItem]:
            return _evaluate_full_run_recon(
                context,
                parsed=parsed,
                payload_dir=payload_dir,
            )

        return TrainValEvalParts(
            loaders=EvalLoaders(named={FULL_RUN_EVAL_LOADER_NAME: val_loader}),
            evaluate_fn=evaluate_fn,
        )


def assemble_full_run_train_val(
    *,
    parsed: FullRunConfigParser,
    spec: FrozenTableSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> TrainValFitParts:
    if not isinstance(train_loader, DataLoader):
        raise TypeError(
            f"train_loader must be DataLoader, got {type(train_loader)!r}"
        )
    if not isinstance(val_loader, DataLoader):
        raise TypeError(
            f"val_loader must be DataLoader, got {type(val_loader)!r}"
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

    metric_specs = [
        (recon_accuracy_metric(), (), EPOCH_REDUCERS),
        (recon_bce_metric(), (), EPOCH_REDUCERS),
    ]
    monitor_key = metric_log_name(BCE, "val", LEVEL_EPOCH, "mean")

    return TrainValFitParts(
        net=net,
        loaders=TrainValLoaders(train=train_loader, val=val_loader),
        steps=reconstruction_train_val_fit_steps(
            masker=masker, val_mask_seed=VAL_MASK_SEED
        ),
        metrics=TrainValMetrics(
            train_metrics=metric_specs,
            val_metrics=metric_specs,
            monitor=MonitorConfig(key=monitor_key, mode="min"),
        ),
        optimization=OptimizationConfig(
            optimizer=optimizer,
            scheduler=OptimizationConfig.ABSENT,
        ),
    )


def _evaluate_full_run_recon(
    context: EvalContext,
    *,
    parsed: FullRunConfigParser,
    payload_dir: Directory,
) -> list[LogItem]:
    spec = table_spec_from_inner_table(parsed.data.inner_table)
    masker = build_masker(
        strategy=parsed.masking.strategy, args=parsed.masking.args, spec=spec
    )
    generator = torch.Generator()
    generator.manual_seed(INSPECT_MASK_SEED)
    steps = reconstruction_fit_steps(masker=masker, generator=generator)
    loader = context.loaders.named[FULL_RUN_EVAL_LOADER_NAME]
    total_loss = 0.0
    total_acc = 0.0
    total_bce = 0.0
    n_items = 0
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise TypeError(
                    f"full_run batch must be a mapping, got {type(batch)!r}"
                )
            prepared = steps.prepare_tensors_fn(
                mapping_to_device(mapping=batch, device=context.device)
            )
            output = steps.forward_fn(context.net, prepared)
            if not isinstance(output, Tensor):
                raise TypeError(
                    f"full_run forward must return a Tensor, got {type(output)!r}"
                )
            loss = steps.loss_fn(output, prepared)
            metric_inputs = steps.build_metric_inputs_fn(output, prepared)
            acc_items = recon_accuracy_metric().metric_function(metric_inputs)
            bce_items = recon_bce_metric().metric_function(metric_inputs)
            n_batch = int(acc_items.shape[0])
            total_loss += float(loss.detach().cpu().item()) * n_batch
            total_acc += float(acc_items.sum().detach().cpu().item())
            total_bce += float(bce_items.sum().detach().cpu().item())
            n_items += n_batch
    if n_items < 1:
        raise RuntimeError("full_run inspect loader yielded no masked positions")
    mean_loss = total_loss / float(n_items)
    mean_acc = total_acc / float(n_items)
    mean_bce = total_bce / float(n_items)
    (payload_dir.path / "recon_loss.txt").write_text(
        f"{mean_loss}\n", encoding="utf-8"
    )
    (payload_dir.path / "recon_acc.txt").write_text(
        f"{mean_acc}\n", encoding="utf-8"
    )
    (payload_dir.path / "bce.txt").write_text(f"{mean_bce}\n", encoding="utf-8")
    return [
        LogItem.scalars(
            step=context.step,
            values={
                "inspect/recon_loss": mean_loss,
                "inspect/recon_acc": mean_acc,
                "inspect/bce": mean_bce,
            },
        )
    ]
