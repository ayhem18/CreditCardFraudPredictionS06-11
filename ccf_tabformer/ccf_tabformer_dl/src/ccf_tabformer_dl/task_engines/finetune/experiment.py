"""Fine-tune construction. Per-fold TrainVal, optional pretrained backbone."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
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

from ccf_tabformer_dl.data_contract.inner_table_dataset import LABEL_KEY, ROW_ID_KEY
from ccf_tabformer_dl.tabformer_components.backbone import load_backbone_from_checkpoint
from ccf_tabformer_dl.tabformer_components.classifier_head.oracle import (
    build_classifier_head,
)
from ccf_tabformer_dl.tabformer_components.classifier_model import build_classifier_model
from ccf_tabformer_dl.tabformer_components.embedding.oracle import build_embedder
from ccf_tabformer_dl.tabformer_components.encoder.oracle import build_encoder
from ccf_tabformer_dl.tabformer_components.reconstruction.steps import IDS_KEY
from ccf_tabformer_dl.tabformer_components.table_spec import FrozenTableSpec
from ccf_tabformer_dl.task_engines.finetune.config import FineTuneConfigParser
from ccf_tabformer_dl.task_engines.finetune.data import (
    build_eval_loaders,
    build_train_val_loaders,
    table_spec_from_inner_table,
)
from ccf_tabformer_dl.task_engines.finetune.steps import (
    AUC,
    AUC_REDUCER,
    auc_metric,
    classification_fit_steps,
)

EVAL_VAL_LOADER_NAME = "val"
EVAL_TEST_LOADER_NAME = "test"


class FineTuneExperiment:
    def build_fit(
        self, experiment_config: Mapping[str, Any], *, project_root: object
    ) -> TrainValFitParts:
        parsed = FineTuneConfigParser(dict(experiment_config))
        spec = table_spec_from_inner_table(parsed.data.inner_table)
        train_loader, val_loader = build_train_val_loaders(parsed.data)
        return assemble_finetune_train_val(
            parsed=parsed,
            spec=spec,
            train_loader=train_loader,
            val_loader=val_loader,
            project_root=project_root,
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
        parsed = FineTuneConfigParser(dict(experiment_config))
        val_loader, test_loader = build_eval_loaders(parsed.data)

        def evaluate_fn(context: EvalContext) -> Sequence[LogItem]:
            return _evaluate_finetune(
                context,
                payload_dir=payload_dir,
            )

        return TrainValEvalParts(
            loaders=EvalLoaders(
                named={
                    EVAL_VAL_LOADER_NAME: val_loader,
                    EVAL_TEST_LOADER_NAME: test_loader,
                }
            ),
            evaluate_fn=evaluate_fn,
        )


def assemble_finetune_train_val(
    *,
    parsed: FineTuneConfigParser,
    spec: FrozenTableSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    project_root: object,
) -> TrainValFitParts:
    if not isinstance(train_loader, DataLoader):
        raise TypeError(
            f"train_loader must be DataLoader, got {type(train_loader)!r}"
        )
    if not isinstance(val_loader, DataLoader):
        raise TypeError(
            f"val_loader must be DataLoader, got {type(val_loader)!r}"
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
    auc_spec = (auc_metric(), (), (AUC_REDUCER,))
    monitor_key = metric_log_name(AUC, "val", LEVEL_EPOCH, AUC_REDUCER)
    trainable = [param for param in net.parameters() if param.requires_grad]
    if len(trainable) == 0:
        raise ValueError("fine-tune net has no trainable parameters")
    optimizer = AdamW(
        trainable,
        lr=parsed.lr.learning_rate,
        weight_decay=parsed.optimizer.weight_decay,
        betas=parsed.optimizer.betas,
    )
    return TrainValFitParts(
        net=net,
        loaders=TrainValLoaders(train=train_loader, val=val_loader),
        steps=classification_fit_steps(),
        metrics=TrainValMetrics(
            train_metrics=[auc_spec],
            val_metrics=[auc_spec],
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


def _evaluate_finetune(
    context: EvalContext, *, payload_dir: Directory
) -> list[LogItem]:
    val_ids, val_logits, val_labels = _collect_loader(
        context,
        loader_name=EVAL_VAL_LOADER_NAME,
        with_labels=True,
    )
    test_ids, test_logits, _test_labels = _collect_loader(
        context,
        loader_name=EVAL_TEST_LOADER_NAME,
        with_labels=False,
    )
    np.save(payload_dir.path / "val_row_ids.npy", val_ids)
    np.save(payload_dir.path / "val_logits.npy", val_logits)
    np.save(payload_dir.path / "val_labels.npy", val_labels)
    np.save(payload_dir.path / "test_row_ids.npy", test_ids)
    np.save(payload_dir.path / "test_logits.npy", test_logits)
    return [
        LogItem.scalars(
            step=context.step,
            values={
                "inspect/n_val": float(val_ids.shape[0]),
                "inspect/n_test": float(test_ids.shape[0]),
            },
        )
    ]


def _collect_loader(
    context: EvalContext, *, loader_name: str, with_labels: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = context.loaders.named[loader_name]
    row_ids: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise TypeError(
                    f"finetune batch must be a mapping, got {type(batch)!r}"
                )
            prepared = mapping_to_device(mapping=batch, device=context.device)
            ids = prepared[IDS_KEY]
            if not isinstance(ids, Tensor):
                raise TypeError(f"ids must be a Tensor, got {type(ids)!r}")
            output = context.net(ids)
            if not isinstance(output, Tensor):
                raise TypeError(
                    f"classifier forward must return a Tensor, got {type(output)!r}"
                )
            row_ids.append(prepared[ROW_ID_KEY].detach().cpu().numpy())
            logits.append(output.detach().cpu().float().numpy())
            if with_labels:
                labels.append(prepared[LABEL_KEY].detach().cpu().numpy())
    if len(row_ids) == 0:
        raise RuntimeError(f"finetune inspect loader {loader_name!r} was empty")
    stacked_ids = np.concatenate(row_ids, axis=0)
    stacked_logits = np.concatenate(logits, axis=0)
    if with_labels:
        stacked_labels = np.concatenate(labels, axis=0)
    else:
        stacked_labels = np.empty((0,), dtype=np.float32)
    return stacked_ids, stacked_logits, stacked_labels
