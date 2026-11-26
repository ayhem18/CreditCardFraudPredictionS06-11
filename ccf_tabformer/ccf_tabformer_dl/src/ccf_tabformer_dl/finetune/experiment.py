"""Assemble a fold's fine-tune run (3 regimes) from a recipe's experiment_config."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.dl_engine.exp_engine.train_val_engine.experiment import (
    TrainValEvalParts,
    TrainValFitParts,
)
from dl_toolbox.dl_engine.metrics.metric_object import LEVEL_EPOCH, metric_log_name
from dl_toolbox.dl_engine.runs.common.run_config import MonitorConfig, OptimizationConfig
from dl_toolbox.dl_engine.runs.train_val_run.train_val_config import (
    TrainValLoaders,
    TrainValMetrics,
)

from ccf_tabformer_dl.data.dataset import TokenizedDataset
from ccf_tabformer_dl.data.tokens import load_tokenized_table
from ccf_tabformer_dl.finetune.config import FineTuneConfig
from ccf_tabformer_dl.finetune.steps import AUC, AUC_REDUCER, auc_metric, classification_fit_steps
from ccf_tabformer_dl.model.classifier import FineTuneModel, HeadConfig, load_pretrained_encoder
from ccf_tabformer_dl.model.network import TabFormerConfig
from ccf_tabformer_dl.model.vocab import build_global_vocab, to_global_ids


class FineTuneExperiment:
    def build_fit(
        self, experiment_config: Mapping[str, Any], *, project_root: object
    ) -> TrainValFitParts:
        config = FineTuneConfig.parse(experiment_config)
        tokens_dir = _resolve(config.tokens_dir, project_root=project_root)
        table = load_tokenized_table(tokens_dir=tokens_dir)
        vocab = build_global_vocab(table.specs)

        global_ids = to_global_ids(table.train_codes, vocab=vocab)
        labels = table.labels
        is_val = table.fold_ids == config.fold
        train_idx = np.flatnonzero(~is_val)
        val_idx = np.flatnonzero(is_val)
        if config.data.max_rows > 0:
            train_idx = train_idx[: config.data.max_rows]

        train_loader = _loader(
            global_ids[train_idx], labels[train_idx], config=config, shuffle=True
        )
        val_loader = _loader(
            global_ids[val_idx], labels[val_idx], config=config, shuffle=False
        )

        encoder_config = TabFormerConfig(
            d_model=config.model.d_model, n_layers=config.model.n_layers,
            n_heads=config.model.n_heads, ffn_mult=config.model.ffn_mult,
            dropout=config.model.dropout,
        )
        net = FineTuneModel(
            vocab=vocab,
            encoder_config=encoder_config,
            head_config=HeadConfig(hidden=config.head.hidden, dropout=config.head.dropout),
        )
        if config.init == "pretrained":
            checkpoint = _resolve(config.pretrained_checkpoint, project_root=project_root)
            load_pretrained_encoder(net, checkpoint=checkpoint, map_location="cpu")
        net.set_encoder_frozen(frozen=config.freeze_encoder)

        steps = classification_fit_steps()
        auc_spec = (auc_metric(), (), (AUC_REDUCER,))
        monitor_key = metric_log_name(AUC, "val", LEVEL_EPOCH, AUC_REDUCER)
        metrics = TrainValMetrics(
            train_metrics=[auc_spec],
            val_metrics=[auc_spec],
            monitor=MonitorConfig(key=monitor_key, mode="max"),
        )
        return TrainValFitParts(
            net=net,
            loaders=TrainValLoaders(train=train_loader, val=val_loader),
            steps=steps,
            metrics=metrics,
            optimization=OptimizationConfig(
                optimizer=_build_optimizer(net, config=config),
                scheduler=OptimizationConfig.ABSENT,
            ),
        )

    def build_eval(
        self,
        experiment_config: Mapping[str, Any],
        evaluate: dict[str, Any],
        *,
        project_root: Directory,
        payload_dir: Directory,
    ) -> TrainValEvalParts:
        del experiment_config, evaluate, project_root, payload_dir
        raise NotImplementedError(
            "fine-tune OOF/test prediction is produced by the standalone inference "
            "script, not the engine's inspect path"
        )


def _loader(global_ids, labels, *, config: FineTuneConfig, shuffle: bool) -> DataLoader:
    dataset = TokenizedDataset(global_ids=global_ids, labels=labels)
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        drop_last=shuffle,
    )


def _build_optimizer(net: FineTuneModel, *, config: FineTuneConfig) -> AdamW:
    lr = config.optimizer.learning_rate
    groups: list[dict] = [{"params": list(net.head.parameters()), "lr": lr}]
    encoder_params = [p for p in net.encoder.parameters() if p.requires_grad]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": lr * config.optimizer.encoder_lr_mult})
    return AdamW(groups, lr=lr, weight_decay=config.optimizer.weight_decay)


def _resolve(path_str: str, *, project_root: object) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if isinstance(project_root, Directory):
        return project_root.path / path
    return Path(str(project_root)) / path
