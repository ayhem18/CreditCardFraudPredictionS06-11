"""Assemble the MFM pretraining run from a recipe's experiment_config."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch.optim import AdamW
from torch.utils.data import DataLoader

from dl_toolbox.base.file_management.directory import Directory
from dl_toolbox.dl_engine.exp_engine.train_only_engine.experiment import (
    TrainOnlyEvalParts,
    TrainOnlyFitParts,
)
from dl_toolbox.dl_engine.metrics.metric_object import LEVEL_EPOCH, metric_log_name
from dl_toolbox.dl_engine.runs.common.run_config import MonitorConfig, OptimizationConfig
from dl_toolbox.dl_engine.runs.train_only_run.train_only_config import (
    TrainOnlyLoaders,
    TrainOnlyMetrics,
)

from ccf_tabformer_dl.data.dataset import TokenizedDataset
from ccf_tabformer_dl.data.tokens import load_tokenized_table
from ccf_tabformer_dl.mfm.masking import MaskSamplerConfig, TokenMaskSampler
from ccf_tabformer_dl.mfm.steps import EPOCH_REDUCERS, RECON_ACC, masked_feature_fit_steps, recon_accuracy_metric
from ccf_tabformer_dl.model.network import MaskedFeatureModel, TabFormerConfig
from ccf_tabformer_dl.model.vocab import build_global_vocab, to_global_ids
from ccf_tabformer_dl.pretraining.config import PretrainConfig


class PretrainExperiment:
    def build_train_only(
        self, experiment_config: Mapping[str, Any], *, project_root: object
    ) -> TrainOnlyFitParts:

        config = PretrainConfig.parse(experiment_config)

        tokens_dir = _resolve_tokens_dir(config.tokens_dir, project_root=project_root)

        # what is table here exactly ? 
        table = load_tokenized_table(tokens_dir=tokens_dir)

        vocab = build_global_vocab(table.specs)

        local = table.train_codes if config.corpus == "train" else table.token_codes


        if config.data.max_rows > 0:
            local = local[: config.data.max_rows]
        
        global_ids = to_global_ids(local, vocab=vocab)

        dataset = TokenizedDataset(global_ids=global_ids, labels=None)
        
        loader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            drop_last=True,
        )

        net = MaskedFeatureModel(
            vocab=vocab,
            config=TabFormerConfig(
                d_model=config.model.d_model,
                n_layers=config.model.n_layers,
                n_heads=config.model.n_heads,
                ffn_mult=config.model.ffn_mult,
                dropout=config.model.dropout,
            ),
        )


        sampler = TokenMaskSampler(
            feature_index=vocab.feature_index,
            n_features=vocab.n_features,
            config=MaskSamplerConfig(
                token_mask_prob=config.mask.token_mask_prob,
                feature_mask_prob=config.mask.feature_mask_prob,
            ),
        )

        steps = masked_feature_fit_steps(sampler=sampler)

        monitor_key = metric_log_name(RECON_ACC, "train", LEVEL_EPOCH, "mean")

        metrics = TrainOnlyMetrics(
            train_metrics=[(recon_accuracy_metric(), (), EPOCH_REDUCERS)],
            monitor=MonitorConfig(key=monitor_key, mode="max"),
        )

        optimizer = AdamW(
            net.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )

        return TrainOnlyFitParts(
            net=net,
            loaders=TrainOnlyLoaders(train=loader),
            steps=steps,
            metrics=metrics,
            optimization=OptimizationConfig(
                optimizer=optimizer, scheduler=OptimizationConfig.ABSENT
            ),
        )

    def build_eval(
        self,
        experiment_config: Mapping[str, Any],
        evaluate: dict[str, Any],
        *,
        project_root: Directory,
        payload_dir: Directory,
    ) -> TrainOnlyEvalParts:
        del experiment_config, evaluate, project_root, payload_dir
        raise NotImplementedError(
            "pretraining has no inspect/eval; representation quality is read by the "
            "linear-probe script, not the engine's inspect path"
        )


def _resolve_tokens_dir(tokens_dir: str, *, project_root: object) -> Path:
    path = Path(tokens_dir)
    if path.is_absolute():
        return path
    if isinstance(project_root, Directory):
        return project_root.path / path
    return Path(str(project_root)) / path
