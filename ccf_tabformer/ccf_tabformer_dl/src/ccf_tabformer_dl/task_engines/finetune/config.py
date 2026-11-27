"""Parse ``experiment_config`` for fraud-classification fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import (
    parse_non_negative_int,
    parse_positive_int,
    parse_positive_number,
)
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import (
    one_of_message,
    parse_one_of,
    require_exact_keys,
)

from ccf_tabformer_dl.task_engines.overfit.config import (
    OverfitLrConfig,
    OverfitOptimizerConfig,
    OverfitSchedulerConfig,
    StrategyBlock,
    parse_optimizer_config,
    parse_scheduler_config,
    parse_strategy_block,
)

_DATA_KEYS = (
    "inner_table",
    "folds",
    "fold",
    "batch_size",
    "num_workers",
)
_LR_KEYS = ("learning_rate",)
CLASSIFIER_SHARED_KEYS = (
    "regime",
    "pretrained_checkpoint",
    "embedding",
    "encoder",
    "classifier_head",
    "lr_config",
    "optimizer_config",
    "lr_scheduler_config",
)
_KEYS = ("data",) + CLASSIFIER_SHARED_KEYS
REGIME_RANDOM = "random"
REGIME_FROZEN = "frozen"
REGIME_PRETRAINED = "pretrained"
REGIMES = (REGIME_RANDOM, REGIME_FROZEN, REGIME_PRETRAINED)


@dataclass(frozen=True)
class FineTuneDataConfig:
    inner_table: str
    folds: str
    fold: int
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class ClassifierSharedConfig:
    regime: str
    pretrained_checkpoint: str
    embedding: StrategyBlock
    encoder: StrategyBlock
    classifier_head: StrategyBlock
    lr: OverfitLrConfig
    optimizer: OverfitOptimizerConfig
    scheduler: OverfitSchedulerConfig

    def load_backbone(self) -> bool:
        return self.regime != REGIME_RANDOM

    def freeze_encoder(self) -> bool:
        return self.regime == REGIME_FROZEN


class FineTuneConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="experiment_config")
        require_exact_keys(obj, _KEYS, label="experiment_config")
        data_raw = parse_mapping(obj["data"], label="data")
        require_exact_keys(data_raw, _DATA_KEYS, label="data")
        shared = parse_classifier_shared(obj)
        self.data = FineTuneDataConfig(
            inner_table=parse_non_empty_str(
                data_raw["inner_table"], label="data.inner_table"
            ),
            folds=parse_non_empty_str(data_raw["folds"], label="data.folds"),
            fold=parse_non_negative_int(data_raw["fold"], label="data.fold"),
            batch_size=parse_positive_int(
                data_raw["batch_size"], label="data.batch_size"
            ),
            num_workers=parse_non_negative_int(
                data_raw["num_workers"], label="data.num_workers"
            ),
        )
        self.regime = shared.regime
        self.pretrained_checkpoint = shared.pretrained_checkpoint
        self.embedding = shared.embedding
        self.encoder = shared.encoder
        self.classifier_head = shared.classifier_head
        self.lr = shared.lr
        self.optimizer = shared.optimizer
        self.scheduler = shared.scheduler

    def load_backbone(self) -> bool:
        return self.regime != REGIME_RANDOM

    def freeze_encoder(self) -> bool:
        return self.regime == REGIME_FROZEN


def parse_classifier_shared(obj: dict) -> ClassifierSharedConfig:
    """Regime, checkpoint, embedding, encoder, head, and optimization blocks."""
    lr_raw = parse_mapping(obj["lr_config"], label="lr_config")
    require_exact_keys(lr_raw, _LR_KEYS, label="lr_config")
    regime = parse_one_of(
        parse_non_empty_str(obj["regime"], label="regime"),
        allowed=REGIMES,
        label="regime",
        error_type=ValueError,
        error_message=one_of_message,
    )
    if not isinstance(regime, str):
        raise TypeError(f"regime must be str, got {type(regime)!r}")
    checkpoint_raw = obj["pretrained_checkpoint"]
    if regime == REGIME_RANDOM:
        if checkpoint_raw not in ("", None):
            raise ValueError(
                "pretrained_checkpoint must be empty when regime is random"
            )
        checkpoint = ""
    else:
        checkpoint = parse_non_empty_str(
            checkpoint_raw, label="pretrained_checkpoint"
        )
    return ClassifierSharedConfig(
        regime=regime,
        pretrained_checkpoint=checkpoint,
        embedding=parse_strategy_block(obj["embedding"], label="embedding"),
        encoder=parse_strategy_block(obj["encoder"], label="encoder"),
        classifier_head=parse_strategy_block(
            obj["classifier_head"], label="classifier_head"
        ),
        lr=OverfitLrConfig(
            learning_rate=parse_positive_number(
                lr_raw["learning_rate"], label="lr_config.learning_rate"
            )
        ),
        optimizer=parse_optimizer_config(obj["optimizer_config"]),
        scheduler=parse_scheduler_config(obj["lr_scheduler_config"]),
    )
