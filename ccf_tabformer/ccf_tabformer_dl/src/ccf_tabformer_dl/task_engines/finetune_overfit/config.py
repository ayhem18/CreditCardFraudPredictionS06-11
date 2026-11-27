"""Parse ``experiment_config`` for classification overfit on a row_subset."""

from __future__ import annotations

from dataclasses import dataclass

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import (
    parse_non_negative_int,
    parse_positive_int,
)
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys

from ccf_tabformer_dl.task_engines.finetune.config import (
    CLASSIFIER_SHARED_KEYS,
    REGIME_FROZEN,
    REGIME_RANDOM,
    parse_classifier_shared,
)

_KEYS = ("data",) + CLASSIFIER_SHARED_KEYS
_DATA_KEYS = ("row_subset", "batch_size", "num_workers")


@dataclass(frozen=True)
class FineTuneOverfitDataConfig:
    row_subset: str
    batch_size: int
    num_workers: int


class FineTuneOverfitConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="experiment_config")
        require_exact_keys(obj, _KEYS, label="experiment_config")
        data_raw = parse_mapping(obj["data"], label="data")
        require_exact_keys(data_raw, _DATA_KEYS, label="data")
        shared = parse_classifier_shared(obj)
        self.data = FineTuneOverfitDataConfig(
            row_subset=parse_non_empty_str(
                data_raw["row_subset"], label="data.row_subset"
            ),
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
