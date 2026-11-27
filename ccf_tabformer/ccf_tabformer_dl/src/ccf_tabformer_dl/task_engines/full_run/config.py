"""Parse ``experiment_config`` for the MFM TrainVal full run."""

from __future__ import annotations

from dataclasses import dataclass

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import (
    parse_non_negative_int,
    parse_positive_int,
    parse_positive_number,
)
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import require_exact_keys

from ccf_tabformer_dl.task_engines.overfit.config import (
    OverfitLrConfig,
    parse_optimizer_config,
    parse_scheduler_config,
    parse_strategy_block,
)

_KEYS = (
    "data",
    "embedding",
    "masking",
    "encoder",
    "head",
    "lr_config",
    "optimizer_config",
    "lr_scheduler_config",
)
_DATA_KEYS = ("inner_table", "batch_size", "num_workers")
_LR_KEYS = ("learning_rate",)


@dataclass(frozen=True)
class FullRunDataConfig:
    inner_table: str
    batch_size: int
    num_workers: int


class FullRunConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="experiment_config")
        require_exact_keys(obj, _KEYS, label="experiment_config")
        data_raw = parse_mapping(obj["data"], label="data")
        require_exact_keys(data_raw, _DATA_KEYS, label="data")
        lr_raw = parse_mapping(obj["lr_config"], label="lr_config")
        require_exact_keys(lr_raw, _LR_KEYS, label="lr_config")
        self.data = FullRunDataConfig(
            inner_table=parse_non_empty_str(
                data_raw["inner_table"], label="data.inner_table"
            ),
            batch_size=parse_positive_int(
                data_raw["batch_size"], label="data.batch_size"
            ),
            num_workers=parse_non_negative_int(
                data_raw["num_workers"], label="data.num_workers"
            ),
        )
        self.embedding = parse_strategy_block(obj["embedding"], label="embedding")
        self.masking = parse_strategy_block(obj["masking"], label="masking")
        self.encoder = parse_strategy_block(obj["encoder"], label="encoder")
        self.head = parse_strategy_block(obj["head"], label="head")
        self.lr = OverfitLrConfig(
            learning_rate=parse_positive_number(
                lr_raw["learning_rate"], label="lr_config.learning_rate"
            )
        )
        self.optimizer = parse_optimizer_config(obj["optimizer_config"])
        self.scheduler = parse_scheduler_config(obj["lr_scheduler_config"])
