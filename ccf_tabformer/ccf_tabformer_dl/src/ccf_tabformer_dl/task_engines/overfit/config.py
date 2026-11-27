"""Parse ``experiment_config`` for the MFM overfit sanity check."""

from __future__ import annotations

from dataclasses import dataclass

from dl_toolbox.base.parsing.parse_iterable import parse_mapping
from dl_toolbox.base.parsing.parse_numerical import (
    parse_non_negative_int,
    parse_non_negative_number,
    parse_positive_int,
    parse_positive_number,
)
from dl_toolbox.base.parsing.parse_str import parse_non_empty_str
from dl_toolbox.base.parsing.parse_utils import (
    one_of_message,
    parse_one_of,
    require_exact_keys,
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
_DATA_KEYS = ("row_subset", "batch_size", "num_workers")
_LR_KEYS = ("learning_rate",)
_OPTIMIZER_KEYS = ("name", "weight_decay", "betas")
_SCHEDULER_KEYS = ("name",)
_STRATEGY_KEYS = ("strategy", "args")
OPTIMIZER_ADAMW = "adamw"
SCHEDULER_CONSTANT = "constant"


@dataclass(frozen=True)
class StrategyBlock:
    strategy: str
    args: dict


@dataclass(frozen=True)
class OverfitDataConfig:
    row_subset: str
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class OverfitLrConfig:
    learning_rate: float


@dataclass(frozen=True)
class OverfitOptimizerConfig:
    name: str
    weight_decay: float
    betas: tuple[float, float]


@dataclass(frozen=True)
class OverfitSchedulerConfig:
    name: str


class OverfitConfigParser:
    def __init__(self, payload: dict) -> None:
        obj = parse_mapping(payload, label="experiment_config")
        require_exact_keys(obj, _KEYS, label="experiment_config")
        data_raw = parse_mapping(obj["data"], label="data")
        require_exact_keys(data_raw, _DATA_KEYS, label="data")
        lr_raw = parse_mapping(obj["lr_config"], label="lr_config")
        require_exact_keys(lr_raw, _LR_KEYS, label="lr_config")
        self.data = OverfitDataConfig(
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


def parse_strategy_block(raw: object, *, label: str) -> StrategyBlock:
    obj = parse_mapping(raw, label=label)
    require_exact_keys(obj, _STRATEGY_KEYS, label=label)
    strategy = parse_non_empty_str(obj["strategy"], label=f"{label}.strategy")
    args = parse_mapping(obj["args"], label=f"{label}.args")
    return StrategyBlock(strategy=strategy, args=dict(args))


def parse_optimizer_config(raw: object) -> OverfitOptimizerConfig:
    opt_raw = parse_mapping(raw, label="optimizer_config")
    require_exact_keys(opt_raw, _OPTIMIZER_KEYS, label="optimizer_config")
    optimizer_name = parse_one_of(
        parse_non_empty_str(opt_raw["name"], label="optimizer_config.name"),
        allowed=(OPTIMIZER_ADAMW,),
        label="optimizer_config.name",
        error_type=ValueError,
        error_message=one_of_message,
    )
    if not isinstance(optimizer_name, str):
        raise TypeError(
            f"optimizer_config.name must be str, got {type(optimizer_name)!r}"
        )
    return OverfitOptimizerConfig(
        name=optimizer_name,
        weight_decay=parse_non_negative_number(
            opt_raw["weight_decay"], label="optimizer_config.weight_decay"
        ),
        betas=_betas(opt_raw["betas"]),
    )


def parse_scheduler_config(raw: object) -> OverfitSchedulerConfig:
    sched_raw = parse_mapping(raw, label="lr_scheduler_config")
    require_exact_keys(sched_raw, _SCHEDULER_KEYS, label="lr_scheduler_config")
    scheduler_name = parse_one_of(
        parse_non_empty_str(sched_raw["name"], label="lr_scheduler_config.name"),
        allowed=(SCHEDULER_CONSTANT,),
        label="lr_scheduler_config.name",
        error_type=ValueError,
        error_message=one_of_message,
    )
    if not isinstance(scheduler_name, str):
        raise TypeError(
            f"lr_scheduler_config.name must be str, got {type(scheduler_name)!r}"
        )
    return OverfitSchedulerConfig(name=scheduler_name)


def _betas(raw: object) -> tuple[float, float]:
    if not isinstance(raw, list):
        raise TypeError(f"optimizer_config.betas must be a list, got {type(raw)!r}")
    if len(raw) != 2:
        raise ValueError(
            f"optimizer_config.betas must have length 2, got {len(raw)}"
        )
    first = parse_positive_number(raw[0], label="optimizer_config.betas[0]")
    second = parse_positive_number(raw[1], label="optimizer_config.betas[1]")
    return (first, second)
