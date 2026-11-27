"""Pydantic parser for the fine-tune recipe's ``experiment_config`` block."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ccf_tabformer_dl.task_engines.pretraining.config import DataCfg, ModelCfg

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


class HeadCfg(BaseModel):
    model_config = _STRICT
    hidden: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)


class FineTuneOptimCfg(BaseModel):
    model_config = _STRICT
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    encoder_lr_mult: float = Field(ge=0.0)  # encoder lr = learning_rate * this (ignored if frozen)


class FineTuneConfig(BaseModel):
    model_config = _STRICT
    tokens_dir: str
    fold: int = Field(ge=0)
    init: Literal["pretrained", "random"]
    freeze_encoder: bool
    pretrained_checkpoint: str  # path (abs or project-relative); "" allowed only when init == "random"
    model: ModelCfg
    head: HeadCfg
    data: DataCfg
    optimizer: FineTuneOptimCfg

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "FineTuneConfig":
        config = cls.model_validate(dict(raw))
        if config.init == "pretrained" and not config.pretrained_checkpoint:
            raise ValueError("pretrained_checkpoint is required when init == 'pretrained'")
        return config
