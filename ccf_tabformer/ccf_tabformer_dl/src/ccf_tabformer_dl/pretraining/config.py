"""Pydantic parser for the pretraining recipe's ``experiment_config`` block."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelCfg(BaseModel):
    model_config = _STRICT
    d_model: int = Field(ge=8)
    n_layers: int = Field(ge=1)
    n_heads: int = Field(ge=1)
    ffn_mult: int = Field(ge=1)
    dropout: float = Field(ge=0.0, lt=1.0)


class DataCfg(BaseModel):
    model_config = _STRICT
    batch_size: int = Field(ge=1)
    num_workers: int = Field(ge=0)
    max_rows: int = Field(ge=0)  # 0 means use the whole corpus


class MaskCfg(BaseModel):
    model_config = _STRICT
    token_mask_prob: float = Field(ge=0.0, lt=1.0)
    feature_mask_prob: float = Field(ge=0.0, lt=1.0)


class OptimCfg(BaseModel):
    model_config = _STRICT
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)


class PretrainConfig(BaseModel):
    model_config = _STRICT
    tokens_dir: str
    corpus: Literal["all", "train"]
    model: ModelCfg
    data: DataCfg
    mask: MaskCfg
    optimizer: OptimCfg

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "PretrainConfig":
        return cls.model_validate(dict(raw))
