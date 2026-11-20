"""Dataset version specifications: named column-drop ablations."""

from __future__ import annotations

from typing import NamedTuple

from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN
from ccf_ml.dataset_versions.constants import (
    CARD_TYPE_COLUMN,
    CHANNEL_COLUMN,
    DEVICE_TYPE_COLUMN,
    IS_NEW_MERCHANT_COLUMN,
    RISK_AGGREGATES_TRANSFORM,
    PROFILE_CONTEXT_TRANSFORM,
    RISK_CONTEXT_TRANSFORM,
    TIME_COLUMNS,
)

_PROTECTED_COLUMNS = (ID_COLUMN, TARGET_COLUMN)


class DatasetVersionSpec(NamedTuple):
    """One reproducible dataset version: a name and the columns to drop.

    A version is ``id + target + feature columns minus ``drop_columns``.
    Folds come from the frozen flag-stratified partition, not from the spec.
    ``transform`` (default: none) names a registered fold-safe derived-feature
    transform applied to the raw frame before the drops; see
    :mod:`ccf_ml.dataset_versions.transforms`.
    """

    name: str
    description: str
    drop_columns: tuple[str, ...]
    transform: str = ""


def validate_version_name(name: str) -> str:
    """Return ``name`` after checking it is one safe path component."""
    if not isinstance(name, str):
        raise TypeError(f"version name must be a str, got {type(name)!r}")
    stem = name.strip()
    if not stem or stem in {".", ".."}:
        raise ValueError(f"version name is empty or invalid: {name!r}")
    if "/" in stem or "\\" in stem:
        raise ValueError(
            f"version name must be a single path component: {name!r}"
        )
    return stem


def validate_spec(
    spec: DatasetVersionSpec,
    *,
    frame_columns: tuple[str, ...] | list[str] | None = None,
    id_column: str = ID_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> tuple[str, ...]:
    """Validate a spec and return its checked ``drop_columns`` tuple.

    When ``frame_columns`` is given, every dropped column must be present in
    it. Id and target can never be dropped.
    """
    if not isinstance(spec, DatasetVersionSpec):
        raise TypeError(
            f"spec must be a DatasetVersionSpec, got {type(spec)!r}"
        )
    validate_version_name(spec.name)
    drop_columns = spec.drop_columns
    if type(drop_columns) is not tuple:
        raise TypeError(
            f"{spec.name!r} drop_columns must be a tuple, got "
            f"{type(drop_columns)!r}"
        )
    if len(drop_columns) == 0:
        raise ValueError(f"{spec.name!r} drops no columns; a version must drop at least one")
    seen: set[str] = set()
    for column in drop_columns:
        if not isinstance(column, str) or not column:
            raise TypeError(
                f"{spec.name!r} drop_columns entries must be non-empty str, "
                f"got {column!r}"
            )
        if column in seen:
            raise ValueError(f"{spec.name!r} drops {column!r} more than once")
        seen.add(column)
    for column in _PROTECTED_COLUMNS:
        if column in seen:
            raise ValueError(
                f"{spec.name!r} drops protected column {column!r}"
            )
    if frame_columns is not None:
        available = set(frame_columns)
        for column in drop_columns:
            if column not in available:
                raise ValueError(
                    f"{spec.name!r} drops {column!r} which is not a frame column"
                )
    return tuple(drop_columns)


def kept_columns_from(
    frame_columns: tuple[str, ...] | list[str],
    drop_columns: tuple[str, ...],
) -> tuple[str, ...]:
    """Return ``frame_columns`` minus ``drop_columns``, order preserved."""
    dropped = set(drop_columns)
    return tuple(column for column in frame_columns if column not in dropped)


DATASET_VERSION_SPECS: tuple[DatasetVersionSpec, ...] = (
    DatasetVersionSpec(
        name="no_time",
        description=(
            "Ablation: drop the time features (time_of_day_hour, "
            "day_of_week). Validates the finding that fraud is independent "
            "of the time cell."
        ),
        drop_columns=TIME_COLUMNS,
    ),
    DatasetVersionSpec(
        name="no_time_no_card_type",
        description=(
            "Ablation: no time features and no card_type. card_type was "
            "found to carry very little signal."
        ),
        drop_columns=TIME_COLUMNS + (CARD_TYPE_COLUMN,),
    ),
    DatasetVersionSpec(
        name="no_time_no_channel",
        description=(
            "Ablation: no time features and no channel. channel was found "
            "to carry very little signal."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN,),
    ),
    DatasetVersionSpec(
        name="no_time_no_channel_no_card_type",
        description=(
            "Ablation: no time features, no channel, no card_type — the "
            "full drop of all three low-signal feature groups."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
    ),
    DatasetVersionSpec(
        name="risk_context",
        description=(
            "First feature-engineered version. Drops time, channel, "
            "card_type, device_type and is_new_merchant. Adds fold-safe "
            "(merchant_category x auth_method) statistics of "
            "merchant_risk_score (mean, p50, p75, p90, computed on each "
            "row's fold-training part; global fallback for unseen groups). "
            "merchant_risk_score becomes conditional: is_new_merchant rows "
            "get their fold's group mean (their raw risk was "
            "anti-informative: lower risk, ~2x fraud), other rows keep the "
            "raw score; the flag stays recoverable via "
            "risk == risk_group_mean. Extra payload "
            "risk_context_groups.parquet maps test rows to full-train "
            "group statistics."
        ),
        drop_columns=TIME_COLUMNS
        + (CHANNEL_COLUMN, CARD_TYPE_COLUMN, DEVICE_TYPE_COLUMN, IS_NEW_MERCHANT_COLUMN),
        transform=RISK_CONTEXT_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="risk_aggregates",
        description=(
            "Diagnostic companion to risk_context: same drops (time, "
            "channel, card_type, device_type, is_new_merchant) and the same "
            "fold-safe (merchant_category x auth_method) risk aggregates "
            "(mean, p50, p75, p90), but merchant_risk_score keeps its raw "
            "value everywhere. Isolates the aggregate features from the "
            "conditional new-merchant replacement."
        ),
        drop_columns=TIME_COLUMNS
        + (CHANNEL_COLUMN, CARD_TYPE_COLUMN, DEVICE_TYPE_COLUMN, IS_NEW_MERCHANT_COLUMN),
        transform=RISK_AGGREGATES_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="risk_aggregates_full",
        description=(
            "Aggregates only: same fold-safe (merchant_category x "
            "auth_method) risk aggregates as risk_aggregates, but no "
            "feature drops beyond the established free ones (time, channel, "
            "card_type). device_type, is_new_merchant and raw "
            "merchant_risk_score all stay. Isolates the pure value of the "
            "aggregate features."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=RISK_AGGREGATES_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="profile_context",
        description=(
            "Within-profile percentile features. Profiles are the "
            "categorical space minus device_type and is_new_merchant "
            "(merchant_category x auth_method x channel x card_type x alarm "
            "flags), with a size-guarded hierarchy (full profile -> without "
            "flags -> merchant x auth -> global, at >= 100 fold-train rows). "
            "Each row gains its percentile of the 9 profile numerics inside "
            "its profile, computed fold-safely on its fold-training part, "
            "plus the profile's train size. Raw columns untouched; drops "
            "only the established free ones (time, channel, card_type as "
            "columns - they stay inside the profile definition)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=PROFILE_CONTEXT_TRANSFORM,
    ),
)
