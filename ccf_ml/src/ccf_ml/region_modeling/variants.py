"""Region-model variant specs: the drop ladder plus profile features."""

from __future__ import annotations

from typing import NamedTuple

from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN
from ccf_ml.dataset_versions.constants import (
    CARD_TYPE_COLUMN,
    CHANNEL_COLUMN,
    TIME_COLUMNS,
)
from ccf_ml.region_modeling.constants import PROFILE_ID_COLUMN


class RegionVariantSpec(NamedTuple):
    """One variant of the no-alarm region model.

    ``drop_columns`` are raw columns removed from the model matrix (the
    alarm flags are always dropped inside the region and are not listed
    here). ``add_profile_id`` adds the 240-level categorical profile id.
    ``add_profile_quantiles`` adds fold-safe within-profile quantiles of the
    numeric columns (``pq_`` prefix, see :mod:`ccf_ml.region_modeling.features`).
    """

    name: str
    description: str
    drop_columns: tuple[str, ...] = ()
    add_profile_id: bool = False
    add_profile_quantiles: bool = False


REGION_MODEL_VARIANTS: tuple[RegionVariantSpec, ...] = (
    RegionVariantSpec(
        name="full",
        description="All original features minus the (constant) alarm flags.",
    ),
    RegionVariantSpec(
        name="no_time",
        description="Drop the time features.",
        drop_columns=TIME_COLUMNS,
    ),
    RegionVariantSpec(
        name="no_time_no_channel",
        description="Drop the time features and channel.",
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN,),
    ),
    RegionVariantSpec(
        name="no_time_no_card_type",
        description="Drop the time features and card_type.",
        drop_columns=TIME_COLUMNS + (CARD_TYPE_COLUMN,),
    ),
    RegionVariantSpec(
        name="no_time_no_channel_no_card_type",
        description="Drop the time features, channel and card_type (variant 4).",
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
    ),
    RegionVariantSpec(
        name="v4_profile_id",
        description=(
            "Variant 4 plus the 90-level profile_id categorical "
            "(merchant_tier x auth x device_group x new merchant)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        add_profile_id=True,
    ),
    RegionVariantSpec(
        name="v4_profile_id_quantiles",
        description=(
            "Variant 4 plus profile_id and fold-safe within-profile "
            "quantiles of the 8 numeric columns."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        add_profile_id=True,
        add_profile_quantiles=True,
    ),
    RegionVariantSpec(
        name="v4_quantiles",
        description=(
            "Variant 4 plus the within-profile quantiles only (no "
            "profile_id) - attributes any v4_profile_id_quantiles gain "
            "between the id shortcut and the quantile features."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        add_profile_quantiles=True,
    ),
)


def validate_variant(
    spec: RegionVariantSpec,
    *,
    frame_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    """Validate a variant spec; return its checked ``drop_columns``."""
    if not spec.name or "/" in spec.name or "\\" in spec.name:
        raise ValueError(f"invalid variant name: {spec.name!r}")
    drop_columns = spec.drop_columns
    if type(drop_columns) is not tuple:
        raise TypeError(f"{spec.name!r} drop_columns must be a tuple")
    seen: set[str] = set()
    for column in drop_columns:
        if not isinstance(column, str) or not column:
            raise TypeError(f"{spec.name!r} drops invalid column {column!r}")
        if column in seen:
            raise ValueError(f"{spec.name!r} drops {column!r} more than once")
        if column in (ID_COLUMN, TARGET_COLUMN):
            raise ValueError(f"{spec.name!r} drops protected column {column!r}")
        if column == PROFILE_ID_COLUMN:
            raise ValueError(f"{spec.name!r} drops {PROFILE_ID_COLUMN!r}; use add_profile_id")
        seen.add(column)
    if frame_columns is not None:
        available = set(frame_columns)
        for column in drop_columns:
            if column not in available:
                raise ValueError(
                    f"{spec.name!r} drops {column!r} which is not a frame column"
                )
    return drop_columns


def variant_by_name(name: str) -> RegionVariantSpec:
    """Return the registered variant with this name."""
    for spec in REGION_MODEL_VARIANTS:
        if spec.name == name:
            return spec
    raise KeyError(
        f"unknown variant {name!r}; known: {[s.name for s in REGION_MODEL_VARIANTS]}"
    )
