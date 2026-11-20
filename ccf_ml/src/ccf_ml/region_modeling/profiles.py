"""Profile construction: merchant tier and the categorical profile id."""

from __future__ import annotations

import pandas as pd

from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN
from ccf_ml.cross_validation.alarm import alarm_flag_mask
from ccf_ml.cross_validation.constants import ALARM_FLAG_COLUMNS
from ccf_ml.region_modeling.constants import (
    DEVICE_GROUP_COLUMN,
    GROUP_BY_DEVICE,
    PROFILE_COLUMNS,
    PROFILE_ID_COLUMN,
    TIER_BY_MERCHANT,
)


def with_profile_columns(train_frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the frame with tier, device_group and ``profile_id``.

    Works on any row subset - the profile columns are defined on every row,
    not only inside the no-alarm region. The alarm-flag definition of the
    region is not applied here.
    """
    profiled = train_frame.copy()
    add_merchant_tier(profiled)
    add_device_group(profiled)
    add_profile_id(profiled)
    return profiled


def build_no_alarm_frame(train_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the no-alarm rows with tier, device_group and ``profile_id``.

    The region is defined purely by the alarm flags (no model scores are
    involved). The flag columns stay in the frame; callers drop them when
    building a model matrix.
    """
    has_alarm = alarm_flag_mask(train_frame, columns=ALARM_FLAG_COLUMNS)
    return with_profile_columns(train_frame.loc[~has_alarm])


def add_device_group(frame: pd.DataFrame) -> None:
    """Add ``device_group`` (atm_pos / computer / personal) in place."""
    if "device_type" not in frame.columns:
        raise ValueError("frame is missing device_type")
    frame[DEVICE_GROUP_COLUMN] = frame["device_type"].map(GROUP_BY_DEVICE)
    if frame[DEVICE_GROUP_COLUMN].isna().any():
        unmapped = sorted(set(frame.loc[frame[DEVICE_GROUP_COLUMN].isna(), "device_type"]))
        raise ValueError(f"unmapped device_type values: {unmapped}")


def add_merchant_tier(frame: pd.DataFrame) -> None:
    """Add ``merchant_tier`` (hot / warm / background) in place."""
    missing = [
        column for column in ("merchant_category",) if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")
    frame["merchant_tier"] = frame["merchant_category"].map(TIER_BY_MERCHANT)
    if frame["merchant_tier"].isna().any():
        raise ValueError("unmapped merchant category")


def add_profile_id(frame: pd.DataFrame) -> None:
    """Add ``profile_id``: the joined profile column values, in place."""
    missing = [
        column for column in PROFILE_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"frame is missing profile columns: {missing}")
    frame[PROFILE_ID_COLUMN] = (
        frame[list(PROFILE_COLUMNS)].astype(str).agg("|".join, axis=1)
    )


def profile_stratum(
    frame: pd.DataFrame,
    *,
    min_stratum_rows: int,
    id_column: str = ID_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> pd.Series:
    """Stratum labels for the profile-stratified CV.

    Strata are ``profile_id|target`` pairs; a pair with fewer than
    ``min_stratum_rows`` rows collapses into the ``_small_|target`` bucket
    so that every stratum can fill all folds. Small profiles remain
    distinct as a feature - this only affects the split.
    """
    labels = frame[PROFILE_ID_COLUMN].astype(str) + "|" + frame[target_column].astype(str)
    counts = labels.value_counts()
    small = counts.index[counts < min_stratum_rows]
    return labels.where(~labels.isin(small), "_small_|" + frame[target_column].astype(str))
