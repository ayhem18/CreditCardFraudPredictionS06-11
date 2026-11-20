"""Model-matrix construction for the no-alarm region variants."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN
from ccf_ml.region_modeling.constants import (
    DEVICE_GROUP_COLUMN,
    PROFILE_ID_COLUMN,
    QUANTILE_MIN_GROUP,
    QUANTILE_NUMERIC_COLUMNS,
    QUANTILE_PREFIX,
    REGION_ALWAYS_DROP,
)
from ccf_ml.region_modeling.profile_cv import ProfileStratifiedCv
from ccf_ml.region_modeling.variants import RegionVariantSpec, validate_variant

# Raw categorical columns (after the alarm flags are dropped). profile_id is
# handled separately through the variant's add_profile_id flag.
CATEGORICAL_FEATURE_COLUMNS = (
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
    "is_new_merchant",
)


def add_profile_quantiles(
    region_frame: pd.DataFrame,
    cv: ProfileStratifiedCv,
    *,
    columns: tuple[str, ...] = QUANTILE_NUMERIC_COLUMNS,
    min_group: int = QUANTILE_MIN_GROUP,
) -> tuple[str, ...]:
    """Add fold-safe within-profile quantile columns (``pq_<column>``).

    A row's quantile is the fraction of its fold-train profile peers with a
    strictly smaller value, plus half the ties — measured against the rows
    of the same profile in the *other* folds only. Profiles with fewer than
    ``min_group`` fold-train rows fall back to the whole fold-train. Returns
    the added column names.
    """
    ids = region_frame[ID_COLUMN].to_numpy(dtype=np.int64)
    pos_by_id = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    pos_by_id[ids] = np.arange(len(ids), dtype=np.int64)
    profile = region_frame[PROFILE_ID_COLUMN].to_numpy()
    values = {
        column: region_frame[column].to_numpy(dtype=np.float64) for column in columns
    }
    quantile_cols = {column: np.full(len(ids), np.nan) for column in columns}
    fold_of = np.full(len(ids), -1, dtype=np.int64)
    for fold in cv.folds:
        for record_id in fold.val_ids:
            fold_of[pos_by_id[int(record_id)]] = fold.fold_id
    if np.any(fold_of < 0):
        raise ValueError("some region rows are in no validation fold")
    for fold_id in range(cv.n_splits):
        train_mask = fold_of != fold_id
        sel = fold_of == fold_id
        train_rows = np.flatnonzero(train_mask)
        train_profiles = profile[train_rows]
        order_local = np.argsort(train_profiles, kind="stable")
        # full-frame positions of the fold-train rows, grouped and sorted
        train_positions = train_rows[order_local]
        sorted_profiles = train_profiles[order_local]
        val_profiles = profile[sel]
        val_groups = np.unique(val_profiles)
        # global fold-train reference, sorted once per column on first use
        global_sorted: dict[str, np.ndarray | None] = {column: None for column in columns}
        for group in val_groups:
            lo = np.searchsorted(sorted_profiles, group, side="left")
            hi = np.searchsorted(sorted_profiles, group, side="right")
            train_pos = train_positions[lo:hi]
            sel_positions = np.flatnonzero(sel & (profile == group))
            use_group = len(train_pos) >= min_group
            for column in columns:
                if use_group:
                    ref_sorted = np.sort(values[column][train_pos])
                else:
                    if global_sorted[column] is None:
                        global_sorted[column] = np.sort(values[column][train_mask])
                    ref_sorted = global_sorted[column]
                x = values[column][sel_positions]
                left = np.searchsorted(ref_sorted, x, side="left")
                right = np.searchsorted(ref_sorted, x, side="right")
                quantile_cols[column][sel_positions] = (
                    left + 0.5 * (right - left)
                ) / len(ref_sorted)
    added = []
    for column in columns:
        name = QUANTILE_PREFIX + column
        if np.isnan(quantile_cols[column]).any():
            raise ValueError(f"profile quantiles left NaNs in {name}")
        region_frame[name] = quantile_cols[column]
        added.append(name)
    return tuple(added)


def variant_model_matrix(
    region_frame: pd.DataFrame,
    spec: RegionVariantSpec,
    *,
    cv: ProfileStratifiedCv | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Build the model matrix for one variant.

    Returns ``(matrix, feature_columns, categorical_columns, numeric_columns)``
    where ``matrix`` keeps id and target plus the features, with categorical
    columns cast to pandas categories (levels frozen in this order - the
    training script is responsible for using the same order for test).
    """
    drop_columns = validate_variant(spec, frame_columns=tuple(region_frame.columns))
    working = region_frame
    quantile_columns: tuple[str, ...] = ()
    if spec.add_profile_quantiles:
        if cv is None:
            raise ValueError(f"{spec.name!r} needs the frozen cv for quantiles")
        quantile_columns = add_profile_quantiles(working, cv)
    excluded = set(drop_columns) | set(REGION_ALWAYS_DROP) | {
        ID_COLUMN,
        TARGET_COLUMN,
        "merchant_tier",
        DEVICE_GROUP_COLUMN,
        PROFILE_ID_COLUMN,
    }
    feature_columns = [
        column
        for column in working.columns
        if column not in excluded
        and not column.startswith(QUANTILE_PREFIX)
    ]
    feature_columns.extend(quantile_columns)
    if spec.add_profile_id:
        feature_columns.append(PROFILE_ID_COLUMN)
    categorical_columns = tuple(
        column
        for column in feature_columns
        if column in CATEGORICAL_FEATURE_COLUMNS or column == PROFILE_ID_COLUMN
    )
    matrix = working.loc[:, [ID_COLUMN, TARGET_COLUMN] + feature_columns].copy()
    for column in categorical_columns:
        matrix[column] = matrix[column].astype(str).astype("category")
    numeric_columns = tuple(
        column for column in feature_columns if column not in categorical_columns
    )
    return matrix, tuple(feature_columns), categorical_columns, numeric_columns


def add_profile_quantiles_vs_train(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = QUANTILE_NUMERIC_COLUMNS,
    min_group: int = QUANTILE_MIN_GROUP,
) -> tuple[str, ...]:
    """Add within-profile quantile columns to ``test_frame`` in place.

    Inference-time counterpart of :func:`add_profile_quantiles`: the
    reference for every test row is the full no-alarm training frame's
    profile group (profiles under ``min_group`` rows fall back to the whole
    training region), mirroring the fold-safe training computation.
    """
    train_profiles = train_frame[PROFILE_ID_COLUMN].to_numpy()
    train_order = np.argsort(train_profiles, kind="stable")
    sorted_profiles = train_profiles[train_order]
    test_profiles = test_frame[PROFILE_ID_COLUMN].to_numpy()
    added: list[str] = []
    for column in columns:
        train_values = train_frame[column].to_numpy(dtype=np.float64)
        train_sorted_by_profile = train_values[train_order]
        x = test_frame[column].to_numpy(dtype=np.float64)
        out = np.full(len(test_frame), np.nan, dtype=np.float64)
        global_sorted = None
        for group in np.unique(test_profiles):
            positions = np.flatnonzero(test_profiles == group)
            lo = np.searchsorted(sorted_profiles, group, side="left")
            hi = np.searchsorted(sorted_profiles, group, side="right")
            if hi - lo >= min_group:
                ref_sorted = np.sort(train_sorted_by_profile[lo:hi])
            else:
                if global_sorted is None:
                    global_sorted = np.sort(train_values)
                ref_sorted = global_sorted
            values = x[positions]
            left = np.searchsorted(ref_sorted, values, side="left")
            right = np.searchsorted(ref_sorted, values, side="right")
            out[positions] = (left + 0.5 * (right - left)) / len(ref_sorted)
        if np.isnan(out).any():
            raise ValueError(f"test quantiles left NaNs in {column}")
        name = QUANTILE_PREFIX + column
        test_frame[name] = out
        added.append(name)
    return tuple(added)
