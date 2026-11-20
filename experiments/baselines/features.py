"""Train/test feature matrix for the HistGradientBoosting baseline."""

from __future__ import annotations

import pandas as pd

from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN

CATEGORICAL_COLUMNS = (
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
)
NUMERIC_COLUMNS = (
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
)
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def prepare_features(
    frame: pd.DataFrame,
    *,
    categories: dict[str, list] | None = None,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    categorical_columns: tuple[str, ...] = CATEGORICAL_COLUMNS,
) -> pd.DataFrame:
    """Return the model matrix with categorical columns as pandas categories.

    ``categories`` freezes the levels from train so test uses the same codes.
    ``feature_columns``/``categorical_columns`` default to the full baseline
    schema; dataset versions pass their kept-column subset instead.
    """
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"table is missing id column: {ID_COLUMN!r}")
    missing = [name for name in feature_columns if name not in frame.columns]
    if missing:
        raise ValueError(f"table missing feature columns: {missing}")
    allowed_extra = {ID_COLUMN, TARGET_COLUMN}
    unexpected = [
        name
        for name in frame.columns
        if name not in feature_columns and name not in allowed_extra
    ]
    if unexpected:
        raise ValueError(f"table has unexpected columns: {unexpected}")
    features = frame.loc[:, list(feature_columns)].copy()
    n_cat = len(categorical_columns)
    for i in range(n_cat):
        name = categorical_columns[i]
        if categories is None:
            features[name] = features[name].astype("category")
        else:
            if name not in categories:
                raise KeyError(f"frozen categories missing {name!r}")
            features[name] = pd.Categorical(
                features[name],
                categories=categories[name],
            )
    return features


def category_levels(
    features: pd.DataFrame,
    *,
    categorical_columns: tuple[str, ...] = CATEGORICAL_COLUMNS,
) -> dict[str, list]:
    """JSON-ready category lists from a prepared train matrix."""
    levels: dict[str, list] = {}
    n_cat = len(categorical_columns)
    for i in range(n_cat):
        name = categorical_columns[i]
        levels[name] = features[name].cat.categories.tolist()
    return levels
