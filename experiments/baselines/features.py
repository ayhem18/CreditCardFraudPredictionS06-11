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
) -> pd.DataFrame:
    """Return the model matrix with categorical columns as pandas categories.

    ``categories`` freezes the levels from train so test uses the same codes.
    """
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"table is missing id column: {ID_COLUMN!r}")
    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"table missing feature columns: {missing}")
    allowed_extra = {ID_COLUMN, TARGET_COLUMN}
    unexpected = [
        name
        for name in frame.columns
        if name not in FEATURE_COLUMNS and name not in allowed_extra
    ]
    if unexpected:
        raise ValueError(f"table has unexpected columns: {unexpected}")
    features = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    n_cat = len(CATEGORICAL_COLUMNS)
    for i in range(n_cat):
        name = CATEGORICAL_COLUMNS[i]
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


def category_levels(features: pd.DataFrame) -> dict[str, list]:
    """JSON-ready category lists from a prepared train matrix."""
    levels: dict[str, list] = {}
    n_cat = len(CATEGORICAL_COLUMNS)
    for i in range(n_cat):
        name = CATEGORICAL_COLUMNS[i]
        levels[name] = features[name].cat.categories.tolist()
    return levels
