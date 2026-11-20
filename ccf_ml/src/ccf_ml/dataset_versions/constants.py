"""Names, filenames, and defaults for dataset versions."""

DATASET_VERSION_SCHEME = "column_drop_v1"

TIME_OF_DAY_HOUR_COLUMN = "time_of_day_hour"
DAY_OF_WEEK_COLUMN = "day_of_week"
CARD_TYPE_COLUMN = "card_type"
CHANNEL_COLUMN = "channel"
DEVICE_TYPE_COLUMN = "device_type"
IS_NEW_MERCHANT_COLUMN = "is_new_merchant"
MERCHANT_CATEGORY_COLUMN = "merchant_category"
AUTH_METHOD_COLUMN = "auth_method"
MERCHANT_RISK_SCORE_COLUMN = "merchant_risk_score"

RISK_CONTEXT_TRANSFORM = "risk_context"
RISK_AGGREGATES_TRANSFORM = "risk_aggregates"
RISK_CONTEXT_GROUP_COLUMNS = (MERCHANT_CATEGORY_COLUMN, AUTH_METHOD_COLUMN)
RISK_CONTEXT_ADDED_COLUMNS = (
    "risk_group_mean",
    "risk_group_p50",
    "risk_group_p75",
    "risk_group_p90",
)
RISK_CONTEXT_GROUPS_FILENAME = "risk_context_groups.parquet"

TIME_COLUMNS = (
    TIME_OF_DAY_HOUR_COLUMN,
    DAY_OF_WEEK_COLUMN,
)

DATASET_VERSION_MANIFEST_FILENAME = "manifest.json"
DATASET_VERSION_FOLDS_FILENAME = "folds.json"
DATASET_VERSION_DATA_FILENAME = "data.parquet"

PROFILE_CONTEXT_TRANSFORM = "profile_context"
PROFILE_NUMERIC_COLUMNS = (
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
)
PROFILE_BASE_COLUMNS = (
    MERCHANT_CATEGORY_COLUMN,
    AUTH_METHOD_COLUMN,
    CHANNEL_COLUMN,
    CARD_TYPE_COLUMN,
)
PROFILE_CONTEXT_SIZE_COLUMN = "prof_n"
PROFILE_MIN_GROUP = 100
