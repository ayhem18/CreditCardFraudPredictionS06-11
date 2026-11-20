"""Names and fixed choices for the region-modeling subpackage."""

from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN
from ccf_ml.dataset_versions.constants import (
    CARD_TYPE_COLUMN,
    CHANNEL_COLUMN,
    TIME_COLUMNS,
)

REGION_NAME = "no_alarm"

# The categorical profile (v3, from the categorical_profile_rarity analysis).
# device_type enters as a 3-level semantic group (atm_pos / computer /
# personal), so the profile space is 3 x 5 x 3 x 2 = 90 theoretical cells.
DEVICE_GROUP_COLUMN = "device_group"
DEVICE_GROUPS = {
    "atm_pos": ("ATM Machine", "POS Terminal"),
    "computer": ("Mac", "Windows PC"),
    "personal": ("Android Phone", "iPhone", "Smart Watch", "Tablet"),
}
GROUP_BY_DEVICE = {
    device: group
    for group, devices in DEVICE_GROUPS.items()
    for device in devices
}
PROFILE_COLUMNS = (
    "merchant_tier",
    "auth_method",
    DEVICE_GROUP_COLUMN,
    "is_new_merchant",
)
PROFILE_ID_COLUMN = "profile_id"

MERCHANT_TIERS = {
    "hot": ("Gift Cards", "Crypto Exchange", "Gaming"),
    "warm": ("Electronics", "Streaming"),
    "background": (
        "Travel",
        "Utilities",
        "Healthcare",
        "Fuel",
        "Restaurants",
        "Groceries",
        "Online Retail",
    ),
}
TIER_BY_MERCHANT = {
    merchant: tier
    for tier, merchants in MERCHANT_TIERS.items()
    for merchant in merchants
}

# The alarm flags are constant (all False) inside the no-alarm region and are
# always dropped from the region model matrix.
REGION_ALWAYS_DROP = (
    "is_foreign_transaction",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
)

# Numerics that receive fold-safe within-profile quantile features
# (cvv_retry_count is excluded by decision: 4 unique values).
QUANTILE_NUMERIC_COLUMNS = (
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
)
QUANTILE_PREFIX = "pq_"
QUANTILE_MIN_GROUP = 100

PROFILE_CV_SCHEME = "profile_stratified"
PROFILE_CV_FILENAME = "profile_stratified_cv.json"
PROFILE_CV_MIN_STRATUM_PROFILE = 25
DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42

# Unified CV shared by global and region models: one fold assignment over
# ALL rows, stratified on alarm x profile x target (2 x 90 x 2 = 360
# theoretical cells). Aggregated to alarm x target it refines the frozen
# flag-stratified scheme; restricted to the no-alarm rows it induces the
# profile x target stratification, so OOF numbers from both modeling
# approaches become directly comparable.
ALARM_PROFILE_CV_SCHEME = "alarm_profile_stratified"
ALARM_PROFILE_CV_FILENAME = "alarm_profile_stratified_cv.json"
ALARM_PROFILE_CV_MIN_STRATUM = 25
ALARM_LABEL = "alarm"
NO_ALARM_LABEL = "no_alarm"
SMALL_PROFILE_LABEL = "_small_"

REGION_COLUMNS = (ID_COLUMN, TARGET_COLUMN, TIME_COLUMNS, CARD_TYPE_COLUMN, CHANNEL_COLUMN)
