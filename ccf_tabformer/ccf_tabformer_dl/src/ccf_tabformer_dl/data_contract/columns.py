"""Declared feature columns, roles, and in-between routing.

This is the human-visible contract the ``declared`` field_spec strategy consumes.
``is_fraud`` is not a feature column (unsupervised reconstruction; label rides
alongside the inner table).
"""

from __future__ import annotations

ID_COLUMN = "id"
LABEL_COLUMN = "is_fraud"

ROLE_NUMERIC = "numeric"
ROLE_IN_BETWEEN = "in_between"
ROLE_CATEGORICAL = "categorical"
ROLE_BOOLEAN = "boolean"
ROLES = (ROLE_NUMERIC, ROLE_IN_BETWEEN, ROLE_CATEGORICAL, ROLE_BOOLEAN)

ROUTING_AS_CAT = "as_cat"
ROUTING_AS_NUM = "as_num"

BOOLEAN_FALSE = "False"
BOOLEAN_TRUE = "True"
BOOLEAN_WRITTEN = (BOOLEAN_FALSE, BOOLEAN_TRUE)

IP_COUNTRY_MISMATCH_COLUMN = "ip_country_mismatch"
IS_AI_GENERATED_SCAM_ATTEMPT_COLUMN = "is_ai_generated_scam_attempt"
BILLING_SHIPPING_MISMATCH_COLUMN = "billing_shipping_mismatch"
IS_FOREIGN_TRANSACTION_COLUMN = "is_foreign_transaction"
USED_VPN_COLUMN = "used_vpn"

ALARM_FLAG_COLUMNS = (
    IP_COUNTRY_MISMATCH_COLUMN,
    IS_AI_GENERATED_SCAM_ATTEMPT_COLUMN,
    BILLING_SHIPPING_MISMATCH_COLUMN,
    IS_FOREIGN_TRANSACTION_COLUMN,
    USED_VPN_COLUMN,
)

COLUMN_DECLARATIONS: tuple[dict, ...] = (
    {"name": "amount_usd", "role": ROLE_NUMERIC},
    {"name": "distance_from_home_km", "role": ROLE_NUMERIC},
    {"name": "account_balance_usd", "role": ROLE_NUMERIC},
    {"name": "velocity_score", "role": ROLE_NUMERIC},
    {"name": "merchant_risk_score", "role": ROLE_NUMERIC},
    {"name": "hours_since_last_txn", "role": ROLE_IN_BETWEEN, "routing": ROUTING_AS_CAT},
    {"name": "txn_count_last_24h", "role": ROLE_IN_BETWEEN, "routing": ROUTING_AS_CAT},
    {"name": "card_age_months", "role": ROLE_IN_BETWEEN, "routing": ROUTING_AS_CAT},
    {"name": "customer_age", "role": ROLE_IN_BETWEEN, "routing": ROUTING_AS_CAT},
    {"name": "cvv_retry_count", "role": ROLE_IN_BETWEEN, "routing": ROUTING_AS_CAT},
    {"name": "time_of_day_hour", "role": ROLE_IN_BETWEEN, "routing": ROUTING_AS_CAT},
    {"name": "day_of_week", "role": ROLE_IN_BETWEEN, "routing": ROUTING_AS_CAT},
    {"name": "prior_disputes", "role": ROLE_IN_BETWEEN, "routing": ROUTING_AS_CAT},
    {"name": "merchant_category", "role": ROLE_CATEGORICAL},
    {"name": "card_type", "role": ROLE_CATEGORICAL},
    {"name": "auth_method", "role": ROLE_CATEGORICAL},
    {"name": "channel", "role": ROLE_CATEGORICAL},
    {"name": "device_type", "role": ROLE_CATEGORICAL},
    {"name": "is_foreign_transaction", "role": ROLE_BOOLEAN},
    {"name": "is_new_merchant", "role": ROLE_BOOLEAN},
    {"name": "used_vpn", "role": ROLE_BOOLEAN},
    {"name": "ip_country_mismatch", "role": ROLE_BOOLEAN},
    {"name": "billing_shipping_mismatch", "role": ROLE_BOOLEAN},
    {"name": "is_ai_generated_scam_attempt", "role": ROLE_BOOLEAN},
)


def feature_names() -> tuple[str, ...]:
    n = len(COLUMN_DECLARATIONS)
    names = [""] * n
    for i in range(n):
        names[i] = COLUMN_DECLARATIONS[i]["name"]
    return tuple(names)
