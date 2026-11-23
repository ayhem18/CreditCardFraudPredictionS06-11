"""Names and pre-registered floors for categorical-profile analysis."""

from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN

NOTEBOOK_STEM = "cat_profiling"

CATEGORICAL_COLUMNS = (
    "merchant_category",
    "is_new_merchant",
    "auth_method",
    "device_type",
    "card_type",
    "channel",
)

HOUR_COLUMN = "time_of_day_hour"
AGE_COLUMN = "customer_age"
DEVICE_COLUMN = "device_type"
CHANNEL_COLUMN = "channel"
MERCHANT_COLUMN = "merchant_category"
NEW_MERCHANT_COLUMN = "is_new_merchant"
AUTH_COLUMN = "auth_method"
CARD_COLUMN = "card_type"
AMOUNT_COLUMN = "amount_usd"
VELOCITY_COLUMN = "velocity_score"
RISK_COLUMN = "merchant_risk_score"
DISTANCE_COLUMN = "distance_from_home_km"

FEATURED_NUMERICS = (
    AMOUNT_COLUMN,
    VELOCITY_COLUMN,
    RISK_COLUMN,
    DISTANCE_COLUMN,
    AGE_COLUMN,
)

EXTRA_NUMERICS = (
    "hours_since_last_txn",
    "card_age_months",
    "account_balance_usd",
)

EXTRA_NUMERICS = (
    "hours_since_last_txn",
    "card_age_months",
    "account_balance_usd",
)

JOINT_COORDINATES = (
    DISTANCE_COLUMN,
    AGE_COLUMN,
    AMOUNT_COLUMN,
)

N_HOURS = 24
HOUR_WINDOW_LENGTHS = (4, 6, 8)
AGE_WINDOW_LENGTHS = (8, 12, 16)

CELL_JOIN = "|"
K_NEIGHBORS = 5
PAIR_CAP = 20_000_000
DOUBLE_ABS = (0.01, 0.05)
JOINT_RADIUS = (0.05, 0.10)

CENSUS_BULKY_FRAUDS = 100
CENSUS_HOT_LIFT = 2.0
CENSUS_HOT_FRAUDS = 20

SHIFT_MIN_SIDE = 500
SHIFT_KS_FLOOR = 0.05
SHIFT_MARGIN = 1.1
SHUFFLE_SEED = 0

COMPACT_MIN_FRAUD = 20
COMPACT_MIN_NONFRAUD = 500
SEPARATE_MEAN_D = 0.10

PAIRS_DIRNAME = "pairs"
COMPACTNESS_FRAUD_FILENAME = "compactness_fraud.parquet"
COMPACTNESS_CELL_FILENAME = "compactness_cell.csv"
CENSUS_FILENAME = "census.csv"
PARENT_SHIFT_FILENAME = "parent_shift.csv"
