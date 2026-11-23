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

# Grid decomposition of the continuous numerics, after the 2026-11-20
# verification battery (experiments/eda/data_analysis/jitter_skeleton_check.py):
# every value sits on an exact 2-decimal grid; the fractional slots are lumpy,
# anchor-dependent, and carry conditional fraud signal - so the grid is
# represented explicitly instead of leaving it to the trees to carve from
# raw floats. anchor = floor(x); slot = cents above the anchor (0..99;
# the 0.1-grid columns only use multiples of 10).
GRID_SLOTS_TRANSFORM = "grid_slots"
GRID_SLOTS_EXACT_TRANSFORM = "grid_slots_exact"
GRID_NUMERIC_COLUMNS = (
    "amount_usd",
    "account_balance_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "velocity_score",
    "merchant_risk_score",
)
# Columns whose native grid is 0.1 steps (only ~11 fractional slots, no
# cent-level structure): also get an exact-value id (x * 10, integer).
GRID_EXACT_VALUE_COLUMNS = (
    "velocity_score",
    "merchant_risk_score",
)
GRID_ANCHOR_PREFIX = "grid_anchor_"
GRID_SLOT_PREFIX = "grid_slot_"
GRID_VALUE_PREFIX = "grid_value_"

# Slot-reference features: numeric re-expressions of the fractional-slot
# signal (the grid IDs poisoned trees; numeric references don't). slot_z
# is target-free (per-anchor slot mean/std -> z-scored slot); slot_refs is
# target-aware and fold-safe (per-anchor fraud/clean slot centroids,
# shrunk k=50; feature = |slot - clean_c| - |slot - fraud_c|).
SLOT_Z_TRANSFORM = "slot_z"
SLOT_REFS_TRANSFORM = "slot_refs"
SLOT_Z_STATS_FILENAME = "slot_z_stats.parquet"
SLOT_REFS_FILENAME = "slot_refs.parquet"
SLOT_Z_PREFIX = "slotz_"
SLOT_LDA_PREFIX = "slotlda_"
SLOT_MIN_ANCHOR_ROWS = 10
SLOT_CENTROID_SHRINKAGE = 50

# Anchor-frequency features: log1p of the train count of a row's floor
# anchor. Rarity is non-monotone in value (popular 100, rare 101), so
# numeric thresholds cannot recover it from the raw column - unlike the
# slot structure, which the trees already reach. Screened 2026-11-20:
# marginal AUC vs fraud 0.63 (velocity), 0.63 (risk), 0.55-0.56 (hours,
# amount). Target-free; train counts ship as payload for test rows.
ANCHOR_FREQ_TRANSFORM = "anchor_freq"
ANCHOR_FREQ_FILENAME = "anchor_freq_counts.parquet"
ANCHOR_FREQ_PREFIX = "afreq_"

# Fold-safe shrunk anchor fraud rate (k = 20): stacks target-aware anchor
# risk on top of the target-free frequency. Leaky full-train screen: 0.67
# (balance), 0.61 (amount), 0.57 (hours) single-feature AUC.
ANCHOR_FULL_TRANSFORM = "anchor_full"
ANCHOR_RATE_FILENAME = "anchor_rate.parquet"
ANCHOR_RATE_PREFIX = "arate_"
ANCHOR_RATE_SHRINKAGE = 20


def anchor_freq_column(column: str) -> str:
    """Name of the log-anchor-frequency feature column."""
    return ANCHOR_FREQ_PREFIX + column


def anchor_rate_column(column: str) -> str:
    """Name of the shrunk anchor fraud-rate feature column."""
    return ANCHOR_RATE_PREFIX + column


def slot_z_column(column: str) -> str:
    """Name of the z-scored-slot feature column for ``column``."""
    return SLOT_Z_PREFIX + column


def slot_lda_column(column: str) -> str:
    """Name of the slot-centroid-distance feature column."""
    return SLOT_LDA_PREFIX + column


def grid_anchor_column(column: str) -> str:
    """Name of the floor-anchor column for ``column``."""
    return GRID_ANCHOR_PREFIX + column


def grid_slot_column(column: str) -> str:
    """Name of the fractional-slot (cents above anchor) column."""
    return GRID_SLOT_PREFIX + column


def grid_value_column(column: str) -> str:
    """Name of the exact-grid-value id column for ``column``."""
    return GRID_VALUE_PREFIX + column

# Attempt-5 family: the anchor rate extended to the native 0.1 grid of
# velocity/risk (leaky screen 0.72 / 0.76) and to small-int columns with
# real rate spread (customer_age 0.65, cvv 0.71, txn 0.62 - upper bounds).
ANCHOR_V2_TRANSFORM = "anchor_v2"
ANCHOR_V2_RATES_FILENAME = "anchor_v2_rates.parquet"
IS_ALARM_COLUMN = "is_alarm"
ANCHOR_V2_IS_ALARM_TRANSFORM = "anchor_v2_is_alarm"
N_ALARM_FLAGS_COLUMN = "n_alarm_flags"
ANCHOR_V2_N_ALARM_TRANSFORM = "anchor_v2_n_alarm"
NATIVE_RATE_COLUMNS = ("velocity_score", "merchant_risk_score")
SMALLINT_RATE_COLUMNS = ("customer_age", "cvv_retry_count", "txn_count_last_24h")
ARATE10_PREFIX = "arate10_"

# Attempt-6: interaction rates at the native grid crossed with the alarm
# flag for velocity/risk (leaky screen 0.81 / 0.83; groups ~400 rows).
ANCHOR_V3_TRANSFORM = "anchor_v3"
ANCHOR_V3_RATES_FILENAME = "anchor_v3_rates.parquet"
ARATE10A_PREFIX = "arate10a_"

# Attempt-7: common decimal tilt for the 0.1-grid columns. The tilt
# analysis (q3_decimal_signal + slot_tilt_stability notebooks, 2026-11-22)
# found one shared within-anchor slot lift g(slot) per column in the
# resolved low blocks (risk decisive, permutation p = 0.010; velocity
# p = 0.075), replicating across folds and explaining every anchor
# (per-anchor deviations fully noise-sized). Feature = the row's slot
# lift, fold-safe (block rows of the fold-train only), rows outside the
# block carry the neutral 1.0. Ten levels per column, each holding ~100k
# fold-train rows, so no shrinkage. Full-train tilts ship as payload.
ANCHOR_V2T_TRANSFORM = "anchor_v2t"
ANCHOR_TILT_FILENAME = "anchor_tilt.parquet"
ANCHOR_TILT_PREFIX = "atilt_"
TILT_COLUMNS = {"velocity_score": 40, "merchant_risk_score": 55}


def anchor_tilt_column(column: str) -> str:
    """Name of the common-slot-tilt feature column for ``column``."""
    return ANCHOR_TILT_PREFIX + column

# Attempt-8: combined fraud rate of the two score columns' anchors. The
# joint-anchor analysis (q1_joint_anchor_structure.ipynb, 2026-11-22)
# showed the (velocity anchor x risk anchor) cell field is essentially the
# product of the two marginal anchor rate stories: within-anchor variance
# is explained by the partner (4.3-4.5x binomial) but the lift map is flat
# in dense cells and the independence simulation gives p = 0.796. Feature:
# ac = sigmoid(logit(p_va) + logit(p_ra) - logit(global)), marginals =
# shrunk k=20 integer-anchor rates, fold-safe; test rows use the
# full-train payload of the two marginal tables.
ANCHOR_V2C_TRANSFORM = "anchor_v2c"
ANCHOR_COMBINED_FILENAME = "anchor_combined_rate.parquet"
ANCHOR_COMBINED_COLUMN = "acomb_velocity_risk"
COMBINED_RATE_COLUMNS = ("velocity_score", "merchant_risk_score")
COMBINED_RATE_SHRINKAGE = 20

# ---------------------------------------------------------------------------
# intersection_conditional family: anchor_v2 features (channel + card_type
# kept) plus self-describing fold-safe rate features whose keys may compose
# several columns (value-mode numerics and raw-mode categoricals). One
# payload carries the full-train rate table and its own key recipe, so the
# test side rebuilds keys without a hardcoded source list.
IC_RATE_PREFIX = "icrate_"
IC_RATES_FILENAME = "ic_rates.parquet"
IC_RATE_SHRINKAGE = 20
IC_DISPUTES_TRANSFORM = "intersection_conditional_disputes"
IC_CVV_TRANSFORM = "intersection_conditional_cvv"
IC_RISK_RATES_TRANSFORM = "intersection_conditional_risk_rates"
IC_VELOCITY_RATES_TRANSFORM = "intersection_conditional_velocity_rates"

# Target-free within-cell robust-z features: (value - cell median) / cell
# scale, scale = 1.4826 * cell MAD (falls back to the global scale for
# degenerate or unseen cells). No label is used, so these are computed on
# the full train frame like the anchor frequencies. Signed and absolute
# (non-monotone) columns are both emitted. One self-describing payload.
IC_CELLZ_FILENAME = "ic_cellz.parquet"
IC_ZSIGNED_PREFIX = "iczs_"
IC_ZABS_PREFIX = "icza_"
IC_CELLZ_MIN_ROWS = 200
IC_RISKZ_TRANSFORM = "intersection_conditional_riskz"
IC_CVV_RISKZ_TRANSFORM = "intersection_conditional_cvv_riskz"

# Fine-cell risk robust-z: (risk - cell median) / cell scale within the full
# 7-atom behavioural profile, with drop-one escalation (finest cell with
# >= IC_CELLZ_MIN_ROWS, else the largest single-atom-dropped cell, else the
# global stats). Target-free, so computed on the full train frame. The
# payload stores the per-(level, cell) median/scale; the test side rebuilds
# the same escalation from it.
IC_RISKZ_FINE_TRANSFORM = "intersection_conditional_riskz_fine"
IC_RISKZ_FINE_FILENAME = "ic_riskz_fine.parquet"
IC_RISKZ_FINE_ATOMS = (
    "channel", "device_type", "is_new_merchant", "merchant_category",
    "is_alarm", "card_type", "auth_method",
)
