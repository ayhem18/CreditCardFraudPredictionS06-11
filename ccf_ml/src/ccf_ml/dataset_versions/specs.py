"""Dataset version specifications: named column-drop ablations."""

from __future__ import annotations

from typing import NamedTuple

from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN
from ccf_ml.dataset_versions.constants import (
    ANCHOR_FREQ_TRANSFORM,
    ANCHOR_FULL_TRANSFORM,
    ANCHOR_V2_IS_ALARM_TRANSFORM,
    ANCHOR_V2_N_ALARM_TRANSFORM,
    ANCHOR_V2_TRANSFORM,
    ANCHOR_V2C_TRANSFORM,
    ANCHOR_V2T_TRANSFORM,
    ANCHOR_V3_TRANSFORM,
    CARD_TYPE_COLUMN,
    CHANNEL_COLUMN,
    DEVICE_TYPE_COLUMN,
    GRID_SLOTS_EXACT_TRANSFORM,
    GRID_SLOTS_TRANSFORM,
    IC_CVV_RISKZ_TRANSFORM,
    IC_CVV_TRANSFORM,
    IC_DISPUTES_TRANSFORM,
    IC_RISK_RATES_TRANSFORM,
    IC_VELOCITY_RATES_TRANSFORM,
    IC_RISKZ_FINE_TRANSFORM,
    IC_RISKZ_TRANSFORM,
    IS_NEW_MERCHANT_COLUMN,
    RISK_AGGREGATES_TRANSFORM,
    PROFILE_CONTEXT_TRANSFORM,
    RISK_CONTEXT_TRANSFORM,
    SLOT_REFS_TRANSFORM,
    SLOT_Z_TRANSFORM,
    TIME_COLUMNS,
)

_PROTECTED_COLUMNS = (ID_COLUMN, TARGET_COLUMN)


class DatasetVersionSpec(NamedTuple):
    """One reproducible dataset version: a name and the columns to drop.

    A version is ``id + target + feature columns minus ``drop_columns``.
    Folds come from the frozen flag-stratified partition, not from the spec.
    ``transform`` (default: none) names a registered fold-safe derived-feature
    transform applied to the raw frame before the drops; see
    :mod:`ccf_ml.dataset_versions.transforms`.
    """

    name: str
    description: str
    drop_columns: tuple[str, ...]
    transform: str = ""


def validate_version_name(name: str) -> str:
    """Return ``name`` after checking it is one safe path component."""
    if not isinstance(name, str):
        raise TypeError(f"version name must be a str, got {type(name)!r}")
    stem = name.strip()
    if not stem or stem in {".", ".."}:
        raise ValueError(f"version name is empty or invalid: {name!r}")
    if "/" in stem or "\\" in stem:
        raise ValueError(
            f"version name must be a single path component: {name!r}"
        )
    return stem


def validate_spec(
    spec: DatasetVersionSpec,
    *,
    frame_columns: tuple[str, ...] | list[str] | None = None,
    id_column: str = ID_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> tuple[str, ...]:
    """Validate a spec and return its checked ``drop_columns`` tuple.

    When ``frame_columns`` is given, every dropped column must be present in
    it. Id and target can never be dropped.
    """
    if not isinstance(spec, DatasetVersionSpec):
        raise TypeError(
            f"spec must be a DatasetVersionSpec, got {type(spec)!r}"
        )
    validate_version_name(spec.name)
    drop_columns = spec.drop_columns
    if type(drop_columns) is not tuple:
        raise TypeError(
            f"{spec.name!r} drop_columns must be a tuple, got "
            f"{type(drop_columns)!r}"
        )
    if len(drop_columns) == 0:
        raise ValueError(f"{spec.name!r} drops no columns; a version must drop at least one")
    seen: set[str] = set()
    for column in drop_columns:
        if not isinstance(column, str) or not column:
            raise TypeError(
                f"{spec.name!r} drop_columns entries must be non-empty str, "
                f"got {column!r}"
            )
        if column in seen:
            raise ValueError(f"{spec.name!r} drops {column!r} more than once")
        seen.add(column)
    for column in _PROTECTED_COLUMNS:
        if column in seen:
            raise ValueError(
                f"{spec.name!r} drops protected column {column!r}"
            )
    if frame_columns is not None:
        available = set(frame_columns)
        for column in drop_columns:
            if column not in available:
                raise ValueError(
                    f"{spec.name!r} drops {column!r} which is not a frame column"
                )
    return tuple(drop_columns)


def kept_columns_from(
    frame_columns: tuple[str, ...] | list[str],
    drop_columns: tuple[str, ...],
) -> tuple[str, ...]:
    """Return ``frame_columns`` minus ``drop_columns``, order preserved."""
    dropped = set(drop_columns)
    return tuple(column for column in frame_columns if column not in dropped)


DATASET_VERSION_SPECS: tuple[DatasetVersionSpec, ...] = (
    DatasetVersionSpec(
        name="no_time",
        description=(
            "Ablation: drop the time features (time_of_day_hour, "
            "day_of_week). Validates the finding that fraud is independent "
            "of the time cell."
        ),
        drop_columns=TIME_COLUMNS,
    ),
    DatasetVersionSpec(
        name="no_time_no_card_type",
        description=(
            "Ablation: no time features and no card_type. card_type was "
            "found to carry very little signal."
        ),
        drop_columns=TIME_COLUMNS + (CARD_TYPE_COLUMN,),
    ),
    DatasetVersionSpec(
        name="no_time_no_channel",
        description=(
            "Ablation: no time features and no channel. channel was found "
            "to carry very little signal."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN,),
    ),
    DatasetVersionSpec(
        name="no_time_no_channel_no_card_type",
        description=(
            "Ablation: no time features, no channel, no card_type — the "
            "full drop of all three low-signal feature groups."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
    ),
    DatasetVersionSpec(
        name="risk_context",
        description=(
            "First feature-engineered version. Drops time, channel, "
            "card_type, device_type and is_new_merchant. Adds fold-safe "
            "(merchant_category x auth_method) statistics of "
            "merchant_risk_score (mean, p50, p75, p90, computed on each "
            "row's fold-training part; global fallback for unseen groups). "
            "merchant_risk_score becomes conditional: is_new_merchant rows "
            "get their fold's group mean (their raw risk was "
            "anti-informative: lower risk, ~2x fraud), other rows keep the "
            "raw score; the flag stays recoverable via "
            "risk == risk_group_mean. Extra payload "
            "risk_context_groups.parquet maps test rows to full-train "
            "group statistics."
        ),
        drop_columns=TIME_COLUMNS
        + (CHANNEL_COLUMN, CARD_TYPE_COLUMN, DEVICE_TYPE_COLUMN, IS_NEW_MERCHANT_COLUMN),
        transform=RISK_CONTEXT_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="risk_aggregates",
        description=(
            "Diagnostic companion to risk_context: same drops (time, "
            "channel, card_type, device_type, is_new_merchant) and the same "
            "fold-safe (merchant_category x auth_method) risk aggregates "
            "(mean, p50, p75, p90), but merchant_risk_score keeps its raw "
            "value everywhere. Isolates the aggregate features from the "
            "conditional new-merchant replacement."
        ),
        drop_columns=TIME_COLUMNS
        + (CHANNEL_COLUMN, CARD_TYPE_COLUMN, DEVICE_TYPE_COLUMN, IS_NEW_MERCHANT_COLUMN),
        transform=RISK_AGGREGATES_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="risk_aggregates_full",
        description=(
            "Aggregates only: same fold-safe (merchant_category x "
            "auth_method) risk aggregates as risk_aggregates, but no "
            "feature drops beyond the established free ones (time, channel, "
            "card_type). device_type, is_new_merchant and raw "
            "merchant_risk_score all stay. Isolates the pure value of the "
            "aggregate features."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=RISK_AGGREGATES_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="profile_context",
        description=(
            "Within-profile percentile features. Profiles are the "
            "categorical space minus device_type and is_new_merchant "
            "(merchant_category x auth_method x channel x card_type x alarm "
            "flags), with a size-guarded hierarchy (full profile -> without "
            "flags -> merchant x auth -> global, at >= 100 fold-train rows). "
            "Each row gains its percentile of the 9 profile numerics inside "
            "its profile, computed fold-safely on its fold-training part, "
            "plus the profile's train size. Raw columns untouched; drops "
            "only the established free ones (time, channel, card_type as "
            "columns - they stay inside the profile definition)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=PROFILE_CONTEXT_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="grid_slots",
        description=(
            "Probe: explicit grid decomposition of the six continuous "
            "numerics. Each gains grid_anchor_<col> = floor(x) and "
            "grid_slot_<col> = cents above the anchor (0..99), both "
            "categorical. Grounded in the 2026-11-20 verification battery: "
            "values sit on exact 2-decimal grids, slot distributions are "
            "lumpy and anchor-dependent, and slots carry conditional fraud "
            "signal (within-anchor AUC up to 0.74 on merchant_risk). Tests "
            "whether tuned XGB depth-6 trees already carve anchor x slot "
            "cells from the raw floats. Drops only the established free "
            "ones (time, channel, card_type)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=GRID_SLOTS_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="grid_slots_exact",
        description=(
            "Probe: grid_slots plus exact-grid-value ids for the two "
            "0.1-grid columns (grid_value_velocity_score, "
            "grid_value_merchant_risk_score = x*10 as integer; 674 / 959 "
            "levels). These have no cent-level structure, so their full "
            "value is a small id space the trees can memorize directly."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=GRID_SLOTS_EXACT_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="slot_z",
        description=(
            "Slot-reference probe v1 (target-free): slotz_<col> = "
            "(slot - mean(slot|anchor)) / std(slot|anchor) for the six "
            "continuous numerics; anchors under 10 rows and unseen anchors "
            "fall back to the column's global slot mean/std. Numeric "
            "re-expression of the verified within-anchor decimal structure "
            "after the grid-ID encoding poisoned trees (-0.022). Drops only "
            "the established free ones (time, channel, card_type)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=SLOT_Z_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="slot_refs",
        description=(
            "Slot-reference probe v2 (target-aware, fold-safe): "
            "slotlda_<col> = |slot - clean_centroid(anchor)| - "
            "|slot - fraud_centroid(anchor)| with centroids from the row's "
            "fold-training part, shrunk k=50 toward global class centroids; "
            "test rows use full-train centroids. Encodes the battery's "
            "strongest finding (fraud-vs-clean slot shifts within anchors, "
            "|z| up to 10) as one numeric threshold. Drops only the "
            "established free ones (time, channel, card_type)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=SLOT_REFS_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="anchor_freq",
        description=(
            "Rarity probe (target-free): afreq_<col> = log1p(train count "
            "of the row's floor anchor) for the six continuous numerics. "
            "Rarity is non-monotone in value, so numeric thresholds cannot "
            "recover it from the raw column. Screened marginal fraud AUC: "
            "0.63 (velocity), 0.63 (risk), 0.56 (hours), 0.55 (amount). "
            "Unseen test anchors score log1p(0). Drops only the "
            "established free ones (time, channel, card_type)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=ANCHOR_FREQ_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="anchor_full",
        description=(
            "Rarity probe stacked with target-aware anchor risk: afreq_<col> "
            "(log train frequency of the floor anchor, as anchor_freq) plus "
            "arate_<col> = (frauds + 20*global_rate) / (rows + 20) of the "
            "anchor, fold-safe (fold-train only; test uses full-train "
            "payload). Leaky full-train screen of anchor rate: 0.67 "
            "(balance), 0.61 (amount). Drops only the established free "
            "ones (time, channel, card_type)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=ANCHOR_FULL_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="anchor_v2",
        description=(
            "Rate family v2: anchor_full (afreq + integer-anchor rates) "
            "plus arate10_<col> at the native 0.1 grid of velocity/risk "
            "(leaky screen 0.72 / 0.76; ~740 rows per level so shrinkage "
            "is light) and arate per value for customer_age, "
            "cvv_retry_count, txn_count_last_24h. All rates fold-safe "
            "(k=20), test rows use the full-train payload. Drops only the "
            "established free ones (time, channel, card_type)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=ANCHOR_V2_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="anchor_v2_is_alarm",
        description=(
            "anchor_v2 plus is_alarm, the OR of the five alarm flags. The "
            "five flags stay in the matrix. The collapsed bit is the "
            "alarm-region indicator the unified CV already stratifies on, "
            "offered to the tree as one column. Same drops as anchor_v2 "
            "(time, channel, card_type)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=ANCHOR_V2_IS_ALARM_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="anchor_v2_n_alarm",
        description=(
            "anchor_v2 plus n_alarm_flags, the count of the five alarm "
            "flags (0 through 5). The five flags stay. The count keeps the "
            "dose that is_alarm collapses: fraud rate rises from 0.011 at "
            "0 flags to 0.069, 0.26, and 0.65 at 1, 2, and 3. Same drops "
            "as anchor_v2 (time, channel, card_type)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=ANCHOR_V2_N_ALARM_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="anchor_v2t",
        description=(
            "Rate family v2 + common decimal tilt: anchor_v2 plus "
            "atilt_<col> for velocity/risk — the row's slot lift under one "
            "shared within-anchor tilt g(slot), estimated fold-safely on the "
            "column's low block (velocity <= 40, risk <= 55) where the tilt "
            "was validated (q3_decimal_signal / slot_tilt_stability "
            "notebooks, 2026-11-22: risk permutation p = 0.010, velocity "
            "p = 0.075; fold-replicating; one tilt explains every anchor). "
            "Rows above the block bound carry the neutral 1.0. Ten levels "
            "per column, ~100k fold-train rows each, no shrinkage. Drops "
            "only the established free ones (time, channel, card_type)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=ANCHOR_V2T_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="anchor_v2c",
        description=(
            "Rate family v2 + combined anchor rate: anchor_v2 plus "
            "acomb_velocity_risk = sigmoid(logit(p_va) + logit(p_ra) - "
            "logit(global)) — the product of the two score columns' marginal "
            "anchor-rate stories made explicit per row. Grounded in the "
            "joint-anchor analysis (q1_joint_anchor_structure.ipynb, "
            "2026-11-22): within-anchor variance is explained by the partner "
            "anchor (4.3-4.5x binomial) but the joint field equals the "
            "marginals' product (independence simulation p = 0.796, lift flat "
            "in dense cells). Marginals are shrunk (k=20) integer-anchor "
            "rates, fold-safe; unseen anchors fall back to the global rate "
            "(anchor -1 rows in the payload). Drops only the established "
            "free ones (time, channel, card_type)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=ANCHOR_V2C_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="anchor_v3",
        description=(
            "Rate family v3: anchor_v2 plus arate10a_<col> = fold-safe "
            "shrunk fraud rate of (native 0.1 value x alarm flag) for "
            "velocity/risk (leaky screen 0.81 / 0.83; ~400 rows per "
            "group, k=20). Drops only the established free ones (time, "
            "channel, card_type)."
        ),
        drop_columns=TIME_COLUMNS + (CHANNEL_COLUMN, CARD_TYPE_COLUMN),
        transform=ANCHOR_V3_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="intersection_conditional_base",
        description=(
            "Base for the intersection_conditional feature family: the full "
            "anchor_v2 feature set (afreq + integer-anchor rates, arate10 "
            "0.1-grid rates for velocity/risk, per-value rates for "
            "customer_age / cvv_retry_count / txn_count_last_24h), but with "
            "channel and card_type ADDED BACK (only the time columns are "
            "dropped). The channel-conditional and device-conditional "
            "features built on this base need the raw channel column in the "
            "matrix, and card_type is restored alongside it. channel and "
            "card_type were free drops for anchor_v2, so this base is "
            "expected to match the champion within noise; attempt 1 measures "
            "exactly that, so later gains are credited to the conditional "
            "features and not the re-added categoricals."
        ),
        drop_columns=TIME_COLUMNS,
        transform=ANCHOR_V2_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="intersection_conditional_disputes",
        description=(
            "intersection_conditional_base plus icrate_prior_disputes = the "
            "fold-safe, k=20 shrunk per-value fraud rate of prior_disputes. "
            "prior_disputes has no rate in anchor_v2, so this is a fully new "
            "column (not an interaction yet): the marginal dispute-count "
            "rate. Attempt 2 of the feature plan. Drops only the time "
            "columns (channel and card_type kept)."
        ),
        drop_columns=TIME_COLUMNS,
        transform=IC_DISPUTES_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="intersection_conditional_riskz",
        description=(
            "intersection_conditional_base plus the within-cell robust-z of "
            "merchant_risk_score for the cell merchant_category x "
            "is_new_merchant: iczs_merchant_risk_score (signed) and "
            "icza_merchant_risk_score (absolute, non-monotone). Targets the "
            "conditional risk outliers the champion misses (e.g. risk 55 in "
            "a cell whose typical risk is ~34). Stats are target-free "
            "(median / 1.4826*MAD, cells >= 200 rows, global fallback), so "
            "no fold split is needed. Attempt 3 of the feature plan. Drops "
            "only the time columns (channel and card_type kept)."
        ),
        drop_columns=TIME_COLUMNS,
        transform=IC_RISKZ_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="intersection_conditional_cvv",
        description=(
            "intersection_conditional_base plus icrate_cvv_ctx = the "
            "fold-safe, k=20 shrunk fraud rate of (cvv_retry_count x channel "
            "x device_type). The base already holds the global cvv value "
            "rate, so this is an interaction test: it pays only if the cvv "
            "lift varies by channel and device. Attempt 4 of the feature "
            "plan. Drops only the time columns (channel and card_type kept)."
        ),
        drop_columns=TIME_COLUMNS,
        transform=IC_CVV_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="intersection_conditional_cvv_riskz",
        description=(
            "intersection_conditional_base plus both day-1 keeps: "
            "icrate_cvv_ctx (cvv x channel x device fraud rate) and the "
            "merchant_category x is_new_merchant robust-z of "
            "merchant_risk_score (iczs_ / icza_). The day-2 combine of "
            "attempts 3 and 4, each a sub-bar positive that lifted the hard "
            "folds. Drops only the time columns (channel and card_type kept)."
        ),
        drop_columns=TIME_COLUMNS,
        transform=IC_CVV_RISKZ_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="intersection_conditional_riskz_fine",
        description=(
            "intersection_conditional_base plus the fine-cell robust-z of "
            "merchant_risk_score: iczs_risk_fine (signed) and icza_risk_fine "
            "(absolute) within the full 7-atom behavioural profile with "
            "drop-one escalation (finest cell >= 200 rows, else the largest "
            "single-atom-dropped cell, else global). Encodes a local risk "
            "norm the trees cannot assemble. Target-free stats. Attempt 5 of "
            "the feature plan (the real B1). Drops only the time columns."
        ),
        drop_columns=TIME_COLUMNS,
        transform=IC_RISKZ_FINE_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="intersection_conditional_risk_rates",
        description=(
            "intersection_conditional_base plus three fold-safe (k=20) fraud "
            "rates of floor(merchant_risk_score) conditioned on "
            "merchant_category, auth_method, and merchant_category x "
            "auth_method (icrate_risk_merchant / _auth / _merchant_auth). "
            "The rate analog of the risk distribution-shift. Drops only the "
            "time columns (channel and card_type kept)."
        ),
        drop_columns=TIME_COLUMNS,
        transform=IC_RISK_RATES_TRANSFORM,
    ),
    DatasetVersionSpec(
        name="intersection_conditional_velocity_rates",
        description=(
            "intersection_conditional_base plus three fold-safe (k=20) fraud "
            "rates of floor(velocity_score) conditioned on merchant_category, "
            "auth_method, and merchant_category x auth_method "
            "(icrate_vel_merchant / _auth / _merchant_auth). Drops only the "
            "time columns (channel and card_type kept)."
        ),
        drop_columns=TIME_COLUMNS,
        transform=IC_VELOCITY_RATES_TRANSFORM,
    ),
)
