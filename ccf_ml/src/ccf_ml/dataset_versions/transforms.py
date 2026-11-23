"""Named, fold-safe derived-feature transforms for dataset versions.

A transform sees the raw frame plus the frozen partition and returns the
transformed frame together with extra payload tables that the builder writes
next to ``data.parquet``. Transforms never change the row count or the id
order; id and target columns pass through untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.alarm import alarm_flag_mask, as_boolean_array
from ccf_ml.cross_validation.constants import (
    ALARM_FLAG_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
)
from ccf_ml.cross_validation.flag_stratified import FlagStratifiedCv
from ccf_ml.dataset_versions.constants import (
    GRID_EXACT_VALUE_COLUMNS,
    GRID_NUMERIC_COLUMNS,
    GRID_SLOTS_EXACT_TRANSFORM,
    GRID_SLOTS_TRANSFORM,
    GRID_VALUE_PREFIX,
    IS_NEW_MERCHANT_COLUMN,
    MERCHANT_RISK_SCORE_COLUMN,
    PROFILE_BASE_COLUMNS,
    PROFILE_CONTEXT_SIZE_COLUMN,
    PROFILE_CONTEXT_TRANSFORM,
    PROFILE_MIN_GROUP,
    PROFILE_NUMERIC_COLUMNS,
    RISK_AGGREGATES_TRANSFORM,
    RISK_CONTEXT_ADDED_COLUMNS,
    RISK_CONTEXT_GROUP_COLUMNS,
    RISK_CONTEXT_GROUPS_FILENAME,
    RISK_CONTEXT_TRANSFORM,
    SLOT_CENTROID_SHRINKAGE,
    SLOT_MIN_ANCHOR_ROWS,
    SLOT_REFS_FILENAME,
    SLOT_REFS_TRANSFORM,
    SLOT_Z_STATS_FILENAME,
    SLOT_Z_TRANSFORM,
    ANCHOR_FREQ_FILENAME,
    ANCHOR_FREQ_TRANSFORM,
    ANCHOR_FULL_TRANSFORM,
    ANCHOR_RATE_FILENAME,
    ANCHOR_RATE_SHRINKAGE,
    ANCHOR_V2_IS_ALARM_TRANSFORM,
    ANCHOR_V2_N_ALARM_TRANSFORM,
    ANCHOR_V2_RATES_FILENAME,
    N_ALARM_FLAGS_COLUMN,
    ANCHOR_V2_TRANSFORM,
    IS_ALARM_COLUMN,
    ANCHOR_V2C_TRANSFORM,
    ANCHOR_V2T_TRANSFORM,
    ANCHOR_COMBINED_FILENAME,
    ANCHOR_TILT_FILENAME,
    ANCHOR_V3_RATES_FILENAME,
    ANCHOR_V3_TRANSFORM,
    ARATE10A_PREFIX,
    ARATE10_PREFIX,
    COMBINED_RATE_COLUMNS,
    COMBINED_RATE_SHRINKAGE,
    ANCHOR_COMBINED_COLUMN,
    IC_CELLZ_FILENAME,
    IC_CELLZ_MIN_ROWS,
    IC_CVV_RISKZ_TRANSFORM,
    IC_CVV_TRANSFORM,
    IC_DISPUTES_TRANSFORM,
    IC_RATE_PREFIX,
    IC_RATE_SHRINKAGE,
    IC_RATES_FILENAME,
    IC_RISK_RATES_TRANSFORM,
    IC_VELOCITY_RATES_TRANSFORM,
    IC_RISKZ_FINE_ATOMS,
    IC_RISKZ_FINE_FILENAME,
    IC_RISKZ_FINE_TRANSFORM,
    IC_RISKZ_TRANSFORM,
    IC_ZABS_PREFIX,
    IC_ZSIGNED_PREFIX,
    NATIVE_RATE_COLUMNS,
    SMALLINT_RATE_COLUMNS,
    TILT_COLUMNS,
    anchor_freq_column,
    anchor_rate_column,
    anchor_tilt_column,
    grid_anchor_column,
    grid_slot_column,
    grid_value_column,
    slot_lda_column,
    slot_z_column,
)

TransformFn = Callable[
    [pd.DataFrame, FlagStratifiedCv],
    "tuple[pd.DataFrame, dict[str, pd.DataFrame]]",
]


def _fold_of_row(frame: pd.DataFrame, folds: FlagStratifiedCv) -> np.ndarray:
    """Validation-fold id of every row, -1 when a row is in no val fold."""
    fold_of = np.full(len(frame), -1, dtype=np.int64)
    ids = frame[ID_COLUMN].to_numpy(dtype=np.int64)
    pos_by_id = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    pos_by_id[ids] = np.arange(len(ids), dtype=np.int64)
    for fold in folds.folds:
        for record_id in fold.val_ids:
            pos = pos_by_id[int(record_id)]
            if pos < 0:
                raise ValueError(
                    f"fold {fold.fold_id} val id {record_id} is not in the frame"
                )
            if fold_of[pos] >= 0:
                raise ValueError(f"id {record_id} appears in two validation folds")
            fold_of[pos] = fold.fold_id
    if np.any(fold_of < 0):
        raise ValueError("some rows are in no validation fold")
    return fold_of


def _group_codes(frame: pd.DataFrame) -> np.ndarray:
    """Dense integer codes of the (merchant_category, auth_method) groups."""
    first, second = RISK_CONTEXT_GROUP_COLUMNS
    first_codes, _ = pd.factorize(frame[first])
    second_codes, _ = pd.factorize(frame[second])
    n_second = int(frame[second].nunique())
    return first_codes.astype(np.int64) * n_second + second_codes.astype(np.int64)


def _fold_safe_group_stats(
    frame: pd.DataFrame,
    folds: FlagStratifiedCv,
) -> dict[str, np.ndarray]:
    """Per-row (merchant, auth) risk statistics from the row's fold-train."""
    fold_of = _fold_of_row(frame, folds)
    group_codes = _group_codes(frame)
    risk = frame[MERCHANT_RISK_SCORE_COLUMN].to_numpy(dtype=np.float64)
    stat_cols = {
        name: np.full(len(frame), np.nan, dtype=np.float64)
        for name in RISK_CONTEXT_ADDED_COLUMNS
    }
    for fold_id in range(folds.n_splits):
        train_mask = fold_of != fold_id
        sel = fold_of == fold_id
        train_codes = group_codes[train_mask]
        train_risk = risk[train_mask]
        grouped = pd.Series(train_risk).groupby(train_codes)
        fold_stats = {
            "risk_group_mean": grouped.mean(),
            "risk_group_p50": grouped.quantile(0.50),
            "risk_group_p75": grouped.quantile(0.75),
            "risk_group_p90": grouped.quantile(0.90),
        }
        global_stats = {
            "risk_group_mean": float(train_risk.mean()),
            "risk_group_p50": float(np.quantile(train_risk, 0.50)),
            "risk_group_p75": float(np.quantile(train_risk, 0.75)),
            "risk_group_p90": float(np.quantile(train_risk, 0.90)),
        }
        val_codes = group_codes[sel]
        for name in RISK_CONTEXT_ADDED_COLUMNS:
            mapped = fold_stats[name].reindex(val_codes).to_numpy(dtype=np.float64)
            stat_cols[name][sel] = np.where(
                np.isnan(mapped), global_stats[name], mapped
            )
    for name in RISK_CONTEXT_ADDED_COLUMNS:
        if np.isnan(stat_cols[name]).any():
            raise ValueError(f"group stats left NaNs in {name}")
    return stat_cols


def _require_risk_columns(frame: pd.DataFrame, *, need_new_flag: bool) -> None:
    required = RISK_CONTEXT_GROUP_COLUMNS + (MERCHANT_RISK_SCORE_COLUMN,)
    if need_new_flag:
        required = required + (IS_NEW_MERCHANT_COLUMN,)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"transform is missing columns: {missing}")


def risk_context_transform(
    frame: pd.DataFrame,
    folds: FlagStratifiedCv,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fold-safe merchant-risk context features with conditional risk.

    Per row (a row lives in exactly one validation fold, so "its fold's
    training part" is the other folds' rows):

    - ``risk_group_mean`` / ``p50`` / ``p75`` / ``p90``: statistics of
      ``merchant_risk_score`` over the row's (merchant_category, auth_method)
      group computed on that training part only; groups absent from the
      training part fall back to the training part's global statistics.
    - ``merchant_risk_score`` itself becomes conditional: rows with
      ``is_new_merchant = True`` get their fold's group mean instead of the
      raw value; all other rows keep the raw value. The flag remains
      exactly recoverable downstream via ``risk == risk_group_mean``.

    Extra payload: full-train group statistics for mapping test rows.
    """
    _require_risk_columns(frame, need_new_flag=True)
    stat_cols = _fold_safe_group_stats(frame, folds)
    out = frame.copy()
    is_new = frame[IS_NEW_MERCHANT_COLUMN].to_numpy(dtype=bool)
    risk = frame[MERCHANT_RISK_SCORE_COLUMN].to_numpy(dtype=np.float64)
    out[MERCHANT_RISK_SCORE_COLUMN] = np.where(
        is_new, stat_cols["risk_group_mean"], risk
    )
    for name in RISK_CONTEXT_ADDED_COLUMNS:
        out[name] = stat_cols[name]
    extras = {RISK_CONTEXT_GROUPS_FILENAME: _full_train_group_table(frame, risk)}
    return out, extras


def risk_aggregates_transform(
    frame: pd.DataFrame,
    folds: FlagStratifiedCv,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fold-safe merchant-risk group aggregates, raw risk untouched.

    Same four (merchant_category, auth_method) statistics as
    :func:`risk_context_transform`, but ``merchant_risk_score`` keeps its
    raw value everywhere. Isolates the aggregate features from the
    conditional replacement.
    """
    _require_risk_columns(frame, need_new_flag=False)
    stat_cols = _fold_safe_group_stats(frame, folds)
    out = frame.copy()
    for name in RISK_CONTEXT_ADDED_COLUMNS:
        out[name] = stat_cols[name]
    risk = frame[MERCHANT_RISK_SCORE_COLUMN].to_numpy(dtype=np.float64)
    extras = {RISK_CONTEXT_GROUPS_FILENAME: _full_train_group_table(frame, risk)}
    return out, extras


def _full_train_group_table(
    frame: pd.DataFrame, risk: np.ndarray
) -> pd.DataFrame:
    """Full-train (merchant, auth) risk statistics, for test-time mapping."""
    first, second = RISK_CONTEXT_GROUP_COLUMNS
    working = pd.DataFrame(
        {
            first: frame[first].to_numpy(),
            second: frame[second].to_numpy(),
            "risk": risk,
        }
    )
    grouped = working.groupby([first, second], observed=True, sort=True)["risk"]
    table = pd.DataFrame(
        {
            "n_rows": grouped.size(),
            "risk_group_mean": grouped.mean(),
            "risk_group_p50": grouped.quantile(0.50),
            "risk_group_p75": grouped.quantile(0.75),
            "risk_group_p90": grouped.quantile(0.90),
        }
    ).reset_index()
    return table


def _profile_levels() -> tuple[tuple[str, ...], ...]:
    """Profile hierarchy, most specific first; last level is global."""
    level0 = PROFILE_BASE_COLUMNS + tuple(ALARM_FLAG_COLUMNS)
    level1 = PROFILE_BASE_COLUMNS
    level2 = (PROFILE_BASE_COLUMNS[0], PROFILE_BASE_COLUMNS[1])
    return (level0, level1, level2, ())


def _codes_for_columns(
    frame: pd.DataFrame, columns: tuple[str, ...]
) -> np.ndarray:
    """Dense-ish integer codes of the given column tuple (product encoding)."""
    if not columns:
        return np.zeros(len(frame), dtype=np.int64)
    codes = np.zeros(len(frame), dtype=np.int64)
    for column in columns:
        col_codes, uniques = pd.factorize(frame[column])
        codes = codes * max(len(uniques), 1) + col_codes.astype(np.int64)
    return codes


def _profile_percentile_features(
    train_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    min_group: int = PROFILE_MIN_GROUP,
) -> pd.DataFrame:
    """Within-profile percentile of each profile numeric for target rows.

    A target row's percentile is measured against the train rows of the same
    profile: ``(n_less + 0.5 * n_equal) / n_train_group``. The profile level
    is the most specific one whose train group holds at least ``min_group``
    rows (hierarchy: full profile -> without alarm flags -> merchant x auth ->
    global). Returns ``prof_pct_<numeric>`` columns plus ``prof_n``, the
    train size behind the chosen level, indexed like ``target_frame``.
    """
    n_target = len(target_frame)
    pct_names = [f"prof_pct_{num}" for num in PROFILE_NUMERIC_COLUMNS]
    pct_cols: dict[str, np.ndarray] = {
        name: np.full(n_target, np.nan, dtype=np.float64) for name in pct_names
    }
    size_col = np.full(n_target, -1, dtype=np.int64)
    assigned = np.zeros(n_target, dtype=bool)
    train_values = {
        num: train_frame[num].to_numpy(dtype=np.float64)
        for num in PROFILE_NUMERIC_COLUMNS
    }
    target_values = {
        num: target_frame[num].to_numpy(dtype=np.float64)
        for num in PROFILE_NUMERIC_COLUMNS
    }
    for columns in _profile_levels():
        if assigned.all():
            break
        train_codes = _codes_for_columns(train_frame, columns)
        target_codes = _codes_for_columns(target_frame, columns)
        sizes = np.bincount(
            train_codes, minlength=int(train_codes.max()) + 1 if len(train_codes) else 1
        )
        effective_min = min_group if columns else 1
        train_order = np.argsort(train_codes, kind="stable")
        train_sorted_codes = train_codes[train_order]
        order = np.argsort(target_codes, kind="stable")
        sorted_codes = target_codes[order]
        bounds = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
        ends = np.r_[bounds[1:], n_target]
        for start, end in zip(bounds, ends):
            code = int(sorted_codes[start])
            if code >= len(sizes) or sizes[code] < effective_min:
                continue
            positions = order[start:end]
            positions = positions[~assigned[positions]]
            if positions.size == 0:
                continue
            lo_train = int(np.searchsorted(train_sorted_codes, code, side="left"))
            hi_train = int(np.searchsorted(train_sorted_codes, code, side="right"))
            train_pos = train_order[lo_train:hi_train]
            for num in PROFILE_NUMERIC_COLUMNS:
                sorted_vals = np.sort(train_values[num][train_pos])
                vals = target_values[num][positions]
                lo = np.searchsorted(sorted_vals, vals, side="left")
                hi = np.searchsorted(sorted_vals, vals, side="right")
                pct_cols[f"prof_pct_{num}"][positions] = (
                    lo + 0.5 * (hi - lo)
                ) / len(sorted_vals)
            size_col[positions] = len(train_pos)
            assigned[positions] = True
    if not assigned.all():
        raise ValueError("profile percentile fallback failed to assign every row")
    out = pd.DataFrame(index=target_frame.index)
    for name in pct_names:
        out[name] = pct_cols[name]
    out[PROFILE_CONTEXT_SIZE_COLUMN] = size_col
    return out


def profile_context_transform(
    frame: pd.DataFrame,
    folds: FlagStratifiedCv,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fold-safe within-profile percentile features.

    Profiles are the categorical space minus ``device_type`` and
    ``is_new_merchant``: (merchant_category x auth_method x channel x
    card_type x alarm flags), with the size-guarded hierarchy of
    :func:`_profile_percentile_features`. Each row gains its percentile of
    the 9 profile numerics inside its profile (computed on the row's
    fold-training part) plus the profile's train size. Raw columns are
    untouched. Test rows are mapped by :func:`apply_profile_context_to_test`
    against the full train.
    """
    required = (
        PROFILE_BASE_COLUMNS
        + tuple(ALARM_FLAG_COLUMNS)
        + PROFILE_NUMERIC_COLUMNS
    )
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"profile_context transform is missing columns: {missing}")
    fold_of = _fold_of_row(frame, folds)
    out = frame.copy()
    pieces: list[pd.DataFrame] = []
    for fold_id in range(folds.n_splits):
        sel = fold_of == fold_id
        train_part = frame.loc[~sel]
        target_part = frame.loc[sel]
        pieces.append(
            _profile_percentile_features(train_part, target_part)
        )
    features = pd.concat(pieces)
    if len(features) != len(frame) or features.index.duplicated().any():
        raise ValueError("profile features lost or duplicated rows")
    for column in features.columns:
        out.loc[features.index, column] = features[column]
        if out[column].isna().any():
            raise ValueError(f"profile feature {column!r} has NaNs")
    return out, {}


def apply_profile_context_to_test(
    test_frame: pd.DataFrame, train_frame: pd.DataFrame
) -> pd.DataFrame:
    """Add the within-profile percentile features to test rows.

    Uses the same hierarchy and percentile definition as the train-side
    transform, measured against the full training frame.
    """
    features = _profile_percentile_features(train_frame, test_frame)
    out = test_frame.copy()
    for column in features.columns:
        out[column] = features[column].to_numpy()
    return out


def _add_grid_columns(
    target: pd.DataFrame, *, include_exact: bool
) -> pd.DataFrame:
    """Return a copy with grid decomposition columns for the grid numerics.

    ``grid_anchor_<col>`` = floor(x); ``grid_slot_<col>`` = cents above the
    anchor (0..99; the 0.1-grid columns only use multiples of 10). Values
    must sit on the exact 2-decimal grid (verified up to 2e-3 float noise);
    anything further off-grid raises.
    """
    missing = [name for name in GRID_NUMERIC_COLUMNS if name not in target.columns]
    if missing:
        raise ValueError(f"grid transform is missing columns: {missing}")
    out = target.copy()
    for column in GRID_NUMERIC_COLUMNS:
        x = out[column].to_numpy(dtype=np.float64)
        off_grid = np.abs(x * 100.0 - np.rint(x * 100.0))
        if off_grid.max() > 0.5:
            raise ValueError(
                f"{column} has values further than half a cent off the "
                f"2-decimal grid (max deviation {off_grid.max():.4f})"
            )
        anchor = np.floor(x)
        out[grid_anchor_column(column)] = anchor.astype(np.int64)
        out[grid_slot_column(column)] = np.clip(
            np.rint((x - anchor) * 100.0), 0, 99
        ).astype(np.int64)
    if include_exact:
        for column in GRID_EXACT_VALUE_COLUMNS:
            x = out[column].to_numpy(dtype=np.float64)
            out[grid_value_column(column)] = np.rint(x * 10.0).astype(np.int64)
    return out


def grid_slots_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Add the grid anchor/slot columns for the six continuous numerics.

    Deterministic per-row (no fold-train state), so the frozen partition is
    unused; the signature matches the transform contract.
    """
    return _add_grid_columns(frame, include_exact=False), {}


def grid_slots_exact_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Grid slots plus exact-grid-value ids for the 0.1-grid columns.

    velocity_score and merchant_risk_score have no cent-level structure
    (~11 slots, all multiples of 10 cents), so their full grid value is a
    small id space (674 / 959 levels) worth memorizing directly.
    """
    return _add_grid_columns(frame, include_exact=True), {}


def apply_grid_slots_to_test(
    test_frame: pd.DataFrame, *, include_exact: bool
) -> pd.DataFrame:
    """Test-side counterpart of the grid transforms (same deterministic rule)."""
    return _add_grid_columns(test_frame, include_exact=include_exact)


def _anchor_slot(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(anchor, slot) integer arrays per grid numeric column."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for column in GRID_NUMERIC_COLUMNS:
        x = frame[column].to_numpy(dtype=np.float64)
        anchor = np.floor(x)
        slot = np.clip(np.rint((x - anchor) * 100.0), 0, 99)
        out[column] = (anchor.astype(np.int64), slot.astype(np.int64))
    return out


def _map_by_anchor(
    values: np.ndarray, table: dict[int, float], fallback: float
) -> np.ndarray:
    """Vectorized anchor->float lookup with a global fallback."""
    return (
        pd.Series(values).map(table).fillna(fallback).to_numpy(dtype=np.float64)
    )


def _slot_z_table(
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Long per-(column, anchor) slot mean/std table; anchor -1 is global."""
    rows: list[dict[str, object]] = []
    for column, (anchor, slot) in pairs.items():
        grouped = pd.Series(slot).groupby(anchor)
        means = grouped.mean()
        stds = grouped.std(ddof=0)
        counts = grouped.size()
        rows.append(
            {
                "column": column,
                "anchor": -1,
                "slot_mean": float(slot.mean()),
                "slot_std": float(slot.std()),
                "n_rows": len(slot),
            }
        )
        for anchor_value in means.index:
            n = int(counts.loc[anchor_value])
            if n < SLOT_MIN_ANCHOR_ROWS:
                continue
            rows.append(
                {
                    "column": column,
                    "anchor": int(anchor_value),
                    "slot_mean": float(means.loc[anchor_value]),
                    "slot_std": float(stds.loc[anchor_value]),
                    "n_rows": n,
                }
            )
    return pd.DataFrame(rows)


def _apply_slot_z(
    target: pd.DataFrame,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Add slotz_<col> = (slot - mean_a) / std_a using a stats table."""
    out = target.copy()
    global_rows = table[table["anchor"] == -1].set_index("column")
    for column, (anchor, slot) in pairs.items():
        global_mean = float(global_rows.loc[column, "slot_mean"])
        global_std = float(global_rows.loc[column, "slot_std"])
        sub = table[(table["column"] == column) & (table["anchor"] != -1)]
        mean_map = dict(zip(sub["anchor"], sub["slot_mean"]))
        std_map = dict(zip(sub["anchor"], sub["slot_std"]))
        means = _map_by_anchor(anchor, mean_map, global_mean)
        stds = _map_by_anchor(anchor, std_map, global_std)
        stds = np.where(stds < 1e-9, global_std, stds)
        out[slot_z_column(column)] = (slot - means) / stds
    return out


def slot_z_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Target-free z-score of the fractional slot within its anchor.

    ``slotz_<col> = (slot - mean(slot | anchor)) / std(slot | anchor)``
    with the anchor's statistics from the full training frame (no labels
    involved); anchors with fewer than ``SLOT_MIN_ANCHOR_ROWS`` rows and
    unseen anchors fall back to the column's global mean/std. Exposes the
    verified within-anchor decimal structure as one numeric threshold.
    """
    pairs = _anchor_slot(frame)
    table = _slot_z_table(pairs)
    out = _apply_slot_z(frame, pairs, table)
    return out, {SLOT_Z_STATS_FILENAME: table}


def _slot_centroids(
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    target: np.ndarray,
) -> pd.DataFrame:
    """Long per-(column, anchor) shrunk fraud/clean slot centroids."""
    shrink = SLOT_CENTROID_SHRINKAGE
    rows: list[dict[str, object]] = []
    for column, (anchor, slot) in pairs.items():
        stats = pd.DataFrame({"anchor": anchor, "slot": slot, "y": target})
        fraud = stats[stats["y"]].groupby("anchor")["slot"]
        clean = stats[~stats["y"]].groupby("anchor")["slot"]
        fraud_mean = fraud.mean()
        fraud_n = fraud.size()
        clean_mean = clean.mean()
        clean_n = clean.size()
        global_fraud = float(slot[target].mean())
        global_clean = float(slot[~target].mean())
        rows.append(
            {
                "column": column,
                "anchor": -1,
                "fraud_centroid": global_fraud,
                "clean_centroid": global_clean,
                "n_fraud": int(target.sum()),
                "n_clean": int((~target).sum()),
            }
        )
        for anchor_value in fraud_mean.index.union(clean_mean.index):
            n_fraud = int(fraud_n.get(anchor_value, 0))
            n_clean = int(clean_n.get(anchor_value, 0))
            local_fraud = float(fraud_mean.get(anchor_value, global_fraud))
            local_clean = float(clean_mean.get(anchor_value, global_clean))
            rows.append(
                {
                    "column": column,
                    "anchor": int(anchor_value),
                    "fraud_centroid": (
                        n_fraud * local_fraud + shrink * global_fraud
                    )
                    / (n_fraud + shrink),
                    "clean_centroid": (
                        n_clean * local_clean + shrink * global_clean
                    )
                    / (n_clean + shrink),
                    "n_fraud": n_fraud,
                    "n_clean": n_clean,
                }
            )
    return pd.DataFrame(rows)


def _apply_slot_refs(
    target: pd.DataFrame,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Add slotlda_<col> = |slot - clean_c| - |slot - fraud_c| per anchor."""
    out = target.copy()
    global_rows = table[table["anchor"] == -1].set_index("column")
    for column, (anchor, slot) in pairs.items():
        global_fraud = float(global_rows.loc[column, "fraud_centroid"])
        global_clean = float(global_rows.loc[column, "clean_centroid"])
        sub = table[(table["column"] == column) & (table["anchor"] != -1)]
        fraud_map = dict(zip(sub["anchor"], sub["fraud_centroid"]))
        clean_map = dict(zip(sub["anchor"], sub["clean_centroid"]))
        fraud_c = _map_by_anchor(anchor, fraud_map, global_fraud)
        clean_c = _map_by_anchor(anchor, clean_map, global_clean)
        out[slot_lda_column(column)] = np.abs(slot - clean_c) - np.abs(
            slot - fraud_c
        )
    return out


def slot_refs_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fold-safe per-anchor slot centroids; positive = closer to fraud.

    The verification battery found fraud-vs-clean fractional-slot shifts
    within anchors (up to |z| = 10). Each row's feature is
    ``|slot - clean_centroid(anchor)| - |slot - fraud_centroid(anchor)|``,
    centroids computed on the row's fold-training part only, shrunk toward
    the global class centroids with k = ``SLOT_CENTROID_SHRINKAGE`` (most
    anchors hold few frauds). Extra payload carries full-train centroids
    for test rows.
    """
    target = frame[TARGET_COLUMN].to_numpy(dtype=bool)
    pairs = _anchor_slot(frame)
    fold_of = _fold_of_row(frame, folds)
    pieces: list[pd.DataFrame] = []
    for fold_id in range(folds.n_splits):
        sel = fold_of == fold_id
        train_part = frame.loc[~sel]
        train_pairs = _anchor_slot(train_part)
        train_target = train_part[TARGET_COLUMN].to_numpy(dtype=bool)
        table = _slot_centroids(train_pairs, train_target)
        val_pairs = {
            column: (pairs[column][0][sel], pairs[column][1][sel])
            for column in pairs
        }
        pieces.append(_apply_slot_refs(frame.loc[sel], val_pairs, table))
    features = pd.concat(pieces)
    if len(features) != len(frame) or features.index.duplicated().any():
        raise ValueError("slot_refs features lost or duplicated rows")
    out = frame.copy()
    for column in features.columns:
        out.loc[features.index, column] = features[column]
        if out[column].isna().any():
            raise ValueError(f"slot feature {column!r} has NaNs")
    full_table = _slot_centroids(pairs, target)
    return out, {SLOT_REFS_FILENAME: full_table}


def apply_slot_z_to_test(
    test_frame: pd.DataFrame, stats_path: Path
) -> pd.DataFrame:
    """Test-side slot_z using the version's frozen stats table."""
    table = pd.read_parquet(stats_path)
    pairs = _anchor_slot(test_frame)
    return _apply_slot_z(test_frame, pairs, table)


def apply_slot_refs_to_test(
    test_frame: pd.DataFrame, refs_path: Path
) -> pd.DataFrame:
    """Test-side slot_lda using the version's full-train centroids."""
    table = pd.read_parquet(refs_path)
    pairs = _anchor_slot(test_frame)
    return _apply_slot_refs(test_frame, pairs, table)


def _apply_anchor_freq(
    target: pd.DataFrame,
    anchors: dict[str, np.ndarray],
    counts: pd.DataFrame,
) -> pd.DataFrame:
    """Add afreq_<col> = log1p(train count of the row's anchor)."""
    out = target.copy()
    for column, anchor in anchors.items():
        sub = counts[counts["column"] == column]
        count_map = dict(zip(sub["anchor"], sub["n_rows"]))
        n = (
            pd.Series(anchor)
            .map(count_map)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        out[anchor_freq_column(column)] = np.log1p(n)
    return out


def anchor_freq_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Target-free log train-frequency of each numeric's floor anchor.

    Rarity is non-monotone in value, so trees cannot carve it from the raw
    floats; the screen found marginal fraud AUC 0.63 on velocity/risk
    anchors and 0.55-0.56 on hours/amount. Counts come from the training
    frame only (no labels); test rows read the same table, unseen anchors
    scoring log1p(0).
    """
    anchors = {
        column: np.floor(frame[column].to_numpy(dtype=np.float64)).astype(np.int64)
        for column in GRID_NUMERIC_COLUMNS
    }
    rows: list[dict[str, object]] = []
    for column, anchor in anchors.items():
        counts = pd.Series(anchor).value_counts()
        for anchor_value, n in counts.items():
            rows.append(
                {"column": column, "anchor": int(anchor_value), "n_rows": int(n)}
            )
    table = pd.DataFrame(rows)
    out = _apply_anchor_freq(frame, anchors, table)
    return out, {ANCHOR_FREQ_FILENAME: table}


def apply_anchor_freq_to_test(
    test_frame: pd.DataFrame, counts_path: Path
) -> pd.DataFrame:
    """Test-side anchor frequency using the version's frozen train counts."""
    table = pd.read_parquet(counts_path)
    anchors = {
        column: np.floor(test_frame[column].to_numpy(dtype=np.float64)).astype(np.int64)
        for column in GRID_NUMERIC_COLUMNS
    }
    return _apply_anchor_freq(test_frame, anchors, table)


def _anchor_rate_table(
    anchors: dict[str, np.ndarray], target: np.ndarray, *, shrink: int
) -> pd.DataFrame:
    """Long per-(column, anchor) shrunk fraud-rate table."""
    global_rate = float(target.mean())
    rows: list[dict[str, object]] = []
    for column, anchor in anchors.items():
        stats = pd.DataFrame({"anchor": anchor, "y": target})
        agg = stats.groupby("anchor")["y"].agg(["sum", "size"])
        for anchor_value, row in agg.iterrows():
            rows.append(
                {
                    "column": column,
                    "anchor": int(anchor_value),
                    "fraud_rate": (float(row["sum"]) + shrink * global_rate)
                    / (float(row["size"]) + shrink),
                    "n_rows": int(row["size"]),
                    "n_fraud": int(row["sum"]),
                }
            )
    return pd.DataFrame(rows)


def _apply_anchor_rate(
    target_frame: pd.DataFrame,
    anchors: dict[str, np.ndarray],
    table: pd.DataFrame,
    *,
    fallback_rate: float,
) -> pd.DataFrame:
    """Add arate_<col> = shrunk fraud rate of the row's anchor."""
    out = target_frame.copy()
    for column, anchor in anchors.items():
        sub = table[table["column"] == column]
        rate_map = dict(zip(sub["anchor"], sub["fraud_rate"]))
        rates = (
            pd.Series(anchor)
            .map(rate_map)
            .fillna(fallback_rate)
            .to_numpy(dtype=np.float64)
        )
        out[anchor_rate_column(column)] = rates
    return out


def anchor_full_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Anchor frequency plus fold-safe shrunk anchor fraud rate.

    Adds afreq_<col> (target-free log train frequency, as
    :func:`anchor_freq_transform`) and arate_<col> = (frauds + k*global) /
    (rows + k) of the row's anchor, k = ``ANCHOR_RATE_SHRINKAGE``, computed
    on the row's fold-training part only; test rows use full-train rates
    from the payload.
    """
    shrink = ANCHOR_RATE_SHRINKAGE
    anchors = {
        column: np.floor(frame[column].to_numpy(dtype=np.float64)).astype(np.int64)
        for column in GRID_NUMERIC_COLUMNS
    }
    target = frame[TARGET_COLUMN].to_numpy(dtype=bool)
    freq_rows: list[dict[str, object]] = []
    for column, anchor in anchors.items():
        counts = pd.Series(anchor).value_counts()
        for anchor_value, n in counts.items():
            freq_rows.append(
                {"column": column, "anchor": int(anchor_value), "n_rows": int(n)}
            )
    freq_table = pd.DataFrame(freq_rows)
    out = _apply_anchor_freq(frame, anchors, freq_table)

    fold_of = _fold_of_row(frame, folds)
    global_rate = float(target.mean())
    pieces: list[pd.DataFrame] = []
    for fold_id in range(folds.n_splits):
        sel = fold_of == fold_id
        train_part = frame.loc[~sel]
        train_anchors = {
            column: np.floor(
                train_part[column].to_numpy(dtype=np.float64)
            ).astype(np.int64)
            for column in GRID_NUMERIC_COLUMNS
        }
        table = _anchor_rate_table(
            train_anchors,
            train_part[TARGET_COLUMN].to_numpy(dtype=bool),
            shrink=shrink,
        )
        val_anchors = {column: anchors[column][sel] for column in anchors}
        pieces.append(
            _apply_anchor_rate(
                frame.loc[sel], val_anchors, table, fallback_rate=global_rate
            )
        )
    features = pd.concat(pieces)
    if len(features) != len(frame) or features.index.duplicated().any():
        raise ValueError("anchor rate features lost or duplicated rows")
    for column in features.columns:
        out.loc[features.index, column] = features[column]
        if out[column].isna().any():
            raise ValueError(f"anchor feature {column!r} has NaNs")
    full_table = _anchor_rate_table(anchors, target, shrink=shrink)
    extras = {
        ANCHOR_FREQ_FILENAME: freq_table,
        ANCHOR_RATE_FILENAME: full_table,
    }
    return out, extras


def apply_anchor_rate_to_test(
    test_frame: pd.DataFrame, rate_path: Path
) -> pd.DataFrame:
    """Test-side anchor fraud rate using the version's full-train table."""
    table = pd.read_parquet(rate_path)
    anchors = {
        column: np.floor(test_frame[column].to_numpy(dtype=np.float64)).astype(np.int64)
        for column in GRID_NUMERIC_COLUMNS
    }
    fallback = float(table["fraud_rate"].dropna().mean())
    return _apply_anchor_rate(
        test_frame, anchors, table, fallback_rate=fallback
    )


def _rate_keys(source: pd.Series, mode: str) -> np.ndarray:
    """Integer key per row for a rate group: floor / tenth / value."""
    values = source.to_numpy(dtype=np.float64)
    if mode == "floor":
        return np.floor(values).astype(np.int64)
    if mode == "tenth":
        return np.rint(values * 10.0).astype(np.int64)
    return np.rint(values).astype(np.int64)


def _v2_rate_sources() -> tuple[tuple[str, str, str], ...]:
    """(output column, source column, key mode) for the v2 rate family."""
    sources = [
        (anchor_rate_column(column), column, "floor")
        for column in GRID_NUMERIC_COLUMNS
    ]
    sources += [
        (ARATE10_PREFIX + column, column, "tenth")
        for column in NATIVE_RATE_COLUMNS
    ]
    sources += [
        (anchor_rate_column(column), column, "value")
        for column in SMALLINT_RATE_COLUMNS
    ]
    return tuple(sources)


def _v2_rate_table(
    frame: pd.DataFrame,
    sources: tuple[tuple[str, str, str], ...],
    target: np.ndarray,
    *,
    shrink: int,
) -> pd.DataFrame:
    """Long per-(output column, key) shrunk fraud-rate table."""
    global_rate = float(target.mean())
    rows: list[dict[str, object]] = []
    for out_column, source_column, mode in sources:
        keys = _rate_keys(frame[source_column], mode)
        stats = pd.DataFrame({"key": keys, "y": target})
        agg = stats.groupby("key")["y"].agg(["sum", "size"])
        for key_value, row in agg.iterrows():
            rows.append(
                {
                    "output": out_column,
                    "key": int(key_value),
                    "fraud_rate": (float(row["sum"]) + shrink * global_rate)
                    / (float(row["size"]) + shrink),
                    "n_rows": int(row["size"]),
                    "n_fraud": int(row["sum"]),
                }
            )
    return pd.DataFrame(rows)


def _apply_v2_rates(
    target_frame: pd.DataFrame,
    keys_by_output: dict[str, np.ndarray],
    table: pd.DataFrame,
    *,
    fallback_rate: float,
) -> pd.DataFrame:
    """Add each rate output column from its (output, key) table."""
    out = target_frame.copy()
    for out_column, keys in keys_by_output.items():
        sub = table[table["output"] == out_column]
        rate_map = dict(zip(sub["key"], sub["fraud_rate"]))
        rates = (
            pd.Series(keys)
            .map(rate_map)
            .fillna(fallback_rate)
            .to_numpy(dtype=np.float64)
        )
        out[out_column] = rates
    return out


def anchor_v2_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Frequency plus the full fold-safe rate family (anchor_v2).

    Adds afreq_<col> (target-free log train frequency), arate_<col> for
    the six continuous numerics at integer anchors (both as
    :func:`anchor_full_transform`), plus arate10_<col> for velocity/risk
    at their native 0.1 grid (levels hold ~740 rows each, so k =
    ``ANCHOR_RATE_SHRINKAGE`` barely shrinks them) and arate_<col> for
    customer_age / cvv_retry_count / txn_count_last_24h per value. All
    rates fold-safe; test rows use the full-train payload.
    """
    shrink = ANCHOR_RATE_SHRINKAGE
    sources = _v2_rate_sources()
    anchors = {
        column: np.floor(frame[column].to_numpy(dtype=np.float64)).astype(np.int64)
        for column in GRID_NUMERIC_COLUMNS
    }
    target = frame[TARGET_COLUMN].to_numpy(dtype=bool)
    freq_rows: list[dict[str, object]] = []
    for column, anchor in anchors.items():
        counts = pd.Series(anchor).value_counts()
        for anchor_value, n in counts.items():
            freq_rows.append(
                {"column": column, "anchor": int(anchor_value), "n_rows": int(n)}
            )
    out = _apply_anchor_freq(frame, anchors, pd.DataFrame(freq_rows))

    fold_of = _fold_of_row(frame, folds)
    global_rate = float(target.mean())
    pieces: list[pd.DataFrame] = []
    for fold_id in range(folds.n_splits):
        sel = fold_of == fold_id
        train_part = frame.loc[~sel]
        table = _v2_rate_table(
            train_part,
            sources,
            train_part[TARGET_COLUMN].to_numpy(dtype=bool),
            shrink=shrink,
        )
        keys_by_output = {
            out_column: _rate_keys(frame.loc[sel][source_column], mode)
            for out_column, source_column, mode in sources
        }
        pieces.append(
            _apply_v2_rates(
                frame.loc[sel], keys_by_output, table, fallback_rate=global_rate
            )
        )
    features = pd.concat(pieces)
    if len(features) != len(frame) or features.index.duplicated().any():
        raise ValueError("anchor v2 features lost or duplicated rows")
    for column in features.columns:
        out.loc[features.index, column] = features[column]
        if out[column].isna().any():
            raise ValueError(f"anchor v2 feature {column!r} has NaNs")
    full_table = _v2_rate_table(frame, sources, target, shrink=shrink)
    extras = {
        ANCHOR_FREQ_FILENAME: pd.DataFrame(freq_rows),
        ANCHOR_V2_RATES_FILENAME: full_table,
    }
    return out, extras


def _interaction_rate_table(
    frame: pd.DataFrame,
    target: np.ndarray,
    *,
    shrink: int,
) -> pd.DataFrame:
    """Shrunk fraud rates of (native 0.1 value x alarm) for velocity/risk."""
    global_rate = float(target.mean())
    alarm = alarm_flag_mask(frame, columns=ALARM_FLAG_COLUMNS).to_numpy(dtype=bool)
    rows: list[dict[str, object]] = []
    for column in NATIVE_RATE_COLUMNS:
        tenths = np.rint(frame[column].to_numpy(dtype=np.float64) * 10.0)
        keys = (tenths * 2 + alarm.astype(np.int64)).astype(np.int64)
        stats = pd.DataFrame({"key": keys, "y": target})
        agg = stats.groupby("key")["y"].agg(["sum", "size"])
        for key_value, row in agg.iterrows():
            rows.append(
                {
                    "output": ARATE10A_PREFIX + column,
                    "key": int(key_value),
                    "fraud_rate": (float(row["sum"]) + shrink * global_rate)
                    / (float(row["size"]) + shrink),
                    "n_rows": int(row["size"]),
                    "n_fraud": int(row["sum"]),
                }
            )
    return pd.DataFrame(rows)


def _apply_interaction_rates(
    target_frame: pd.DataFrame,
    keys_by_output: dict[str, np.ndarray],
    table: pd.DataFrame,
    *,
    fallback_rate: float,
) -> pd.DataFrame:
    out = target_frame.copy()
    for out_column, keys in keys_by_output.items():
        sub = table[table["output"] == out_column]
        rate_map = dict(zip(sub["key"], sub["fraud_rate"]))
        out[out_column] = (
            pd.Series(keys)
            .map(rate_map)
            .fillna(fallback_rate)
            .to_numpy(dtype=np.float64)
        )
    return out


def _interaction_keys(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    alarm = alarm_flag_mask(frame, columns=ALARM_FLAG_COLUMNS).to_numpy(dtype=bool)
    keys: dict[str, np.ndarray] = {}
    for column in NATIVE_RATE_COLUMNS:
        tenths = np.rint(frame[column].to_numpy(dtype=np.float64) * 10.0)
        keys[ARATE10A_PREFIX + column] = (
            tenths * 2 + alarm.astype(np.int64)
        ).astype(np.int64)
    return keys


def attach_is_alarm(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with ``is_alarm``, the OR of the five alarm flags.

    The bit is a function of columns already on the row, so every fold and
    the test rows see the same value the alarm-profile CV stratifies on.
    """
    if IS_ALARM_COLUMN in frame.columns:
        raise ValueError(f"{IS_ALARM_COLUMN} is already present")
    out = frame.copy()
    out[IS_ALARM_COLUMN] = alarm_flag_mask(
        frame, columns=ALARM_FLAG_COLUMNS
    ).to_numpy(dtype=bool)
    return out


def anchor_v2_is_alarm_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """anchor_v2 plus the explicit alarm-region bit.

    The five alarm flags stay. ``is_alarm`` gives the tree the collapsed
    region in one column.
    """
    out, extras = anchor_v2_transform(frame, folds)
    return attach_is_alarm(out), extras


def attach_n_alarm_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with ``n_alarm_flags``, how many alarm flags are on.

    The count keeps the dose that ``is_alarm`` collapses. It is a function
    of columns already on the row, so every fold and the test rows agree.
    """
    if N_ALARM_FLAGS_COLUMN in frame.columns:
        raise ValueError(f"{N_ALARM_FLAGS_COLUMN} is already present")
    n_flags = len(ALARM_FLAG_COLUMNS)
    bits: list[np.ndarray | None] = [None] * n_flags
    for index, name in enumerate(ALARM_FLAG_COLUMNS):
        bits[index] = as_boolean_array(frame[name], column=name)
    filled = [bit for bit in bits if bit is not None]
    if len(filled) != n_flags:
        raise ValueError("failed to read every alarm flag column")
    out = frame.copy()
    out[N_ALARM_FLAGS_COLUMN] = np.stack(filled, axis=1).sum(axis=1).astype(np.int64)
    return out


def anchor_v2_n_alarm_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """anchor_v2 plus the count of alarm flags.

    The five flags stay. ``n_alarm_flags`` is their sum, an ordered dose
    from 0 through 5.
    """
    out, extras = anchor_v2_transform(frame, folds)
    return attach_n_alarm_flags(out), extras


def anchor_v3_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """anchor_v2 plus fold-safe (native value x alarm) rates.

    The alarm flag splits the value space into two fraud-rate regimes;
    rates per (0.1-grid value, alarm) group capture that interaction
    directly. Groups hold ~400 rows on average, so k = 20 shrinkage stays
    light. Same fold-safe pattern as anchor_v2; test rows use the
    full-train payload.
    """
    shrink = ANCHOR_RATE_SHRINKAGE
    out, extras = anchor_v2_transform(frame, folds)
    target = frame[TARGET_COLUMN].to_numpy(dtype=bool)
    fold_of = _fold_of_row(frame, folds)
    global_rate = float(target.mean())
    pieces: list[pd.DataFrame] = []
    for fold_id in range(folds.n_splits):
        sel = fold_of == fold_id
        train_part = frame.loc[~sel]
        table = _interaction_rate_table(
            train_part,
            train_part[TARGET_COLUMN].to_numpy(dtype=bool),
            shrink=shrink,
        )
        keys = {
            out_column: keys_full[sel]
            for out_column, keys_full in _interaction_keys(frame).items()
        }
        pieces.append(
            _apply_interaction_rates(
                frame.loc[sel], keys, table, fallback_rate=global_rate
            )
        )
    features = pd.concat(pieces)
    if len(features) != len(frame) or features.index.duplicated().any():
        raise ValueError("anchor v3 features lost or duplicated rows")
    for column in features.columns:
        out.loc[features.index, column] = features[column]
        if out[column].isna().any():
            raise ValueError(f"anchor v3 feature {column!r} has NaNs")
    extras[ANCHOR_V3_RATES_FILENAME] = _interaction_rate_table(
        frame, target, shrink=shrink
    )
    return out, extras


def apply_anchor_v3_rates_to_test(
    test_frame: pd.DataFrame, rates_path: Path
) -> pd.DataFrame:
    """Test-side interaction rates using the full-train table."""
    table = pd.read_parquet(rates_path)
    keys = _interaction_keys(test_frame)
    fallback = float(table["fraud_rate"].dropna().mean())
    return _apply_interaction_rates(
        test_frame, keys, table, fallback_rate=fallback
    )


def _tilt_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Common slot lift per tilt column, from ``frame``'s block rows.

    For each column, the lift of slot s is observed frauds at s divided
    by the frauds expected at s if slots were indifferent within each
    anchor (expected = slot rows x anchor rate, summed over the block's
    anchors). Anchors above the column's block bound do not enter the
    estimate.
    """
    target = frame[TARGET_COLUMN].to_numpy(dtype=bool)
    rows: list[dict[str, object]] = []
    for column, bound in TILT_COLUMNS.items():
        x = frame[column].to_numpy(dtype=np.float64)
        anchor = np.floor(x)
        slot = np.rint((x - anchor) * 10.0).astype(np.int64)
        block = anchor <= bound
        stats = pd.DataFrame(
            {"anchor": anchor[block].astype(np.int64), "slot": slot[block], "y": target[block]}
        )
        rate_a = stats.groupby("anchor")["y"].mean()
        grouped = stats.groupby(["anchor", "slot"])["y"].agg(fraud="sum", rows="size")
        obs = grouped.reset_index().groupby("slot")["fraud"].sum()
        cell_rows = grouped.reset_index().groupby("slot")["rows"].sum()
        exp = (
            grouped["rows"]
            .mul(grouped.index.get_level_values("anchor").map(rate_a))
            .groupby(grouped.index.get_level_values("slot"))
            .sum()
        )
        for slot_value in range(10):
            observed = float(obs.get(slot_value, 0.0))
            expected = float(exp.get(slot_value, 0.0))
            tilt = observed / expected if expected > 0 else 1.0
            rows.append(
                {
                    "column": column,
                    "slot": slot_value,
                    "tilt": tilt,
                    "obs_frauds": observed,
                    "exp_frauds": expected,
                    "n_rows": int(cell_rows.get(slot_value, 0)),
                }
            )
    return pd.DataFrame(rows)


def _apply_tilt(
    target_frame: pd.DataFrame,
    slot_by_column: dict[str, tuple[np.ndarray, np.ndarray]],
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Add atilt_<col> = the row's slot tilt inside the block, 1.0 outside."""
    out = target_frame.copy()
    for column, (anchor, slot) in slot_by_column.items():
        sub = table[table["column"] == column]
        tilt_map = dict(zip(sub["slot"], sub["tilt"]))
        tilts = (
            pd.Series(slot).map(tilt_map).fillna(1.0).to_numpy(dtype=np.float64)
        )
        bound = TILT_COLUMNS[column]
        out[anchor_tilt_column(column)] = np.where(
            anchor <= bound, tilts, 1.0
        )
    return out


def anchor_v2t_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """anchor_v2 plus the fold-safe common decimal tilt for velocity/risk.

    The tilt analysis found one shared within-anchor slot lift per column
    (risk: permutation p = 0.010; velocity: p = 0.075) that replicates
    across folds and explains every low-block anchor. Each row's feature
    is its slot's lift, estimated on the row's fold-training part inside
    the column's low block only (velocity <= 40, risk <= 55); rows above
    the block bound carry the neutral 1.0. Test rows read the full-train
    tilt table from the payload.
    """
    out, extras = anchor_v2_transform(frame, folds)
    fold_of = _fold_of_row(frame, folds)
    slot_by_column = {
        column: (
            np.floor(frame[column].to_numpy(dtype=np.float64)).astype(np.int64),
            np.rint(
                (frame[column].to_numpy(dtype=np.float64)
                 - np.floor(frame[column].to_numpy(dtype=np.float64)))
                * 10.0
            ).astype(np.int64),
        )
        for column in TILT_COLUMNS
    }
    pieces: list[pd.DataFrame] = []
    for fold_id in range(folds.n_splits):
        sel = fold_of == fold_id
        table = _tilt_table(frame.loc[~sel])
        pieces.append(
            _apply_tilt(
                frame.loc[sel],
                {c: (a[sel], s[sel]) for c, (a, s) in slot_by_column.items()},
                table,
            )
        )
    features = pd.concat(pieces)
    if len(features) != len(frame) or features.index.duplicated().any():
        raise ValueError("anchor tilt features lost or duplicated rows")
    for column in features.columns:
        out.loc[features.index, column] = features[column]
        if out[column].isna().any():
            raise ValueError(f"anchor tilt feature {column!r} has NaNs")
    extras[ANCHOR_TILT_FILENAME] = _tilt_table(frame)
    return out, extras


def apply_anchor_tilt_to_test(
    test_frame: pd.DataFrame, tilt_path: Path
) -> pd.DataFrame:
    """Test-side common tilt using the version's full-train table."""
    table = pd.read_parquet(tilt_path)
    slot_by_column = {
        column: (
            np.floor(test_frame[column].to_numpy(dtype=np.float64)).astype(np.int64),
            np.rint(
                (test_frame[column].to_numpy(dtype=np.float64)
                 - np.floor(test_frame[column].to_numpy(dtype=np.float64)))
                * 10.0
            ).astype(np.int64),
        )
        for column in TILT_COLUMNS
    }
    return _apply_tilt(test_frame, slot_by_column, table)


def _combined_marginal_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column shrunk integer-anchor rates of the two score columns.

    One ``anchor = -1`` row per column carries the global rate — the
    neutral marginal for unseen anchors.
    """
    shrink = COMBINED_RATE_SHRINKAGE
    target = frame[TARGET_COLUMN].to_numpy(dtype=bool)
    global_rate = float(target.mean())
    rows: list[dict[str, object]] = []
    for column in COMBINED_RATE_COLUMNS:
        rows.append(
            {"column": column, "anchor": -1, "fraud_rate": global_rate, "n_rows": len(frame)}
        )
        anchor = np.floor(frame[column].to_numpy(dtype=np.float64)).astype(np.int64)
        stats = pd.DataFrame({"anchor": anchor, "y": target})
        agg = stats.groupby("anchor")["y"].agg(["sum", "size"])
        for anchor_value, row in agg.iterrows():
            rows.append(
                {
                    "column": column,
                    "anchor": int(anchor_value),
                    "fraud_rate": (float(row["sum"]) + shrink * global_rate)
                    / (float(row["size"]) + shrink),
                    "n_rows": int(row["size"]),
                }
            )
    return pd.DataFrame(rows)


def _apply_combined_rate(
    target_frame: pd.DataFrame,
    anchors: dict[str, np.ndarray],
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Add acomb = sigmoid(logit(p_va) + logit(p_ra) - logit(global))."""
    first, second = COMBINED_RATE_COLUMNS
    maps = {}
    global_rate = None
    for column in (first, second):
        sub = table[table["column"] == column]
        maps[column] = dict(zip(sub["anchor"], sub["fraud_rate"]))
        neutral = float(sub.loc[sub["anchor"] == -1, "fraud_rate"].iloc[0])
        if global_rate is None:
            global_rate = neutral
        elif abs(neutral - global_rate) > 1e-12:
            raise ValueError("marginal tables disagree on the global rate")
    p_first = (
        pd.Series(anchors[first]).map(maps[first]).fillna(global_rate)
        .to_numpy(dtype=np.float64)
    )
    p_second = (
        pd.Series(anchors[second]).map(maps[second]).fillna(global_rate)
        .to_numpy(dtype=np.float64)
    )
    p_first = np.clip(p_first, 1e-9, 1 - 1e-9)
    p_second = np.clip(p_second, 1e-9, 1 - 1e-9)
    logit_g = np.log(global_rate / (1.0 - global_rate))
    combined = 1.0 / (
        1.0 + np.exp(-(np.log(p_first / (1 - p_first)) + np.log(p_second / (1 - p_second)) - logit_g))
    )
    out = target_frame.copy()
    out[ANCHOR_COMBINED_COLUMN] = combined
    return out


def anchor_v2c_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """anchor_v2 plus the combined fraud rate of the two score anchors.

    The joint-anchor analysis found the (velocity anchor x risk anchor)
    cell field is the product of the two marginal anchor rates (lift flat
    in dense cells; independence simulation p = 0.796). This feature makes
    that product explicit per row: the two shrunk integer-anchor rates
    combined on the log-odds scale, computed fold-safely on the row's
    fold-training part; unseen anchors fall back to the training part's
    global rate. Test rows use the full-train marginal payload.
    """
    out, extras = anchor_v2_transform(frame, folds)
    fold_of = _fold_of_row(frame, folds)
    anchors = {
        column: np.floor(frame[column].to_numpy(dtype=np.float64)).astype(np.int64)
        for column in COMBINED_RATE_COLUMNS
    }
    pieces: list[pd.DataFrame] = []
    for fold_id in range(folds.n_splits):
        sel = fold_of == fold_id
        train_part = frame.loc[~sel]
        table = _combined_marginal_table(train_part)
        pieces.append(
            _apply_combined_rate(
                frame.loc[sel],
                {c: a[sel] for c, a in anchors.items()},
                table,
            )
        )
    features = pd.concat(pieces)
    if len(features) != len(frame) or features.index.duplicated().any():
        raise ValueError("combined rate features lost or duplicated rows")
    out.loc[features.index, ANCHOR_COMBINED_COLUMN] = features[ANCHOR_COMBINED_COLUMN]
    if out[ANCHOR_COMBINED_COLUMN].isna().any():
        raise ValueError("combined rate feature has NaNs")
    extras[ANCHOR_COMBINED_FILENAME] = _combined_marginal_table(frame)
    return out, extras


def apply_anchor_combined_to_test(
    test_frame: pd.DataFrame, rates_path: Path
) -> pd.DataFrame:
    """Test-side combined rate using the full-train marginal payload."""
    table = pd.read_parquet(rates_path)
    anchors = {
        column: np.floor(test_frame[column].to_numpy(dtype=np.float64)).astype(np.int64)
        for column in COMBINED_RATE_COLUMNS
    }
    return _apply_combined_rate(test_frame, anchors, table)


def apply_anchor_v2_rates_to_test(
    test_frame: pd.DataFrame, rates_path: Path
) -> pd.DataFrame:
    """Test-side v2 rates using the version's full-train table."""
    table = pd.read_parquet(rates_path)
    sources = _v2_rate_sources()
    keys_by_output = {
        out_column: _rate_keys(test_frame[source_column], mode)
        for out_column, source_column, mode in sources
    }
    fallback = float(table["fraud_rate"].dropna().mean())
    return _apply_v2_rates(
        test_frame, keys_by_output, table, fallback_rate=fallback
    )


# ---------------------------------------------------------------------------
# intersection_conditional rate layer
#
# A self-describing fold-safe rate feature. Each IcRate names an output
# column and the columns whose (discretized) values compose the group key.
# ``modes`` says how to turn each key column into a key part: "value",
# "floor" or "tenth" for numerics (via :func:`_rate_keys`), or "raw" for a
# categorical used as-is. The same recipe is stored in the payload so the
# test side rebuilds the key without a hardcoded source list. Shared by
# every intersection_conditional_* rate version (marginal and context).


class IcRate(NamedTuple):
    """One rate feature: output column and its composite group key."""

    output: str
    key_cols: tuple[str, ...]
    modes: tuple[str, ...]


def _ic_key(frame: pd.DataFrame, key_cols: tuple[str, ...], modes: tuple[str, ...]) -> np.ndarray:
    """String group key per row, parts joined with '|'."""
    parts: list[pd.Series] = []
    for column, mode in zip(key_cols, modes):
        if mode == "raw":
            part = frame[column].astype(str)
        else:
            part = pd.Series(
                _rate_keys(frame[column], mode).astype(str), index=frame.index
            )
        parts.append(part)
    key = parts[0]
    for part in parts[1:]:
        key = key.str.cat(part, sep="|")
    return key.to_numpy()


def _ic_rate_table(
    frame: pd.DataFrame,
    rates: tuple[IcRate, ...],
    target: np.ndarray,
    *,
    shrink: int,
) -> pd.DataFrame:
    """Long per-(output, key) shrunk fraud-rate table, recipe embedded."""
    global_rate = float(target.mean())
    rows: list[dict[str, object]] = []
    for rate in rates:
        keys = _ic_key(frame, rate.key_cols, rate.modes)
        stats = pd.DataFrame({"key": keys, "y": target})
        agg = stats.groupby("key")["y"].agg(["sum", "size"])
        key_cols_text = "|".join(rate.key_cols)
        modes_text = "|".join(rate.modes)
        for key_value, row in agg.iterrows():
            rows.append(
                {
                    "output": rate.output,
                    "key": str(key_value),
                    "fraud_rate": (float(row["sum"]) + shrink * global_rate)
                    / (float(row["size"]) + shrink),
                    "n_rows": int(row["size"]),
                    "n_fraud": int(row["sum"]),
                    "key_cols": key_cols_text,
                    "modes": modes_text,
                }
            )
    return pd.DataFrame(rows)


def _apply_ic_rates(
    target_frame: pd.DataFrame,
    rates: tuple[IcRate, ...],
    table: pd.DataFrame,
    *,
    fallback_rate: float,
) -> pd.DataFrame:
    """Map each IcRate output onto ``target_frame`` from its (output, key) table."""
    out = target_frame.copy()
    for rate in rates:
        sub = table[table["output"] == rate.output]
        rate_map = dict(zip(sub["key"], sub["fraud_rate"]))
        keys = _ic_key(target_frame, rate.key_cols, rate.modes)
        values = (
            pd.Series(keys, index=target_frame.index)
            .map(rate_map)
            .fillna(fallback_rate)
        )
        out[rate.output] = values.to_numpy()
    return out


def _ic_rate_features(
    frame: pd.DataFrame,
    folds: FlagStratifiedCv,
    rates: tuple[IcRate, ...],
    *,
    shrink: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOF-safe IcRate columns (per-fold) and the full-train payload table."""
    target = frame[TARGET_COLUMN].to_numpy(dtype=bool)
    fold_of = _fold_of_row(frame, folds)
    global_rate = float(target.mean())
    outputs = [rate.output for rate in rates]
    pieces: list[pd.DataFrame] = []
    for fold_id in range(folds.n_splits):
        sel = fold_of == fold_id
        train_part = frame.loc[~sel]
        table = _ic_rate_table(
            train_part,
            rates,
            train_part[TARGET_COLUMN].to_numpy(dtype=bool),
            shrink=shrink,
        )
        applied = _apply_ic_rates(
            frame.loc[sel], rates, table, fallback_rate=global_rate
        )
        pieces.append(applied[outputs])
    features = pd.concat(pieces)
    if len(features) != len(frame) or features.index.duplicated().any():
        raise ValueError("ic rate features lost or duplicated rows")
    features = features.loc[frame.index]
    for column in outputs:
        if features[column].isna().any():
            raise ValueError(f"ic rate feature {column!r} has NaNs")
    full_table = _ic_rate_table(frame, rates, target, shrink=shrink)
    return features, full_table


def _ic_rates_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv, rates: tuple[IcRate, ...]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """anchor_v2 features (channel + card_type kept) plus the given IcRates."""
    out, extras = anchor_v2_transform(frame, folds)
    features, table = _ic_rate_features(frame, folds, rates, shrink=IC_RATE_SHRINKAGE)
    for column in features.columns:
        out[column] = features[column].to_numpy()
    extras[IC_RATES_FILENAME] = table
    return out, extras


IC_DISPUTES_RATES: tuple[IcRate, ...] = (
    IcRate(IC_RATE_PREFIX + "prior_disputes", ("prior_disputes",), ("value",)),
)


def ic_disputes_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Base features plus the marginal per-value fraud rate of prior_disputes.

    prior_disputes has no rate in anchor_v2 (it is not in
    ``SMALLINT_RATE_COLUMNS``), so icrate_prior_disputes is a fully new
    column: the fold-safe, k-shrunk fraud rate of the row's dispute count.
    """
    return _ic_rates_transform(frame, folds, IC_DISPUTES_RATES)


IC_CVV_RATES: tuple[IcRate, ...] = (
    IcRate(
        IC_RATE_PREFIX + "cvv_ctx",
        ("cvv_retry_count", "channel", "device_type"),
        ("value", "raw", "raw"),
    ),
)


def ic_cvv_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Base plus the fraud rate of (cvv_retry_count x channel x device_type).

    The base already carries the GLOBAL cvv value rate, so icrate_cvv_ctx is
    an interaction test: it only helps if the cvv lift varies by channel and
    device beyond that marginal. Fold-safe, k-shrunk; unseen cells fall back
    to the global rate.
    """
    return _ic_rates_transform(frame, folds, IC_CVV_RATES)


def _anchor_cat_rates(numeric: str, tag: str) -> tuple[IcRate, ...]:
    """Three fold-safe rates of floor(numeric) conditioned on merchant / auth / both."""
    return (
        IcRate(IC_RATE_PREFIX + f"{tag}_merchant",
               (numeric, "merchant_category"), ("floor", "raw")),
        IcRate(IC_RATE_PREFIX + f"{tag}_auth",
               (numeric, "auth_method"), ("floor", "raw")),
        IcRate(IC_RATE_PREFIX + f"{tag}_merchant_auth",
               (numeric, "merchant_category", "auth_method"), ("floor", "raw", "raw")),
    )


IC_RISK_RATES: tuple[IcRate, ...] = _anchor_cat_rates("merchant_risk_score", "risk")
IC_VELOCITY_RATES: tuple[IcRate, ...] = _anchor_cat_rates("velocity_score", "vel")


def ic_risk_rates_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Base plus fraud rate of floor(risk) within merchant / auth / both.

    The rate analog of the risk distribution-shift: the base has the GLOBAL
    floor(risk) rate (arate_merchant_risk_score); these condition it on the
    cats whose level moves risk's distribution, testing whether a given risk
    level means a different fraud rate in different merchant/auth contexts.
    """
    return _ic_rates_transform(frame, folds, IC_RISK_RATES)


def ic_velocity_rates_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Base plus fraud rate of floor(velocity) within merchant / auth / both."""
    return _ic_rates_transform(frame, folds, IC_VELOCITY_RATES)


def apply_ic_rates_to_test(
    test_frame: pd.DataFrame, rates_path: Path
) -> pd.DataFrame:
    """Test-side IcRates using the version's self-describing full-train table."""
    table = pd.read_parquet(rates_path)
    rates = tuple(
        IcRate(
            output=str(output),
            key_cols=tuple(sub["key_cols"].iloc[0].split("|")),
            modes=tuple(sub["modes"].iloc[0].split("|")),
        )
        for output, sub in table.groupby("output")
    )
    fallback = float(table["fraud_rate"].dropna().mean())
    return _apply_ic_rates(test_frame, rates, table, fallback_rate=fallback)


# ---------------------------------------------------------------------------
# intersection_conditional cell-z layer (target-free)
#
# A within-cell robust-z of a numeric: (value - cell median) / cell scale,
# with scale = 1.4826 * MAD. Both the signed z (direction) and the absolute
# z (non-monotone outlierness) are emitted. No label is used, so the cell
# statistics are computed on the full train frame (like the anchor
# frequencies) and reused unchanged for the test frame; the payload embeds
# the key recipe so the test side rebuilds cells without a hardcoded source.


class CellZ(NamedTuple):
    """One within-cell robust-z: a source numeric and its grouping cell."""

    source: str
    key_cols: tuple[str, ...]
    modes: tuple[str, ...]


def _mad_scale(values: np.ndarray) -> float:
    """1.4826 * median-absolute-deviation; nan when it cannot be formed."""
    if len(values) == 0:
        return float("nan")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad


def _cellz_table(frame: pd.DataFrame, specs: tuple[CellZ, ...]) -> pd.DataFrame:
    """Per-(source, cell key) median and robust scale, recipe embedded.

    Global stats are stored under the sentinel key ``*`` and used as the
    fallback for small or unseen cells (and for a degenerate zero scale).
    """
    rows: list[dict[str, object]] = []
    for spec in specs:
        values_all = frame[spec.source].to_numpy(dtype=np.float64)
        global_median = float(np.median(values_all))
        global_scale = _mad_scale(values_all)
        if not np.isfinite(global_scale) or global_scale == 0.0:
            global_scale = float(np.std(values_all)) or 1.0
        keys = _ic_key(frame, spec.key_cols, spec.modes)
        key_cols_text = "|".join(spec.key_cols)
        modes_text = "|".join(spec.modes)
        common = {
            "source": spec.source,
            "key_cols": key_cols_text,
            "modes": modes_text,
        }
        rows.append(
            {**common, "key": "*", "median": global_median, "scale": global_scale, "n_rows": int(len(values_all))}
        )
        grouped = pd.Series(values_all).groupby(keys)
        for key_value, series in grouped:
            vals = series.to_numpy()
            if len(vals) < IC_CELLZ_MIN_ROWS:
                continue
            scale = _mad_scale(vals)
            if not np.isfinite(scale) or scale == 0.0:
                scale = global_scale
            rows.append(
                {
                    **common,
                    "key": str(key_value),
                    "median": float(np.median(vals)),
                    "scale": float(scale),
                    "n_rows": int(len(vals)),
                }
            )
    return pd.DataFrame(rows)


def _apply_cellz(
    target_frame: pd.DataFrame, specs: tuple[CellZ, ...], table: pd.DataFrame
) -> pd.DataFrame:
    """Add iczs_<source> (signed) and icza_<source> (absolute) z columns."""
    out = target_frame.copy()
    for spec in specs:
        sub = table[table["source"] == spec.source]
        glob = sub[sub["key"] == "*"].iloc[0]
        g_median, g_scale = float(glob["median"]), float(glob["scale"])
        median_map = dict(zip(sub["key"], sub["median"]))
        scale_map = dict(zip(sub["key"], sub["scale"]))
        keys = pd.Series(_ic_key(target_frame, spec.key_cols, spec.modes))
        median = keys.map(median_map).fillna(g_median).to_numpy()
        scale = keys.map(scale_map).fillna(g_scale).to_numpy()
        scale = np.where(scale > 0.0, scale, g_scale)
        value = target_frame[spec.source].to_numpy(dtype=np.float64)
        signed = (value - median) / scale
        out[IC_ZSIGNED_PREFIX + spec.source] = signed
        out[IC_ZABS_PREFIX + spec.source] = np.abs(signed)
    return out


def _ic_cellz_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv, specs: tuple[CellZ, ...]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """anchor_v2 features (channel + card_type kept) plus within-cell z scores."""
    out, extras = anchor_v2_transform(frame, folds)
    table = _cellz_table(frame, specs)
    applied = _apply_cellz(frame, specs, table)
    for spec in specs:
        for column in (IC_ZSIGNED_PREFIX + spec.source, IC_ZABS_PREFIX + spec.source):
            out[column] = applied[column].to_numpy()
            if out[column].isna().any():
                raise ValueError(f"ic cell-z feature {column!r} has NaNs")
    extras[IC_CELLZ_FILENAME] = table
    return out, extras


IC_RISKZ_SPECS: tuple[CellZ, ...] = (
    CellZ("merchant_risk_score", ("merchant_category", "is_new_merchant"), ("raw", "raw")),
)


def ic_riskz_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Base plus robust-z of merchant_risk_score within merchant x is_new.

    iczs_ is the signed z (how far above/below the cell's typical risk);
    icza_ is its magnitude (non-monotone outlierness, the shape no raw-risk
    threshold reproduces). Stats are target-free (median / MAD), so they are
    formed on the full train frame and reused for the test frame.
    """
    return _ic_cellz_transform(frame, folds, IC_RISKZ_SPECS)


def _ic_full_transform(
    frame: pd.DataFrame,
    folds: FlagStratifiedCv,
    rates: tuple[IcRate, ...],
    cellz: tuple[CellZ, ...],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Base features plus any IcRates and any CellZ features, in one matrix."""
    out, extras = anchor_v2_transform(frame, folds)
    if rates:
        features, table = _ic_rate_features(
            frame, folds, rates, shrink=IC_RATE_SHRINKAGE
        )
        for column in features.columns:
            out[column] = features[column].to_numpy()
        extras[IC_RATES_FILENAME] = table
    if cellz:
        ctable = _cellz_table(frame, cellz)
        applied = _apply_cellz(frame, cellz, ctable)
        for spec in cellz:
            for column in (
                IC_ZSIGNED_PREFIX + spec.source,
                IC_ZABS_PREFIX + spec.source,
            ):
                out[column] = applied[column].to_numpy()
                if out[column].isna().any():
                    raise ValueError(f"ic cell-z feature {column!r} has NaNs")
        extras[IC_CELLZ_FILENAME] = ctable
    return out, extras


def ic_cvv_riskz_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Base plus the cvv context rate and the risk within-cell robust-z.

    The day-2 combine of the two day-1 keeps (attempts 3 and 4): each was a
    sub-bar positive that lifted the hard folds, so the combine tests
    whether they stack past the submission bar.
    """
    return _ic_full_transform(frame, folds, IC_CVV_RATES, IC_RISKZ_SPECS)


def apply_cellz_to_test(
    test_frame: pd.DataFrame, cellz_path: Path
) -> pd.DataFrame:
    """Test-side cell-z using the version's self-describing stat table."""
    table = pd.read_parquet(cellz_path)
    specs = tuple(
        CellZ(
            source=str(source),
            key_cols=tuple(sub["key_cols"].iloc[0].split("|")),
            modes=tuple(sub["modes"].iloc[0].split("|")),
        )
        for source, sub in table.groupby("source")
    )
    return _apply_cellz(test_frame, specs, table)


# ---------------------------------------------------------------------------
# fine-cell risk robust-z (full 7-atom profile, drop-one escalation)
#
# iczs_risk_fine / icza_risk_fine: the signed and absolute robust-z of
# merchant_risk_score inside the finest behavioural cell with enough rows.
# Unlike the coarse merchant x is_new version, this encodes a LOCAL risk
# norm a depth-5/6 tree cannot assemble (reach a 7-way cell AND rank within
# it). Target-free (median / MAD), so no fold split; the cell assignment is
# label-free too.

IC_RISKZ_FINE_SOURCE = "merchant_risk_score"
IC_RISKZ_FINE_ZSIGNED = IC_ZSIGNED_PREFIX + "risk_fine"
IC_RISKZ_FINE_ZABS = IC_ZABS_PREFIX + "risk_fine"
_RISKZ_FINE_DROP_LEVELS = tuple(
    tuple(atom for atom in IC_RISKZ_FINE_ATOMS if atom != dropped)
    for dropped in IC_RISKZ_FINE_ATOMS
)


def _with_alarm(frame: pd.DataFrame) -> pd.DataFrame:
    """Return frame with the boolean is_alarm column present."""
    if "is_alarm" in frame.columns:
        return frame
    out = frame.copy()
    out["is_alarm"] = alarm_flag_mask(frame).to_numpy(dtype=bool)
    return out


def _combo_key(frame: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    """'|'-joined string key over several categorical columns."""
    key = frame[cols[0]].astype(str)
    for column in cols[1:]:
        key = key.str.cat(frame[column].astype(str), sep="|")
    return key.to_numpy()


def _riskz_fine_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-(level, cell) median and robust scale for cells >= min rows.

    Level ``*`` holds the global stats (fallback). ``full7`` is the whole
    profile; ``drop:<atom>`` each single-atom-dropped profile. Cells below
    the row threshold are omitted, so presence in the table means eligible.
    """
    work = _with_alarm(frame)
    risk = work[IC_RISKZ_FINE_SOURCE].to_numpy(dtype=np.float64)
    global_median = float(np.median(risk))
    global_scale = _mad_scale(risk)
    if not np.isfinite(global_scale) or global_scale == 0.0:
        global_scale = float(np.std(risk)) or 1.0
    rows: list[dict[str, object]] = [
        {"level": "*", "key": "*", "median": global_median,
         "scale": global_scale, "n_rows": int(len(risk))}
    ]
    levels = [("full7", IC_RISKZ_FINE_ATOMS)] + [
        (f"drop:{dropped}", cols)
        for dropped, cols in zip(IC_RISKZ_FINE_ATOMS, _RISKZ_FINE_DROP_LEVELS)
    ]
    for level, cols in levels:
        keys = _combo_key(work, cols)
        for key_value, series in pd.Series(risk).groupby(keys):
            vals = series.to_numpy()
            if len(vals) < IC_CELLZ_MIN_ROWS:
                continue
            scale = _mad_scale(vals)
            if not np.isfinite(scale) or scale == 0.0:
                scale = global_scale
            rows.append({
                "level": level, "key": str(key_value),
                "median": float(np.median(vals)), "scale": float(scale),
                "n_rows": int(len(vals)),
            })
    return pd.DataFrame(rows)


def _apply_riskz_fine(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Add iczs_risk_fine / icza_risk_fine using the escalation table."""
    work = _with_alarm(frame)
    risk = work[IC_RISKZ_FINE_SOURCE].to_numpy(dtype=np.float64)
    n = len(work)
    glob = table[table["level"] == "*"].iloc[0]
    g_median, g_scale = float(glob["median"]), float(glob["scale"])
    maps = {
        level: (
            dict(zip(sub["key"], sub["median"])),
            dict(zip(sub["key"], sub["scale"])),
            dict(zip(sub["key"], sub["n_rows"])),
        )
        for level, sub in table.groupby("level")
    }
    median = np.full(n, g_median)
    scale = np.full(n, g_scale)
    assigned = np.zeros(n, dtype=bool)

    full_median, full_scale, _ = maps.get("full7", ({}, {}, {}))
    full_key = pd.Series(_combo_key(work, IC_RISKZ_FINE_ATOMS))
    fm = full_key.map(full_median).to_numpy()
    hit = ~pd.isna(fm)
    median[hit] = fm[hit]
    scale[hit] = full_key.map(full_scale).to_numpy()[hit]
    assigned |= hit

    # remaining rows: best (largest-n) single-atom-dropped cell
    drop_n = np.zeros((n, len(IC_RISKZ_FINE_ATOMS)))
    drop_med = np.full((n, len(IC_RISKZ_FINE_ATOMS)), np.nan)
    drop_scl = np.full((n, len(IC_RISKZ_FINE_ATOMS)), np.nan)
    for j, (dropped, cols) in enumerate(zip(IC_RISKZ_FINE_ATOMS, _RISKZ_FINE_DROP_LEVELS)):
        dm, ds, dn = maps.get(f"drop:{dropped}", ({}, {}, {}))
        key = pd.Series(_combo_key(work, cols))
        drop_n[:, j] = key.map(dn).fillna(0).to_numpy()
        drop_med[:, j] = key.map(dm).to_numpy()
        drop_scl[:, j] = key.map(ds).to_numpy()
    best = drop_n.argmax(axis=1)
    rows_idx = np.arange(n)
    best_n = drop_n[rows_idx, best]
    use_drop = (~assigned) & (best_n > 0)
    median[use_drop] = drop_med[rows_idx, best][use_drop]
    scale[use_drop] = drop_scl[rows_idx, best][use_drop]
    assigned |= use_drop

    scale = np.where((scale > 0) & np.isfinite(scale), scale, g_scale)
    signed = (risk - median) / scale
    out = frame.copy()
    out[IC_RISKZ_FINE_ZSIGNED] = signed
    out[IC_RISKZ_FINE_ZABS] = np.abs(signed)
    return out


def ic_riskz_fine_transform(
    frame: pd.DataFrame, folds: FlagStratifiedCv
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Base plus risk robust-z within the escalated full behavioural cell."""
    out, extras = anchor_v2_transform(frame, folds)
    table = _riskz_fine_table(frame)
    applied = _apply_riskz_fine(frame, table)
    for column in (IC_RISKZ_FINE_ZSIGNED, IC_RISKZ_FINE_ZABS):
        out[column] = applied[column].to_numpy()
        if out[column].isna().any():
            raise ValueError(f"fine risk-z feature {column!r} has NaNs")
    extras[IC_RISKZ_FINE_FILENAME] = table
    return out, extras


def apply_riskz_fine_to_test(
    test_frame: pd.DataFrame, table_path: Path
) -> pd.DataFrame:
    """Test-side fine risk-z using the version's escalation table."""
    return _apply_riskz_fine(test_frame, pd.read_parquet(table_path))


TRANSFORMS: dict[str, TransformFn] = {
    RISK_CONTEXT_TRANSFORM: risk_context_transform,
    RISK_AGGREGATES_TRANSFORM: risk_aggregates_transform,
    PROFILE_CONTEXT_TRANSFORM: profile_context_transform,
    GRID_SLOTS_TRANSFORM: grid_slots_transform,
    GRID_SLOTS_EXACT_TRANSFORM: grid_slots_exact_transform,
    SLOT_Z_TRANSFORM: slot_z_transform,
    SLOT_REFS_TRANSFORM: slot_refs_transform,
    ANCHOR_FREQ_TRANSFORM: anchor_freq_transform,
    ANCHOR_FULL_TRANSFORM: anchor_full_transform,
    ANCHOR_V2_TRANSFORM: anchor_v2_transform,
    ANCHOR_V2_IS_ALARM_TRANSFORM: anchor_v2_is_alarm_transform,
    ANCHOR_V2_N_ALARM_TRANSFORM: anchor_v2_n_alarm_transform,
    ANCHOR_V2C_TRANSFORM: anchor_v2c_transform,
    ANCHOR_V2T_TRANSFORM: anchor_v2t_transform,
    ANCHOR_V3_TRANSFORM: anchor_v3_transform,
    IC_DISPUTES_TRANSFORM: ic_disputes_transform,
    IC_CVV_TRANSFORM: ic_cvv_transform,
    IC_RISK_RATES_TRANSFORM: ic_risk_rates_transform,
    IC_VELOCITY_RATES_TRANSFORM: ic_velocity_rates_transform,
    IC_RISKZ_TRANSFORM: ic_riskz_transform,
    IC_CVV_RISKZ_TRANSFORM: ic_cvv_riskz_transform,
    IC_RISKZ_FINE_TRANSFORM: ic_riskz_fine_transform,
}


def apply_transform(
    name: str,
    frame: pd.DataFrame,
    folds: FlagStratifiedCv,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run the registered transform ``name``; check the output contract."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"transform name must be a non-empty str, got {name!r}")
    if name not in TRANSFORMS:
        raise ValueError(
            f"unknown transform {name!r}; known: {sorted(TRANSFORMS)}"
        )
    out, extras = TRANSFORMS[name](frame, folds)
    if len(out) != len(frame):
        raise ValueError(f"transform {name!r} changed the row count")
    if not np.array_equal(
        out[ID_COLUMN].to_numpy(dtype=np.int64),
        frame[ID_COLUMN].to_numpy(dtype=np.int64),
    ):
        raise ValueError(f"transform {name!r} changed the id column or its order")
    for filename, extra_frame in extras.items():
        if not isinstance(extra_frame, pd.DataFrame):
            raise TypeError(
                f"transform {name!r} extra {filename!r} is not a DataFrame"
            )
    return out, extras
