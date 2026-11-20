"""Named, fold-safe derived-feature transforms for dataset versions.

A transform sees the raw frame plus the frozen partition and returns the
transformed frame together with extra payload tables that the builder writes
next to ``data.parquet``. Transforms never change the row count or the id
order; id and target columns pass through untouched.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.constants import ALARM_FLAG_COLUMNS, ID_COLUMN
from ccf_ml.cross_validation.flag_stratified import FlagStratifiedCv
from ccf_ml.dataset_versions.constants import (
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


TRANSFORMS: dict[str, TransformFn] = {
    RISK_CONTEXT_TRANSFORM: risk_context_transform,
    RISK_AGGREGATES_TRANSFORM: risk_aggregates_transform,
    PROFILE_CONTEXT_TRANSFORM: profile_context_transform,
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
