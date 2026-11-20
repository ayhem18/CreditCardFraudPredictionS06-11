"""Region modeling: profiles, profile-stratified CV, variants, quantiles."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.constants import (
    ALARM_FLAG_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
)
from ccf_ml.region_modeling import (
    add_profile_quantiles,
    build_no_alarm_frame,
    build_profile_stratified_cv,
    dump_to_path,
    load_profile_stratified_cv,
    profile_stratum,
    validate_variant,
    variant_by_name,
    variant_model_matrix,
)
from ccf_ml.region_modeling.constants import (
    PROFILE_ID_COLUMN,
    QUANTILE_PREFIX,
)
from ccf_ml.region_modeling.variants import REGION_MODEL_VARIANTS

N_PER_PROFILE = 30


def _region_frame() -> pd.DataFrame:
    """Two profiles x (fraud, clean) x 30 rows, plus one tiny profile."""
    n_rows = N_PER_PROFILE * 4 + 8
    target = np.concatenate(
        [
            np.zeros(N_PER_PROFILE, dtype=int),
            np.ones(N_PER_PROFILE, dtype=int),
            np.zeros(N_PER_PROFILE, dtype=int),
            np.ones(N_PER_PROFILE, dtype=int),
            np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int),
        ]
    )
    frame = pd.DataFrame(
        {
            ID_COLUMN: np.arange(n_rows, dtype=np.int64),
            TARGET_COLUMN: target,
            "amount_usd": np.linspace(1.0, 100.0, n_rows),
            "hours_since_last_txn": np.linspace(0.5, 48.0, n_rows),
            "distance_from_home_km": np.linspace(0.0, 60.0, n_rows),
            "card_age_months": np.arange(1, n_rows + 1, dtype=float),
            "customer_age": np.linspace(20.0, 80.0, n_rows),
            "account_balance_usd": np.linspace(10.0, 9000.0, n_rows),
            "velocity_score": np.linspace(0.0, 1.0, n_rows),
            "merchant_risk_score": np.linspace(0.0, 100.0, n_rows),
            "cvv_retry_count": np.zeros(n_rows, dtype=int),
            "txn_count_last_24h": np.tile(np.arange(5), n_rows // 5 + 1)[:n_rows],
            "prior_disputes": np.zeros(n_rows, dtype=int),
            "time_of_day_hour": np.tile(np.arange(24), n_rows // 24 + 1)[:n_rows],
            "day_of_week": np.arange(n_rows) % 7,
            "merchant_category": np.array(
                ["Groceries"] * (N_PER_PROFILE * 2)
                + ["Gaming"] * (N_PER_PROFILE * 2)
                + ["Travel"] * 8
            ),
            "card_type": np.tile(["Visa", "Amex"], n_rows // 2 + 1)[:n_rows],
            "auth_method": np.array(
                ["PIN"] * (n_rows - 8) + ["OTP"] * 8
            ),
            "channel": np.tile(["Online", "POS"], n_rows // 2 + 1)[:n_rows],
            "device_type": np.array(["iPhone"] * n_rows),
            "is_new_merchant": np.array(
                [False] * (N_PER_PROFILE * 2)
                + [True] * (N_PER_PROFILE * 2)
                + [False] * 8
            ),
        }
    )
    for name in ALARM_FLAG_COLUMNS:
        frame[name] = False
    return frame


class TestProfiles(unittest.TestCase):
    def test_build_no_alarm_frame_filters_and_adds(self) -> None:
        frame = _region_frame()
        frame.loc[0, ALARM_FLAG_COLUMNS[0]] = True
        region = build_no_alarm_frame(frame)
        self.assertEqual(len(region), len(frame) - 1)
        self.assertIn("merchant_tier", region.columns)
        self.assertIn(PROFILE_ID_COLUMN, region.columns)
        self.assertTrue(
            region["merchant_tier"].isin({"hot", "warm", "background"}).all()
        )

    def test_small_profiles_collapse_in_stratum(self) -> None:
        region = build_no_alarm_frame(_region_frame())
        stratum = profile_stratum(region, min_stratum_rows=25)
        # the tiny Travel|OTP|personal profile collapses per target; the two
        # big profiles (background|PIN|personal, hot|PIN|personal) keep
        # their own strata
        self.assertIn("_small_|0", set(stratum))
        self.assertIn("_small_|1", set(stratum))
        self.assertIn("background|PIN|personal|False|0", set(stratum))
        self.assertIn("hot|PIN|personal|True|1", set(stratum))


class TestProfileCv(unittest.TestCase):
    def test_build_partition_and_round_trip(self) -> None:
        region = build_no_alarm_frame(_region_frame())
        cv = build_profile_stratified_cv(region, n_splits=2, random_state=0)
        self.assertEqual(cv.n_rows, len(region))
        val_union: set[int] = set()
        for i, fold in enumerate(cv.folds):
            self.assertEqual(fold.fold_id, i)
            val_union.update(fold.val_ids)
        self.assertEqual(val_union, set(region[ID_COLUMN]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cv.json"
            dump_to_path(cv, path=path)
            loaded = load_profile_stratified_cv(path=path)
            self.assertEqual(loaded.n_rows, cv.n_rows)
            self.assertEqual(
                [list(f.val_ids) for f in loaded.folds],
                [list(f.val_ids) for f in cv.folds],
            )
            self.assertEqual(
                [list(f.train_ids) for f in loaded.folds],
                [list(f.train_ids) for f in cv.folds],
            )


class TestVariants(unittest.TestCase):
    def test_registered_variants_validate(self) -> None:
        names = [spec.name for spec in REGION_MODEL_VARIANTS]
        self.assertEqual(len(names), len(set(names)))
        columns = tuple(_region_frame().columns) + (
            "merchant_tier",
            PROFILE_ID_COLUMN,
        )
        for spec in REGION_MODEL_VARIANTS:
            validate_variant(spec, frame_columns=columns)

    def test_unknown_variant_raises(self) -> None:
        with self.assertRaises(KeyError):
            variant_by_name("nope")


class TestQuantilesAndMatrix(unittest.TestCase):
    def test_quantiles_measured_on_fold_train_only(self) -> None:
        region = build_no_alarm_frame(_region_frame())
        cv = build_profile_stratified_cv(region, n_splits=2, random_state=0)
        frame = region.copy()
        added = add_profile_quantiles(frame, cv, min_group=10)
        self.assertIn(QUANTILE_PREFIX + "amount_usd", added)
        fold_of = np.full(len(frame), -1, dtype=np.int64)
        ids = frame[ID_COLUMN].to_numpy()
        for fold in cv.folds:
            for record_id in fold.val_ids:
                fold_of[ids == record_id] = fold.fold_id
        raw = frame["amount_usd"].to_numpy(dtype=np.float64)
        profile = frame[PROFILE_ID_COLUMN].to_numpy()
        pq = frame[QUANTILE_PREFIX + "amount_usd"].to_numpy(dtype=np.float64)
        for i in range(0, len(frame), 7):
            train_mask = fold_of != fold_of[i]
            same = train_mask & (profile == profile[i])
            if same.sum() >= 10:
                ref = np.sort(raw[same])
            else:
                ref = np.sort(raw[train_mask])
            expected = (np.sum(ref < raw[i]) + 0.5 * np.sum(ref == raw[i])) / len(ref)
            self.assertAlmostEqual(pq[i], expected, places=9)

    def test_variant_matrix_shapes(self) -> None:
        region = build_no_alarm_frame(_region_frame())
        cv = build_profile_stratified_cv(region, n_splits=2, random_state=0)
        full = variant_by_name("full")
        _matrix, features, _cats, _nums = variant_model_matrix(region, full, cv=cv)
        self.assertIn("time_of_day_hour", features)
        self.assertIn("channel", features)
        self.assertNotIn(PROFILE_ID_COLUMN, features)
        for flag in ALARM_FLAG_COLUMNS:
            self.assertNotIn(flag, features)
        v4 = variant_by_name("v4_profile_id_quantiles")
        matrix4, features4, cats4, _nums4 = variant_model_matrix(region, v4, cv=cv)
        self.assertNotIn("time_of_day_hour", features4)
        self.assertNotIn("channel", features4)
        self.assertNotIn("card_type", features4)
        self.assertIn(PROFILE_ID_COLUMN, features4)
        self.assertTrue(any(c.startswith(QUANTILE_PREFIX) for c in features4))
        self.assertIn(PROFILE_ID_COLUMN, cats4)
        self.assertNotIn("device_group", features4)
        self.assertIn("device_type", features4)
        self.assertEqual(matrix4.shape[0], len(region))


if __name__ == "__main__":
    unittest.main()
