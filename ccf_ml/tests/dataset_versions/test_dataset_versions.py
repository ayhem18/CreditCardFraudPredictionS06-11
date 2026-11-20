"""Dataset versions: spec validation, build, load, folds, fingerprints."""

from __future__ import annotations

import json
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
from ccf_ml.cross_validation.flag_stratified import build_flag_stratified_cv
from ccf_ml.dataset_versions.builder import build_dataset_version
from ccf_ml.dataset_versions.constants import (
    PROFILE_NUMERIC_COLUMNS,

    PROFILE_CONTEXT_TRANSFORM,

    AUTH_METHOD_COLUMN,
    CARD_TYPE_COLUMN,
    CHANNEL_COLUMN,
    DEVICE_TYPE_COLUMN,
    IS_NEW_MERCHANT_COLUMN,
    MERCHANT_CATEGORY_COLUMN,
    MERCHANT_RISK_SCORE_COLUMN,
    RISK_CONTEXT_ADDED_COLUMNS,
    RISK_CONTEXT_GROUPS_FILENAME,
    RISK_CONTEXT_TRANSFORM,
    TIME_OF_DAY_HOUR_COLUMN,
)
from ccf_ml.dataset_versions.loader import (
    iter_folds,
    list_dataset_versions,
    load_dataset_version,
    load_fold,
)
from ccf_ml.dataset_versions.manifest import (
    load_dataset_version_manifest,
    parse_dataset_version_manifest,
)
from ccf_ml.dataset_versions.transforms import (
    _profile_percentile_features,
    apply_transform,
)
from ccf_ml.dataset_versions.specs import (
    DATASET_VERSION_SPECS,
    DatasetVersionSpec,
    validate_spec,
)

N_PER_CELL = 20
N_SPLITS = 2
SEED = 0

SPEC = DatasetVersionSpec(
    name="tiny_version",
    description="test ablation",
    drop_columns=(TIME_OF_DAY_HOUR_COLUMN, CARD_TYPE_COLUMN),
)


def _grid_frame(*, n_per_cell: int = N_PER_CELL) -> pd.DataFrame:
    n_rows = n_per_cell * 4
    alarm = np.repeat([False, False, True, True], n_per_cell)
    fraud = np.repeat([0, 1, 0, 1], n_per_cell)
    frame = pd.DataFrame(
        {
            ID_COLUMN: np.arange(n_rows, dtype=np.int64),
            TARGET_COLUMN: fraud,
            "amount_usd": np.linspace(1.0, 100.0, n_rows),
            TIME_OF_DAY_HOUR_COLUMN: np.tile(np.arange(n_per_cell), 4) % 24,
            "day_of_week": np.arange(n_rows) % 7,
            CARD_TYPE_COLUMN: np.repeat(["Visa", "Mastercard"], n_rows // 2),
            CHANNEL_COLUMN: np.tile(["Online", "ATM"], n_rows // 2),
            DEVICE_TYPE_COLUMN: np.tile(["POS Terminal", "iPhone"], n_rows // 2),
            IS_NEW_MERCHANT_COLUMN: np.arange(n_rows) % 3 == 0,
            MERCHANT_CATEGORY_COLUMN: np.repeat(
                ["Groceries", "Gaming"], n_rows // 2
            ),
            AUTH_METHOD_COLUMN: np.tile(["PIN", "OTP"], n_rows // 2),
            MERCHANT_RISK_SCORE_COLUMN: np.linspace(10.0, 90.0, n_rows),
        }
    )
    for i, name in enumerate(ALARM_FLAG_COLUMNS):
        frame[name] = alarm if i == 0 else fraud == 1
    return frame


class TestValidateSpec(unittest.TestCase):
    def test_accepts_valid_spec(self) -> None:
        frame = _grid_frame()
        drop_columns = validate_spec(SPEC, frame_columns=tuple(frame.columns))
        self.assertEqual(drop_columns, SPEC.drop_columns)

    def test_rejects_protected_column(self) -> None:
        spec = DatasetVersionSpec(
            name="bad", description="", drop_columns=(TARGET_COLUMN,)
        )
        with self.assertRaisesRegex(ValueError, "protected"):
            validate_spec(spec)

    def test_rejects_unknown_column(self) -> None:
        spec = DatasetVersionSpec(
            name="bad", description="", drop_columns=("not_a_column",)
        )
        frame = _grid_frame()
        with self.assertRaisesRegex(ValueError, "not a frame column"):
            validate_spec(spec, frame_columns=tuple(frame.columns))

    def test_rejects_duplicate_drops(self) -> None:
        spec = DatasetVersionSpec(
            name="bad",
            description="",
            drop_columns=(TIME_OF_DAY_HOUR_COLUMN, TIME_OF_DAY_HOUR_COLUMN),
        )
        with self.assertRaisesRegex(ValueError, "more than once"):
            validate_spec(spec)

    def test_rejects_empty_drops(self) -> None:
        spec = DatasetVersionSpec(name="bad", description="", drop_columns=())
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_spec(spec)

    def test_rejects_path_separator_in_name(self) -> None:
        spec = DatasetVersionSpec(
            name="a/b", description="", drop_columns=(TIME_OF_DAY_HOUR_COLUMN,)
        )
        with self.assertRaisesRegex(ValueError, "single path component"):
            validate_spec(spec)

    def test_preset_specs_validate_against_train_schema(self) -> None:
        columns = tuple(_grid_frame().columns)
        for spec in DATASET_VERSION_SPECS:
            validate_spec(spec, frame_columns=columns)


class TestBuildAndLoad(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.versions_dir = Path(self._tmp.name)
        self.frame = _grid_frame()
        self.folds = build_flag_stratified_cv(
            self.frame, n_splits=N_SPLITS, random_state=SEED
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_build_and_load_round_trip(self) -> None:
        version_dir = build_dataset_version(
            self.frame,
            SPEC,
            folds=self.folds,
            versions_dir=self.versions_dir,
        )
        self.assertTrue(version_dir.is_dir())
        self.assertEqual(
            list_dataset_versions(versions_dir=self.versions_dir),
            (SPEC.name,),
        )
        version = load_dataset_version(
            SPEC.name, versions_dir=self.versions_dir
        )
        self.assertEqual(version.manifest.name, SPEC.name)
        self.assertEqual(
            version.manifest.drop_columns, SPEC.drop_columns
        )
        self.assertNotIn(TIME_OF_DAY_HOUR_COLUMN, version.frame.columns)
        self.assertNotIn(CARD_TYPE_COLUMN, version.frame.columns)
        self.assertIn(ID_COLUMN, version.frame.columns)
        self.assertIn(TARGET_COLUMN, version.frame.columns)
        self.assertEqual(len(version.frame), len(self.frame))
        for expected, actual in zip(self.folds.folds, version.folds.folds):
            self.assertEqual(expected.fold_id, actual.fold_id)
            self.assertEqual(expected.val_ids, actual.val_ids)
            self.assertEqual(expected.train_ids, actual.train_ids)

    def test_folds_partition_the_version_frame(self) -> None:
        build_dataset_version(
            self.frame,
            SPEC,
            folds=self.folds,
            versions_dir=self.versions_dir,
        )
        version = load_dataset_version(
            SPEC.name, versions_dir=self.versions_dir
        )
        seen_val: set[int] = set()
        n_train_total = 0
        for version_fold in iter_folds(version):
            train_ids = set(version_fold.train[ID_COLUMN])
            val_ids = set(version_fold.val[ID_COLUMN])
            self.assertFalse(train_ids & val_ids)
            seen_val |= val_ids
            n_train_total += len(train_ids)
            self.assertNotIn(TIME_OF_DAY_HOUR_COLUMN, version_fold.train.columns)
            self.assertEqual(
                len(version_fold.train) + len(version_fold.val), len(self.frame)
            )
        self.assertEqual(seen_val, set(self.frame[ID_COLUMN]))
        self.assertEqual(n_train_total, len(self.frame) * (N_SPLITS - 1))

    def test_load_fold_selects_columns(self) -> None:
        build_dataset_version(
            self.frame,
            SPEC,
            folds=self.folds,
            versions_dir=self.versions_dir,
        )
        version = load_dataset_version(
            SPEC.name, versions_dir=self.versions_dir
        )
        version_fold = load_fold(
            version, 1, columns=(ID_COLUMN, "amount_usd")
        )
        self.assertEqual(
            list(version_fold.train.columns), [ID_COLUMN, "amount_usd"]
        )

    def test_rebuild_requires_overwrite(self) -> None:
        build_dataset_version(
            self.frame, SPEC, folds=self.folds, versions_dir=self.versions_dir
        )
        with self.assertRaises(FileExistsError):
            build_dataset_version(
                self.frame, SPEC, folds=self.folds, versions_dir=self.versions_dir
            )
        build_dataset_version(
            self.frame,
            SPEC,
            folds=self.folds,
            overwrite=True,
            versions_dir=self.versions_dir,
        )

    def test_frame_ids_must_match_partition(self) -> None:
        trimmed = self.frame.iloc[:-1]
        with self.assertRaisesRegex(ValueError, "rows"):
            build_dataset_version(
                trimmed,
                SPEC,
                folds=self.folds,
                versions_dir=self.versions_dir,
            )

    def test_fingerprint_verification_detects_swapped_payload(self) -> None:
        build_dataset_version(
            self.frame, SPEC, folds=self.folds, versions_dir=self.versions_dir
        )
        version = load_dataset_version(
            SPEC.name, versions_dir=self.versions_dir
        )
        data_path = (
            self.versions_dir / SPEC.name / "data.parquet"
        )
        # a valid, same-shape, but different payload under the same manifest
        swapped = version.frame.copy()
        swapped["amount_usd"] = swapped["amount_usd"] + 1.0
        swapped.to_parquet(data_path, index=False)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_dataset_version(SPEC.name, versions_dir=self.versions_dir)

    def test_manifest_name_must_match_directory(self) -> None:
        build_dataset_version(
            self.frame, SPEC, folds=self.folds, versions_dir=self.versions_dir
        )
        manifest_path = self.versions_dir / SPEC.name / "manifest.json"
        payload = parse_dataset_version_manifest(_load_json(manifest_path))
        tampered = payload._replace(name="other_name")
        manifest_path.write_text(
            json.dumps(_manifest_doc(tampered)), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "not"):
            load_dataset_version_manifest(
                SPEC.name, path=manifest_path
            )


def _load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_doc(manifest) -> dict:
    from ccf_ml.dataset_versions.manifest import dump_dataset_version_manifest

    return dump_dataset_version_manifest(manifest)


if __name__ == "__main__":
    unittest.main()


def _risk_frame() -> pd.DataFrame:
    """12 rows, 3 (merchant, auth) groups; group G3 is a single row."""
    n_rows = 12
    frame = pd.DataFrame(
        {
            ID_COLUMN: np.arange(n_rows, dtype=np.int64),
            TARGET_COLUMN: np.repeat([0, 1], n_rows // 2),
            "amount_usd": np.linspace(1.0, 50.0, n_rows),
            MERCHANT_CATEGORY_COLUMN: [
                "M1", "M1", "M1", "M1", "M1",
                "M2", "M2", "M2", "M2",
                "M3",
                "M1", "M2",
            ],
            AUTH_METHOD_COLUMN: ["A1"] * n_rows,
            DEVICE_TYPE_COLUMN: ["POS Terminal"] * n_rows,
            CHANNEL_COLUMN: ["Online"] * n_rows,
            CARD_TYPE_COLUMN: ["Visa"] * n_rows,
            "hours_since_last_txn": np.linspace(1.0, 12.0, n_rows),
            "txn_count_last_24h": np.arange(n_rows) % 5,
            "distance_from_home_km": np.linspace(0.5, 30.0, n_rows),
            "card_age_months": np.arange(1, n_rows + 1),
            "customer_age": np.arange(20, 20 + n_rows),
            "account_balance_usd": np.linspace(100.0, 9000.0, n_rows),
            "velocity_score": np.linspace(0.05, 0.95, n_rows),
            MERCHANT_RISK_SCORE_COLUMN: np.array(
                [10.0, 20.0, 100.0, 30.0, 40.0,
                 1.0, 2.0, 3.0, 4.0,
                 7.0,
                 50.0, 60.0]
            ),
            IS_NEW_MERCHANT_COLUMN: np.array(
                [False, True, False, False, False,
                 False, False, True, False,
                 True,
                 False, False]
            ),
        }
    )
    for i, name in enumerate(ALARM_FLAG_COLUMNS):
        frame[name] = frame[TARGET_COLUMN] == 1 if i else False
    return frame


class TestRiskContextTransform(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.versions_dir = Path(self._tmp.name)
        self.frame = _risk_frame()
        self.folds = build_flag_stratified_cv(
            self.frame, n_splits=2, random_state=SEED
        )
        self.spec = DatasetVersionSpec(
            name="tiny_risk_context",
            description="test transform version",
            drop_columns=(DEVICE_TYPE_COLUMN, IS_NEW_MERCHANT_COLUMN),
            transform=RISK_CONTEXT_TRANSFORM,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _fold_train_mask(self, fold_id: int) -> np.ndarray:
        val_ids = set(self.folds.fold(fold_id).val_ids)
        return ~self.frame[ID_COLUMN].isin(val_ids).to_numpy()

    def test_new_merchant_rows_get_fold_train_group_mean(self) -> None:
        out, _extras = apply_transform(
            RISK_CONTEXT_TRANSFORM, self.frame, self.folds
        )
        fold_of = np.full(len(self.frame), -1, dtype=np.int64)
        for fold in self.folds.folds:
            for record_id in fold.val_ids:
                fold_of[self.frame[ID_COLUMN].to_numpy() == record_id] = fold.fold_id
        raw = self.frame[MERCHANT_RISK_SCORE_COLUMN].to_numpy(dtype=np.float64)
        new_mask = self.frame[IS_NEW_MERCHANT_COLUMN].to_numpy(dtype=bool)
        for i in range(len(self.frame)):
            if not new_mask[i]:
                self.assertEqual(out[MERCHANT_RISK_SCORE_COLUMN].iloc[i], raw[i])
                continue
            train_mask = fold_of != fold_of[i]
            group = (
                self.frame[MERCHANT_CATEGORY_COLUMN].iloc[i],
                self.frame[AUTH_METHOD_COLUMN].iloc[i],
            )
            in_group = (
                (self.frame[MERCHANT_CATEGORY_COLUMN] == group[0])
                & (self.frame[AUTH_METHOD_COLUMN] == group[1])
            ).to_numpy()
            expected = float(raw[train_mask & in_group].mean())
            if not np.any(train_mask & in_group):
                expected = float(raw[train_mask].mean())
            self.assertEqual(out[MERCHANT_RISK_SCORE_COLUMN].iloc[i], expected)

    def test_aggregates_match_fold_train_stats(self) -> None:
        out, _extras = apply_transform(
            RISK_CONTEXT_TRANSFORM, self.frame, self.folds
        )
        fold_of = np.full(len(self.frame), -1, dtype=np.int64)
        for fold in self.folds.folds:
            for record_id in fold.val_ids:
                fold_of[self.frame[ID_COLUMN].to_numpy() == record_id] = fold.fold_id
        raw = self.frame[MERCHANT_RISK_SCORE_COLUMN].to_numpy(dtype=np.float64)
        for i in range(len(self.frame)):
            train_mask = fold_of != fold_of[i]
            group = (
                self.frame[MERCHANT_CATEGORY_COLUMN].iloc[i],
                self.frame[AUTH_METHOD_COLUMN].iloc[i],
            )
            in_group = (
                (self.frame[MERCHANT_CATEGORY_COLUMN] == group[0])
                & (self.frame[AUTH_METHOD_COLUMN] == group[1])
            ).to_numpy()
            values = raw[train_mask & in_group]
            if len(values) == 0:
                values = raw[train_mask]
            self.assertAlmostEqual(
                out["risk_group_mean"].iloc[i], float(values.mean()), places=9
            )
            self.assertAlmostEqual(
                out["risk_group_p50"].iloc[i], float(np.quantile(values, 0.5)), places=9
            )
            self.assertAlmostEqual(
                out["risk_group_p75"].iloc[i], float(np.quantile(values, 0.75)), places=9
            )
            self.assertAlmostEqual(
                out["risk_group_p90"].iloc[i], float(np.quantile(values, 0.9)), places=9
            )

    def test_unseen_group_falls_back_to_global(self) -> None:
        out, _extras = apply_transform(
            RISK_CONTEXT_TRANSFORM, self.frame, self.folds
        )
        solo_pos = int(
            np.flatnonzero(
                (self.frame[MERCHANT_CATEGORY_COLUMN] == "M3").to_numpy()
            )[0]
        )
        fold_id = None
        for fold in self.folds.folds:
            if solo_pos in set(fold.val_ids):
                fold_id = fold.fold_id
                break
        assert fold_id is not None
        val_ids = set(self.folds.fold(fold_id).val_ids)
        train_mask = ~self.frame[ID_COLUMN].isin(val_ids).to_numpy()
        expected = float(
            self.frame.loc[train_mask, MERCHANT_RISK_SCORE_COLUMN].mean()
        )
        self.assertEqual(out["risk_group_mean"].iloc[solo_pos], expected)

    def test_flag_is_recoverable_from_equality(self) -> None:
        out, _extras = apply_transform(
            RISK_CONTEXT_TRANSFORM, self.frame, self.folds
        )
        recovered = (
            out[MERCHANT_RISK_SCORE_COLUMN] == out["risk_group_mean"]
        ).to_numpy()
        self.assertEqual(
            recovered.tolist(),
            self.frame[IS_NEW_MERCHANT_COLUMN].to_numpy(dtype=bool).tolist(),
        )

    def test_build_load_and_extras_round_trip(self) -> None:
        version_dir = build_dataset_version(
            self.frame,
            self.spec,
            folds=self.folds,
            versions_dir=self.versions_dir,
        )
        for column in RISK_CONTEXT_ADDED_COLUMNS:
            self.assertIn(
                column, pd.read_parquet(version_dir / "data.parquet").columns
            )
        self.assertTrue(
            (version_dir / RISK_CONTEXT_GROUPS_FILENAME).exists()
        )
        version = load_dataset_version(
            self.spec.name, versions_dir=self.versions_dir
        )
        self.assertNotIn(IS_NEW_MERCHANT_COLUMN, version.frame.columns)
        self.assertIn("risk_group_mean", version.frame.columns)
        groups = pd.read_parquet(version_dir / RISK_CONTEXT_GROUPS_FILENAME)
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups["n_rows"].sum(), len(self.frame))

    def test_unknown_transform_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown transform"):
            apply_transform("nope", self.frame, self.folds)

    def test_missing_required_column_raises(self) -> None:
        broken = self.frame.drop(columns=[MERCHANT_RISK_SCORE_COLUMN])
        with self.assertRaisesRegex(ValueError, "missing columns"):
            apply_transform(RISK_CONTEXT_TRANSFORM, broken, self.folds)


class TestProfileContextTransform(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.versions_dir = Path(self._tmp.name)
        self.frame = _risk_frame()
        self.folds = build_flag_stratified_cv(
            self.frame, n_splits=2, random_state=SEED
        )
        self.spec = DatasetVersionSpec(
            name="tiny_profile_context",
            description="test profile transform version",
            drop_columns=(DEVICE_TYPE_COLUMN, IS_NEW_MERCHANT_COLUMN),
            transform=PROFILE_CONTEXT_TRANSFORM,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_percentile_and_hierarchical_fallback(self) -> None:
        train = pd.DataFrame(
            {
                MERCHANT_CATEGORY_COLUMN: ["A"] * 8 + ["B"] * 2,
                AUTH_METHOD_COLUMN: ["X"] * 10,
                CHANNEL_COLUMN: ["Online"] * 10,
                CARD_TYPE_COLUMN: ["Visa"] * 10,
                "amount_usd": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 20.0],
            }
        )
        for flag in ALARM_FLAG_COLUMNS:
            train[flag] = False
        target = pd.DataFrame(
            {
                MERCHANT_CATEGORY_COLUMN: ["A", "B"],
                AUTH_METHOD_COLUMN: ["X", "X"],
                CHANNEL_COLUMN: ["Online", "Online"],
                CARD_TYPE_COLUMN: ["Visa", "Visa"],
                "amount_usd": [2.5, 15.0],
            }
        )
        for flag in ALARM_FLAG_COLUMNS:
            target[flag] = False
        for num in PROFILE_NUMERIC_COLUMNS:
            if num == "amount_usd":
                continue
            train[num] = 1.0
            target[num] = 1.0
        features = _profile_percentile_features(
            train, target, min_group=3
        )
        expected_a = (2 + 0.5 * 0) / 8
        self.assertAlmostEqual(
            features["prof_pct_amount_usd"].iloc[0], expected_a, places=9
        )
        self.assertEqual(int(features["prof_n"].iloc[0]), 8)
        expected_b = (9 + 0.5 * 0) / 10
        self.assertAlmostEqual(
            features["prof_pct_amount_usd"].iloc[1], expected_b, places=9
        )
        self.assertEqual(int(features["prof_n"].iloc[1]), 10)

    def test_transform_end_to_end_fold_safe(self) -> None:
        out, extras = apply_transform(
            PROFILE_CONTEXT_TRANSFORM, self.frame, self.folds
        )
        self.assertEqual(extras, {})
        self.assertFalse(out["prof_pct_amount_usd"].isna().any())
        fold_of = np.full(len(self.frame), -1, dtype=np.int64)
        for fold in self.folds.folds:
            for record_id in fold.val_ids:
                fold_of[self.frame[ID_COLUMN].to_numpy() == record_id] = fold.fold_id
        for fold_id in range(2):
            sel = fold_of == fold_id
            self.assertTrue(
                (out.loc[sel, "prof_n"] == int((~sel).sum())).all(),
                "with 12 rows every profile must fall back to the global level",
            )
        raw = self.frame["amount_usd"].to_numpy(dtype=np.float64)
        i = 0
        train_mask = fold_of != fold_of[i]
        values = raw[train_mask]
        expected = (
            np.sum(values < raw[i]) + 0.5 * np.sum(values == raw[i])
        ) / len(values)
        self.assertAlmostEqual(
            out["prof_pct_amount_usd"].iloc[i], expected, places=9
        )

    def test_build_round_trip(self) -> None:
        version_dir = build_dataset_version(
            self.frame,
            self.spec,
            folds=self.folds,
            versions_dir=self.versions_dir,
        )
        version = load_dataset_version(
            self.spec.name, versions_dir=self.versions_dir
        )
        self.assertIn("prof_n", version.frame.columns)
        self.assertIn("prof_pct_amount_usd", version.frame.columns)
        self.assertNotIn(IS_NEW_MERCHANT_COLUMN, version.frame.columns)
        self.assertEqual(len(version.frame), len(self.frame))
