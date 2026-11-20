"""Unified alarm x profile x target CV: stratum, build, freeze, restriction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.alarm import flag_target_stratum, stratum_count_names
from ccf_ml.cross_validation.constants import (
    ALARM_FLAG_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
)
from ccf_ml.region_modeling import (
    alarm_profile_fold_summaries,
    alarm_profile_stratum,
    alarm_target_totals,
    build_alarm_profile_stratified_cv,
    load_alarm_profile_stratified_cv,
    profile_stratum,
    restrict_to_no_alarm,
    variant_by_name,
    variant_model_matrix,
    with_profile_columns,
    write_alarm_profile_stratified_cv,
)
from tests.region_modeling.test_region_modeling import _region_frame


def _full_frame() -> pd.DataFrame:
    """The region test frame with an alarm side added.

    ids 0-59 carry the alarm flag split evenly across the target (two alarm
    strata of 30), ids 120-121 are a 2-row alarm cell that must collapse.
    """
    frame = _region_frame()
    frame.loc[0:59, ALARM_FLAG_COLUMNS[0]] = True
    frame.loc[[120, 121], ALARM_FLAG_COLUMNS[0]] = True
    return frame


def _profiled_frame() -> pd.DataFrame:
    return with_profile_columns(_full_frame())


def _region_only(profiled_frame: pd.DataFrame) -> pd.DataFrame:
    """The profiled frame's no-alarm rows (flags still present)."""
    mask = profiled_frame[ALARM_FLAG_COLUMNS[0]]
    for column in ALARM_FLAG_COLUMNS[1:]:
        mask = mask | profiled_frame[column]
    return profiled_frame.loc[~mask].copy()


class TestAlarmProfileStratum(unittest.TestCase):
    def test_labels_and_collapse(self) -> None:
        stratum = alarm_profile_stratum(_profiled_frame(), min_stratum_rows=25)
        expected = {
            "no_alarm|hot|PIN|personal|True|0",
            "no_alarm|hot|PIN|personal|True|1",
            "no_alarm|_small_|0",
            "no_alarm|_small_|1",
            "alarm|background|PIN|personal|False|0",
            "alarm|background|PIN|personal|False|1",
            "alarm|_small_|0",
        }
        self.assertEqual(set(stratum), expected)
        counts = stratum.value_counts()
        self.assertEqual(int(counts["no_alarm|_small_|1"]), 4)
        self.assertEqual(int(counts["alarm|_small_|0"]), 2)


class TestBuildAndInvariants(unittest.TestCase):
    def test_partition_and_per_fold_stratum_coverage(self) -> None:
        frame = _profiled_frame()
        cv = build_alarm_profile_stratified_cv(frame, n_splits=2, random_state=0)
        self.assertEqual(cv.n_rows, len(frame))
        val_union: set[int] = set()
        for i, fold in enumerate(cv.folds):
            self.assertEqual(fold.fold_id, i)
            val_union.update(fold.val_ids)
        self.assertEqual(val_union, set(frame[ID_COLUMN]))
        labels = alarm_profile_stratum(frame, min_stratum_rows=25)
        summaries = alarm_profile_fold_summaries(cv, frame)
        for summary in summaries:
            # every stratum, however small, reaches every fold's val list
            self.assertEqual(
                set(summary.stratum_counts_val), set(labels.value_counts().index)
            )
            self.assertEqual(
                summary.n_train + summary.n_val, len(frame)
            )

    def test_alarm_target_totals_refine_flag_stratum(self) -> None:
        frame = _profiled_frame()
        cv = build_alarm_profile_stratified_cv(frame, n_splits=2, random_state=0)
        totals = alarm_target_totals(cv.stratum_counts)
        expected = stratum_count_names(flag_target_stratum(frame))
        self.assertEqual(totals, expected)
        self.assertEqual(totals["no_alarm_not_fraud"], 32)
        self.assertEqual(totals["no_alarm_fraud"], 34)
        self.assertEqual(totals["alarm_not_fraud"], 32)
        self.assertEqual(totals["alarm_fraud"], 30)
        summaries = alarm_profile_fold_summaries(cv, frame)
        for summary in summaries:
            fold_totals = alarm_target_totals(summary.stratum_counts_val)
            self.assertEqual(fold_totals["no_alarm_fraud"], 17)
            self.assertEqual(fold_totals["alarm_fraud"], 15)


class TestRestrictToNoAlarm(unittest.TestCase):
    def test_restriction_preserves_fold_assignment(self) -> None:
        frame = _profiled_frame()
        cv = build_alarm_profile_stratified_cv(frame, n_splits=2, random_state=0)
        region = _region_only(frame)
        restricted = restrict_to_no_alarm(cv, region)
        self.assertEqual(restricted.n_rows, len(region))
        region_ids = set(region[ID_COLUMN])
        val_union: set[int] = set()
        for fold, unified_fold in zip(restricted.folds, cv.folds):
            self.assertEqual(fold.fold_id, unified_fold.fold_id)
            self.assertEqual(
                set(fold.val_ids), set(unified_fold.val_ids) & region_ids
            )
            self.assertEqual(
                set(fold.train_ids), region_ids - set(fold.val_ids)
            )
            val_union.update(fold.val_ids)
        self.assertEqual(val_union, region_ids)
        expected_counts = {
            str(k): int(v)
            for k, v in profile_stratum(
                region, min_stratum_rows=cv.min_stratum_rows
            ).value_counts().items()
        }
        self.assertEqual(restricted.stratum_counts, expected_counts)

    def test_restricted_cv_feeds_the_variant_ladder(self) -> None:
        frame = _profiled_frame()
        cv = build_alarm_profile_stratified_cv(frame, n_splits=2, random_state=0)
        region = _region_only(frame)
        restricted = restrict_to_no_alarm(cv, region)
        spec = variant_by_name("v4_profile_id_quantiles")
        matrix, _features, _cats, _nums = variant_model_matrix(region, spec, cv=restricted)
        self.assertEqual(matrix.shape[0], len(region))
        quantile_cols = [c for c in matrix.columns if c.startswith("pq_")]
        self.assertTrue(quantile_cols)
        self.assertFalse(matrix[quantile_cols].isna().to_numpy().any())

    def test_restriction_rejects_foreign_id(self) -> None:
        frame = _profiled_frame()
        cv = build_alarm_profile_stratified_cv(frame, n_splits=2, random_state=0)
        region = _region_only(frame)
        intruder = region.iloc[[0]].copy()
        intruder[ID_COLUMN] = 999_999
        with self.assertRaises(ValueError):
            restrict_to_no_alarm(cv, pd.concat([region, intruder]))


class TestRoundTrip(unittest.TestCase):
    def test_write_load_round_trip(self) -> None:
        frame = _profiled_frame()
        cv = build_alarm_profile_stratified_cv(frame, n_splits=2, random_state=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unified_cv.json"
            written = write_alarm_profile_stratified_cv(cv, path=path)
            self.assertTrue(written.exists())
            loaded = load_alarm_profile_stratified_cv(path=path)
            self.assertEqual(loaded.n_rows, cv.n_rows)
            self.assertEqual(loaded.alarm_flag_columns, cv.alarm_flag_columns)
            self.assertEqual(loaded.stratum_counts, cv.stratum_counts)
            self.assertEqual(
                [list(f.val_ids) for f in loaded.folds],
                [list(f.val_ids) for f in cv.folds],
            )
            self.assertEqual(
                [list(f.train_ids) for f in loaded.folds],
                [list(f.train_ids) for f in cv.folds],
            )
            with self.assertRaises(FileExistsError):
                write_alarm_profile_stratified_cv(cv, path=path)
            write_alarm_profile_stratified_cv(cv, path=path, overwrite=True)

    def test_parse_rejects_tampered_documents(self) -> None:
        frame = _profiled_frame()
        cv = build_alarm_profile_stratified_cv(frame, n_splits=2, random_state=0)
        from ccf_ml.region_modeling.alarm_profile_cv import (
            dump_alarm_profile_stratified_cv,
            parse_alarm_profile_stratified_cv,
        )

        document = dump_alarm_profile_stratified_cv(cv)
        bad_scheme = dict(document, scheme="flag_stratified")
        with self.assertRaises(ValueError):
            parse_alarm_profile_stratified_cv(bad_scheme)
        missing_key = {k: v for k, v in document.items() if k != "n_rows"}
        with self.assertRaises(ValueError):
            parse_alarm_profile_stratified_cv(missing_key)
        broken_counts = dict(document)
        broken_counts["stratum_counts"] = dict(broken_counts["stratum_counts"])
        first_key = next(iter(broken_counts["stratum_counts"]))
        broken_counts["stratum_counts"][first_key] = (
            broken_counts["stratum_counts"][first_key] + 1
        )
        with self.assertRaises(ValueError):
            parse_alarm_profile_stratified_cv(broken_counts)


if __name__ == "__main__":
    unittest.main()
