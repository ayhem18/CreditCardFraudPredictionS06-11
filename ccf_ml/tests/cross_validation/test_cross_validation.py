"""Alarm × target stratified k-fold: partition, balance, frozen artifact."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import check_cv

from ccf_ml.cross_validation.alarm import (
    alarm_flag_mask,
    binary_pair_stratum,
    flag_target_stratum,
)
from ccf_ml.cross_validation.constants import (
    ALARM_FLAG_COLUMNS,
    ID_COLUMN,
    STRATUM_ALARM_FRAUD,
    STRATUM_ALARM_NOT_FRAUD,
    STRATUM_CODE_TO_NAME,
    STRATUM_NO_ALARM_FRAUD,
    STRATUM_NO_ALARM_NOT_FRAUD,
    TARGET_COLUMN,
)
from ccf_ml.cross_validation.flag_stratified import (
    build_flag_stratified_cv,
    load_flag_stratified_cv,
    write_flag_stratified_cv,
)
from ccf_ml.cross_validation.predefined import (
    predefined_split_from_folds,
    val_fold_vector,
)
from ccf_ml.cross_validation.stratified import stratified_kfold_ids


N_PER_CELL = 20
N_SPLITS = 2
SEED = 0


def _grid_frame(*, n_per_cell: int = N_PER_CELL) -> pd.DataFrame:
    n_rows = n_per_cell * 4
    alarm = np.repeat([False, False, True, True], n_per_cell)
    fraud = np.repeat([0, 1, 0, 1], n_per_cell)
    frame = pd.DataFrame(
        {
            ID_COLUMN: np.arange(n_rows, dtype=np.int64),
            TARGET_COLUMN: fraud,
        }
    )
    n_flags = len(ALARM_FLAG_COLUMNS)
    for i in range(n_flags):
        name = ALARM_FLAG_COLUMNS[i]
        if i == 0:
            frame[name] = alarm
        else:
            frame[name] = False
    return frame


class AlarmStratumTests(unittest.TestCase):
    def test_binary_pair_codes(self) -> None:
        left = np.array([0, 0, 1, 1])
        right = np.array([0, 1, 0, 1])
        codes = binary_pair_stratum(left=left, right=right)
        np.testing.assert_array_equal(
            codes,
            np.array(
                [
                    STRATUM_NO_ALARM_NOT_FRAUD,
                    STRATUM_NO_ALARM_FRAUD,
                    STRATUM_ALARM_NOT_FRAUD,
                    STRATUM_ALARM_FRAUD,
                ]
            ),
        )

    def test_any_flag_is_alarm(self) -> None:
        frame = _grid_frame()
        frame.loc[0, ALARM_FLAG_COLUMNS[0]] = False
        frame.loc[0, ALARM_FLAG_COLUMNS[-1]] = True
        mask = alarm_flag_mask(frame)
        self.assertTrue(bool(mask.iloc[0]))

    def test_flag_target_stratum_counts(self) -> None:
        frame = _grid_frame()
        stratum = flag_target_stratum(frame)
        unique, counts = np.unique(stratum, return_counts=True)
        self.assertEqual(set(unique.tolist()), set(STRATUM_CODE_TO_NAME))
        self.assertTrue(np.all(counts == N_PER_CELL))


class StratifiedKfoldTests(unittest.TestCase):
    def test_vals_partition_and_stratum_balance(self) -> None:
        frame = _grid_frame()
        record_ids = frame[ID_COLUMN].to_numpy()
        stratum = flag_target_stratum(frame)
        folds = stratified_kfold_ids(
            record_ids=record_ids,
            stratum=stratum,
            n_splits=N_SPLITS,
            random_state=SEED,
        )
        self.assertEqual(len(folds), N_SPLITS)
        eligible = set(int(x) for x in record_ids)
        val_union: set[int] = set()
        position = {int(record_ids[i]): i for i in range(len(record_ids))}
        expected_per_cell = N_PER_CELL // N_SPLITS
        for train_ids, val_ids in ((fold.train_ids, fold.val_ids) for fold in folds):
            self.assertEqual(set(train_ids).union(val_ids), eligible)
            self.assertFalse(set(train_ids).intersection(val_ids))
            self.assertEqual(tuple(sorted(train_ids)), train_ids)
            self.assertEqual(tuple(sorted(val_ids)), val_ids)
            leaked = val_union.intersection(val_ids)
            self.assertFalse(leaked)
            val_union.update(val_ids)
            val_index = np.array([position[record_id] for record_id in val_ids])
            val_stratum = stratum[val_index]
            unique, counts = np.unique(val_stratum, return_counts=True)
            self.assertEqual(set(unique.tolist()), set(STRATUM_CODE_TO_NAME))
            for count in counts.tolist():
                self.assertLessEqual(abs(count - expected_per_cell), 1)
        self.assertEqual(val_union, eligible)

    def test_duplicate_ids_raise(self) -> None:
        with self.assertRaises(ValueError):
            stratified_kfold_ids(
                record_ids=np.array([1, 1, 2, 3]),
                stratum=np.array([0, 1, 0, 1]),
                n_splits=2,
                random_state=0,
            )


class FlagStratifiedCvTests(unittest.TestCase):
    def test_build_is_deterministic(self) -> None:
        frame = _grid_frame()
        first = build_flag_stratified_cv(
            frame,
            n_splits=N_SPLITS,
            random_state=SEED,
        )
        second = build_flag_stratified_cv(
            frame,
            n_splits=N_SPLITS,
            random_state=SEED,
        )
        self.assertEqual(first.folds, second.folds)
        other = build_flag_stratified_cv(
            frame,
            n_splits=N_SPLITS,
            random_state=SEED + 1,
        )
        self.assertNotEqual(first.folds, other.folds)

    def test_json_roundtrip_and_refuse_overwrite(self) -> None:
        frame = _grid_frame()
        document = build_flag_stratified_cv(
            frame,
            n_splits=N_SPLITS,
            random_state=SEED,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flag_stratified_cv.json"
            write_flag_stratified_cv(document, path=path)
            loaded = load_flag_stratified_cv(path=path)
            self.assertEqual(loaded.folds, document.folds)
            self.assertEqual(loaded.stratum_counts, document.stratum_counts)
            with self.assertRaises(FileExistsError):
                write_flag_stratified_cv(document, path=path)

    def test_predefined_split_matches_val_lists(self) -> None:
        frame = _grid_frame()
        document = build_flag_stratified_cv(
            frame,
            n_splits=N_SPLITS,
            random_state=SEED,
        )
        record_ids = tuple(int(x) for x in frame[ID_COLUMN].to_numpy())
        vector = val_fold_vector(
            record_ids=record_ids,
            val_ids_by_fold=tuple(fold.val_ids for fold in document.folds),
        )
        splitter = predefined_split_from_folds(
            record_ids=record_ids,
            folds=document.folds,
        )
        self.assertEqual(splitter.get_n_splits(), N_SPLITS)
        splits = list(splitter.split())
        for fold_id in range(N_SPLITS):
            val_pos = np.where(vector == fold_id)[0]
            _, val_index = splits[fold_id]
            np.testing.assert_array_equal(val_index, val_pos)
            val_ids = tuple(record_ids[int(i)] for i in val_index)
            self.assertEqual(tuple(sorted(val_ids)), document.folds[fold_id].val_ids)
        checked = check_cv(splitter)
        self.assertEqual(checked.get_n_splits(), N_SPLITS)

    def test_missing_alarm_column_raises(self) -> None:
        frame = _grid_frame().drop(columns=[ALARM_FLAG_COLUMNS[0]])
        with self.assertRaises(ValueError):
            build_flag_stratified_cv(frame, n_splits=N_SPLITS, random_state=SEED)
