"""Per-column inverted index: raw keys, percentile ranges, postings lookup."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.constants import ID_COLUMN
from ccf_ml.index.column_index import (
    RAW_UNIQUE_LIMIT,
    ColumnRole,
    Keying,
    PercentileRange,
    column_index_from_frame,
    frame_index_from_frame,
    parse_percentile_range,
)


class PercentileRangeTests(unittest.TestCase):
    def test_str_is_half_open_interval(self) -> None:
        key = parse_percentile_range(start=1, stop=2)
        self.assertEqual(str(key), "[1, 2[")

    def test_start_not_less_than_stop_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "start < stop"):
            parse_percentile_range(start=2, stop=2)


class CategoricalColumnIndexTests(unittest.TestCase):
    def test_value_maps_to_matching_ids(self) -> None:
        frame = pd.DataFrame(
            {
                ID_COLUMN: np.array([10, 11, 12, 13], dtype=np.int64),
                "channel": ["POS", "Online", "POS", "POS"],
            }
        )
        index = column_index_from_frame(
            frame,
            column="channel",
            id_column=ID_COLUMN,
            role=ColumnRole.CATEGORICAL,
        )
        self.assertEqual(index.keying, Keying.CATEGORICAL)
        self.assertEqual(index.ids_for("POS"), frozenset({10, 12, 13}))
        self.assertEqual(index.ids_for("Online"), frozenset({11}))
        self.assertEqual(index.ids_for("ATM"), frozenset())

    def test_boolean_values_are_python_bools(self) -> None:
        frame = pd.DataFrame(
            {
                ID_COLUMN: np.array([0, 1, 2], dtype=np.int64),
                "used_vpn": [False, True, False],
            }
        )
        index = column_index_from_frame(
            frame,
            column="used_vpn",
            id_column=ID_COLUMN,
            role=ColumnRole.CATEGORICAL,
        )
        self.assertIn(False, index.keys())
        self.assertIn(True, index.keys())
        self.assertEqual(index.ids_for(False), frozenset({0, 2}))
        self.assertEqual(index.ids_for_value(True), frozenset({1}))


class NumericRawColumnIndexTests(unittest.TestCase):
    def test_below_unique_limit_uses_raw_values(self) -> None:
        n = RAW_UNIQUE_LIMIT - 1
        values = np.arange(n, dtype=np.int64)
        frame = pd.DataFrame(
            {
                ID_COLUMN: np.arange(n, dtype=np.int64),
                "txn_count_last_24h": values,
            }
        )
        index = column_index_from_frame(
            frame,
            column="txn_count_last_24h",
            id_column=ID_COLUMN,
            role=ColumnRole.NUMERIC,
        )
        self.assertEqual(index.keying, Keying.NUMERIC_RAW)
        self.assertIsNone(index.percentile_edges)
        self.assertEqual(index.key_of(3), 3)
        self.assertEqual(index.ids_for(3), frozenset({3}))


class NumericPercentileColumnIndexTests(unittest.TestCase):
    def test_at_unique_limit_uses_percentile_range_keys(self) -> None:
        n = RAW_UNIQUE_LIMIT
        frame = pd.DataFrame(
            {
                ID_COLUMN: np.arange(n, dtype=np.int64),
                "amount_usd": np.arange(n, dtype=np.float64),
            }
        )
        index = column_index_from_frame(
            frame,
            column="amount_usd",
            id_column=ID_COLUMN,
            role=ColumnRole.NUMERIC,
        )
        self.assertEqual(index.keying, Keying.NUMERIC_PERCENTILE)
        self.assertIsNotNone(index.percentile_edges)

    def test_one_two_range_is_first_to_second_percentile(self) -> None:
        n = 1000
        frame = pd.DataFrame(
            {
                ID_COLUMN: np.arange(n, dtype=np.int64),
                "amount_usd": np.arange(n, dtype=np.float64),
            }
        )
        index = column_index_from_frame(
            frame,
            column="amount_usd",
            id_column=ID_COLUMN,
            role=ColumnRole.NUMERIC,
        )
        edges = index.percentile_edges
        assert edges is not None
        self.assertGreater(edges[2], edges[1])
        mid = 0.5 * (edges[1] + edges[2])
        self.assertEqual(index.key_of(mid), PercentileRange(start=1, stop=2))
        ids = index.ids_for(PercentileRange(start=1, stop=2))
        self.assertGreater(len(ids), 0)
        values = frame.loc[frame[ID_COLUMN].isin(ids), "amount_usd"].to_numpy()
        self.assertTrue(np.all(values >= edges[1]))
        self.assertTrue(np.all(values < edges[2]))

    def test_maximum_is_in_last_bin(self) -> None:
        n = 200
        values = np.arange(n, dtype=np.float64)
        frame = pd.DataFrame(
            {
                ID_COLUMN: np.arange(n, dtype=np.int64),
                "amount_usd": values,
            }
        )
        index = column_index_from_frame(
            frame,
            column="amount_usd",
            id_column=ID_COLUMN,
            role=ColumnRole.NUMERIC,
        )
        self.assertEqual(
            index.key_of(float(n - 1)),
            PercentileRange(start=99, stop=100),
        )
        last_ids = index.ids_for(PercentileRange(start=99, stop=100))
        self.assertIn(n - 1, last_ids)

    def test_value_outside_indexed_range_raises(self) -> None:
        n = RAW_UNIQUE_LIMIT
        frame = pd.DataFrame(
            {
                ID_COLUMN: np.arange(n, dtype=np.int64),
                "amount_usd": np.arange(n, dtype=np.float64),
            }
        )
        index = column_index_from_frame(
            frame,
            column="amount_usd",
            id_column=ID_COLUMN,
            role=ColumnRole.NUMERIC,
        )
        with self.assertRaisesRegex(ValueError, "outside the indexed range"):
            index.key_of(-1.0)

    def test_width_greater_than_one_raises(self) -> None:
        n = RAW_UNIQUE_LIMIT
        frame = pd.DataFrame(
            {
                ID_COLUMN: np.arange(n, dtype=np.int64),
                "amount_usd": np.arange(n, dtype=np.float64),
            }
        )
        index = column_index_from_frame(
            frame,
            column="amount_usd",
            id_column=ID_COLUMN,
            role=ColumnRole.NUMERIC,
        )
        with self.assertRaisesRegex(ValueError, "width 1"):
            index.ids_for(PercentileRange(start=1, stop=3))


class FrameIndexTests(unittest.TestCase):
    def test_indexes_each_named_column(self) -> None:
        frame = pd.DataFrame(
            {
                ID_COLUMN: np.array([5, 6, 7], dtype=np.int64),
                "channel": ["POS", "POS", "Online"],
                "txn_count_last_24h": np.array([1, 1, 2], dtype=np.int64),
            }
        )
        index = frame_index_from_frame(
            frame,
            id_column=ID_COLUMN,
            categorical_columns=("channel",),
            numeric_columns=("txn_count_last_24h",),
        )
        self.assertEqual(len(index.columns), 2)
        self.assertEqual(index.ids, frozenset({5, 6, 7}))
        self.assertEqual(index.ids_for("channel", "POS"), frozenset({5, 6}))
        self.assertEqual(
            index.ids_for_value("txn_count_last_24h", 1),
            frozenset({5, 6}),
        )
        self.assertEqual(
            index.ids_matching(values={"channel": "POS", "txn_count_last_24h": 1}),
            frozenset({5, 6}),
        )
        self.assertEqual(
            index.ids_matching(values={"channel": "Online", "txn_count_last_24h": 1}),
            frozenset(),
        )

    def test_ids_matching_empty_values_raises(self) -> None:
        frame = pd.DataFrame(
            {
                ID_COLUMN: np.array([0, 1], dtype=np.int64),
                "channel": ["POS", "Online"],
            }
        )
        index = frame_index_from_frame(
            frame,
            id_column=ID_COLUMN,
            categorical_columns=("channel",),
            numeric_columns=(),
        )
        with self.assertRaisesRegex(ValueError, "values is empty"):
            index.ids_matching(values={})

    def test_duplicate_column_listing_raises(self) -> None:
        frame = pd.DataFrame(
            {
                ID_COLUMN: np.array([0, 1], dtype=np.int64),
                "channel": ["POS", "Online"],
            }
        )
        with self.assertRaisesRegex(ValueError, "listed more than once"):
            frame_index_from_frame(
                frame,
                id_column=ID_COLUMN,
                categorical_columns=("channel", "channel"),
                numeric_columns=(),
            )

    def test_na_raises(self) -> None:
        frame = pd.DataFrame(
            {
                ID_COLUMN: np.array([0, 1], dtype=np.int64),
                "channel": ["POS", None],
            }
        )
        with self.assertRaisesRegex(ValueError, "contains NA"):
            column_index_from_frame(
                frame,
                column="channel",
                id_column=ID_COLUMN,
                role=ColumnRole.CATEGORICAL,
            )


if __name__ == "__main__":
    unittest.main()
