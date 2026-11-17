"""Fraud-conditional ECDF: estimate, apply, fold-to-fold distance."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.constants import TARGET_COLUMN
from ccf_ml.features.fraud_ecdf import (
    common_grid,
    fraud_ecdf_from_arrays,
    fraud_ecdf_from_frame,
    max_abs_difference,
    transform_frame,
)


class FraudEcdfTests(unittest.TestCase):
    def test_evaluate_matches_count_formula(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1, 0, 1, 0])
        curve = fraud_ecdf_from_arrays(column="age", x=x, y=y)
        query = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        got = curve.evaluate(query)
        expected = np.array([0.0, 0.5, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(got, expected)
        self.assertEqual(curve.n, 2)

    def test_ties_count_once_per_row(self) -> None:
        x = np.array([1.0, 1.0, 2.0])
        y = np.array([1, 1, 1])
        curve = fraud_ecdf_from_arrays(column="age", x=x, y=y)
        got = curve.evaluate(np.array([1.0, 2.0]))
        np.testing.assert_allclose(got, np.array([2.0 / 3.0, 1.0]))

    def test_curve_table_ends_at_one(self) -> None:
        x = np.array([5.0, 1.0, 1.0, 3.0])
        y = np.array([1, 1, 0, 1])
        table = fraud_ecdf_from_arrays(column="age", x=x, y=y).curve_table()
        np.testing.assert_array_equal(table["age"].to_numpy(), np.array([1.0, 3.0, 5.0]))
        np.testing.assert_allclose(
            table["cdf"].to_numpy(),
            np.array([1.0 / 3.0, 2.0 / 3.0, 1.0]),
        )

    def test_from_frame_uses_target_column(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [10, 20, 30],
                TARGET_COLUMN: [1, 0, 1],
            }
        )
        curve = fraud_ecdf_from_frame(frame, column="age")
        np.testing.assert_allclose(curve.evaluate(np.array([10.0, 30.0])), [0.5, 1.0])

    def test_no_fraud_raises(self) -> None:
        x = np.array([1.0, 2.0])
        y = np.array([0, 0])
        with self.assertRaisesRegex(ValueError, "no fraud rows"):
            fraud_ecdf_from_arrays(column="age", x=x, y=y)

    def test_pairwise_max_abs_on_union_grid(self) -> None:
        left = fraud_ecdf_from_arrays(
            column="age",
            x=np.array([1.0, 3.0]),
            y=np.array([1, 1]),
        )
        right = fraud_ecdf_from_arrays(
            column="age",
            x=np.array([1.0, 2.0]),
            y=np.array([1, 1]),
        )
        grid = common_grid((left, right))
        np.testing.assert_array_equal(grid, np.array([1.0, 2.0, 3.0]))
        gap = max_abs_difference(left, right, grid=grid)
        self.assertAlmostEqual(gap, 0.5)

    def test_transform_frame_replaces_column(self) -> None:
        frame = pd.DataFrame({"age": [1.0, 2.0, 3.0], "other": [0.0, 0.0, 0.0]})
        curve = fraud_ecdf_from_arrays(
            column="age",
            x=np.array([1.0, 3.0]),
            y=np.array([1, 1]),
        )
        out = transform_frame(frame, curves=(curve,))
        np.testing.assert_allclose(out["age"].to_numpy(), np.array([0.5, 0.5, 1.0]))
        np.testing.assert_array_equal(out["other"].to_numpy(), frame["other"].to_numpy())


if __name__ == "__main__":
    unittest.main()
