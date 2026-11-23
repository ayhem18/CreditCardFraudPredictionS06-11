"""Windows, compactness, census, scramble, and a tiny catalog run."""

from __future__ import annotations

import shutil
import unittest

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN
from ccf_ml.data_analysis.census import census_row, region_totals, tag_cell
from ccf_ml.data_analysis.compactness import (
    compactness_1d,
    maybe_outer_difference,
    midrank_quantiles,
)
from ccf_ml.data_analysis.constants import (
    AGE_COLUMN,
    AMOUNT_COLUMN,
    AUTH_COLUMN,
    CARD_COLUMN,
    CHANNEL_COLUMN,
    DEVICE_COLUMN,
    DISTANCE_COLUMN,
    HOUR_COLUMN,
    MERCHANT_COLUMN,
    NEW_MERCHANT_COLUMN,
    RISK_COLUMN,
    VELOCITY_COLUMN,
)
from ccf_ml.data_analysis.parent_shift import (
    ShiftRow,
    ks_d_cell_vs_parent,
    mark_distilled,
    occupancy_ok,
    scramble_window_d,
)
from ccf_ml.data_analysis.profiles import catalog, iter_cells, region_view_from_frame
from ccf_ml.data_analysis.run import artifacts_exist, run_cat_profiling
from ccf_ml.data_analysis.windows import (
    Window,
    age_membership,
    age_windows,
    hour_membership,
    hour_windows,
)
from ccf_ml.file_utils import get_artifacts_dir


class WindowTests(unittest.TestCase):
    def test_hour_family_has_seventy_two(self) -> None:
        self.assertEqual(len(hour_windows()), 72)

    def test_hour_wrap_includes_midnight(self) -> None:
        window = Window(kind="hour", start=22, length=6, label="[22, 04)")
        hours = np.array([21, 22, 23, 0, 1, 2, 3, 4])
        got = hour_membership(hours, window=window)
        expected = np.array([False, True, True, True, True, True, True, False])
        np.testing.assert_array_equal(got, expected)

    def test_age_end_exclusive(self) -> None:
        window = Window(kind="age", start=18, length=8, label="[18, 26)")
        ages = np.array([17.0, 18.0, 25.9, 26.0])
        got = age_membership(ages, window=window)
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(got, expected)

    def test_age_windows_cover_range(self) -> None:
        windows = age_windows(age_min=18, age_max=33)
        starts_len8 = [w.start for w in windows if w.length == 8]
        self.assertEqual(starts_len8[0], 18)
        self.assertEqual(starts_len8[-1], 26)
        self.assertTrue(any(w.label == "[18, 26)" for w in windows))


class CompactnessTests(unittest.TestCase):
    def test_midrank_ties_average(self) -> None:
        values = np.array([1.0, 1.0, 2.0])
        got = midrank_quantiles(values)
        np.testing.assert_allclose(got, np.array([1.5 / 3.0, 1.5 / 3.0, 3.0 / 3.0]))

    def test_signed_extrema_are_rank_one(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0])
        y = np.array([0, 0, 0, 0, 1, 1])
        ids = np.arange(6)
        per_fraud, rollup, extras = compactness_1d(values=values, y=y, ids=ids)
        q1 = extras["q1"]
        q0 = extras["q0"]
        np.testing.assert_allclose(per_fraud["min_d"], q1 - q0.max())
        np.testing.assert_allclose(per_fraud["max_d"], q1 - q0.min())
        self.assertGreater(rollup["mean_d"], 0.0)

    def test_pair_cap_skips_allocation(self) -> None:
        q1 = np.arange(4, dtype=np.float64)
        q0 = np.arange(5, dtype=np.float64)
        self.assertIsNone(maybe_outer_difference(q1, q0, pair_cap=10))
        d_ij = maybe_outer_difference(q1, q0, pair_cap=100)
        self.assertIsNotNone(d_ij)
        np.testing.assert_allclose(d_ij[0], q1[0] - q0)


class CensusTests(unittest.TestCase):
    def test_tags_use_registered_floors(self) -> None:
        self.assertEqual(tag_cell(n_fraud=100, rate=0.01, region_rate=0.02), "bulky_cold")
        self.assertEqual(tag_cell(n_fraud=20, rate=0.05, region_rate=0.02), "hot")
        self.assertEqual(tag_cell(n_fraud=50, rate=0.03, region_rate=0.02), "other")

    def test_census_shares(self) -> None:
        y = np.array([0, 0, 1, 1, 0])
        totals = region_totals(y=y)
        y_cell = np.array([1, 0, 0])
        row = census_row(
            profile_id="P1",
            spec_name="P1:device_type",
            cell_key="computer",
            parent_key="all",
            y_cell=y_cell,
            totals=totals,
        )
        self.assertEqual(row.n, 3)
        self.assertEqual(row.n_fraud, 1)
        self.assertAlmostEqual(row.share_frauds, 0.5)
        self.assertAlmostEqual(row.share_rows, 0.6)


class ParentShiftTests(unittest.TestCase):
    def test_occupancy_needs_both_sides(self) -> None:
        self.assertFalse(occupancy_ok(n_cell=500, n_parent=999))
        self.assertTrue(occupancy_ok(n_cell=500, n_parent=1000))

    def test_ks_detects_location_shift(self) -> None:
        parent = np.concatenate([np.zeros(200), np.ones(200)])
        cell = np.ones(200)
        d = ks_d_cell_vs_parent(cell, parent)
        self.assertGreaterEqual(d, 0.4)

    def test_scramble_hour_returns_finite_or_none(self) -> None:
        parent_numeric = np.linspace(0.0, 1.0, 1200)
        hours = np.arange(1200) % 24
        window = Window(kind="hour", start=0, length=8, label="[00, 08)")
        rng = np.random.default_rng(0)
        got = scramble_window_d(
            parent_numeric=parent_numeric,
            parent_axis_values=hours.astype(np.float64),
            window=window,
            rng=rng,
        )
        self.assertTrue(got is None or np.isfinite(got))

    def test_distill_keeps_one_window_per_parent_numeric(self) -> None:
        def _row(*, cell_key: str, numeric: str, ks_d: float, verdict: str) -> ShiftRow:
            return ShiftRow(
                profile_id="P7",
                spec_name="P7:merchant_category|hour_window",
                cell_key=cell_key,
                parent_key="Groceries",
                parent_axis="merchant_category",
                numeric=numeric,
                n_cell=800,
                n_parent=2000,
                ks_d=ks_d,
                delta_f_p50=0.0,
                delta_f_p90=0.0,
                mean_parent_rank=0.5,
                scramble_d=0.02,
                verdict=verdict,
                distilled=False,
                is_window=True,
            )

        rows = [
            _row(cell_key="a|[00, 08)", numeric="amount_usd", ks_d=0.10, verdict="fe_candidate"),
            _row(cell_key="a|[08, 16)", numeric="amount_usd", ks_d=0.20, verdict="fe_candidate"),
            _row(cell_key="a|[00, 08)", numeric="velocity_score", ks_d=0.12, verdict="fe_candidate"),
            _row(cell_key="a|[08, 16)", numeric="amount_usd", ks_d=0.50, verdict="below_floor"),
        ]
        marked = mark_distilled(rows)
        flags = [row.distilled for row in marked]
        self.assertEqual(flags, [False, True, True, False])


class CatalogTests(unittest.TestCase):
    def test_p5_skips_device_channel_duplicate(self) -> None:
        specs = catalog()
        p5_names = [spec.spec_name for spec in specs if spec.profile_id == "P5"]
        self.assertNotIn("P5:channel|device_type", p5_names)
        p4_names = [spec.spec_name for spec in specs if spec.profile_id == "P4"]
        self.assertIn("P4:device_type|channel", p4_names)

    def test_p6_does_not_impute_missing_pairs(self) -> None:
        frame = _toy_frame(n_rows=1200, seed=1)
        frame = frame.loc[
            ~((frame[DEVICE_COLUMN] == "computer") & (frame[CHANNEL_COLUMN] == "pos"))
        ].reset_index(drop=True)
        view = region_view_from_frame(frame)
        spec = [item for item in catalog() if item.profile_id == "P6"][0]
        keys = {cell.cell_key for cell in iter_cells(view, spec)}
        self.assertNotIn("computer|pos", keys)
        self.assertIn("computer|web", keys)

    def test_p9_joint_drops_age(self) -> None:
        spec = [item for item in catalog() if item.profile_id == "P9"][0]
        self.assertEqual(spec.joint_coordinates, (DISTANCE_COLUMN, AMOUNT_COLUMN))
        self.assertEqual(spec.window_axis, AGE_COLUMN)


class RunTests(unittest.TestCase):
    def test_tiny_run_writes_census_and_skips_rerun(self) -> None:
        stem = "cat_profiling_unit_test"
        artifact_root = get_artifacts_dir() / stem
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
        frame = _toy_frame(n_rows=2400, seed=2)
        specs = tuple(
            spec
            for spec in catalog()
            if spec.spec_name in {"P1:merchant_category", "P6:device_type|channel"}
        )
        tables = run_cat_profiling(
            frame=frame,
            specs=specs,
            script_name=stem,
            force=True,
            make_plots=False,
        )
        self.assertGreater(len(tables.census), 0)
        self.assertTrue((tables.census["profile_id"] == "P6").any())
        self.assertTrue(artifacts_exist(script_name=stem))
        again = run_cat_profiling(frame=frame, specs=specs, script_name=stem, force=False)
        pd.testing.assert_frame_equal(tables.census, again.census)
        shutil.rmtree(artifact_root)


def _toy_frame(*, n_rows: int, seed: int) -> pd.DataFrame:
    """A region-shaped table with a distance shift on computer × web frauds."""
    rng = np.random.default_rng(seed)
    merchants = np.array(["Groceries", "Crypto"])
    devices = np.array(["computer", "mobile"])
    channels = np.array(["web", "pos"])
    merchant = merchants[rng.integers(0, 2, size=n_rows)]
    device = devices[rng.integers(0, 2, size=n_rows)]
    channel = channels[rng.integers(0, 2, size=n_rows)]
    is_new = rng.integers(0, 2, size=n_rows)
    amount = rng.normal(80.0, 20.0, size=n_rows)
    amount[merchant == "Crypto"] += 40.0
    distance = rng.normal(10.0, 3.0, size=n_rows)
    y = (rng.random(n_rows) < 0.04).astype(np.int64)
    far = (device == "computer") & (channel == "web") & (y == 1)
    distance[far] += 25.0
    extra = {
        "hours_since_last_txn": rng.uniform(0.0, 48.0, size=n_rows),
        "card_age_months": rng.uniform(1.0, 120.0, size=n_rows),
        "account_balance_usd": rng.uniform(100.0, 5000.0, size=n_rows),
        RISK_COLUMN: rng.uniform(0.0, 1.0, size=n_rows),
        VELOCITY_COLUMN: rng.uniform(0.0, 1.0, size=n_rows),
        AUTH_COLUMN: np.array(["pin", "otp"])[rng.integers(0, 2, size=n_rows)],
        CARD_COLUMN: np.array(["visa", "mc"])[rng.integers(0, 2, size=n_rows)],
        NEW_MERCHANT_COLUMN: is_new,
        AGE_COLUMN: rng.integers(18, 70, size=n_rows),
        HOUR_COLUMN: rng.integers(0, 24, size=n_rows),
    }
    frame = pd.DataFrame(
        {
            ID_COLUMN: np.arange(n_rows, dtype=np.int64),
            TARGET_COLUMN: y,
            MERCHANT_COLUMN: merchant,
            DEVICE_COLUMN: device,
            CHANNEL_COLUMN: channel,
            AMOUNT_COLUMN: amount,
            DISTANCE_COLUMN: distance,
            **extra,
        }
    )
    return frame


if __name__ == "__main__":
    unittest.main()
