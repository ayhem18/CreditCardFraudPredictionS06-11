"""Label census of a profile cell."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ccf_ml.data_analysis.constants import (
    CENSUS_BULKY_FRAUDS,
    CENSUS_HOT_FRAUDS,
    CENSUS_HOT_LIFT,
)


class CensusRow(NamedTuple):
    """One observed cell's label counts and tags."""

    profile_id: str
    spec_name: str
    cell_key: str
    parent_key: str
    n: int
    n_fraud: int
    rate: float
    lift: float
    share_frauds: float
    share_rows: float
    tag: str


class RegionTotals(NamedTuple):
    """No-alarm region size used as the census denominator."""

    n: int
    n_fraud: int
    rate: float


def region_totals(*, y: np.ndarray) -> RegionTotals:
    """Count rows and frauds on the region label vector."""
    if y.ndim != 1:
        raise ValueError(f"y must be 1-d, got ndim={y.ndim}")
    n = int(y.size)
    if n == 0:
        raise ValueError("region is empty")
    n_fraud = int(np.sum(y.astype(np.int64) == 1))
    return RegionTotals(n=n, n_fraud=n_fraud, rate=float(n_fraud) / float(n))


def tag_cell(*, n_fraud: int, rate: float, region_rate: float) -> str:
    """Return ``bulky_cold``, ``hot``, or ``other`` from pre-registered floors."""
    if n_fraud >= CENSUS_BULKY_FRAUDS and rate <= region_rate:
        return "bulky_cold"
    if n_fraud >= CENSUS_HOT_FRAUDS and rate >= CENSUS_HOT_LIFT * region_rate:
        return "hot"
    return "other"


def census_row(
    *,
    profile_id: str,
    spec_name: str,
    cell_key: str,
    parent_key: str,
    y_cell: np.ndarray,
    totals: RegionTotals,
) -> CensusRow:
    """Label census for one observed cell. Empty cells are not called."""
    n = int(y_cell.size)
    if n == 0:
        raise ValueError(f"cell {cell_key!r} is empty")
    n_fraud = int(np.sum(y_cell.astype(np.int64) == 1))
    rate = float(n_fraud) / float(n)
    lift = rate / totals.rate if totals.rate > 0.0 else float("nan")
    share_frauds = (
        float(n_fraud) / float(totals.n_fraud) if totals.n_fraud > 0 else 0.0
    )
    share_rows = float(n) / float(totals.n)
    tag = tag_cell(n_fraud=n_fraud, rate=rate, region_rate=totals.rate)
    return CensusRow(
        profile_id=profile_id,
        spec_name=spec_name,
        cell_key=cell_key,
        parent_key=parent_key,
        n=n,
        n_fraud=n_fraud,
        rate=rate,
        lift=lift,
        share_frauds=share_frauds,
        share_rows=share_rows,
        tag=tag,
    )


def census_as_dict(row: CensusRow) -> dict[str, object]:
    """Flat record for a census table."""
    return row._asdict()
